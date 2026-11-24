import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Literal, Optional

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

FPL_BASE = "https://fantasy.premierleague.com/api"
FPL_TOKEN_URL = "https://account.premierleague.com/as/token"
SESSION_CACHE_FILE = ".fpl_session_cache.json"

TEAM_ID = os.environ["FPL_TEAM_ID"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
FPL_CLIENT_ID = os.environ.get("FPL_CLIENT_ID", "").strip()
FPL_REFRESH_TOKEN = os.environ.get("FPL_REFRESH_TOKEN", "").strip()
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"
FORCE_RUN = os.environ.get("FORCE_RUN", "false").lower() == "true"

VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
    (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3),
]

ACTIVE_BEARER_TOKEN: str = ""


class PlayerPressUpdate(BaseModel):
    player_id: int
    player_name: str
    status_label: Literal[
        "RULED_OUT",
        "MAJOR_DOUBT",
        "LATE_FITNESS_TEST",
        "CONFIRMED_FIT",
        "NO_UPDATE",
    ]
    revised_chance_of_playing: int = Field(
        description="0 if RULED_OUT, 25 if MAJOR_DOUBT, 50-75 if LATE_FITNESS_TEST, 100 if CONFIRMED_FIT."
    )
    manager_quote_summary: str = Field(
        description="1-sentence summary of what the manager said in the latest press conference."
    )


class PressConferenceReport(BaseModel):
    updates: list[PlayerPressUpdate]


class TacticalAudit(BaseModel):
    executive_summary: str = Field(
        description="Concise 2-3 sentence summary of the Gameweek strategy."
    )
    transfer_rationale: list[str] = Field(
        description="Bullet-by-bullet explanation for each recommended transfer or why transfers were rolled."
    )
    chip_rationale: str = Field(
        description="Explanation of why a boost/chip was activated or held."
    )
    captain_rationale: str = Field(
        description="Why the chosen Captain and Vice-Captain maximize expected points."
    )
    risk_flag: Literal["LOW", "MEDIUM", "HIGH"]


def mark_github_state_changed() -> None:
    """Signals GitHub Actions to persist updated cache files even if a later step errors."""
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write("state_changed=true\n")


def resolve_fpl_access_token() -> str:
    """
    Exchanges the OIDC refresh token for a fresh short-lived bearer token and saves
    the newly rotated refresh token immediately to SESSION_CACHE_FILE.
    """
    global ACTIVE_BEARER_TOKEN
    if ACTIVE_BEARER_TOKEN:
        return ACTIVE_BEARER_TOKEN

    if not (FPL_REFRESH_TOKEN and FPL_CLIENT_ID):
        print("Warning: FPL_CLIENT_ID or FPL_REFRESH_TOKEN not set; running in public read-only mode.")
        return ""

    secret_fingerprint = hashlib.sha256(FPL_REFRESH_TOKEN.encode()).hexdigest()[:16]
    token_to_use = FPL_REFRESH_TOKEN
    source = "GitHub Secret"

    # Use rotated refresh token from cache unless the user updated the GitHub Secret
    if os.path.exists(SESSION_CACHE_FILE):
        try:
            with open(SESSION_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("secret_fingerprint") == secret_fingerprint
                and cached.get("refresh_token")
            ):
                token_to_use = cached["refresh_token"]
                source = "rotated session cache"
        except (OSError, json.JSONDecodeError):
            pass

    resp = requests.post(
        FPL_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": FPL_CLIENT_ID,
            "refresh_token": token_to_use,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"FPL OIDC token exchange failed ({resp.status_code}) using {source}: {resp.text}"
        )

    payload = resp.json()
    ACTIVE_BEARER_TOKEN = payload["access_token"].strip()
    rotated_refresh = payload.get("refresh_token", token_to_use).strip()

    with open(SESSION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "secret_fingerprint": secret_fingerprint,
                "refresh_token": rotated_refresh,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    mark_github_state_changed()
    print(f"Authenticated via {source} (new rotated refresh token persisted).")
    return ACTIVE_BEARER_TOKEN


def get_auth_headers() -> dict:
    headers = {"User-Agent": "FPL-AutoPilot/2.0"}
    token = resolve_fpl_access_token()
    if token:
        headers["X-Api-Authorization"] = f"Bearer {token}"
    return headers


def fetch_fpl_data() -> tuple[dict, list[dict]]:
    bootstrap = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=20).json()
    fixtures = requests.get(f"{FPL_BASE}/fixtures/", timeout=20).json()
    return bootstrap, fixtures


def check_deadline_window(bootstrap: dict) -> tuple[bool, int, str, float]:
    """Returns (should_run, next_gw, deadline_iso, minutes_remaining)."""
    now = datetime.now(timezone.utc)
    next_event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if not next_event:
        return False, 0, "", 0.0

    deadline_str = next_event["deadline_time"]
    deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
    mins_left = (deadline_dt - now).total_seconds() / 60.0

    # Executes in the hourly run falling 30 to 90 minutes before the GW deadline
    in_window = 30.0 <= mins_left <= 90.0
    return in_window, next_event["id"], deadline_str, mins_left


def score_horizon_xp(base_rate: float, horizon_fdrs: dict[int, list[int]], chance: int) -> tuple[float, float]:
    """Computes (xp_next, xp_4gw) from base rate, fixture difficulties, and availability %."""
    availability = max(0.0, min(1.0, chance / 100.0))
    gw_xp = {}
    for idx, gw in enumerate(sorted(horizon_fdrs.keys())):
        fdrs = horizon_fdrs[gw]
        if not fdrs:
            gw_xp[gw] = 0.0
            continue
        fixture_mult = sum(max(0.65, 1.45 - 0.15 * d) for d in fdrs)
        avail_mult = availability if idx == 0 else min(1.0, availability + 0.25)
        gw_xp[gw] = round(base_rate * fixture_mult * avail_mult, 2)

    first_gw = min(horizon_fdrs.keys())
    return gw_xp.get(first_gw, 0.0), round(sum(gw_xp.values()), 2)


def compute_player_projections(bootstrap: dict, fixtures: list[dict], next_gw: int) -> dict[int, dict]:
    """Calculates baseline 1-GW and 4-GW xP from official FPL API metrics."""
    teams_by_id = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    horizon = set(range(next_gw, min(next_gw + 4, 39)))
    team_fixtures: dict[int, dict[int, list[int]]] = {
        t["id"]: {gw: [] for gw in horizon} for t in bootstrap["teams"]
    }

    for f in fixtures:
        gw = f.get("event")
        if gw in horizon:
            team_fixtures[f["team_h"]][gw].append(f["team_a_difficulty"])
            team_fixtures[f["team_a"]][gw].append(f["team_h_difficulty"])

    projections = {}
    for p in bootstrap["elements"]:
        pid = p["id"]
        chance = p.get("chance_of_playing_next_round")
        if chance is None:
            chance = 100 if p.get("status") == "a" else 0

        form = float(p.get("form") or 0.0)
        ppg = float(p.get("points_per_game") or 0.0)
        ict = float(p.get("ict_index") or 0.0) / max(next_gw - 1, 1)
        base_rate = (0.50 * form) + (0.35 * ppg) + (0.15 * min(ict, 12.0))

        horizon_fdrs = team_fixtures[p["team"]]
        xp_next, xp_4gw = score_horizon_xp(base_rate, horizon_fdrs, chance)
        next_fdrs = horizon_fdrs.get(next_gw, [])

        projections[pid] = {
            "id": pid,
            "name": p["web_name"],
            "full_name": f"{p['first_name']} {p['second_name']}",
            "team": p["team"],
            "club": teams_by_id.get(p["team"], ""),
            "pos": p["element_type"],
            "now_cost": p["now_cost"],
            "chance": chance,
            "news": p.get("news", ""),
            "press_note": "",
            "base_rate": base_rate,
            "horizon_fdrs": horizon_fdrs,
            "xp_next": xp_next,
            "xp_4gw": xp_4gw,
            "next_fixture_count": len(next_fdrs),
            "next_avg_fdr": round(sum(next_fdrs) / len(next_fdrs), 2) if next_fdrs else 5.0,
        }
    return projections


def load_current_squad(bootstrap: dict, next_gw: int) -> tuple[list[dict], int, int, list[str]]:
    """Loads live squad, exact selling prices, bank, free transfers, and used chips."""
    headers = get_auth_headers()
    if "X-Api-Authorization" in headers:
        resp = requests.get(f"{FPL_BASE}/my-team/{TEAM_ID}/", headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        picks = data["picks"]
        bank = data["transfers"]["bank"]
        free_transfers = data["transfers"]["limit"]
        used_chips = [
            c["name"] for c in data.get("chips", []) if c.get("status_for_entry") == "played"
        ]
        return picks, bank, free_transfers, used_chips

    # Public fallback if OIDC secrets are not configured yet
    prev_gw = max(1, next_gw - 1)
    picks_data = requests.get(
        f"{FPL_BASE}/entry/{TEAM_ID}/event/{prev_gw}/picks/", timeout=20
    ).json()
    history_data = requests.get(f"{FPL_BASE}/entry/{TEAM_ID}/history/", timeout=20).json()
    elements_by_id = {e["id"]: e for e in bootstrap["elements"]}

    picks = []
    for p in picks_data["picks"]:
        p["selling_price"] = elements_by_id[p["element"]]["now_cost"]
        picks.append(p)

    bank = picks_data["entry_history"]["bank"]
    used_chips = [c["name"] for c in history_data.get("chips", [])]
    return picks, bank, 1, used_chips


def check_press_conference_injuries(
    squad_ids: list[int],
    proj: dict[int, dict],
    next_gw: int,
) -> tuple[list[PlayerPressUpdate], list[dict[str, str]]]:
    """Checks Premier League manager press conferences via Gemini Google Search grounding."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    top_targets = sorted(
        [p for pid, p in proj.items() if pid not in squad_ids and p["xp_4gw"] > 0],
        key=lambda x: x["xp_4gw"],
        reverse=True,
    )[:15]

    watchlist = [proj[pid] for pid in squad_ids] + top_targets
    watchlist_payload = [
        {
            "player_id": p["id"],
            "name": p["full_name"],
            "club": p["club"],
            "current_fpl_chance": p["chance"],
            "current_fpl_news": p["news"],
        }
        for p in watchlist
    ]

    search_prompt = (
        f"Today's date is {today_utc} (ahead of Fantasy Premier League Gameweek {next_gw}). "
        "Search recent Premier League manager press conferences and reliable team news from the "
        "last 48 hours for injury updates, late fitness tests, illness, or confirmed returns "
        "affecting any of these players:\n\n"
        f"{json.dumps(watchlist_payload)}\n\n"
        "Report ONLY verified updates from manager press conferences or official club announcements. "
        "Explicitly distinguish between players RULED OUT (0%), MAJOR DOUBTS (25%), "
        "LATE FITNESS TESTS / 50-50 (50-75%), and players previously flagged who were CONFIRMED FIT (100%)."
    )

    grounded_resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=search_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=1.0,
        ),
    )

    sources: list[dict[str, str]] = []
    seen_uris = set()
    candidate = grounded_resp.candidates[0] if grounded_resp.candidates else None
    gm = getattr(candidate, "grounding_metadata", None)
    for chunk in getattr(gm, "grounding_chunks", None) or []:
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None)
        title = getattr(web, "title", None) or uri
        if uri and uri not in seen_uris:
            seen_uris.add(uri)
            sources.append({"title": title, "uri": uri})

    parse_prompt = (
        "Convert the following Premier League press conference injury report into structured JSON. "
        "Only include players from the watchlist who have an actionable injury, fitness, or return update "
        "(omit players with NO_UPDATE).\n\n"
        f"Watchlist:\n{json.dumps(watchlist_payload)}\n\n"
        f"Grounded Press Conference Findings:\n{grounded_resp.text}"
    )

    structured_resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parse_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PressConferenceReport,
        ),
    )

    report: PressConferenceReport = structured_resp.parsed
    actionable = [u for u in report.updates if u.status_label != "NO_UPDATE"]
    return actionable, sources[:8]


def apply_press_conference_overrides(
    proj: dict[int, dict],
    updates: list[PlayerPressUpdate],
) -> list[str]:
    """Updates player availability and recalculates xP based on press conference findings."""
    change_log = []
    for u in updates:
        if u.player_id not in proj:
            continue
        p = proj[u.player_id]
        old_chance = p["chance"]
        new_chance = max(0, min(100, u.revised_chance_of_playing))
        p["press_note"] = f"[{u.status_label}] {u.manager_quote_summary}"

        if old_chance != new_chance:
            old_xp = p["xp_next"]
            p["chance"] = new_chance
            p["xp_next"], p["xp_4gw"] = score_horizon_xp(
                p["base_rate"], p["horizon_fdrs"], new_chance
            )
            change_log.append(
                f"{p['name']} ({p['club']}): `{old_chance}%` -> `{new_chance}%` "
                f"(`{old_xp}` -> `{p['xp_next']} xP`) — {u.manager_quote_summary}"
            )
        else:
            change_log.append(
                f"{p['name']} ({p['club']}): Confirmed `{new_chance}%` ({u.status_label}) — {u.manager_quote_summary}"
            )
    return change_log


def select_best_starting_xi(squad_ids: list[int], proj: dict[int, dict]) -> tuple[list[int], list[int]]:
    """Finds the legal FPL starting XI maximizing next-GW xP and orders the bench."""
    gks = sorted([pid for pid in squad_ids if proj[pid]["pos"] == 1], key=lambda x: proj[x]["xp_next"], reverse=True)
    defs = sorted([pid for pid in squad_ids if proj[pid]["pos"] == 2], key=lambda x: proj[x]["xp_next"], reverse=True)
    mids = sorted([pid for pid in squad_ids if proj[pid]["pos"] == 3], key=lambda x: proj[x]["xp_next"], reverse=True)
    fwds = sorted([pid for pid in squad_ids if proj[pid]["pos"] == 4], key=lambda x: proj[x]["xp_next"], reverse=True)

    best_score = -1.0
    best_xi: list[int] = []

    for d_cnt, m_cnt, f_cnt in VALID_FORMATIONS:
        if len(defs) >= d_cnt and len(mids) >= m_cnt and len(fwds) >= f_cnt:
            xi = [gks[0]] + defs[:d_cnt] + mids[:m_cnt] + fwds[:f_cnt]
            score = sum(proj[pid]["xp_next"] for pid in xi)
            if score > best_score:
                best_score = score
                best_xi = xi

    bench_gk = [gks[1]]
    bench_outfield = sorted(
        [pid for pid in squad_ids if pid not in best_xi and pid not in bench_gk],
        key=lambda x: proj[x]["xp_next"],
        reverse=True,
    )
    return best_xi, bench_gk + bench_outfield


def optimize_transfers_and_chips(
    picks: list[dict],
    bank: int,
    free_transfers: int,
    used_chips: list[str],
    proj: dict[int, dict],
) -> dict:
    """Enforces Free Transfers + MAX ONE -4 hit rule and evaluates optimal chip timing."""
    squad_ids = [p["element"] for p in picks]
    sell_prices = {p["element"]: p.get("selling_price", proj[p["element"]]["now_cost"]) for p in picks}
    team_counts: dict[int, int] = {}
    for pid in squad_ids:
        t = proj[pid]["team"]
        team_counts[t] = team_counts.get(t, 0) + 1

    candidate_swaps = []
    for out_id in squad_ids:
        out_p = proj[out_id]
        budget = bank + sell_prices[out_id]
        for in_id, in_p in proj.items():
            if in_id in squad_ids or in_p["pos"] != out_p["pos"]:
                continue
            if in_p["now_cost"] > budget or in_p["chance"] < 75:
                continue
            if in_p["team"] != out_p["team"] and team_counts.get(in_p["team"], 0) >= 3:
                continue

            gain_4gw = round(in_p["xp_4gw"] - out_p["xp_4gw"], 2)
            if gain_4gw > 1.5:
                candidate_swaps.append({
                    "out_id": out_id,
                    "out_name": out_p["name"],
                    "out_price": sell_prices[out_id],
                    "out_chance": out_p["chance"],
                    "in_id": in_id,
                    "in_name": in_p["name"],
                    "in_price": in_p["now_cost"],
                    "gain_4gw": gain_4gw,
                    "team_out": out_p["team"],
                    "team_in": in_p["team"],
                })

    candidate_swaps.sort(key=lambda x: x["gain_4gw"], reverse=True)

    approved_transfers = []
    used_out = set()
    used_in = set()
    running_bank = bank
    running_teams = dict(team_counts)

    # Hard cap: all available Free Transfers + AT MOST 1 paid hit (-4 pts)
    max_total_moves = free_transfers + 1

    for swap in candidate_swaps:
        if len(approved_transfers) >= max_total_moves:
            break
        if swap["out_id"] in used_out or swap["in_id"] in used_in:
            continue

        price_diff = swap["in_price"] - swap["out_price"]
        if price_diff > running_bank:
            continue
        if swap["team_in"] != swap["team_out"] and running_teams.get(swap["team_in"], 0) >= 3:
            continue

        is_paid_hit = len(approved_transfers) >= free_transfers
        if is_paid_hit and not (swap["gain_4gw"] >= 6.5 or swap["out_chance"] < 75):
            continue

        approved_transfers.append({**swap, "cost": -4 if is_paid_hit else 0})
        used_out.add(swap["out_id"])
        used_in.add(swap["in_id"])
        running_bank -= price_diff
        running_teams[swap["team_out"]] -= 1
        running_teams[swap["team_in"]] = running_teams.get(swap["team_in"], 0) + 1

    new_squad_ids = [
        next((t["in_id"] for t in approved_transfers if t["out_id"] == pid), pid)
        for pid in squad_ids
    ]
    starting_xi, bench = select_best_starting_xi(new_squad_ids, proj)

    ranked_xi = sorted(starting_xi, key=lambda x: proj[x]["xp_next"], reverse=True)
    captain_id = ranked_xi[0]
    vice_captain_id = ranked_xi[1]

    selected_chip: Optional[str] = None
    injured_count = sum(1 for pid in new_squad_ids if proj[pid]["chance"] < 75)
    blank_starters = sum(1 for pid in starting_xi if proj[pid]["next_fixture_count"] == 0)
    dgw_squad_count = sum(1 for pid in new_squad_ids if proj[pid]["next_fixture_count"] >= 2)
    cap = proj[captain_id]

    if "freehit" not in used_chips and blank_starters >= 4:
        selected_chip = "freehit"
    elif "wildcard" not in used_chips and injured_count >= 4:
        selected_chip = "wildcard"
    elif "3xc" not in used_chips and cap["next_fixture_count"] >= 2 and cap["next_avg_fdr"] <= 2.2:
        selected_chip = "3xc"
    elif "bboost" not in used_chips and injured_count == 0 and dgw_squad_count >= 10:
        selected_chip = "bboost"

    hit_cost = sum(t["cost"] for t in approved_transfers)
    assert hit_cost in (0, -4), f"Safety invariant violated: hit_cost={hit_cost}"

    return {
        "transfers": approved_transfers,
        "hit_cost": hit_cost,
        "chip": selected_chip,
        "starting_xi": starting_xi,
        "bench": bench,
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
    }


def generate_gemini_audit(
    next_gw: int,
    plan: dict,
    proj: dict[int, dict],
    press_changes: list[str],
) -> TacticalAudit:
    client = genai.Client(api_key=GEMINI_API_KEY)
    summary_payload = {
        "gameweek": next_gw,
        "press_conference_updates": press_changes,
        "hit_cost": plan["hit_cost"],
        "chip": plan["chip"] or "NONE",
        "transfers": plan["transfers"],
        "captain": {k: v for k, v in proj[plan["captain_id"]].items() if k != "horizon_fdrs"},
        "vice_captain": {k: v for k, v in proj[plan["vice_captain_id"]].items() if k != "horizon_fdrs"},
        "starting_xi": [
            {k: v for k, v in proj[pid].items() if k != "horizon_fdrs"}
            for pid in plan["starting_xi"]
        ],
        "bench": [
            {k: v for k, v in proj[pid].items() if k != "horizon_fdrs"}
            for pid in plan["bench"]
        ],
    }

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "You are an elite Fantasy Premier League analyst. Audit this automated Gameweek plan "
            "(which already incorporates live press conference injury updates) and explain the "
            "tactical reasoning behind transfers, hit cost (capped at -4 max), captaincy, and boosts:\n\n"
            f"{json.dumps(summary_payload)}"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TacticalAudit,
        ),
    )
    return response.parsed


def execute_fpl_moves(next_gw: int, plan: dict) -> str:
    if DRY_RUN:
        return "DRY_RUN=true — Plan simulated and logged without modifying live FPL team."

    headers = get_auth_headers()
    if "X-Api-Authorization" not in headers:
        return "SKIPPED — DRY_RUN=false set, but no valid FPL token could be resolved."

    headers["Content-Type"] = "application/json"

    if plan["transfers"] or plan["chip"] in ("wildcard", "freehit"):
        transfer_payload = {
            "chip": plan["chip"],
            "entry": int(TEAM_ID),
            "event": int(next_gw),
            "transfers": [
                {
                    "element_in": t["in_id"],
                    "element_out": t["out_id"],
                    "purchase_price": t["in_price"],
                    "selling_price": t["out_price"],
                }
                for t in plan["transfers"]
            ],
        }
        r_tx = requests.post(f"{FPL_BASE}/transfers/", headers=headers, json=transfer_payload, timeout=20)
        r_tx.raise_for_status()

    ordered_15 = plan["starting_xi"] + plan["bench"]
    lineup_payload = {
        "picks": [
            {
                "element": pid,
                "position": idx + 1,
                "is_captain": pid == plan["captain_id"],
                "is_vice_captain": pid == plan["vice_captain_id"],
            }
            for idx, pid in enumerate(ordered_15)
        ]
    }
    if plan["chip"] in ("3xc", "bboost"):
        lineup_payload["chip"] = plan["chip"]

    r_lineup = requests.post(f"{FPL_BASE}/my-team/{TEAM_ID}/", headers=headers, json=lineup_payload, timeout=20)
    r_lineup.raise_for_status()
    return "SUCCESS — Transfers, captaincy, and chip settings committed to FPL."


def write_outputs(
    next_gw: int,
    deadline_str: str,
    plan: dict,
    proj: dict[int, dict],
    audit: TacticalAudit,
    press_changes: list[str],
    sources: list[dict[str, str]],
    exec_status: str,
):
    press_lines = [f"- {line}" for line in press_changes] or [
        "- No new injury doubts or late fitness overrides reported across squad or top targets."
    ]
    source_lines = [f"- [{s['title']}]({s['uri']})" for s in sources] or ["- No external web sources cited."]

    tx_lines = [
        f"- **OUT:** {t['out_name']} -> **IN:** {t['in_name']} (`+{t['gain_4gw']} xP`, cost: `{t['cost']} pts`)"
        for t in plan["transfers"]
    ] or ["- No transfers made (rolling Free Transfer)."]

    xi_names = ", ".join(proj[pid]["name"] for pid in plan["starting_xi"])
    bench_names = ", ".join(proj[pid]["name"] for pid in plan["bench"])

    report = f"""## FPL Auto-Pilot — Gameweek {next_gw} Report
**Deadline (UTC):** `{deadline_str}` | **Status:** `{exec_status}`

### Press Conference Team News (Google Search Grounded)
{chr(10).join(press_lines)}

**Verified Sources:**
{chr(10).join(source_lines)}

### Tactical Summary
{audit.executive_summary}

### Transfers & Boosts (`{plan['hit_cost']} pts` hit | Chip: `{plan['chip'] or 'NONE'}`)
{chr(10).join(tx_lines)}

- **Chip Decision:** {audit.chip_rationale}
- **Captain:** **{proj[plan['captain_id']]['name']}** (`{proj[plan['captain_id']]['xp_next']} xP`) | **Vice:** **{proj[plan['vice_captain_id']]['name']}**
- **Starting XI:** {xi_names}
- **Bench:** {bench_names}
"""
    print(report)

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(report)

    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": report[:1950]}, timeout=10)


def main():
    bootstrap, fixtures = fetch_fpl_data()
    should_run, next_gw, deadline_str, mins_left = check_deadline_window(bootstrap)

    if not should_run and not FORCE_RUN:
        print(f"Skipping: Next GW {next_gw} deadline is in {mins_left:.1f} mins (outside 30-90m window).")
        return

    marker_file = f".fpl_ran_gw_{next_gw}"
    if os.path.exists(marker_file) and not FORCE_RUN:
        print(f"Skipping: Gameweek {next_gw} has already been processed.")
        return

    proj = compute_player_projections(bootstrap, fixtures, next_gw)
    picks, bank, free_transfers, used_chips = load_current_squad(bootstrap, next_gw)

    squad_ids = [p["element"] for p in picks]
    press_updates, sources = check_press_conference_injuries(squad_ids, proj, next_gw)
    press_changes = apply_press_conference_overrides(proj, press_updates)

    plan = optimize_transfers_and_chips(picks, bank, free_transfers, used_chips, proj)
    audit = generate_gemini_audit(next_gw, plan, proj, press_changes)
    exec_status = execute_fpl_moves(next_gw, plan)
    write_outputs(next_gw, deadline_str, plan, proj, audit, press_changes, sources, exec_status)

    with open(marker_file, "w", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    mark_github_state_changed()


if __name__ == "__main__":
    main()
