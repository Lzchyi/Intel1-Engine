from __future__ import annotations

import unicodedata
from typing import Any

from .structured_data import WeekendContext


RACE_POINTS_BY_POSITION = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}

SPRINT_POINTS_BY_POSITION = {
    1: 8.0,
    2: 7.0,
    3: 6.0,
    4: 5.0,
    5: 4.0,
    6: 3.0,
    7: 2.0,
    8: 1.0,
}


def apply_session_results_to_standings(
    *,
    weekend: WeekendContext,
    driver_standings: list[dict[str, Any]],
    constructor_standings: list[dict[str, Any]],
    session_results: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drivers, constructors, _ = apply_pending_session_results_to_standings(
        weekend=weekend,
        driver_standings=driver_standings,
        constructor_standings=constructor_standings,
        session_results=session_results,
        standings_update_state=None,
        previous_standings_payload=None,
    )
    return drivers, constructors


def apply_pending_session_results_to_standings(
    *,
    weekend: WeekendContext,
    driver_standings: list[dict[str, Any]],
    constructor_standings: list[dict[str, Any]],
    session_results: dict[str, list[dict[str, Any]]],
    standings_update_state: dict[str, Any] | None,
    previous_standings_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    state = normalize_update_state(standings_update_state)
    driver_overlay_needed = standings_need_result_overlay(weekend, driver_standings)
    constructor_overlay_needed = standings_need_result_overlay(weekend, constructor_standings)
    if not driver_overlay_needed and not constructor_overlay_needed:
        return driver_standings, constructor_standings, state

    pending_events = pending_standings_events(weekend, session_results, state)
    if not pending_events:
        previous = standings_from_payload(previous_standings_payload)
        if previous and state_has_weekend_events(state, weekend.weekend_id):
            return previous[0], previous[1], state
        return driver_standings, constructor_standings, state

    previous = standings_from_payload(previous_standings_payload) if state_has_weekend_events(state, weekend.weekend_id) else None
    drivers = [dict(item) for item in (previous[0] if previous else driver_standings)]
    constructors = [dict(item) for item in (previous[1] if previous else constructor_standings)]
    applied = False
    for event in pending_events:
        applied |= apply_classification_points(
            weekend=weekend,
            drivers=drivers,
            constructors=constructors,
            rows=event["rows"],
            points_by_position=event["points_by_position"],
            count_race_win=event["count_race_win"],
            apply_drivers=driver_overlay_needed,
            apply_constructors=constructor_overlay_needed,
        )
        state["applied_events"].append(
            {
                "event_key": event["event_key"],
                "weekend_id": weekend.weekend_id,
                "round": weekend.round_number,
                "session": event["session"],
                "winner": classification_winner(event["rows"]),
                "classified_rows": len(event["rows"]),
            }
        )
    if not applied:
        return driver_standings, constructor_standings, state

    return rank_standings(drivers, "driver"), rank_standings(constructors, "team"), state


def normalize_update_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    events = []
    if isinstance(payload, dict):
        events = [
            dict(item)
            for item in payload.get("applied_events", [])
            if isinstance(item, dict) and item.get("event_key")
        ]
    return {
        "schema_version": "1.0",
        "applied_events": events[-120:],
    }


def pending_standings_events(
    weekend: WeekendContext,
    session_results: dict[str, list[dict[str, Any]]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    applied_keys = {str(item.get("event_key")) for item in state.get("applied_events", []) if isinstance(item, dict)}
    candidates = [
        ("sprint", SPRINT_POINTS_BY_POSITION, False),
        ("race", RACE_POINTS_BY_POSITION, True),
    ]
    events = []
    for session_key, points_by_position, count_race_win in candidates:
        rows = session_results.get(session_key, [])
        event_key = standings_event_key(weekend, session_key)
        if event_key in applied_keys or not classification_ready(rows, points_by_position):
            continue
        events.append(
            {
                "session": session_key,
                "event_key": event_key,
                "rows": rows,
                "points_by_position": points_by_position,
                "count_race_win": count_race_win,
            }
        )
    return events


def standings_event_key(weekend: WeekendContext, session_key: str) -> str:
    round_part = weekend.round_number if weekend.round_number is not None else weekend.weekend_id
    return f"{weekend.year}:{round_part}:{weekend.weekend_id}:{session_key}"


def classification_ready(rows: list[dict[str, Any]], points_by_position: dict[int, float]) -> bool:
    return any(
        safe_int(row.get("position")) > 0
        and is_points_eligible(row)
        and classification_points(row, points_by_position) > 0
        for row in rows
    )


def classification_winner(rows: list[dict[str, Any]]) -> str | None:
    winners = [row for row in rows if safe_int(row.get("position")) == 1]
    if not winners:
        return None
    return str(winners[0].get("driver") or "") or None


def state_has_weekend_events(state: dict[str, Any], weekend_id: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("weekend_id") == weekend_id
        for item in state.get("applied_events", [])
    )


def standings_from_payload(payload: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not isinstance(payload, dict):
        return None
    driver_rows = payload.get("driver_standings")
    constructor_rows = payload.get("constructor_standings")
    if not isinstance(driver_rows, list) and not isinstance(constructor_rows, list):
        return None
    drivers = [driver_standing_from_payload(item) for item in driver_rows or [] if isinstance(item, dict)]
    constructors = [constructor_standing_from_payload(item) for item in constructor_rows or [] if isinstance(item, dict)]
    if not drivers and not constructors:
        return None
    return drivers, constructors


def driver_standing_from_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "driver": item.get("primary_label") or item.get("id") or "Unknown Driver",
        "driver_id": item.get("driver_id") or item.get("id"),
        "team": item.get("team_label") or item.get("secondary_label"),
        "team_id": item.get("team_id"),
        "position": safe_int(item.get("position")) or 99,
        "points": safe_float(item.get("points")),
        "wins": safe_int(item.get("wins")),
        "_source": "previous_overlay",
    }


def constructor_standing_from_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "team": item.get("team_label") or item.get("primary_label") or item.get("id") or "Unknown Team",
        "team_id": item.get("team_id") or item.get("id"),
        "position": safe_int(item.get("position")) or 99,
        "points": safe_float(item.get("points")),
        "wins": safe_int(item.get("wins")),
        "_source": "previous_overlay",
    }


def standings_need_result_overlay(weekend: WeekendContext, standings: list[dict[str, Any]]) -> bool:
    if not standings:
        return True
    sources = {str(item.get("_source")) for item in standings if item.get("_source")}
    if "reference" in sources:
        return True
    rounds = [safe_int(item.get("_round")) for item in standings if safe_int(item.get("_round")) > 0]
    if weekend.round_number and rounds:
        return max(rounds) < weekend.round_number
    return "jolpica" not in sources


def apply_classification_points(
    *,
    weekend: WeekendContext,
    drivers: list[dict[str, Any]],
    constructors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    points_by_position: dict[int, float],
    count_race_win: bool,
    apply_drivers: bool,
    apply_constructors: bool,
) -> bool:
    applied = False
    for row in sorted(rows, key=lambda item: safe_int(item.get("position"))):
        position = safe_int(row.get("position"))
        if position <= 0 or not is_points_eligible(row):
            continue
        points = classification_points(row, points_by_position)
        if points <= 0:
            continue

        driver_name = str(row.get("driver") or "").strip()
        constructor_name = str(row.get("constructor") or row.get("team") or "Unknown").strip() or "Unknown"
        if apply_drivers and driver_name:
            driver = find_or_create_driver(drivers, driver_name, constructor_name, weekend)
            driver["points"] = safe_float(driver.get("points")) + points
            if count_race_win and position == 1:
                driver["wins"] = safe_int(driver.get("wins")) + 1
            applied = True
        if apply_constructors:
            constructor = find_or_create_constructor(constructors, constructor_name, weekend)
            constructor["points"] = safe_float(constructor.get("points")) + points
            if count_race_win and position == 1:
                constructor["wins"] = safe_int(constructor.get("wins")) + 1
            applied = True
    return applied


def classification_points(row: dict[str, Any], points_by_position: dict[int, float]) -> float:
    if "points" in row:
        return max(0.0, safe_float(row.get("points")))
    return points_by_position.get(safe_int(row.get("position")), 0.0)


def is_points_eligible(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    return "disqualified" not in status and "excluded" not in status


def find_or_create_driver(
    drivers: list[dict[str, Any]],
    driver_name: str,
    constructor_name: str,
    weekend: WeekendContext,
) -> dict[str, Any]:
    lookup = driver_lookup(drivers)
    for key in driver_keys(driver_name):
        if key in lookup:
            return lookup[key]

    entry = {
        "driver": driver_name,
        "driver_id": slug_id(driver_name),
        "team": constructor_name,
        "team_id": canonical_constructor_key(constructor_name),
        "position": len(drivers) + 1,
        "points": 0.0,
        "wins": 0,
        "_source": "result_overlay",
        "_round": weekend.round_number,
    }
    drivers.append(entry)
    return entry


def find_or_create_constructor(
    constructors: list[dict[str, Any]],
    constructor_name: str,
    weekend: WeekendContext,
) -> dict[str, Any]:
    target_key = canonical_constructor_key(constructor_name)
    for constructor in constructors:
        keys = {
            canonical_constructor_key(str(constructor.get("team") or "")),
            canonical_constructor_key(str(constructor.get("team_id") or "")),
        }
        if target_key in keys:
            return constructor

    entry = {
        "team": constructor_name,
        "team_id": target_key,
        "position": len(constructors) + 1,
        "points": 0.0,
        "wins": 0,
        "_source": "result_overlay",
        "_round": weekend.round_number,
    }
    constructors.append(entry)
    return entry


def rank_standings(entries: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    for entry in entries:
        entry["_previous_position"] = safe_int(entry.get("position")) or 99
    ranked = sorted(
        entries,
        key=lambda item: (
            -safe_float(item.get("points")),
            -safe_int(item.get("wins")),
            safe_int(item.get("_previous_position")) or 99,
            str(item.get(label_key) or ""),
        ),
    )
    for position, entry in enumerate(ranked, start=1):
        entry["position"] = position
    return ranked


def driver_lookup(drivers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    surname_counts: dict[str, int] = {}
    for driver in drivers:
        keys = driver_keys(str(driver.get("driver") or ""))
        for key in keys:
            if key:
                lookup.setdefault(key, driver)
        if keys:
            surname_counts[keys[-1]] = surname_counts.get(keys[-1], 0) + 1

    for key, count in surname_counts.items():
        if count > 1:
            lookup.pop(key, None)
    return lookup


def driver_keys(name: str) -> list[str]:
    normalized = normalized_text(name)
    tokens = normalized.split()
    keys = [normalized]
    if tokens:
        keys.append(tokens[-1])
    return [key for index, key in enumerate(keys) if key and key not in keys[:index]]


def canonical_constructor_key(name: str) -> str:
    normalized = normalized_text(name)
    if "mercedes" in normalized:
        return "mercedes"
    if "ferrari" in normalized:
        return "ferrari"
    if "mclaren" in normalized:
        return "mclaren"
    if "red bull" in normalized and "racing bulls" not in normalized:
        return "red_bull"
    if "racing bulls" in normalized or "rb f1" in normalized or "visa" in normalized:
        return "racing_bulls"
    if "alpine" in normalized:
        return "alpine"
    if "haas" in normalized:
        return "haas"
    if "williams" in normalized:
        return "williams"
    if "audi" in normalized or "sauber" in normalized or "kick" in normalized:
        return "audi"
    if "cadillac" in normalized:
        return "cadillac"
    if "aston" in normalized:
        return "aston_martin"
    return slug_id(normalized)


def normalized_text(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    chars = [char.lower() if char.isalnum() else " " for char in ascii_text]
    return " ".join("".join(chars).split())


def slug_id(value: str) -> str:
    return normalized_text(value).replace(" ", "_") or "unknown"


def safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
