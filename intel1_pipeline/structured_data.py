from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

from .http import FetchError, fetch_json
from .race_calendar import load_static_current_weekend
from .reference_standings import reference_constructor_standings, reference_driver_standings
from .time_utils import parse_iso_date, utc_now


JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE = "https://api.openf1.org/v1"

SESSION_RESULT_KEYS = ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"]


@dataclass
class WeekendContext:
    weekend_id: str
    grand_prix_name: str
    circuit_name: str
    country: str
    year: int
    round_number: int | None
    race_date: str | None
    is_sprint_weekend: bool
    stage: str
    next_relevant_session: str | None
    session_schedule: list[dict[str, str]]


def load_current_weekend(force_weekend_id: str | None, force_stage: str | None) -> WeekendContext:
    static_weekend = load_static_current_weekend(force_weekend_id, force_stage)
    # Prefer the verified static calendar when exact session times are known.
    # For newly added events whose session timetable is still TBD,
    # fall through to the live calendar provider so official times can take over without
    # inventing a provisional timetable.
    if static_weekend is not None and static_weekend.session_schedule:
        return static_weekend

    try:
        payload = fetch_json(f"{JOLPICA_BASE}/current.json")
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    except (FetchError, ValueError, KeyError):
        races = []

    race = select_current_race(races)
    year = int(payload.get("MRData", {}).get("RaceTable", {}).get("season", utc_now().year)) if races else utc_now().year
    if not race:
        if static_weekend is not None:
            return static_weekend
        weekend_id = force_weekend_id or f"{year}-unknown-weekend"
        return WeekendContext(
            weekend_id=weekend_id,
            grand_prix_name="F1 Weekend",
            circuit_name="Unknown circuit",
            country="Unknown",
            year=year,
            round_number=None,
            race_date=None,
            is_sprint_weekend=False,
            stage=force_stage or "pre_weekend",
            next_relevant_session=None,
            session_schedule=[],
        )

    round_number = int(race["round"]) if race.get("round") else None
    race_name = race.get("raceName", "F1 Weekend")
    weekend_id = force_weekend_id or make_weekend_id(year, round_number, race_name)
    schedule = make_schedule(race)
    return WeekendContext(
        weekend_id=weekend_id,
        grand_prix_name=race_name,
        circuit_name=race.get("Circuit", {}).get("circuitName", "Unknown circuit"),
        country=race.get("Circuit", {}).get("Location", {}).get("country", "Unknown"),
        year=year,
        round_number=round_number,
        race_date=race.get("date"),
        is_sprint_weekend=any(item["session"] == "sprint" for item in schedule),
        stage=force_stage or infer_stage(schedule),
        next_relevant_session=next_session(schedule),
        session_schedule=schedule,
    )


def load_driver_standings() -> list[dict[str, Any]]:
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/current/driverStandings.json")
    except (FetchError, ValueError):
        return standings_with_source(reference_driver_standings(utc_now().year), "reference")
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return standings_with_source(reference_driver_standings(utc_now().year), "reference")
    standings_list = lists[0]
    standings_round = safe_optional_int(standings_list.get("round"))
    standings = standings_list.get("DriverStandings", [])
    drivers: list[dict[str, Any]] = []
    for standing in standings:
        driver = standing.get("Driver", {})
        constructors = standing.get("Constructors", [])
        constructor = constructors[0] if constructors else {}
        given = driver.get("givenName", "")
        family = driver.get("familyName", "")
        name = " ".join(part for part in [given, family] if part).strip() or driver.get("driverId", "TBD")
        drivers.append(
            {
                "driver": name,
                "driver_id": driver.get("driverId"),
                "team": constructor.get("name", "Unknown"),
                "team_id": constructor.get("constructorId"),
                "position": int(standing.get("position", 99)),
                "points": float(standing.get("points", 0)),
                "wins": int(standing.get("wins", 0)),
                "_source": "jolpica",
                "_round": standings_round,
            }
        )
    return drivers or standings_with_source(reference_driver_standings(utc_now().year), "reference")


def load_constructor_standings() -> list[dict[str, Any]]:
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/current/constructorStandings.json")
    except (FetchError, ValueError):
        return standings_with_source(reference_constructor_standings(utc_now().year), "reference")
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return standings_with_source(reference_constructor_standings(utc_now().year), "reference")
    standings_list = lists[0]
    standings_round = safe_optional_int(standings_list.get("round"))
    standings = standings_list.get("ConstructorStandings", [])
    constructors: list[dict[str, Any]] = []
    for standing in standings:
        constructor = standing.get("Constructor", {})
        constructors.append(
            {
                "team": constructor.get("name", "Unknown"),
                "team_id": constructor.get("constructorId"),
                "position": int(standing.get("position", 99)),
                "points": float(standing.get("points", 0)),
                "wins": int(standing.get("wins", 0)),
                "_source": "jolpica",
                "_round": standings_round,
            }
        )
    return constructors or standings_with_source(reference_constructor_standings(utc_now().year), "reference")


def standings_with_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**item, "_source": source} for item in rows]


def load_historical_race_data(year: int | None = None) -> dict[str, Any]:
    season = year or utc_now().year
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/{season}/results.json?limit=2000")
    except (FetchError, ValueError):
        return {
            "season": season,
            "updated_at": utc_now().isoformat(),
            "races": [],
            "source": "jolpica",
            "fetch_status": "failed",
        }
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    normalized = []
    for race in races:
        results = race.get("Results", [])
        podium = [normalize_result(result) for result in results[:3]]
        winner = podium[0] if podium else None
        normalized.append(
            {
                "season": int(race.get("season", season)),
                "round": int(race.get("round", 0) or 0),
                "race_name": race.get("raceName", "Unknown Grand Prix"),
                "date": race.get("date"),
                "circuit_name": race.get("Circuit", {}).get("circuitName"),
                "country": race.get("Circuit", {}).get("Location", {}).get("country"),
                "winner": winner,
                "podium": podium,
                "classified_results": [normalize_result(result) for result in results],
            }
        )
    return {
        "schema_version": "1.0",
        "season": season,
        "updated_at": utc_now().isoformat(),
        "races": normalized,
        "source": "jolpica",
        "fetch_status": "success",
    }


def load_session_results(weekend: WeekendContext) -> dict[str, list[dict[str, Any]]]:
    results = {key: [] for key in SESSION_RESULT_KEYS}
    if not weekend.round_number:
        return results

    jolpica_loaders = {
        "qualifying": load_jolpica_qualifying_results,
        "sprint": load_jolpica_sprint_results,
        "race": load_jolpica_race_results,
    }
    for session_key, loader in jolpica_loaders.items():
        results[session_key] = loader(weekend.year, weekend.round_number)

    openf1_session_names = {
        "fp1": "Practice 1",
        "fp2": "Practice 2",
        "fp3": "Practice 3",
        "sprint_qualifying": "Sprint Qualifying",
        "sprint": "Sprint",
        "qualifying": "Qualifying",
        "race": "Race",
    }
    for session_key, session_name in openf1_session_names.items():
        if results[session_key]:
            continue
        results[session_key] = load_openf1_session_result(weekend, session_name)

    return results


def load_jolpica_qualifying_results(year: int, round_number: int) -> list[dict[str, Any]]:
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/{year}/{round_number}/qualifying.json")
    except (FetchError, ValueError):
        return []
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    rows = races[0].get("QualifyingResults", [])
    return [normalize_jolpica_qualifying_row(row) for row in rows]


def load_jolpica_sprint_results(year: int, round_number: int) -> list[dict[str, Any]]:
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/{year}/{round_number}/sprint.json")
    except (FetchError, ValueError):
        return []
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    rows = races[0].get("SprintResults", [])
    return [normalize_jolpica_classification_row(row) for row in rows]


def load_jolpica_race_results(year: int, round_number: int) -> list[dict[str, Any]]:
    try:
        payload = fetch_json(f"{JOLPICA_BASE}/{year}/{round_number}/results.json")
    except (FetchError, ValueError):
        return []
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    rows = races[0].get("Results", [])
    return [normalize_jolpica_classification_row(row) for row in rows]


def normalize_jolpica_qualifying_row(row: dict[str, Any]) -> dict[str, Any]:
    driver = row.get("Driver", {})
    constructor = row.get("Constructor", {})
    return session_result_row(
        position=safe_int(row.get("position")),
        driver=driver_name(driver),
        constructor=constructor.get("name") or "Unknown",
        time_or_gap=row.get("Q3") or row.get("Q2") or row.get("Q1") or "",
        laps=None,
        status="classified",
        source="Jolpica",
        is_official=True,
    )


def normalize_jolpica_classification_row(row: dict[str, Any]) -> dict[str, Any]:
    driver = row.get("Driver", {})
    constructor = row.get("Constructor", {})
    result = session_result_row(
        position=safe_int(row.get("position")),
        driver=driver_name(driver),
        constructor=constructor.get("name") or "Unknown",
        time_or_gap=jolpica_time_or_status(row),
        laps=safe_optional_int(row.get("laps")),
        status=str(row.get("status") or ""),
        source="Jolpica",
        is_official=True,
    )
    result["points"] = safe_float(row.get("points"))
    return result


def jolpica_time_or_status(row: dict[str, Any]) -> str:
    if isinstance(row.get("Time"), dict):
        return str(row["Time"].get("time") or "")
    return str(row.get("status") or "")


def load_openf1_session_result(weekend: WeekendContext, session_name: str) -> list[dict[str, Any]]:
    session = openf1_session(weekend, session_name)
    if not session:
        return []
    session_key = session.get("session_key")
    if session_key is None:
        return []
    try:
        result_payload = fetch_json(f"{OPENF1_BASE}/session_result?{urlencode({'session_key': session_key})}")
        drivers_payload = fetch_json(f"{OPENF1_BASE}/drivers?{urlencode({'session_key': session_key})}")
    except (FetchError, ValueError):
        return []
    if not isinstance(result_payload, list):
        return []
    drivers = openf1_driver_lookup(drivers_payload if isinstance(drivers_payload, list) else [])
    rows = [normalize_openf1_session_result_row(row, drivers) for row in result_payload]
    return [row for row in rows if row["position"] > 0]


def openf1_session(weekend: WeekendContext, session_name: str) -> dict[str, Any] | None:
    query = {
        "year": weekend.year,
        "session_name": session_name,
    }
    if weekend.country and weekend.country != "Unknown":
        query["country_name"] = weekend.country
    try:
        payload = fetch_json(f"{OPENF1_BASE}/sessions?{urlencode(query)}")
    except (FetchError, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    sessions = [item for item in payload if item.get("session_name") == session_name]
    if not sessions:
        sessions = payload
    race_date = parse_iso_date(weekend.race_date)
    if race_date:
        window_start = race_date - timedelta(days=3)
        window_end = race_date + timedelta(days=1)
        window_matches = [
            item
            for item in sessions
            if (date_start := parse_iso_date(item.get("date_start"))) and window_start <= date_start <= window_end
        ]
        if window_matches:
            sessions = window_matches
    return sorted(sessions, key=lambda item: str(item.get("date_start") or ""))[-1] if sessions else None


def openf1_driver_lookup(rows: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    lookup: dict[int, dict[str, str]] = {}
    for row in rows:
        number = safe_int(row.get("driver_number"))
        if not number:
            continue
        lookup[number] = {
            "driver": str(row.get("full_name") or row.get("broadcast_name") or row.get("name_acronym") or number),
            "constructor": str(row.get("team_name") or "Unknown"),
        }
    return lookup


def normalize_openf1_session_result_row(row: dict[str, Any], drivers: dict[int, dict[str, str]]) -> dict[str, Any]:
    driver_number = safe_int(row.get("driver_number"))
    driver = drivers.get(driver_number, {})
    return session_result_row(
        position=safe_int(row.get("position")),
        driver=driver.get("driver") or str(driver_number or "Unknown"),
        constructor=driver.get("constructor") or "Unknown",
        time_or_gap=openf1_time_or_gap(row),
        laps=safe_optional_int(row.get("number_of_laps")),
        status=str(row.get("status") or ""),
        source="OpenF1",
        is_official=True,
    )


def openf1_time_or_gap(row: dict[str, Any]) -> str:
    duration = last_non_empty(row.get("duration"))
    gap = last_non_empty(row.get("gap_to_leader"))
    if gap not in (None, "", 0, 0.0):
        return format_gap(gap)
    if duration not in (None, ""):
        return format_duration(duration)
    return ""


def last_non_empty(value: Any) -> Any:
    if isinstance(value, list):
        for item in reversed(value):
            if item not in (None, ""):
                return item
        return None
    return value


def format_duration(value: Any) -> str:
    if isinstance(value, (int, float)):
        minutes = int(value // 60)
        seconds = value - minutes * 60
        return f"{minutes}:{seconds:06.3f}" if minutes else f"{seconds:.3f}s"
    return str(value)


def format_gap(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"+{value:.3f}s" if value > 0 else ""
    text = str(value)
    return text if text.startswith("+") or not text else f"+{text}"


def session_result_row(
    *,
    position: int,
    driver: str,
    constructor: str,
    time_or_gap: str,
    laps: int | None,
    status: str,
    source: str,
    is_official: bool,
) -> dict[str, Any]:
    return {
        "position": position,
        "driver": driver,
        "constructor": constructor,
        "time_or_gap": time_or_gap,
        "laps": laps,
        "status": status,
        "source": source,
        "is_official": is_official,
    }


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    driver = result.get("Driver", {})
    constructor = result.get("Constructor", {})
    given = driver.get("givenName", "")
    family = driver.get("familyName", "")
    name = " ".join(part for part in [given, family] if part).strip() or driver.get("driverId", "TBD")
    return {
        "position": int(result.get("position", 0) or 0),
        "grid": int(result.get("grid", 0) or 0),
        "driver": name,
        "driver_id": driver.get("driverId"),
        "constructor": constructor.get("name"),
        "constructor_id": constructor.get("constructorId"),
        "status": result.get("status"),
        "points": float(result.get("points", 0) or 0),
    }


def driver_name(driver: dict[str, Any]) -> str:
    given = driver.get("givenName", "")
    family = driver.get("familyName", "")
    return " ".join(part for part in [given, family] if part).strip() or driver.get("driverId", "TBD")


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def safe_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = safe_int(value)
    return parsed if parsed > 0 else None


def select_current_race(races: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not races:
        return None
    today = utc_now().date()
    dated = [(parse_iso_date(race.get("date")), race) for race in races]
    upcoming = [race for date, race in dated if date and date.date() >= today]
    if upcoming:
        return upcoming[0]
    return races[-1]


def make_schedule(race: dict[str, Any]) -> list[dict[str, str]]:
    sessions = [
        ("fp1", "Practice 1", race.get("FirstPractice", {}).get("date")),
        ("fp2", "Practice 2", race.get("SecondPractice", {}).get("date")),
        ("fp3", "Practice 3", race.get("ThirdPractice", {}).get("date")),
        ("sprint_qualifying", "Sprint Qualifying", race.get("SprintQualifying", {}).get("date")),
        ("sprint", "Sprint", race.get("Sprint", {}).get("date")),
        ("qualifying", "Qualifying", race.get("Qualifying", {}).get("date")),
        ("race", "Grand Prix", race.get("date")),
    ]
    return [{"session": key, "label": label} for key, label, date in sessions if date]


def infer_stage(schedule: list[dict[str, str]]) -> str:
    # Jolpica schedule dates are not fine-grained enough for reliable session-end detection.
    return "pre_weekend" if schedule else "unknown"


def next_session(schedule: list[dict[str, str]]) -> str | None:
    return schedule[0]["session"] if schedule else None


def make_weekend_id(year: int, round_number: int | None, race_name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in race_name)
    slug = "-".join(part for part in slug.split("-") if part)
    if round_number:
        return f"{year}-r{round_number}-{slug}"
    return f"{year}-{slug}"


def is_active_monitoring_window(weekend: WeekendContext) -> bool:
    race_date = parse_iso_date(weekend.race_date)
    if not race_date:
        return False
    now = utc_now()
    return race_date - timedelta(days=4) <= now <= race_date + timedelta(days=2)
