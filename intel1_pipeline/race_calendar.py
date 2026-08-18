from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from .time_utils import utc_now

if TYPE_CHECKING:
    from .structured_data import WeekendContext


RACES_2026 = [
    {"round": 1, "name": "Australian Grand Prix", "circuit": "Albert Park Grand Prix Circuit", "country": "Australia", "date": "2026-03-08", "time": "15:00", "timezone": "Australia/Melbourne", "sprint": False},
    {"round": 2, "name": "Chinese Grand Prix", "circuit": "Shanghai International Circuit", "country": "China", "date": "2026-03-15", "time": "15:00", "timezone": "Asia/Shanghai", "sprint": True},
    {"round": 3, "name": "Japanese Grand Prix", "circuit": "Suzuka Circuit", "country": "Japan", "date": "2026-03-29", "time": "14:00", "timezone": "Asia/Tokyo", "sprint": False},
    {"round": 4, "name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "country": "United States", "date": "2026-05-03", "time": "16:00", "timezone": "America/New_York", "sprint": True},
    {"round": 5, "name": "Canadian Grand Prix", "circuit": "Circuit Gilles-Villeneuve", "country": "Canada", "date": "2026-05-24", "time": "16:00", "timezone": "America/Toronto", "sprint": True},
    {"round": 6, "name": "Monaco Grand Prix", "circuit": "Circuit de Monaco", "country": "Monaco", "date": "2026-06-07", "time": "15:00", "timezone": "Europe/Monaco", "sprint": False},
    {"round": 7, "name": "Barcelona-Catalunya Grand Prix", "circuit": "Circuit de Barcelona-Catalunya", "country": "Spain", "date": "2026-06-14", "time": "15:00", "timezone": "Europe/Madrid", "sprint": False},
    {"round": 8, "name": "Austrian Grand Prix", "circuit": "Red Bull Ring", "country": "Austria", "date": "2026-06-28", "time": "15:00", "timezone": "Europe/Vienna", "sprint": False},
    {"round": 9, "name": "British Grand Prix", "circuit": "Silverstone Circuit", "country": "United Kingdom", "date": "2026-07-05", "time": "15:00", "timezone": "Europe/London", "sprint": True},
    {"round": 10, "name": "Belgian Grand Prix", "circuit": "Circuit de Spa-Francorchamps", "country": "Belgium", "date": "2026-07-19", "time": "15:00", "timezone": "Europe/Brussels", "sprint": False},
    {"round": 11, "name": "Hungarian Grand Prix", "circuit": "Hungaroring", "country": "Hungary", "date": "2026-07-26", "time": "15:00", "timezone": "Europe/Budapest", "sprint": False},
    {"round": 12, "name": "Dutch Grand Prix", "circuit": "Circuit Zandvoort", "country": "Netherlands", "date": "2026-08-23", "time": "15:00", "timezone": "Europe/Amsterdam", "sprint": True},
    {"round": 13, "name": "Italian Grand Prix", "circuit": "Autodromo Nazionale Monza", "country": "Italy", "date": "2026-09-06", "time": "15:00", "timezone": "Europe/Rome", "sprint": False},
    {"round": 14, "name": "Spanish Grand Prix", "circuit": "Madring", "country": "Spain", "date": "2026-09-13", "time": "15:00", "timezone": "Europe/Madrid", "sprint": False},
    {"round": 15, "name": "Azerbaijan Grand Prix", "circuit": "Baku City Circuit", "country": "Azerbaijan", "date": "2026-09-26", "time": "15:00", "timezone": "Asia/Baku", "sprint": False},
    {"round": 16, "name": "Bahrain Grand Prix in Malaysia", "circuit": "Petronas Sepang International Circuit", "country": "Malaysia", "date": "2026-10-04", "time": "19:00", "timezone": "Asia/Kuala_Lumpur", "sprint": False, "session_times_confirmed": True},
    {"round": 17, "name": "Singapore Grand Prix", "circuit": "Marina Bay Street Circuit", "country": "Singapore", "date": "2026-10-11", "time": "20:00", "timezone": "Asia/Singapore", "sprint": True},
    {"round": 18, "name": "United States Grand Prix", "circuit": "Circuit of The Americas", "country": "United States", "date": "2026-10-25", "time": "15:00", "timezone": "America/Chicago", "sprint": False},
    {"round": 19, "name": "Mexico City Grand Prix", "circuit": "Autodromo Hermanos Rodriguez", "country": "Mexico", "date": "2026-11-01", "time": "14:00", "timezone": "America/Mexico_City", "sprint": False},
    {"round": 20, "name": "Sao Paulo Grand Prix", "circuit": "Autodromo Jose Carlos Pace", "country": "Brazil", "date": "2026-11-08", "time": "14:00", "timezone": "America/Sao_Paulo", "sprint": False},
    {"round": 21, "name": "Las Vegas Grand Prix", "circuit": "Las Vegas Strip Circuit", "country": "United States", "date": "2026-11-21", "time": "20:00", "timezone": "America/Los_Angeles", "sprint": False},
    {"round": 22, "name": "Qatar Grand Prix", "circuit": "Lusail International Circuit", "country": "Qatar", "date": "2026-11-29", "time": "19:00", "timezone": "Asia/Qatar", "sprint": False},
    {"round": 23, "name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina Circuit", "country": "United Arab Emirates", "date": "2026-12-06", "time": "17:00", "timezone": "Asia/Dubai", "sprint": False},
]

POST_RACE_RESULT_WINDOW = timedelta(hours=6)


def load_static_current_weekend(
    force_weekend_id: str | None = None,
    force_stage: str | None = None,
    now: datetime | None = None,
) -> WeekendContext | None:
    from .structured_data import WeekendContext

    current_time = (now or utc_now()).astimezone(UTC)
    if current_time.year != 2026:
        return None

    race = current_or_next_race(current_time)
    if not race:
        return None

    sessions = weekend_sessions(race)
    latest_completed = latest_completed_session(sessions, current_time) if sessions else None
    next_relevant = next_relevant_session(sessions, current_time) if sessions else None
    stage = force_stage or (stage_after(latest_completed) if sessions else "pre_weekend")

    return WeekendContext(
        weekend_id=force_weekend_id or make_weekend_id(2026, race["round"], race["name"]),
        grand_prix_name=race["name"],
        circuit_name=race["circuit"],
        country=race["country"],
        year=2026,
        round_number=race["round"],
        race_date=race["date"],
        is_sprint_weekend=bool(race["sprint"]),
        stage=stage,
        next_relevant_session=next_relevant["session"] if next_relevant else None,
        session_schedule=[{"session": item["session"], "label": item["label"]} for item in sessions],
    )


def current_or_next_race(now: datetime) -> dict[str, object] | None:
    for race in RACES_2026:
        sessions = weekend_sessions(race)
        if sessions:
            start = sessions[0]["starts_at"]
            end = session_end(sessions[-1])
        else:
            start, end = date_only_weekend_bounds(race)
        if start <= now <= end + POST_RACE_RESULT_WINDOW:
            return race
    upcoming = []
    for race in RACES_2026:
        sessions = weekend_sessions(race)
        end = session_end(sessions[-1]) if sessions else date_only_weekend_bounds(race)[1]
        if end + POST_RACE_RESULT_WINDOW >= now:
            upcoming.append(race)
    return upcoming[0] if upcoming else RACES_2026[-1]


def date_only_weekend_bounds(race: dict[str, object]) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(str(race["timezone"]))
    local_race_date = date.fromisoformat(str(race["date"]))
    start_local = datetime.combine(local_race_date - timedelta(days=2), time.min, timezone)
    end_local = datetime.combine(local_race_date + timedelta(days=1), time(hour=6), timezone)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def weekend_sessions(race: dict[str, object]) -> list[dict[str, object]]:
    if not race.get("time") or race.get("session_times_confirmed") is False:
        return []
    race_date = str(race["date"])
    race_time = str(race["time"])
    timezone = str(race["timezone"])
    round_number = int(race["round"])
    practice = official_practice_times(round_number)
    qualifying = official_qualifying_time(round_number)
    sessions = [
        session("fp1", "Practice 1", race_date, -2, practice["fp1"], timezone),
    ]
    if race["sprint"]:
        sessions.append(session("sprint_qualifying", "Sprint Qualifying", race_date, -2, practice.get("sq") or qualifying, timezone))
        sessions.append(session("sprint", "Sprint", race_date, -1, official_sprint_time(round_number) or "12:00", timezone))
    else:
        sessions.append(session("fp2", "Practice 2", race_date, -2, practice.get("fp2") or qualifying, timezone))
        sessions.append(session("fp3", "Practice 3", race_date, -1, practice.get("fp3") or qualifying, timezone))
    sessions.append(session("qualifying", "Qualifying", race_date, -1, qualifying, timezone))
    sessions.append(session("race", "Grand Prix", race_date, 0, race_time, timezone))
    return sessions


def session(key: str, label: str, race_date: str, day_offset: int, session_time: str, timezone: str) -> dict[str, object]:
    local_date = date.fromisoformat(race_date) + timedelta(days=day_offset)
    hour, minute = [int(part) for part in session_time.split(":", maxsplit=1)]
    starts_at = datetime.combine(local_date, time(hour=hour, minute=minute), ZoneInfo(timezone)).astimezone(UTC)
    return {"session": key, "label": label, "starts_at": starts_at}


def latest_completed_session(sessions: list[dict[str, object]], now: datetime) -> dict[str, object] | None:
    completed = [item for item in sessions if session_end(item) <= now]
    return completed[-1] if completed else None


def next_relevant_session(sessions: list[dict[str, object]], now: datetime) -> dict[str, object] | None:
    upcoming = [item for item in sessions if session_end(item) > now]
    return upcoming[0] if upcoming else None


def session_end(item: dict[str, object]) -> datetime:
    session_name = str(item["session"])
    duration = timedelta(hours=3) if session_name == "race" else timedelta(minutes=90 if session_name == "sprint" else 75)
    return item["starts_at"] + duration


def stage_after(session_item: dict[str, object] | None) -> str:
    if not session_item:
        return "pre_weekend"
    return {
        "fp1": "after_fp1",
        "fp2": "after_fp2",
        "fp3": "after_fp3",
        "sprint_qualifying": "after_sprint_qualifying",
        "sprint": "after_sprint",
        "qualifying": "after_qualifying",
        "race": "post_race",
    }.get(str(session_item["session"]), "final_pre_race")


def official_practice_times(round_number: int) -> dict[str, str | None]:
    table: dict[int, dict[str, str | None]] = {
        1: {"fp1": "12:30", "fp2": "16:00", "fp3": "12:30", "sq": None},
        2: {"fp1": "11:30", "fp2": None, "fp3": None, "sq": "15:30"},
        3: {"fp1": "11:30", "fp2": "15:00", "fp3": "11:30", "sq": None},
        4: {"fp1": "12:00", "fp2": None, "fp3": None, "sq": "16:30"},
        5: {"fp1": "12:30", "fp2": None, "fp3": None, "sq": "16:30"},
        6: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        7: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        8: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        9: {"fp1": "12:30", "fp2": None, "fp3": None, "sq": "16:30"},
        10: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        11: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        12: {"fp1": "12:30", "fp2": None, "fp3": None, "sq": "16:30"},
        13: {"fp1": "12:30", "fp2": "16:00", "fp3": "12:30", "sq": None},
        14: {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None},
        15: {"fp1": "12:30", "fp2": "16:00", "fp3": "12:30", "sq": None},
        16: {"fp1": "16:30", "fp2": "20:00", "fp3": "16:30", "sq": None},
        17: {"fp1": "16:30", "fp2": None, "fp3": None, "sq": "20:30"},
        18: {"fp1": "12:30", "fp2": "16:00", "fp3": "12:30", "sq": None},
        19: {"fp1": "12:30", "fp2": "16:00", "fp3": "11:30", "sq": None},
        20: {"fp1": "12:30", "fp2": "16:00", "fp3": "11:30", "sq": None},
        21: {"fp1": "16:30", "fp2": "20:00", "fp3": "16:30", "sq": None},
        22: {"fp1": "16:30", "fp2": "20:00", "fp3": "17:30", "sq": None},
        23: {"fp1": "13:30", "fp2": "17:00", "fp3": "14:30", "sq": None},
    }
    return table.get(round_number, {"fp1": "13:30", "fp2": "17:00", "fp3": "12:30", "sq": None})


def official_qualifying_time(round_number: int) -> str:
    if round_number == 16:
        return "20:00"
    if round_number == 21:
        return "20:00"
    if round_number in {17, 22}:
        return "21:00"
    if round_number == 23:
        return "18:00"
    return "15:00" if round_number in {2, 3, 19, 20} else "16:00"


def official_sprint_time(round_number: int) -> str | None:
    return {
        2: "11:00",
        4: "12:00",
        5: "12:00",
        9: "12:00",
        12: "12:00",
        17: "17:00",
    }.get(round_number)


PREDICTION_LEAD_TIME = timedelta(hours=6)
POST_RACE_LEARNING_DELAY = timedelta(hours=2)


def prediction_checkpoints(race: dict[str, object]) -> list[dict[str, object]]:
    """High-value runs only: Friday evidence -> Saturday forecast, Saturday evidence -> race forecast, then result learning."""
    sessions = weekend_sessions(race)
    if not sessions:
        return []
    race_session = next(item for item in sessions if item["session"] == "race")
    saturday_date = date.fromisoformat(str(race["date"])) - timedelta(days=1)
    timezone = ZoneInfo(str(race["timezone"]))
    saturday_sessions = [
        item for item in sessions
        if item["starts_at"].astimezone(timezone).date() == saturday_date
    ]
    first_saturday = saturday_sessions[0] if saturday_sessions else None
    checkpoints: list[dict[str, object]] = []
    if first_saturday is not None:
        checkpoints.append({
            "kind": "saturday_forecast",
            "run_at": first_saturday["starts_at"] - PREDICTION_LEAD_TIME,
            "target_session": first_saturday["session"],
        })
    checkpoints.append({
        "kind": "race_forecast",
        "run_at": race_session["starts_at"] - PREDICTION_LEAD_TIME,
        "target_session": "race",
    })
    checkpoints.append({
        "kind": "post_race_learning",
        "run_at": session_end(race_session) + POST_RACE_LEARNING_DELAY,
        "target_session": "post_race",
    })
    return checkpoints


def make_weekend_id(year: int, round_number: int | None, race_name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in race_name)
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{year}-r{round_number}-{slug}" if round_number else f"{year}-{slug}"
