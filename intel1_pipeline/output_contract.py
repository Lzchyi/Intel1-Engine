from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .source_items import SourceItem
from .structured_data import WeekendContext
from .time_utils import isoformat, utc_now


REQUIRED_FILES = [
    "current_weekend.json",
    "latest_prediction.json",
    "latest_summary.json",
    "latest_chart_data.json",
    "prediction_history.json",
    "source_log.json",
    "extracted_signals.json",
    "stored_signals.json",
    "extraction_errors.json",
    "prediction_arena.json",
    "prediction_evaluation.json",
    "learning_state.json",
    "historical_race_data.json",
    "current_standings.json",
]


def current_weekend_payload(run_id: str, weekend: WeekendContext) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "run_id": run_id,
        "weekend_id": weekend.weekend_id,
        "grand_prix_name": weekend.grand_prix_name,
        "circuit_name": weekend.circuit_name,
        "country": weekend.country,
        "year": weekend.year,
        "is_sprint_weekend": weekend.is_sprint_weekend,
        "stage": weekend.stage,
        "next_relevant_session": weekend.next_relevant_session,
        "last_pipeline_run_at": isoformat(utc_now()),
        "session_schedule": weekend.session_schedule,
    }


def chart_data_payload(run_id: str, weekend: WeekendContext, prediction: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    timestamp = prediction["updated_at"]
    history_rows = history or history_payload(run_id, weekend, prediction, {"headline": ""})
    driver_series: dict[str, list[dict[str, Any]]] = {}
    constructor_series: dict[str, list[dict[str, Any]]] = {}
    stage_markers = []
    for row in history_rows:
        row_timestamp = row.get("timestamp") or timestamp
        stage = row.get("stage") or weekend.stage
        stage_markers.append({"stage": stage, "timestamp": row_timestamp})
        for entry in row.get("top_driver_win_probabilities", []):
            if not entry.get("driver"):
                continue
            driver_series.setdefault(entry["driver"], []).append(
                {
                    "label": stage.replace("_", " ").title(),
                    "timestamp": row_timestamp,
                    "value": entry["probability"],
                    "stage": stage,
                    "driver": entry["driver"],
                }
            )
        for entry in row.get("top_constructor_win_probabilities", []):
            if not entry.get("team"):
                continue
            constructor_series.setdefault(entry["team"], []).append(
                {
                    "label": stage.replace("_", " ").title(),
                    "timestamp": row_timestamp,
                    "value": entry["probability"],
                    "stage": stage,
                    "driver": entry["team"],
                }
            )
    return {
        "schema_version": "1.1",
        "run_id": run_id,
        "updated_at": timestamp,
        "weekend_id": weekend.weekend_id,
        "stage_markers": stage_markers,
        "driver_series": driver_series,
        "constructor_series": constructor_series,
    }


def history_payload(
    run_id: str,
    weekend: WeekendContext,
    prediction: dict[str, Any],
    summary: dict[str, Any],
    previous_history: list[dict[str, Any]] | None = None,
    max_entries: int = 60,
) -> list[dict[str, Any]]:
    entry = {
        "run_id": run_id,
        "timestamp": prediction["updated_at"],
        "weekend_id": weekend.weekend_id,
        "stage": weekend.stage,
        "top_driver_win_probabilities": prediction["race"]["driver_win_probabilities"][:5],
        "top_constructor_win_probabilities": prediction["race"]["constructor_win_probabilities"][:5],
        "safety_car": prediction["safety_car"],
        "summary_headline": summary["headline"],
        "confidence": prediction.get("forecast_confidence") or prediction.get("confidence"),
        "evidence_quality": prediction.get("evidence_quality"),
        "weekend_phase": prediction.get("weekend_phase"),
        "is_evaluation_snapshot": bool((prediction.get("session_results") or {}).get("qualifying")) and not bool((prediction.get("session_results") or {}).get("race")),
    }
    rows = [row for row in (previous_history or []) if isinstance(row, dict) and row.get("run_id") != run_id]
    rows.append(entry)
    return rows[-max_entries:]


def source_log_payload(run_id: str, weekend: WeekendContext, items: list[SourceItem], failed_items: list[SourceItem], signal_ids_by_source: dict[str, list[str]]) -> dict[str, Any]:
    entries = []
    for item in items + failed_items:
        entries.append(
            {
                "source_item_id": item.source_item_id,
                "source_id": item.source_id,
                "source_name": item.source_name,
                "source_tier": item.source_tier,
                "source_type": item.source_type,
                "reliability_weight": item.reliability_weight,
                "connector_type": item.connector_type,
                "title": redacted_title(item),
                "source_url": item.url,
                "canonical_url": item.canonical_url,
                "published_at": item.published_at,
                "updated_at": item.updated_at,
                "fetched_at": item.fetched_at,
                "raw_excerpt": redacted_raw_excerpt(item),
                "language": item.language,
                "fingerprint": item.content_hash,
                "content_hash": item.content_hash,
                "relevance": item.weekend_relevance_status,
                "weekend_relevance_status": item.weekend_relevance_status,
                "fetch_status": item.fetch_status,
                "failure_reason": item.failure_reason,
                "deduped": False,
                "extracted_signal_ids": signal_ids_by_source.get(item.source_item_id, signal_ids_by_source.get(item.source_id, [])),
                "processed_at": isoformat(utc_now()),
            }
        )
    return {
        "schema_version": "1.1",
        "run_id": run_id,
        "weekend_id": weekend.weekend_id,
        "updated_at": isoformat(utc_now()),
        "fetched_sources": entries,
    }


def current_standings_payload(
    *,
    season: int,
    driver_standings: list[dict[str, Any]],
    constructor_standings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "season": season,
        "updated_at": isoformat(utc_now()),
        "driver_standings": [
            {
                "id": item.get("driver_id") or item.get("driver") or "unknown-driver",
                "position": item.get("position", 99),
                "points": item.get("points", 0),
                "wins": item.get("wins", 0),
                "primary_label": item.get("driver") or "Unknown Driver",
                "secondary_label": item.get("team"),
                "short_name": None,
                "team_label": item.get("team"),
                "team_id": item.get("team_id"),
                "driver_id": item.get("driver_id"),
            }
            for item in driver_standings
        ],
        "constructor_standings": [
            {
                "id": item.get("team_id") or item.get("team") or "unknown-team",
                "position": item.get("position", 99),
                "points": item.get("points", 0),
                "wins": item.get("wins", 0),
                "primary_label": item.get("team") or "Unknown Team",
                "secondary_label": None,
                "short_name": None,
                "team_label": item.get("team"),
                "team_id": item.get("team_id"),
                "driver_id": None,
            }
            for item in constructor_standings
        ],
    }


def manifest_payload(
    run_id: str,
    weekend: WeekendContext,
    output_dir: Path,
    file_urls: dict[str, str] | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    now = isoformat(utc_now())
    if file_urls:
        urls = file_urls
    elif public_base_url:
        base_url = public_base_url.rstrip("/")
        urls = {name: f"{base_url}/{name}" for name in REQUIRED_FILES}
    else:
        urls = {name: output_dir.joinpath(name).resolve().as_uri() for name in REQUIRED_FILES}
    return {
        "schema_version": "1.1",
        "run_id": run_id,
        "updated_at": now,
        "weekend_id": weekend.weekend_id,
        "active_weekend_id": weekend.weekend_id,
        "freshness_timestamp": now,
        "last_successful_full_run_at": now,
        "files": [
            {
                "name": name,
                "url": urls[name],
                "updated_at": now,
                "content_type": "application/json",
            }
            for name in REQUIRED_FILES
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_outputs(output_dir: Path) -> None:
    for name in REQUIRED_FILES + ["app_manifest.json"]:
        path = output_dir / name
        if not path.exists():
            raise ValueError(f"Missing output file: {name}")
        json.loads(path.read_text(encoding="utf-8"))

    prediction = json.loads((output_dir / "latest_prediction.json").read_text(encoding="utf-8"))
    required_prediction_keys = {"schema_version", "run_id", "updated_at", "weekend_id", "stage", "session_results", "race", "safety_car", "confidence"}
    missing = required_prediction_keys - set(prediction)
    if missing:
        raise ValueError(f"latest_prediction.json missing keys: {sorted(missing)}")

    session_keys = {"fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"}
    missing_session_keys = session_keys - set(prediction.get("session_results", {}))
    if missing_session_keys:
        raise ValueError(f"session_results missing keys: {sorted(missing_session_keys)}")

    validate_probability_group("race driver", prediction["race"]["driver_win_probabilities"])
    validate_probability_group("constructor", prediction["race"]["constructor_win_probabilities"])
    if prediction.get("sprint", {}).get("enabled"):
        validate_probability_group("sprint driver", prediction["sprint"]["driver_win_probabilities"])

    manifest = json.loads((output_dir / "app_manifest.json").read_text(encoding="utf-8"))
    names = [item["name"] for item in manifest.get("files", [])]
    if len(names) != len(set(names)):
        raise ValueError("app_manifest.json contains duplicate file names")
    validate_no_raw_social_payloads(output_dir)


def validate_probability_group(label: str, entries: list[dict[str, Any]]) -> None:
    probabilities = [item.get("probability", 0.0) for item in entries]
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError(f"{label} probabilities must be between 0 and 1")
    if abs(sum(probabilities) - 1.0) > 0.0001:
        raise ValueError(f"{label} probabilities must sum to 1.0")


def redacted_raw_excerpt(item: SourceItem) -> str:
    if item.source_type in {"reddit", "x"}:
        return ""
    return item.raw_excerpt[:240]


def redacted_title(item: SourceItem) -> str:
    if item.source_type in {"reddit", "x"}:
        return f"{item.source_name} social item"
    return item.title


def validate_no_raw_social_payloads(output_dir: Path) -> None:
    for path in output_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in walk_json(payload):
            if isinstance(entry, dict):
                source_type = entry.get("source_type") or entry.get("sourceType")
                if source_type in {"reddit", "x"} and entry.get("raw_excerpt"):
                    raise ValueError(f"{path.name} contains raw social excerpt")
                if source_type in {"reddit", "x"} and entry.get("title") and not str(entry.get("title", "")).endswith("social item"):
                    raise ValueError(f"{path.name} contains raw social title")
                if entry.get("raw_content"):
                    raise ValueError(f"{path.name} contains raw content")


def walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)
