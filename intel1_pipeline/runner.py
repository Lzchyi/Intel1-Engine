from __future__ import annotations

import os
import uuid
import json
from dataclasses import dataclass
from pathlib import Path

from .ai import build_extraction_cache
from .arena import build_prediction_arena
from .config import load_source_registry
from .deepseek_extractor import extract_hybrid_signals
from .drive_upload import upload_bundle
from .evaluator import evaluate_prediction_arena
from .learning import load_learning_state, update_learning_state
from .output_contract import (
    chart_data_payload,
    current_standings_payload,
    current_weekend_payload,
    history_payload,
    manifest_payload,
    source_log_payload,
    validate_outputs,
    write_json,
)
from .prediction import build_prediction
from .session_result_extractor import supplement_session_results_with_deepseek
from .source_items import fetch_source_items
from .standings_updater import apply_pending_session_results_to_standings
from .structured_data import is_active_monitoring_window, load_current_weekend, load_driver_standings, load_historical_race_data, load_session_results
from .structured_data import load_constructor_standings
from .summary import build_summary


@dataclass
class RunOptions:
    output_dir: Path
    source_registry: Path
    force_weekend_id: str | None
    force_stage: str | None
    scheduled: bool
    dry_run: bool
    skip_ai: bool
    skip_drive_upload: bool
    max_items_per_source: int
    public_base_url: str | None = None


def run(options: RunOptions) -> dict[str, str | bool | int]:
    run_id = str(uuid.uuid4())
    output_dir = options.output_dir
    weekend = load_current_weekend(options.force_weekend_id, options.force_stage)
    github_event = os.getenv("GITHUB_EVENT_NAME", "").strip().lower()
    enforce_monitoring_window = options.scheduled and github_event not in {"push", "workflow_dispatch"}
    if enforce_monitoring_window and not options.force_weekend_id and not is_active_monitoring_window(weekend):
        return {
            "run_id": run_id,
            "skipped": True,
            "reason": "outside_active_monitoring_window",
            "weekend_id": weekend.weekend_id,
        }

    sources = load_source_registry(options.source_registry)
    source_items, failed_items = fetch_source_items(sources, max_items_per_source=options.max_items_per_source)
    drivers = load_driver_standings()
    constructors = load_constructor_standings()
    historical_race_data = load_historical_race_data(weekend.year)
    session_results = supplement_session_results_with_deepseek(
        items=source_items,
        weekend=weekend,
        session_results=load_session_results(weekend),
        skip_ai=options.skip_ai,
    )
    standings_update_state = load_json_object(output_dir / "standings_update_state.json")
    previous_standings_payload = load_json_object(output_dir / "current_standings.json")
    drivers, constructors, standings_update_state = apply_pending_session_results_to_standings(
        weekend=weekend,
        driver_standings=drivers,
        constructor_standings=constructors,
        session_results=session_results,
        standings_update_state=standings_update_state,
        previous_standings_payload=previous_standings_payload,
    )
    prior_learning_state = load_learning_state(options.output_dir / "learning_state.json")
    previous_prediction_history = load_json_array(options.output_dir / "prediction_history.json")
    previous_prediction = load_json_object(output_dir / "latest_prediction.json")
    previous_prediction_arena = load_json_object(output_dir / "prediction_arena.json")
    previous_manifest = load_json_object(output_dir / "app_manifest.json")
    extraction_cache_path = options.output_dir / "ai_extraction_cache.json"
    cached_signals_by_hash = load_extraction_cache(extraction_cache_path)
    extraction_result = extract_hybrid_signals(
        items=source_items,
        weekend_id=weekend.weekend_id,
        run_id=run_id,
        stage=weekend.stage,
        skip_ai=options.skip_ai,
        cached_signals_by_hash=cached_signals_by_hash,
    )
    signals = extraction_result.signals
    prediction = build_prediction(
        run_id=run_id,
        weekend=weekend,
        drivers=drivers,
        signals=signals,
        source_count=len(source_items),
        session_results=session_results,
        learning_state=prior_learning_state,
    )
    enrich_prediction_with_previous(prediction, previous_prediction)
    frozen_arena = race_result_prediction_arena(
        run_id=run_id,
        weekend_id=weekend.weekend_id,
        stage=weekend.stage,
        session_results=session_results,
        previous_prediction=previous_prediction,
        previous_prediction_arena=previous_prediction_arena,
        previous_prediction_history=previous_prediction_history,
        previous_manifest=previous_manifest,
    )
    if frozen_arena:
        prediction_arena = frozen_arena
    else:
        prediction_arena = build_prediction_arena(
            run_id=run_id,
            weekend=weekend,
            baseline_prediction=prediction,
            stored_signals=extraction_result.stored_signals,
            session_results=session_results,
            skip_ai=options.skip_ai,
        )
    prediction_evaluation = evaluate_prediction_arena(
        run_id=run_id,
        weekend_id=weekend.weekend_id,
        arena_payload=prediction_arena,
        session_results=session_results,
    )
    learning_state = update_learning_state(
        prior_learning_state,
        run_id=run_id,
        weekend_id=weekend.weekend_id,
        arena_payload=prediction_arena,
        evaluation=prediction_evaluation,
        session_results=session_results,
    )
    summary = build_summary(run_id, weekend, prediction, signals, skip_ai=options.skip_ai)
    prediction_history = history_payload(run_id, weekend, prediction, summary, previous_prediction_history)
    signal_ids_by_source: dict[str, list[str]] = {}
    for signal in signals:
        signal_ids_by_source.setdefault(signal.source_id, []).append(signal.signal_id)
        signal_ids_by_source.setdefault(signal.source_item_id, []).append(signal.signal_id)

    write_json(output_dir / "current_weekend.json", current_weekend_payload(run_id, weekend))
    write_json(output_dir / "latest_prediction.json", prediction)
    write_json(output_dir / "latest_summary.json", summary)
    write_json(output_dir / "latest_chart_data.json", chart_data_payload(run_id, weekend, prediction, prediction_history))
    write_json(output_dir / "prediction_history.json", prediction_history)
    write_json(output_dir / "extracted_signals.json", [signal.to_dict() for signal in signals])
    write_json(output_dir / "stored_signals.json", [signal.to_dict() for signal in extraction_result.stored_signals])
    write_json(output_dir / "extraction_errors.json", [error.to_dict() for error in extraction_result.extraction_errors])
    write_json(output_dir / "prediction_arena.json", prediction_arena)
    write_json(output_dir / "prediction_evaluation.json", prediction_evaluation)
    write_json(output_dir / "learning_state.json", learning_state)
    write_json(output_dir / "source_log.json", source_log_payload(run_id, weekend, source_items, failed_items, signal_ids_by_source))
    write_json(output_dir / "ai_extraction_cache.json", build_extraction_cache(source_items, signals))
    write_json(output_dir / "historical_race_data.json", historical_race_data)
    write_json(output_dir / "standings_update_state.json", standings_update_state)
    write_json(
        output_dir / "current_standings.json",
        current_standings_payload(
            season=weekend.year,
            driver_standings=drivers,
            constructor_standings=constructors,
        ),
    )
    public_base_url = options.public_base_url or os.getenv("INTEL1_PUBLIC_BASE_URL") or None
    write_json(output_dir / "app_manifest.json", manifest_payload(run_id, weekend, output_dir, public_base_url=public_base_url))
    validate_outputs(output_dir)

    manifest_url = f"{public_base_url.rstrip('/')}/app_manifest.json" if public_base_url and not options.dry_run else ""
    if not options.dry_run and not options.skip_drive_upload:
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        folder_id = os.getenv("GOOGLE_DRIVE_LIVE_FOLDER_ID", "")
        make_public = os.getenv("GOOGLE_DRIVE_MAKE_PUBLIC", "true").lower() == "true"
        if service_account_json and folder_id:
            manifest_url = upload_bundle(
                output_dir=output_dir,
                run_id=run_id,
                weekend=weekend,
                folder_id=folder_id,
                service_account_json=service_account_json,
                make_public=make_public,
            )

    return {
        "run_id": run_id,
        "skipped": False,
        "weekend_id": weekend.weekend_id,
        "stage": weekend.stage,
        "source_items": len(source_items),
        "failed_sources": len(failed_items),
        "signals": len(signals),
        "manifest_url": manifest_url,
    }


def load_extraction_cache(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, list)}


def load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def race_result_prediction_arena(
    *,
    run_id: str,
    weekend_id: str,
    stage: str,
    session_results: dict[str, list[dict[str, object]]],
    previous_prediction: dict,
    previous_prediction_arena: dict,
    previous_prediction_history: list[dict],
    previous_manifest: dict,
) -> dict | None:
    if not session_results.get("race"):
        return None

    if same_weekend(previous_prediction, weekend_id) and not prediction_has_race_result(previous_prediction):
        frozen = dict(previous_prediction_arena) if isinstance(previous_prediction_arena, dict) else {}
        if valid_arena(frozen, weekend_id):
            frozen["run_id"] = run_id
            frozen["stage"] = frozen.get("stage") or stage
            return frozen

    excluded_run_id = str(previous_manifest.get("run_id") or "") if prediction_has_race_result(previous_prediction) else ""
    entry = latest_history_prediction_for_result(previous_prediction_history, weekend_id, excluded_run_id=excluded_run_id)
    if entry:
        return arena_from_history_entry(run_id=run_id, weekend_id=weekend_id, stage=stage, entry=entry)
    return None


def valid_arena(payload: dict, weekend_id: str) -> bool:
    predictions = payload.get("predictions") if isinstance(payload, dict) else None
    return bool(payload.get("weekend_id") == weekend_id and isinstance(predictions, dict) and predictions.get("intel1_consensus"))


def same_weekend(payload: dict, weekend_id: str) -> bool:
    return bool(isinstance(payload, dict) and payload.get("weekend_id") == weekend_id)


def prediction_has_race_result(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    session_results = payload.get("session_results") or {}
    return bool(isinstance(session_results, dict) and session_results.get("race"))


def latest_history_prediction_for_result(history: list[dict], weekend_id: str, *, excluded_run_id: str = "") -> dict | None:
    for entry in reversed(history):
        if not isinstance(entry, dict) or entry.get("weekend_id") != weekend_id:
            continue
        if excluded_run_id and entry.get("run_id") == excluded_run_id:
            continue
        if entry.get("stage") == "post_race":
            continue
        if entry.get("top_driver_win_probabilities"):
            return entry
    return None


def arena_from_history_entry(*, run_id: str, weekend_id: str, stage: str, entry: dict) -> dict:
    win_probabilities = history_probabilities_to_percent(entry.get("top_driver_win_probabilities") or [], "driver")
    constructor_probabilities = history_probabilities_to_percent(entry.get("top_constructor_win_probabilities") or [], "team")
    predicted_winner = str((win_probabilities[0] if win_probabilities else {}).get("driver") or "TBD")
    prediction_stage = str(entry.get("stage") or stage)
    prediction = {
        "provider": "intel1_consensus",
        "version": f"{run_id}:intel1_consensus:history:{entry.get('run_id')}",
        "modelUsed": "intel1-history-snapshot",
        "providerRequestId": None,
        "modelTemperature": None,
        "confidence": 0.5,
        "sourceQuality": 0.72,
        "officialDataAvailability": 1.0,
        "sessionRecency": 1.0,
        "predicted_winner": predicted_winner,
        "win_probabilities": win_probabilities,
        "constructor_win_probabilities": constructor_probabilities or [{"team": "TBD", "probability": 100.0}],
        "podium_probabilities": win_probabilities[:3],
        "top10_probabilities": win_probabilities[:10],
        "dnf_risk": [],
        "safety_car_probability": round(float((entry.get("safety_car") or {}).get("probability_at_least_one", 0.0)) * 100, 2),
        "key_reasons": ["Frozen prediction snapshot recorded before the official race result."],
        "weak_assumptions": [],
        "analyst_report": None,
        "providerStatus": "history_snapshot",
        "validation_errors": [],
    }
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "weekend_id": weekend_id,
        "stage": prediction_stage,
        "updated_at": entry.get("timestamp"),
        "promptVersion": "history-snapshot",
        "predictions": {
            "chatgpt": None,
            "deepseek": None,
            "intel1_consensus": prediction,
        },
    }


def history_probabilities_to_percent(entries: list[dict], label: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get(label) or entry.get("driver") or entry.get("team")
        if not name:
            continue
        try:
            probability = float(entry.get("probability", 0.0))
        except (TypeError, ValueError):
            probability = 0.0
        percent = probability * 100 if probability <= 1 else probability
        output.append({label: name, "probability": round(max(0.0, min(100.0, percent)), 2)})
    output.sort(key=lambda item: float(item.get("probability", 0.0)), reverse=True)
    return output


def enrich_prediction_with_previous(prediction: dict, previous: dict) -> None:
    """Attach an explicit checkpoint-to-checkpoint delta without allowing old data to drive the new forecast."""
    if not isinstance(previous, dict) or previous.get("weekend_id") != prediction.get("weekend_id"):
        return
    previous_by_driver = {
        str(item.get("driver")): float(item.get("probability", 0.0))
        for item in ((previous.get("race") or {}).get("driver_win_probabilities") or [])
        if item.get("driver")
    }
    deltas = []
    for item in ((prediction.get("race") or {}).get("driver_win_probabilities") or []):
        driver = str(item.get("driver") or "")
        if not driver or driver not in previous_by_driver:
            continue
        current = float(item.get("probability", 0.0))
        prior = previous_by_driver[driver]
        delta = current - prior
        item["delta_vs_previous"] = round(delta, 4)
        if abs(delta) < 0.005:
            continue
        direction = "up" if delta > 0 else "down"
        deltas.append({
            "target_type": "driver",
            "target_name": driver,
            "direction": direction,
            "delta": round(delta, 4),
            "previous_probability": round(prior, 4),
            "current_probability": round(current, 4),
            "summary": f"{driver} {direction} {abs(delta):.1%} since the previous checkpoint.",
        })
    deltas.sort(key=lambda row: abs(float(row.get("delta", 0.0))), reverse=True)
    prediction["prediction_delta_vs_previous"] = [
        {"target_type": row["target_type"], "target_name": row["target_name"], "delta": row["delta"], "summary": row["summary"]}
        for row in deltas
    ]
    prediction["change_digest"] = deltas[:5]
    prediction["comparison_baseline_run_id"] = previous.get("run_id")
