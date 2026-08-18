from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .time_utils import isoformat, utc_now


MAX_DRIVER_ADJUSTMENT = 0.15
DECAY = 0.85
SESSION_KEYS = ["race", "sprint", "qualifying", "sprint_qualifying", "fp3", "fp2", "fp1"]


def empty_learning_state() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "updated_at": isoformat(utc_now()),
        "driver_adjustments": {},
        "model_scores": {},
        "calibration": {},
        "events": [],
        "learning_notes": ["No classified session has been evaluated yet."],
    }


def load_learning_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_learning_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_learning_state()
    if not isinstance(payload, dict):
        return empty_learning_state()
    payload.setdefault("schema_version", "1.0")
    payload.setdefault("driver_adjustments", {})
    payload.setdefault("model_scores", {})
    payload.setdefault("calibration", {})
    payload.setdefault("events", [])
    payload.setdefault("learning_notes", [])
    return payload


def update_learning_state(
    previous_state: dict[str, Any],
    *,
    run_id: str,
    weekend_id: str,
    arena_payload: dict[str, Any],
    evaluation: dict[str, Any],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    state = normalize_state(previous_state)
    state["updated_at"] = isoformat(utc_now())

    latest_session = evaluation.get("latest_classified_session")
    if evaluation.get("status") != "evaluated" or not latest_session:
        state["learning_notes"] = list(evaluation.get("learning_notes") or ["No result available for learning yet."])
        return state

    rows = session_results.get(str(latest_session)) or []
    if latest_session not in {"race", "sprint", "qualifying", "sprint_qualifying"} or not rows:
        state["learning_notes"] = list(evaluation.get("learning_notes") or ["Latest session is not a race-learning checkpoint."])
        return state

    consensus = (arena_payload.get("predictions") or {}).get("intel1_consensus") or {}
    evaluations = evaluation.get("evaluations") or {}
    provider_scores = provider_score_updates(evaluations)
    driver_deltas = driver_learning_deltas(consensus, rows)
    update_calibration(state, evaluations.get("intel1_consensus") or {})

    for provider, score in provider_scores.items():
        current = state["model_scores"].get(provider, {"evaluations": 0, "average_score": 0.0})
        count = int(current.get("evaluations", 0)) + 1
        average = (float(current.get("average_score", 0.0)) * (count - 1) + score) / count
        state["model_scores"][provider] = {"evaluations": count, "average_score": round(average, 4)}

    for driver, delta in driver_deltas.items():
        current = state["driver_adjustments"].get(driver, {"score_delta": 0.0, "evaluations": 0})
        blended = float(current.get("score_delta", 0.0)) * DECAY + delta
        state["driver_adjustments"][driver] = {
            "score_delta": round(max(-MAX_DRIVER_ADJUSTMENT, min(MAX_DRIVER_ADJUSTMENT, blended)), 4),
            "evaluations": int(current.get("evaluations", 0)) + 1,
            "last_session": latest_session,
            "last_updated_at": state["updated_at"],
        }

    event = {
        "run_id": run_id,
        "weekend_id": weekend_id,
        "session": latest_session,
        "actual_winner": rows[0].get("driver") if rows else None,
        "model_winner": evaluation.get("model_winner"),
        "driver_deltas": {key: round(value, 4) for key, value in driver_deltas.items()},
    }
    state["events"] = (state.get("events") or [])[-19:] + [event]
    state["learning_notes"] = [
        f"Latest learning checkpoint: {latest_session}.",
        "Driver score nudges are capped and decay over time, so official results guide the model without overpowering fresh session data.",
    ]
    return state


def driver_score_adjustment(learning_state: dict[str, Any] | None, driver: str) -> float:
    if not learning_state:
        return 0.0
    adjustment = (learning_state.get("driver_adjustments") or {}).get(driver) or {}
    try:
        value = float(adjustment.get("score_delta", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(-MAX_DRIVER_ADJUSTMENT, min(MAX_DRIVER_ADJUSTMENT, value))


def normalize_state(state: dict[str, Any] | None) -> dict[str, Any]:
    output = empty_learning_state()
    if not isinstance(state, dict):
        return output
    output.update({key: value for key, value in state.items() if key in output})
    output["driver_adjustments"] = dict(output.get("driver_adjustments") or {})
    output["model_scores"] = dict(output.get("model_scores") or {})
    output["calibration"] = dict(output.get("calibration") or {})
    output["events"] = list(output.get("events") or [])
    output["learning_notes"] = list(output.get("learning_notes") or [])
    return output


def provider_score_updates(evaluations: dict[str, dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for provider, payload in evaluations.items():
        try:
            scores[provider] = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


def driver_learning_deltas(prediction: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    actual_positions = {
        str(row.get("driver")): int(row.get("position") or index + 1)
        for index, row in enumerate(rows)
        if row.get("driver")
    }
    predicted_winner = str(prediction.get("predicted_winner") or "")
    win_probabilities = prediction.get("win_probabilities") or []
    predicted_rank = {
        str(entry.get("driver")): index + 1
        for index, entry in enumerate(sorted(win_probabilities, key=lambda item: item.get("probability", 0), reverse=True))
        if entry.get("driver")
    }

    deltas: dict[str, float] = {}
    actual_winner = str(rows[0].get("driver") or "")
    if actual_winner and predicted_winner != actual_winner:
        deltas[actual_winner] = deltas.get(actual_winner, 0.0) + 0.018
        if predicted_winner:
            deltas[predicted_winner] = deltas.get(predicted_winner, 0.0) - 0.012

    for driver, actual_position in actual_positions.items():
        rank = predicted_rank.get(driver)
        if rank is None:
            if actual_position <= 10:
                deltas[driver] = deltas.get(driver, 0.0) + 0.008
            continue
        if actual_position <= 3 and rank > 5:
            deltas[driver] = deltas.get(driver, 0.0) + 0.012
        elif actual_position <= 10 and rank > actual_position + 4:
            deltas[driver] = deltas.get(driver, 0.0) + 0.006
        elif rank <= 3 and actual_position > 10:
            deltas[driver] = deltas.get(driver, 0.0) - 0.01

    return {driver: max(-0.025, min(0.025, delta)) for driver, delta in deltas.items() if abs(delta) > 0.0001}


def update_calibration(state: dict[str, Any], evaluation: dict[str, Any]) -> None:
    if not evaluation:
        return
    current = dict(state.get("calibration") or {})
    count = int(current.get("evaluations", 0))
    new_count = count + 1
    metrics = {
        "average_brier": float(evaluation.get("probability_calibration", 1.0)),
        "winner_accuracy": float(evaluation.get("prediction_accuracy", 0.0)),
        "podium_hit_rate": float(evaluation.get("podium_hit_rate", 0.0)),
        "top10_hit_rate": float(evaluation.get("top10_hit_rate", 0.0)),
        "average_score": float(evaluation.get("score", 0.0)),
    }
    for key, value in metrics.items():
        old = float(current.get(key, 0.0))
        current[key] = round((old * count + value) / new_count, 4)
    current["evaluations"] = new_count
    current["updated_at"] = isoformat(utc_now())
    state["calibration"] = current
