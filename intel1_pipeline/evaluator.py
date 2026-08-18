from __future__ import annotations

import re
import unicodedata
from typing import Any

from .time_utils import isoformat, utc_now


SESSION_ORDER = ["race", "qualifying", "sprint", "sprint_qualifying", "fp3", "fp2", "fp1"]


def evaluate_prediction_arena(
    *,
    run_id: str,
    weekend_id: str,
    arena_payload: dict[str, Any],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    latest_session, actual_rows = latest_classified_session(session_results)
    if not latest_session or not actual_rows:
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "weekend_id": weekend_id,
            "updated_at": isoformat(utc_now()),
            "status": "no_classified_session",
            "latest_classified_session": None,
            "evaluations": {},
            "model_winner": None,
            "learning_notes": ["No official/latest classified session result is available yet."],
        }
    if latest_session != "race":
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "weekend_id": weekend_id,
            "updated_at": isoformat(utc_now()),
            "status": "awaiting_race_result",
            "latest_classified_session": latest_session,
            "actual_result_source": actual_rows[0].get("source") if actual_rows else None,
            "evaluations": {},
            "model_winner": None,
            "learning_notes": [f"Latest classified session is {latest_session}; race prediction evaluation waits for the Grand Prix result."],
        }

    evaluations = {}
    for name, prediction in (arena_payload.get("predictions") or {}).items():
        if not isinstance(prediction, dict):
            continue
        if name == "intel1_consensus" or prediction.get("win_probabilities"):
            evaluations[name] = evaluate_model_prediction(prediction, actual_rows)

    model_winner = best_model(evaluations)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "weekend_id": weekend_id,
        "updated_at": isoformat(utc_now()),
        "status": "evaluated",
        "latest_classified_session": latest_session,
        "actual_result_source": actual_rows[0].get("source") if actual_rows else None,
        "evaluations": evaluations,
        "model_winner": model_winner,
        "learning_notes": learning_notes(evaluations, latest_session),
    }


def evaluate_model_prediction(prediction: dict[str, Any], actual_rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_winner = str(actual_rows[0].get("driver") or "") if actual_rows else ""
    actual_podium = {str(row.get("driver")) for row in actual_rows[:3] if row.get("driver")}
    actual_top10 = {str(row.get("driver")) for row in actual_rows[:10] if row.get("driver")}
    predicted_podium = {entry.get("driver") for entry in sorted(prediction.get("podium_probabilities", []), key=lambda item: item.get("probability", 0), reverse=True)[:3]}
    predicted_top10 = {entry.get("driver") for entry in sorted(prediction.get("top10_probabilities", []), key=lambda item: item.get("probability", 0), reverse=True)[:10]}
    if not predicted_top10:
        predicted_top10 = {entry.get("driver") for entry in prediction.get("win_probabilities", [])[:10]}
    dnf_actual = {str(row.get("driver")) for row in actual_rows if is_dnf_status(str(row.get("status") or ""))}
    dnf_predicted = {entry.get("driver") for entry in prediction.get("dnf_risk", []) if entry.get("probability", 0) >= 35}
    return {
        "prediction_accuracy": 1.0 if names_match(prediction.get("predicted_winner"), actual_winner) else 0.0,
        "podium_hit_rate": hit_rate(predicted_podium, actual_podium, 3),
        "top10_hit_rate": hit_rate(predicted_top10, actual_top10, min(10, len(actual_top10) or 10)),
        "dnf_accuracy": dnf_accuracy(dnf_predicted, dnf_actual, actual_rows),
        "probability_calibration": winner_brier_score(prediction.get("win_probabilities", []), actual_winner),
        "predicted_winner": prediction.get("predicted_winner"),
        "actual_winner": actual_winner,
        "score": model_score(prediction, actual_winner, predicted_podium, actual_podium, predicted_top10, actual_top10),
    }


def latest_classified_session(session_results: dict[str, list[dict[str, Any]]]) -> tuple[str | None, list[dict[str, Any]]]:
    for key in SESSION_ORDER:
        rows = session_results.get(key) or []
        if rows:
            return key, rows
    return None, []


def hit_rate(predicted: set[str | None], actual: set[str], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    actual_keys = {compact_name_key(item) for item in actual}
    return round(len({item for item in predicted if compact_name_key(item or "") in actual_keys}) / denominator, 3)


def dnf_accuracy(predicted: set[str | None], actual: set[str], rows: list[dict[str, Any]]) -> float:
    drivers = {str(row.get("driver")) for row in rows if row.get("driver")}
    if not drivers:
        return 0.0
    correct = 0
    for driver in drivers:
        correct += (driver in predicted) == (driver in actual)
    return round(correct / len(drivers), 3)


def winner_brier_score(win_probabilities: list[dict[str, Any]], actual_winner: str) -> float:
    if not win_probabilities or not actual_winner:
        return 1.0
    actual_key = compact_name_key(actual_winner)
    score = 0.0
    for entry in win_probabilities:
        probability = float(entry.get("probability", 0.0)) / 100
        expected = 1.0 if compact_name_key(str(entry.get("driver") or "")) == actual_key else 0.0
        score += (probability - expected) ** 2
    return round(score, 4)


def model_score(
    prediction: dict[str, Any],
    actual_winner: str,
    predicted_podium: set[str | None],
    actual_podium: set[str],
    predicted_top10: set[str | None],
    actual_top10: set[str],
) -> float:
    winner = 1.0 if names_match(prediction.get("predicted_winner"), actual_winner) else 0.0
    podium = hit_rate(predicted_podium, actual_podium, 3)
    top10 = hit_rate(predicted_top10, actual_top10, min(10, len(actual_top10) or 10))
    calibration = 1.0 - min(1.0, winner_brier_score(prediction.get("win_probabilities", []), actual_winner))
    return round(winner * 0.4 + podium * 0.25 + top10 * 0.2 + calibration * 0.15, 3)


def best_model(evaluations: dict[str, dict[str, Any]]) -> str | None:
    if not evaluations:
        return None
    return max(evaluations, key=lambda name: evaluations[name].get("score", 0.0))


def learning_notes(evaluations: dict[str, dict[str, Any]], latest_session: str) -> list[str]:
    if not evaluations:
        return [f"{latest_session} result is available, but no prediction outputs were evaluated."]
    winner = best_model(evaluations)
    return [
        f"Evaluation used the latest classified session: {latest_session}.",
        f"{winner} had the strongest combined winner/podium/top10/calibration score." if winner else "No model winner was selected.",
    ]


def is_dnf_status(status: str) -> bool:
    lowered = status.lower()
    return any(term in lowered for term in ["dnf", "retired", "accident", "engine", "gearbox", "not classified"])


def names_match(left: Any, right: Any) -> bool:
    return bool(left and right and compact_name_key(str(left)) == compact_name_key(str(right)))


def compact_name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())
