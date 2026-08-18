from __future__ import annotations

from collections import defaultdict
from math import exp
import re
from typing import Any
import unicodedata

from .learning import driver_score_adjustment
from .signals import ExtractedSignal
from .structured_data import WeekendContext
from .time_utils import isoformat, utc_now


STAGE_CONFIDENCE = {
    "pre_weekend": 0.34,
    "after_fp1": 0.42,
    "after_fp2": 0.52,
    "after_fp3": 0.58,
    "after_sprint_qualifying": 0.55,
    "after_sprint": 0.62,
    "after_qualifying": 0.74,
    "final_pre_race": 0.78,
    "post_race": 0.9,
}

SIGNAL_DELTAS = {
    "confirmed_grid_penalty": -0.55,
    "pending_investigation": -0.18,
    "reliability_concern": -0.20,
    "power_unit_concern": -0.24,
    "cooling_concern": -0.16,
    "brake_or_suspension_concern": -0.18,
    "floor_or_bodywork_damage": -0.22,
    "upgrade_positive": 0.12,
    "upgrade_negative": -0.10,
    "tyre_degradation_negative": -0.14,
    "tyre_degradation_positive": 0.10,
    "race_pace_positive": 0.18,
    "race_pace_negative": -0.14,
    "single_lap_pace_positive": 0.10,
    "single_lap_pace_negative": -0.10,
    "track_specific_suitability_positive": 0.10,
    "track_specific_suitability_negative": -0.10,
    "traffic_or_flags_compromised_lap": 0.04,
}

HIGH_SC_TRACKS = ["monaco", "baku", "singapore", "jeddah", "montreal", "saudi", "las vegas"]
HARD_TO_OVERTAKE = ["monaco", "hungaroring", "zandvoort", "singapore", "imola"]

SESSION_RESULT_KEYS = ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"]

MARKET_THRESHOLDS = {
    "no_clear_favourite_below": 0.15,
    "close_gap_below": 0.03,
    "model_favourite_min": 0.25,
    "model_favourite_gap": 0.08,
    "aggressive_podium_gain": 3,
}

WIN_SCORE_TEMPERATURE = {
    "race": 2.5,
    "sprint": 2.7,
}

LONG_SHOT_MULTIPLIERS = {
    "pointsless": 0.18,
    "backmarker": 0.35,
    "lower_midfield": 0.55,
    "midfield": 0.76,
    "front": 1.0,
}

METRIC_DEFINITIONS = [
    {
        "key": "weather_impact",
        "title": "Weather impact",
        "unit": "Low / Medium / High",
        "scale": "condition",
        "explanation": "Estimates how much rain, temperature, wind, or changing track conditions could disturb the race outlook.",
        "source": "OpenF1 weather where available, FIA/F1/Pirelli notes, and extracted weather signals.",
    },
    {
        "key": "tyre_degradation_risk",
        "title": "Tyre degradation risk",
        "unit": "Low / Medium / High or 0-100 index",
        "scale": "risk",
        "explanation": "Estimates how likely tyre wear will affect pace, stint length, and strategy flexibility.",
        "source": "Pirelli notes, long-run/session signals, circuit profile, and extracted tyre signals.",
    },
    {
        "key": "race_pace_confidence",
        "title": "Race pace confidence",
        "unit": "0-100",
        "scale": "score",
        "explanation": "Shows how strongly the current evidence supports the model's race-pace ranking.",
        "source": "Championship priors, session results, and corroborated pace signals.",
    },
    {
        "key": "strategy_volatility",
        "title": "Strategy volatility",
        "unit": "Low / Medium / High",
        "scale": "condition",
        "explanation": "Captures how likely Safety Cars, tyre offsets, weather, or field spread are to disrupt a normal strategy race.",
        "source": "Circuit profile, Safety Car outlook, weather signals, and strategy-related extracted signals.",
    },
    {
        "key": "safety_car_chance",
        "title": "Safety Car chance",
        "unit": "%",
        "scale": "probability",
        "explanation": "Heuristic probability of at least one Safety Car or major race interruption.",
        "source": "Circuit history/profile and extracted race-control or volatility signals.",
    },
    {
        "key": "forecast_confidence",
        "title": "Forecast confidence",
        "unit": "Low / Medium / High",
        "scale": "confidence",
        "explanation": "Separates how much evidence the model has from how chaotic the race itself may be.",
        "source": "Weekend stage, source count, structured session data, and extracted signal count.",
    },
]


def build_prediction(
    *,
    run_id: str,
    weekend: WeekendContext,
    drivers: list[dict[str, Any]],
    signals: list[ExtractedSignal],
    source_count: int,
    session_results: dict[str, list[dict[str, Any]]] | None = None,
    learning_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = isoformat(utc_now())
    session_results = normalized_session_results(session_results)
    safety_car = safety_car_outlook(weekend, signals)
    confidence = forecast_confidence(weekend.stage, source_count, signals)
    volatility = race_volatility(weekend, signals, safety_car)
    race_grid = starting_positions_for_market("race", session_results)
    sprint_grid = starting_positions_for_market("sprint", session_results)
    driver_entries, adjustment_log = driver_probabilities(
        drivers,
        signals,
        starting_positions=race_grid,
        market="race",
        learning_state=learning_state,
    )
    sprint_driver_entries, _ = driver_probabilities(
        drivers,
        signals,
        starting_positions=sprint_grid,
        market="sprint",
        learning_state=learning_state,
    )
    constructor_entries = constructor_probabilities(driver_entries)
    performance_entries = performance_comparison(driver_entries, signals)
    race_podium_entries = podium_probabilities(
        driver_entries,
        weekend=weekend,
        signals=signals,
        starting_positions=race_grid,
        safety_car=safety_car,
        volatility=volatility,
    )
    sprint_podium_entries = podium_probabilities(
        sprint_driver_entries,
        weekend=weekend,
        signals=signals,
        starting_positions=sprint_grid,
        safety_car=safety_car,
        volatility=volatility,
    )
    race_predicted_podium = race_podium_entries[:3]
    sprint_predicted_podium = sprint_podium_entries[:3]

    top_driver = driver_entries[0] if driver_entries else {"driver": "TBD", "team": None, "probability": 0.0, "rank": 1}
    top_sprint_driver = sprint_driver_entries[0] if sprint_driver_entries else top_driver
    top_constructor = constructor_entries[0] if constructor_entries else {"driver": None, "team": "TBD", "probability": 0.0, "rank": 1}
    prediction = {
        "schema_version": "1.1",
        "run_id": run_id,
        "updated_at": now,
        "weekend_id": weekend.weekend_id,
        "stage": weekend.stage,
        "session_results": session_results,
        "race": {
            "market_outlook": market_outlook(driver_entries, "race"),
            "predicted_winner": winner_payload(top_driver, race_grid, signals, weekend, safety_car, volatility),
            "driver_win_probabilities": driver_entries,
            "driver_podium_probabilities": race_podium_entries,
            "constructor_win_probabilities": constructor_entries,
            "predicted_podium": race_predicted_podium,
            "predicted_winning_constructor": top_constructor,
            "performance_comparison": performance_entries,
        },
        "sprint": {
            "enabled": weekend.is_sprint_weekend,
            "model_scope": "sprint_and_grand_prix" if weekend.is_sprint_weekend else "grand_prix_only",
            "market_outlook": market_outlook(sprint_driver_entries, "sprint") if weekend.is_sprint_weekend else None,
            "predicted_winner": winner_payload(top_sprint_driver, sprint_grid, signals, weekend, safety_car, volatility) if weekend.is_sprint_weekend else None,
            "driver_win_probabilities": sprint_driver_entries if weekend.is_sprint_weekend else [],
            "driver_podium_probabilities": sprint_podium_entries if weekend.is_sprint_weekend else [],
            "predicted_podium": sprint_predicted_podium if weekend.is_sprint_weekend else [],
            "notes": (
                "Sprint outlook is generated on sprint weekends from the same rule-based engine, with sprint qualifying and sprint-session signals lowering confidence until cleaner evidence is available."
                if weekend.is_sprint_weekend
                else None
            ),
        },
        "safety_car": safety_car,
        "confidence": confidence,
        "forecast_confidence": confidence,
        "race_volatility": volatility,
        "biggest_upgrades_since_previous_run": upgrade_summaries(signals, positive=True),
        "biggest_downgrades_since_previous_run": upgrade_summaries(signals, positive=False),
        "key_uncertainties": key_uncertainties(weekend, signals),
        "score_adjustment_log": adjustment_log,
        "data_freshness": {
            "last_source_scan_at": now,
            "last_session_data_at": now,
        },
        "evidence_count": len(signals),
        "metric_definitions": METRIC_DEFINITIONS,
        "material_change_detected": any(signal.material_change for signal in signals),
        "material_change_summary": material_change_summary(signals),
        "prediction_delta_vs_previous": [],
        "change_digest": [],
        "comparison_baseline_run_id": None,
        "weekend_phase": weekend_phase(weekend.stage, session_results),
        "evidence_quality": evidence_quality(signals),
        "evaluation_lock": evaluation_lock_payload(weekend.stage, session_results, now),
    }
    validate_prediction_payload(prediction)
    return prediction


def weekend_phase(stage: str, session_results: dict[str, list[dict[str, Any]]]) -> str:
    """A compact UI-facing phase. Keep this deterministic and independent from prose prompts."""
    if session_results.get("race") or stage == "post_race":
        return "review"
    if session_results.get("qualifying"):
        return "race_forecast"
    if session_results.get("sprint"):
        return "qualifying_and_race_forecast"
    if session_results.get("sprint_qualifying"):
        return "sprint_forecast"
    if session_results.get("fp3"):
        return "qualifying_forecast"
    if session_results.get("fp1") or session_results.get("fp2"):
        return "friday_intelligence"
    return "weekend_preview"


def evidence_quality(signals: list[ExtractedSignal]) -> dict[str, Any]:
    if not signals:
        return {
            "score": 0.0, "grade": "WAITING", "signal_count": 0,
            "official_signal_share": 0.0, "confirmed_signal_share": 0.0,
            "corroborated_signal_share": 0.0,
        }
    total = len(signals)
    official = sum(1 for signal in signals if str(signal.source_tier).lower() in {"a", "tier_a", "tier_1", "tier1", "1", "official"})
    confirmed = sum(1 for signal in signals if signal.is_confirmed)
    corroborated = sum(1 for signal in signals if str(signal.corroboration_status).lower() in {"corroborated", "confirmed", "official", "officially_confirmed"})
    weighted_confidence = sum(max(0.0, min(1.0, float(signal.confidence))) for signal in signals) / total
    official_share = official / total
    confirmed_share = confirmed / total
    corroborated_share = corroborated / total
    score = max(0.0, min(1.0, weighted_confidence * 0.45 + official_share * 0.25 + confirmed_share * 0.2 + corroborated_share * 0.1))
    grade = "HIGH" if score >= 0.72 else "MEDIUM" if score >= 0.48 else "LOW"
    return {
        "score": round(score, 3),
        "grade": grade,
        "signal_count": total,
        "official_signal_share": round(official_share, 3),
        "confirmed_signal_share": round(confirmed_share, 3),
        "corroborated_signal_share": round(corroborated_share, 3),
    }


def evaluation_lock_payload(stage: str, session_results: dict[str, list[dict[str, Any]]], timestamp: str) -> dict[str, Any]:
    if session_results.get("race") or stage == "post_race":
        return {"status": "result_available", "locked_at": None, "reason": "Race result is available; evaluation must use a pre-race history snapshot."}
    if session_results.get("qualifying"):
        return {"status": "lock_candidate", "locked_at": timestamp, "reason": "Qualifying is complete; this snapshot is eligible for post-race evaluation."}
    return {"status": "open", "locked_at": None, "reason": "Prediction remains live while decisive weekend evidence is still arriving."}


def driver_probabilities(
    drivers: list[dict[str, Any]],
    signals: list[ExtractedSignal],
    *,
    starting_positions: dict[str, int] | None = None,
    market: str = "race",
    learning_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not drivers:
        return ([{"driver": "TBD", "team": None, "probability": 1.0, "rank": 1, "probability_band": "unknown", "delta_vs_previous": None}], [])

    max_points = max([driver["points"] for driver in drivers] + [1.0])
    scores: dict[str, float] = {}
    teams: dict[str, str] = {}
    for driver in drivers:
        name = driver["driver"]
        teams[name] = driver["team"]
        points_factor = driver["points"] / max_points if max_points else 0
        position_factor = max(0.0, 1.0 - (driver["position"] - 1) * 0.045)
        scores[name] = 0.35 + points_factor * 0.55 + position_factor * 0.35 + driver["wins"] * 0.03
        if starting_positions:
            scores[name] += grid_win_score_adjustment(starting_position_for_driver(starting_positions, name), market)

    adjustment_log: list[dict[str, Any]] = []
    for driver in list(scores):
        learning_delta = driver_score_adjustment(learning_state, driver)
        if learning_delta == 0:
            continue
        scores[driver] = max(0.01, scores[driver] + learning_delta)
        if market == "race":
            adjustment_log.append(
                {
                    "target_type": "driver",
                    "target_name": driver,
                    "signal_id": "learning_state",
                    "signal_type": "result_learning",
                    "delta": round(learning_delta, 4),
                    "reason": "Past official result evaluation adjusted this driver slightly.",
                }
            )

    for signal in signals:
        if not signal.can_shift_probability:
            continue
        delta = SIGNAL_DELTAS.get(signal.signal_type)
        if delta is None:
            continue
        targets = [driver for driver in signal.drivers if driver in scores] or []
        if not targets and signal.teams:
            targets = [driver for driver, team in teams.items() if team in signal.teams]
        if not targets:
            continue
        weighted_delta = delta * signal.confidence * signal.source_reliability_weight
        for driver in targets:
            scores[driver] = max(0.01, scores[driver] + weighted_delta)
            adjustment_log.append(
                {
                    "target_type": "driver",
                    "target_name": driver,
                    "signal_id": signal.signal_id,
                    "signal_type": signal.signal_type,
                    "delta": round(weighted_delta, 4),
                    "reason": signal.evidence_summary,
                }
            )

    temperature = WIN_SCORE_TEMPERATURE.get(market, WIN_SCORE_TEMPERATURE["race"])
    raw_probabilities = {
        driver["driver"]: exp(scores[driver["driver"]] * temperature) * long_shot_multiplier(driver)
        for driver in drivers
        if driver["driver"] in scores
    }
    probabilities = normalized_probabilities(raw_probabilities)
    entries = []
    for driver, score in scores.items():
        probability = probabilities[driver]
        entries.append(
            {
                "driver": driver,
                "team": teams[driver],
                "probability": round(probability, 4),
                "rank": 0,
                "probability_band": probability_band(probability),
                "delta_vs_previous": None,
            }
        )
    entries.sort(key=lambda item: item["probability"], reverse=True)
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index
    return entries, adjustment_log


def grid_win_score_adjustment(starting_position: int | None, market: str) -> float:
    if starting_position is None:
        return 0.0
    sprint_multiplier = 1.18 if market == "sprint" else 1.0
    if starting_position == 1:
        return 1.15 * sprint_multiplier
    if starting_position == 2:
        return 0.92 * sprint_multiplier
    if starting_position == 3:
        return 0.72 * sprint_multiplier
    if starting_position <= 5:
        return 0.46 * sprint_multiplier
    if starting_position <= 8:
        return 0.18 * sprint_multiplier
    if starting_position <= 12:
        return -0.08 * sprint_multiplier
    return -0.28 * sprint_multiplier


def long_shot_multiplier(driver: dict[str, Any]) -> float:
    points = float(driver.get("points") or 0.0)
    position = int(driver.get("position") or 99)
    if points <= 0:
        return LONG_SHOT_MULTIPLIERS["pointsless"]
    if position > 14:
        return LONG_SHOT_MULTIPLIERS["backmarker"]
    if position > 10:
        return LONG_SHOT_MULTIPLIERS["lower_midfield"]
    if position > 6:
        return LONG_SHOT_MULTIPLIERS["midfield"]
    return LONG_SHOT_MULTIPLIERS["front"]


def podium_probabilities(
    driver_entries: list[dict[str, Any]],
    *,
    weekend: WeekendContext,
    signals: list[ExtractedSignal],
    starting_positions: dict[str, int],
    safety_car: dict[str, Any],
    volatility: dict[str, Any],
) -> list[dict[str, Any]]:
    scored_entries = []
    for entry in driver_entries:
        driver = entry.get("driver") or ""
        team = entry.get("team")
        starting_position = starting_position_for_driver(starting_positions, driver)
        score = entry["probability"] * 2.35 + 0.006 + grid_podium_adjustment(starting_position, weekend)
        scored_entries.append({**entry, "_podium_score": max(0.001, score)})

    for _ in range(12):
        annotated = annotate_podium_entries(
            scored_entries,
            weekend=weekend,
            signals=signals,
            starting_positions=starting_positions,
            safety_car=safety_car,
            volatility=volatility,
        )
        unreasoned = next((entry for entry in annotated[:3] if is_unreasoned_aggressive_call(entry)), None)
        if not unreasoned:
            return annotated
        for entry in scored_entries:
            if entry.get("driver") == unreasoned.get("driver"):
                entry["_podium_score"] *= 0.35
                break

    return annotate_podium_entries(
        scored_entries,
        weekend=weekend,
        signals=signals,
        starting_positions=starting_positions,
        safety_car=safety_car,
        volatility=volatility,
    )


def constructor_probabilities(driver_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    for entry in driver_entries:
        team = entry.get("team") or "Unknown"
        totals[team] += entry["probability"]
    probabilities = normalized_probabilities(dict(totals))
    entries = [
        {
            "driver": None,
            "team": team,
            "probability": probabilities[team],
            "rank": 0,
            "probability_band": probability_band(probabilities[team]),
            "delta_vs_previous": None,
        }
        for team in totals
    ]
    entries.sort(key=lambda item: item["probability"], reverse=True)
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index
    return entries


def performance_comparison(driver_entries: list[dict[str, Any]], signals: list[ExtractedSignal]) -> list[dict[str, Any]]:
    if not driver_entries:
        return []
    max_probability = max(entry["probability"] for entry in driver_entries) or 1.0
    entries = []
    for entry in driver_entries:
        driver = entry["driver"]
        team = entry.get("team")
        base = clamp(48 + (entry["probability"] / max_probability) * 34, 0, 100)
        single_lap = base + signal_index_adjustment(driver, team, signals, {
            "single_lap_pace_positive": 7,
            "single_lap_pace_negative": -7,
            "traffic_or_flags_compromised_lap": 3,
        })
        race_pace = base + signal_index_adjustment(driver, team, signals, {
            "race_pace_positive": 8,
            "race_pace_negative": -8,
            "long_run_data_contaminated": -2,
        })
        tyre = 62 + signal_index_adjustment(driver, team, signals, {
            "tyre_degradation_positive": 9,
            "tyre_degradation_negative": -10,
            "strategic_tyre_offset": 5,
        })
        reliability = 78 + signal_index_adjustment(driver, team, signals, {
            "reliability_concern": -14,
            "power_unit_concern": -16,
            "cooling_concern": -10,
            "brake_or_suspension_concern": -12,
            "floor_or_bodywork_damage": -10,
        })
        strategy = 64 + signal_index_adjustment(driver, team, signals, {
            "strategy_volatility_increase": 4,
            "setup_tradeoff_race_for_quali": -4,
            "setup_tradeoff_quali_for_race": 4,
            "upgrade_positive": 3,
        })
        speed = single_lap * 0.42 + race_pace * 0.46 + tyre * 0.12
        overall = speed * 0.56 + reliability * 0.22 + strategy * 0.22
        entries.append(
            {
                "driver": driver,
                "team": team,
                "single_lap_index": round(clamp(single_lap, 0, 100), 1),
                "race_pace_index": round(clamp(race_pace, 0, 100), 1),
                "tyre_management_index": round(clamp(tyre, 0, 100), 1),
                "reliability_index": round(clamp(reliability, 0, 100), 1),
                "strategy_index": round(clamp(strategy, 0, 100), 1),
                "speed_index": round(clamp(speed, 0, 100), 1),
                "overall_index": round(clamp(overall, 0, 100), 1),
            }
        )
    entries.sort(key=lambda item: item["overall_index"], reverse=True)
    return entries


def normalized_probabilities(raw_probabilities: dict[str, float], precision: int = 4) -> dict[str, float]:
    if not raw_probabilities:
        return {}
    scale = 10**precision
    positive_values = {key: max(0.0, value) for key, value in raw_probabilities.items()}
    total = sum(positive_values.values())
    if total <= 0:
        equal_floor = scale // len(positive_values)
        units = {key: equal_floor for key in positive_values}
        for key in list(positive_values)[: scale - sum(units.values())]:
            units[key] += 1
        return {key: round(value / scale, precision) for key, value in units.items()}

    scaled = {key: (value / total) * scale for key, value in positive_values.items()}
    units = {key: int(value) for key, value in scaled.items()}
    remainder = scale - sum(units.values())
    ordered = sorted(scaled, key=lambda key: scaled[key] - units[key], reverse=True)
    for key in ordered[:remainder]:
        units[key] += 1
    return {key: round(value / scale, precision) for key, value in units.items()}


def normalized_session_results(session_results: dict[str, list[dict[str, Any]]] | None) -> dict[str, list[dict[str, Any]]]:
    payload = {key: [] for key in SESSION_RESULT_KEYS}
    if not session_results:
        return payload
    for key in SESSION_RESULT_KEYS:
        rows = session_results.get(key) or []
        payload[key] = [normalize_session_result_row(row) for row in rows]
    return payload


def normalize_session_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": int(row.get("position") or 0),
        "driver": str(row.get("driver") or "Unknown"),
        "constructor": str(row.get("constructor") or "Unknown"),
        "time_or_gap": str(row.get("time_or_gap") or ""),
        "laps": row.get("laps") if row.get("laps") is None else int(row.get("laps") or 0),
        "status": str(row.get("status") or ""),
        "source": str(row.get("source") or "Unknown"),
        "is_official": bool(row.get("is_official", False)),
    }


def starting_positions_for_market(market: str, session_results: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    if market == "sprint":
        rows = session_results.get("sprint_qualifying") or []
    else:
        rows = session_results.get("qualifying") or []
    positions: dict[str, int] = {}
    for row in rows:
        if not row.get("driver") or not isinstance(row.get("position"), int) or row["position"] <= 0:
            continue
        for key in driver_position_keys(str(row["driver"])):
            positions.setdefault(key, row["position"])
    return positions


def starting_position_for_driver(starting_positions: dict[str, int], driver: str) -> int | None:
    for key in driver_position_keys(driver):
        if key in starting_positions:
            return starting_positions[key]
    return None


def driver_position_keys(driver: str) -> list[str]:
    exact = str(driver).strip()
    keys = [exact, compact_driver_key(exact), last_name_key(exact)]
    unique = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return unique


def compact_driver_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def last_name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    tokens = re.findall(r"[A-Za-z0-9]+", ascii_text)
    return compact_driver_key(tokens[-1]) if tokens else ""


def grid_podium_adjustment(starting_position: int | None, weekend: WeekendContext) -> float:
    if starting_position is None:
        return 0.0
    text = f"{weekend.grand_prix_name} {weekend.circuit_name}".lower()
    hard_to_overtake = any(track in text for track in HARD_TO_OVERTAKE)
    if starting_position <= 3:
        return 0.18 if hard_to_overtake else 0.14
    if starting_position <= 6:
        return 0.07 if hard_to_overtake else 0.09
    if starting_position <= 10:
        return -0.04 if hard_to_overtake else 0.0
    return -0.12 if hard_to_overtake else -0.07


def annotate_podium_entries(
    scored_entries: list[dict[str, Any]],
    *,
    weekend: WeekendContext,
    signals: list[ExtractedSignal],
    starting_positions: dict[str, int],
    safety_car: dict[str, Any],
    volatility: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = sorted(scored_entries, key=lambda item: item["_podium_score"], reverse=True)
    annotated = []
    for index, entry in enumerate(entries, start=1):
        driver = entry.get("driver") or ""
        starting_position = starting_position_for_driver(starting_positions, driver)
        positions_to_gain = starting_position - index if starting_position else None
        reasoning = podium_reasoning_factors(entry, weekend, signals, positions_to_gain, safety_car, volatility)
        probability = min(0.88, max(0.004, entry["_podium_score"]))
        annotated.append(
            {
                **{key: value for key, value in entry.items() if not key.startswith("_")},
                "probability": round(probability, 4),
                "rank": index,
                "starting_position": starting_position,
                "predicted_finish_position": index,
                "positions_to_gain": positions_to_gain,
                "reasoning_factors": reasoning,
            }
        )
    return annotated


def podium_reasoning_factors(
    entry: dict[str, Any],
    weekend: WeekendContext,
    signals: list[ExtractedSignal],
    positions_to_gain: int | None,
    safety_car: dict[str, Any],
    volatility: dict[str, Any],
) -> list[str]:
    driver = entry.get("driver") or ""
    team = entry.get("team")
    factors = signal_reasoning_factors(driver, team, signals)
    if positions_to_gain and positions_to_gain > 0:
        if volatility.get("level") == "high":
            factors.append("high volatility race profile")
        elif safety_car.get("risk_level") in {"high", "medium"}:
            factors.append("Safety Car risk can compress the field")
        if not is_hard_to_overtake(weekend):
            factors.append("overtaking profile is not unusually restrictive")
    if entry.get("probability", 0) >= 0.2:
        factors.append("strong baseline model share")
    return unique_preserving_order(factors)


def signal_reasoning_factors(driver: str, team: str | None, signals: list[ExtractedSignal]) -> list[str]:
    factors: list[str] = []
    labels = {
        "race_pace_positive": "positive race pace signal",
        "tyre_degradation_positive": "strong tyre degradation signal",
        "single_lap_pace_positive": "positive single-lap pace signal",
        "upgrade_positive": "positive upgrade signal",
        "track_specific_suitability_positive": "track suitability signal",
    }
    for signal in signals:
        if not signal.can_shift_probability:
            continue
        driver_match = driver in signal.drivers
        team_match = bool(team and team in signal.teams)
        if driver_match or team_match:
            label = labels.get(signal.signal_type)
            if label:
                factors.append(label)
    return unique_preserving_order(factors)


def unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def is_hard_to_overtake(weekend: WeekendContext) -> bool:
    text = f"{weekend.grand_prix_name} {weekend.circuit_name}".lower()
    return any(track in text for track in HARD_TO_OVERTAKE)


def is_unreasoned_aggressive_call(entry: dict[str, Any]) -> bool:
    gain = entry.get("positions_to_gain")
    if gain is None or gain < MARKET_THRESHOLDS["aggressive_podium_gain"]:
        return False
    return not entry.get("reasoning_factors")


def winner_payload(
    top_driver: dict[str, Any],
    starting_positions: dict[str, int],
    signals: list[ExtractedSignal],
    weekend: WeekendContext,
    safety_car: dict[str, Any],
    volatility: dict[str, Any],
) -> dict[str, Any]:
    driver = top_driver.get("driver") or "TBD"
    starting_position = starting_position_for_driver(starting_positions, driver)
    positions_to_gain = starting_position - 1 if starting_position else None
    return {
        "driver": driver,
        "probability": top_driver.get("probability", 0.0),
        "starting_position": starting_position,
        "predicted_finish_position": 1,
        "positions_to_gain": positions_to_gain,
        "reasoning_factors": podium_reasoning_factors(top_driver, weekend, signals, positions_to_gain, safety_car, volatility),
    }


def market_outlook(entries: list[dict[str, Any]], market_name: str, thresholds: dict[str, float] = MARKET_THRESHOLDS) -> dict[str, Any]:
    if not entries:
        return {
            "state": "unavailable",
            "label": "No model output",
            "explanation": f"No {market_name} probability distribution is available yet.",
            "top_probability": 0.0,
            "gap_to_second": 0.0,
            "thresholds": thresholds,
        }
    top = entries[0]
    second = entries[1] if len(entries) > 1 else None
    top_probability = top["probability"]
    gap = top_probability - (second["probability"] if second else 0.0)
    driver = top.get("driver") or top.get("team") or "the leader"
    if top_probability < thresholds["no_clear_favourite_below"]:
        state = "no_clear_favourite"
        label = "No clear favourite"
        explanation = f"Open {market_name} outlook: {driver} leads a very flat model at {round(top_probability * 100)}%."
    elif gap < thresholds["close_gap_below"]:
        state = "close_front_group"
        label = "Close front group"
        explanation = f"{driver} leads, but the gap to P2 is only {round(gap * 100)} percentage points."
    elif top_probability >= thresholds["model_favourite_min"] and gap >= thresholds["model_favourite_gap"]:
        state = "model_favourite"
        label = "Model favourite"
        explanation = f"{driver} is the model favourite at {round(top_probability * 100)}%."
    else:
        state = "open_outlook"
        label = "Open outlook"
        explanation = f"{driver} leads the model, but the distribution is not decisive."
    return {
        "state": state,
        "label": label,
        "explanation": explanation,
        "top_probability": top_probability,
        "gap_to_second": round(gap, 4),
        "thresholds": thresholds,
    }


def validate_prediction_payload(prediction: dict[str, Any]) -> None:
    race = prediction["race"]
    validate_probability_entries("race driver win", race["driver_win_probabilities"])
    validate_probability_entries("constructor win", race["constructor_win_probabilities"])
    validate_top_winner("race", race["predicted_winner"], race["driver_win_probabilities"])
    validate_podium_entries("race", race["predicted_podium"], race["driver_podium_probabilities"])
    sprint = prediction["sprint"]
    if sprint.get("enabled"):
        validate_probability_entries("sprint driver win", sprint["driver_win_probabilities"])
        validate_top_winner("sprint", sprint["predicted_winner"], sprint["driver_win_probabilities"])
        validate_podium_entries("sprint", sprint.get("predicted_podium", []), sprint["driver_podium_probabilities"])


def validate_probability_entries(label: str, entries: list[dict[str, Any]]) -> None:
    if not entries:
        raise ValueError(f"{label} probabilities are empty")
    probabilities = [entry.get("probability", 0.0) for entry in entries]
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError(f"{label} probabilities must be between 0 and 1")
    if abs(sum(probabilities) - 1.0) > 0.0001:
        raise ValueError(f"{label} probabilities must sum to 1.0")


def validate_top_winner(label: str, winner: dict[str, Any] | None, entries: list[dict[str, Any]]) -> None:
    if not winner:
        raise ValueError(f"{label} predicted winner is missing")
    top = max(entries, key=lambda item: item.get("probability", 0.0))
    if winner.get("driver") != top.get("driver"):
        raise ValueError(f"{label} predicted winner is not the highest-probability driver")


def validate_podium_entries(label: str, predicted_podium: list[dict[str, Any]], podium_entries: list[dict[str, Any]]) -> None:
    podium_names = {entry.get("driver") for entry in podium_entries}
    for entry in predicted_podium:
        if entry.get("driver") not in podium_names:
            raise ValueError(f"{label} podium driver missing from podium probability list")
        if is_unreasoned_aggressive_call(entry):
            raise ValueError(f"{label} podium call for {entry.get('driver')} needs reasoning")


def signal_index_adjustment(driver: str, team: str | None, signals: list[ExtractedSignal], weights: dict[str, float]) -> float:
    total = 0.0
    for signal in signals:
        if not signal.can_shift_probability:
            continue
        weight = weights.get(signal.signal_type)
        if weight is None:
            continue
        driver_match = driver in signal.drivers
        team_match = bool(team and team in signal.teams)
        if not signal.drivers and not signal.teams:
            continue
        if driver_match or team_match:
            total += weight * signal.confidence * signal.source_reliability_weight
    return total


def safety_car_outlook(weekend: WeekendContext, signals: list[ExtractedSignal]) -> dict[str, Any]:
    text = f"{weekend.grand_prix_name} {weekend.circuit_name}".lower()
    base = 0.52
    factors = ["Base heuristic from modern F1 race interruption frequency."]
    if any(track in text for track in HIGH_SC_TRACKS):
        base += 0.14
        factors.append("Circuit profile has elevated interruption risk.")
    if any(signal.signal_type in {"weather_volatility", "safety_car_risk_increase"} for signal in signals):
        base += 0.08
        factors.append("Current source signals increase race volatility.")
    probability = max(0.05, min(0.9, base))
    expected = probability * (1.15 if probability > 0.62 else 0.9)
    return {
        "probability_at_least_one": round(probability, 2),
        "expected_count": round(expected, 1),
        "risk_level": "high" if probability >= 0.68 else "medium" if probability >= 0.45 else "low",
        "reasoning_factors": factors,
    }


def forecast_confidence(stage: str, source_count: int, signals: list[ExtractedSignal]) -> dict[str, Any]:
    base = STAGE_CONFIDENCE.get(stage, 0.4)
    evidence_bonus = min(0.16, source_count * 0.006 + len(signals) * 0.01)
    score = min(0.86, base + evidence_bonus)
    return {
        "level": "high" if score >= 0.72 else "medium" if score >= 0.48 else "low",
        "score": round(score, 2),
        "reasons": [
            f"Stage weighting: {stage.replace('_', ' ')}.",
            f"{source_count} source checks and {len(signals)} race signals available.",
            "Forecast remains heuristic until calibrated against historical outcomes.",
        ],
    }


def race_volatility(weekend: WeekendContext, signals: list[ExtractedSignal], safety_car: dict[str, Any]) -> dict[str, Any]:
    score = 0.4
    reasons = []
    if safety_car["risk_level"] == "high":
        score += 0.18
        reasons.append("High Safety Car risk.")
    if any(signal.signal_type in {"weather_volatility", "pending_investigation"} for signal in signals):
        score += 0.16
        reasons.append("Weather or official-process uncertainty is present.")
    text = f"{weekend.grand_prix_name} {weekend.circuit_name}".lower()
    if any(track in text for track in HARD_TO_OVERTAKE):
        score += 0.08
        reasons.append("Track position is unusually important at this circuit.")
    score = min(0.9, score)
    return {
        "level": "high" if score >= 0.67 else "medium" if score >= 0.44 else "low",
        "score": round(score, 2),
        "reasons": reasons or ["No major volatility amplifier identified in current inputs."],
    }


def upgrade_summaries(signals: list[ExtractedSignal], positive: bool) -> list[str]:
    wanted = {"upgrade_positive", "race_pace_positive", "single_lap_pace_positive"} if positive else {"confirmed_grid_penalty", "reliability_concern", "power_unit_concern", "tyre_degradation_negative"}
    return [signal.evidence_summary for signal in signals if signal.signal_type in wanted][:5]


def key_uncertainties(weekend: WeekendContext, signals: list[ExtractedSignal]) -> list[str]:
    uncertainties = [
        "Practice pace is fuel-load and run-plan sensitive until qualifying and final grid documents are available.",
        "Probabilities are rounded heuristic model outlooks, not betting advice.",
    ]
    if weekend.stage in {"pre_weekend", "after_fp1"}:
        uncertainties.append("The forecast is still prior-heavy at this stage.")
    if any(signal.requires_corroboration for signal in signals):
        uncertainties.append("Some editorial signals are single-source and need corroboration.")
    if weekend.is_sprint_weekend:
        uncertainties.append("Sprint sessions can reveal useful tyre and race-pace signals, but traffic and risk management can distort the read.")
    return uncertainties


def material_change_summary(signals: list[ExtractedSignal]) -> str:
    material = [signal for signal in signals if signal.material_change]
    if not material:
        return "No material source signal changed the model inputs."
    top = material[0]
    return f"{top.signal_type.replace('_', ' ').title()}: {top.evidence_summary}"


def probability_band(probability: float) -> str:
    if probability >= 0.25:
        return "front_pack"
    if probability >= 0.1:
        return "contender"
    if probability >= 0.04:
        return "outside_chance"
    return "long_shot"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
