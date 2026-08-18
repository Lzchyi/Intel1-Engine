from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .providers import AIProvider, DeepSeekProvider, OpenAIProvider
from .signal_store import StoredSignal
from .structured_data import WeekendContext
from .time_utils import isoformat, utc_now


PREDICTION_PROMPT_VERSION = "hybrid-prediction-v5-session-aware-senior-analyst"


PREDICTION_ENGINE_PROMPT = """
You are Intel1's senior Formula 1 race-intelligence analyst.
Think like a race engineer, strategy analyst, technical journalist, and probabilistic forecaster — not a fan, tipster, or generic news summariser.

Use ONLY the structured evidence supplied by the backend. Do not browse, invent results, or fill missing facts from memory.
Treat the evidenceBoard as the distilled weekend brief and explicitly distinguish observed fact from inference.

EVIDENCE PRIORITY
Before a major forecast, expect the backend evidence sweep to include FIA event documents, official F1 session data, car-upgrade/technical submissions where available, Pirelli tyre information, confirmed team/driver statements, trusted specialist reporting, and relevant weather/track conditions. Missing evidence must be treated as missing, never filled from memory.
1. Official FIA classifications, decisions, final grids, scrutineering/technical documents and penalties.
2. Official F1 timing/session data, Pirelli information, and confirmed team/driver statements.
3. Corroborated specialist technical/paddock reporting.
4. Single-source reporting and team PR claims.
5. Rumours/social interpretation: context only; never decisive by itself.

RECENCY RULE
New weekend evidence overrides stale priors when it is representative and credible. Do not keep rewarding an expected upgrade, historical strength, or pre-weekend narrative when later session evidence contradicts it. Explain the contradiction.

SESSION INTERPRETATION
- FP1 is weak evidence unless the signal is unusually clean or officially confirmed.
- FP2 on a normal weekend is usually the strongest practice input for race pace and degradation, but correct for tyre, fuel, traffic, track evolution, run length and timing of the run.
- FP3 is more useful for qualifying direction than race pace.
- Sprint weekends have less practice evidence; Sprint Qualifying and the Sprint add useful single-lap/race evidence but can still be distorted by tyre strategy, traffic and risk management.
- After qualifying, grid position matters strongly, but never treat qualifying order as race pace automatically.
- Final pre-race analysis prioritises final grid, penalties, parc-ferme/component changes, weather, tyre allocation/strategy, long-run evidence, reliability and overtaking difficulty.

REPORT INTENT
Friday checkpoint: forecast the next Saturday competitive session, not the Sunday race as if the weekend were already settled.
Saturday checkpoint: after decisive Saturday running/Qualifying, forecast the Sunday Grand Prix.
Use the newest evidence available at the checkpoint and explicitly state what is still unknown.

ANALYST DISCIPLINE
For the final call, reason in this order: Observed facts -> interpretation -> forecast -> confidence -> main uncertainty.
Check car upgrades and setup changes against actual on-track evidence rather than assuming claimed gains are real.
Separate one-lap pace, long-run pace, tyre degradation, strategy flexibility, driver execution, reliability and track-position effects.
When evidence conflicts, prefer the most recent representative high-quality evidence and preserve the disagreement as uncertainty.
Avoid false precision and hype. Never use betting language.

OUTPUT RULES
Return strict JSON only.
All requested probabilities must be finite, non-negative percentages and form an internally coherent forecast distribution.
Always make one winner pick from your highest win probability, even when confidence is low.
The analyst_report must be consistent with the probabilities, podium outlook, constructor outlook and Safety Car probability.
Use confidence and risk language to express uncertainty rather than pretending certainty.
"""


@dataclass
class PredictionArena:
    chatgpt_provider: AIProvider | None = None
    deepseek_provider: AIProvider | None = None

    def run(
        self,
        *,
        run_id: str,
        weekend: WeekendContext,
        baseline_prediction: dict[str, Any],
        stored_signals: list[StoredSignal],
        session_results: dict[str, list[dict[str, Any]]],
        skip_ai: bool = False,
    ) -> dict[str, Any]:
        evidence_board = build_evidence_board(
            weekend=weekend,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
        )
        chatgpt = ChatGPTPredictionEngine(self.chatgpt_provider or OpenAIProvider()).predict(
            run_id=run_id,
            weekend=weekend,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
            skip_ai=skip_ai,
        )
        deepseek = DeepSeekPredictionEngine(self.deepseek_provider or DeepSeekProvider()).predict(
            run_id=run_id,
            weekend=weekend,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
            skip_ai=skip_ai,
        )
        consensus = consensus_prediction(
            run_id=run_id,
            weekend=weekend,
            predictions=[chatgpt, deepseek],
            session_results=session_results,
            stored_signals=stored_signals,
        )
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "weekend_id": weekend.weekend_id,
            "stage": weekend.stage,
            "updated_at": isoformat(utc_now()),
            "promptVersion": PREDICTION_PROMPT_VERSION,
            "evidenceBoard": evidence_board,
            "predictions": {
                "chatgpt": chatgpt,
                "deepseek": deepseek,
                "intel1_consensus": consensus,
            },
        }


class ChatGPTPredictionEngine:
    provider_name = "chatgpt"

    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def predict(
        self,
        *,
        run_id: str,
        weekend: WeekendContext,
        baseline_prediction: dict[str, Any],
        stored_signals: list[StoredSignal],
        session_results: dict[str, list[dict[str, Any]]],
        skip_ai: bool = False,
    ) -> dict[str, Any]:
        if skip_ai:
            return skipped_provider_prediction(
                provider_name=self.provider_name,
                provider=self.provider,
                run_id=run_id,
                baseline_prediction=baseline_prediction,
                stored_signals=stored_signals,
                session_results=session_results,
            )
        return provider_prediction(
            provider_name=self.provider_name,
            provider=self.provider,
            run_id=run_id,
            weekend=weekend,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
        )


class DeepSeekPredictionEngine(ChatGPTPredictionEngine):
    provider_name = "deepseek"


def build_prediction_arena(
    *,
    run_id: str,
    weekend: WeekendContext,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
    chatgpt_provider: AIProvider | None = None,
    deepseek_provider: AIProvider | None = None,
    skip_ai: bool = False,
) -> dict[str, Any]:
    return PredictionArena(chatgpt_provider=chatgpt_provider, deepseek_provider=deepseek_provider).run(
        run_id=run_id,
        weekend=weekend,
        baseline_prediction=baseline_prediction,
        stored_signals=stored_signals,
        session_results=session_results,
        skip_ai=skip_ai,
    )


def provider_prediction(
    *,
    provider_name: str,
    provider: AIProvider,
    run_id: str,
    weekend: WeekendContext,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    prompt_payload = prediction_prompt_payload(
        weekend=weekend,
        baseline_prediction=baseline_prediction,
        stored_signals=stored_signals,
        session_results=session_results,
    )
    try:
        validate_prediction_prompt(prompt_payload)
        response = provider.complete_json(system_prompt=PREDICTION_ENGINE_PROMPT, user_payload=prompt_payload)
        prediction = prediction_from_provider_payload(
            provider_name=provider_name,
            run_id=run_id,
            payload=response.payload,
            provider_request_id=response.provider_request_id,
            model_used=response.model_used,
            model_temperature=response.model_temperature,
            stored_signals=stored_signals,
            session_results=session_results,
        )
        prediction["providerStatus"] = "ok"
        validate_model_prediction(prediction)
        return prediction
    except Exception as error:
        fallback = baseline_model_prediction(
            provider_name=provider_name,
            run_id=run_id,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
            model_used=getattr(provider, "model_name", f"{provider_name}-unavailable"),
            provider_request_id=None,
            model_temperature=getattr(provider, "model_temperature", None),
        )
        fallback["providerStatus"] = "deterministic_baseline"
        fallback["validation_errors"] = [f"{type(error).__name__}: {str(error)[:180]}"]
        validate_model_prediction(fallback)
        return fallback


def skipped_provider_prediction(
    *,
    provider_name: str,
    provider: AIProvider,
    run_id: str,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    fallback = baseline_model_prediction(
        provider_name=provider_name,
        run_id=run_id,
        baseline_prediction=baseline_prediction,
        stored_signals=stored_signals,
        session_results=session_results,
        model_used=getattr(provider, "model_name", f"{provider_name}-skipped"),
        provider_request_id=None,
        model_temperature=getattr(provider, "model_temperature", None),
    )
    fallback["providerStatus"] = "deterministic_baseline"
    fallback["validation_errors"] = ["AI skipped by run option."]
    validate_model_prediction(fallback)
    return fallback


def prediction_prompt_payload(
    *,
    weekend: WeekendContext,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "task": "Produce a strict JSON F1 prediction from structured signals and official session data only.",
        "promptVersion": PREDICTION_PROMPT_VERSION,
        "eventId": weekend.weekend_id,
        "stage": weekend.stage,
        "officialSessionData": session_results,
        "structuredSignals": [signal.to_dict() for signal in stored_signals],
        "evidenceBoard": build_evidence_board(
            weekend=weekend,
            baseline_prediction=baseline_prediction,
            stored_signals=stored_signals,
            session_results=session_results,
        ),
        "stageWeightingGuide": stage_weighting_guide(weekend.stage),
        "decisionRules": [
            "Use the evidenceBoard as the main race brief; do not blindly copy baselinePrediction when late-session evidence points elsewhere.",
            "For final_pre_race and after_qualifying stages, official grid, qualifying, sprint result, strategy, tyre, weather, and Safety Car signals outrank older championship priors.",
            "Still choose the highest-chance winner even when the race is open; express uncertainty through confidence, probability spread, risk, and chaos labels.",
            "When models or sources disagree, keep the most evidence-supported winner and mention the risk in analyst_report.biggest_risks or weak_assumptions.",
        ],
        "baselinePrediction": compact_baseline_prediction(baseline_prediction),
        "outputSchema": {
            "confidence": "number 0-1",
            "predicted_winner": "string; must match highest win_probabilities driver",
            "win_probabilities": [{"driver": "string", "probability": "percent"}],
            "constructor_win_probabilities": [{"team": "string", "probability": "percent"}],
            "podium_probabilities": [{"driver": "string", "probability": "percent"}],
            "top10_probabilities": [{"driver": "string", "probability": "percent"}],
            "dnf_risk": [{"driver": "string", "probability": "percent"}],
            "safety_car_probability": "percent",
            "key_reasons": ["string"],
            "weak_assumptions": ["string"],
            "analyst_report": {
                "title": "short title, e.g. Canadian GP prediction",
                "assumption": "one short note about stage/data freshness, or empty string",
                "final_call": {
                    "winner_driver": "string; must match predicted_winner",
                    "winner_constructor": "string; must match highest constructor probability",
                    "podium": ["P1 driver", "P2 driver", "P3 driver"],
                    "highest_scoring_team": "string",
                    "safety_car_risk": "plain-language label with optional range",
                    "rain_impact": "Low|Medium|High or short label",
                    "chaos_level": "Low|Medium|High or short label",
                    "most_likely_upset_winner": "string",
                    "dark_horse_podium": ["driver"]
                },
                "narrative": [{"title": "Why this pick|Podium read|Race shape", "body": "1-3 compact sentences"}],
                "strategy": {"dry": "short dry strategy expectation", "wet_mixed": "short wet/mixed strategy expectation"},
                "biggest_risks": [{"risk": "short risk", "benefits": ["driver/team"]}],
                "final_answer": "compact final answer"
            },
        },
    }


def compact_baseline_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    race = prediction.get("race", {})
    return {
        "predicted_winner": race.get("predicted_winner"),
        "driver_win_probabilities": race.get("driver_win_probabilities", [])[:10],
        "constructor_win_probabilities": race.get("constructor_win_probabilities", [])[:10],
        "predicted_podium": race.get("predicted_podium", [])[:3],
        "safety_car": prediction.get("safety_car"),
        "forecast_confidence": prediction.get("forecast_confidence"),
        "race_volatility": prediction.get("race_volatility"),
        "key_uncertainties": prediction.get("key_uncertainties", [])[:6],
    }


def build_evidence_board(
    *,
    weekend: WeekendContext,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    ranked_signals = sorted(stored_signals, key=stored_signal_score, reverse=True)
    target_evidence: dict[str, list[dict[str, Any]]] = {}
    field_evidence: list[dict[str, Any]] = []
    high_impact_evidence: list[dict[str, Any]] = []

    for signal in ranked_signals:
        entry = compact_stored_signal(signal)
        target = signal.target or "field"
        if target == "field" or field_signal_type(signal.signalType):
            field_evidence.append(entry)
        else:
            target_evidence.setdefault(target, []).append(entry)
        if signal.strength >= 0.7 or signal.confidence >= 0.78 or signal.evidenceType == "official":
            high_impact_evidence.append(entry)

    return {
        "eventId": weekend.weekend_id,
        "stage": weekend.stage,
        "stageWeightingGuide": stage_weighting_guide(weekend.stage),
        "sourceMix": source_mix(stored_signals),
        "sessionSnapshot": session_snapshot(session_results),
        "baselineTop": baseline_top(baseline_prediction),
        "targetEvidence": {target: entries[:6] for target, entries in sorted(target_evidence.items())[:24]},
        "fieldEvidence": field_evidence[:12],
        "highImpactEvidence": high_impact_evidence[:12],
        "analystChecklist": analyst_checklist(weekend.stage),
    }


def compact_stored_signal(signal: StoredSignal) -> dict[str, Any]:
    return {
        "target": signal.target,
        "signalType": signal.signalType,
        "evidenceType": signal.evidenceType,
        "summary": signal.summary,
        "strength": round(float(signal.strength), 3),
        "confidence": round(float(signal.confidence), 3),
        "sourceQuality": round(float(signal.sourceQuality), 3),
        "sourceId": signal.sourceId,
        "sourceType": signal.sourceType,
        "evidenceUrl": signal.evidenceUrl,
    }


def stored_signal_score(signal: StoredSignal) -> float:
    score = float(signal.strength) * 0.45 + float(signal.confidence) * 0.35 + float(signal.sourceQuality) * 0.20
    if signal.evidenceType == "official":
        score += 0.12
    if field_signal_type(signal.signalType):
        score += 0.04
    return score


def field_signal_type(signal_type: str) -> bool:
    lowered = signal_type.lower()
    field_terms = ("weather", "safety_car", "strategy", "tyre", "tire", "grid", "penalty", "investigation", "parc_ferme")
    return any(term in lowered for term in field_terms)


def source_mix(stored_signals: list[StoredSignal]) -> dict[str, Any]:
    if not stored_signals:
        return {
            "signalCount": 0,
            "averageSourceQuality": 0.0,
            "bySourceType": {},
            "byEvidenceType": {},
            "topSources": [],
        }
    by_source_type: dict[str, int] = {}
    by_evidence_type: dict[str, int] = {}
    by_source_id: dict[str, int] = {}
    for signal in stored_signals:
        by_source_type[signal.sourceType] = by_source_type.get(signal.sourceType, 0) + 1
        by_evidence_type[signal.evidenceType] = by_evidence_type.get(signal.evidenceType, 0) + 1
        by_source_id[signal.sourceId] = by_source_id.get(signal.sourceId, 0) + 1
    top_sources = [
        {"sourceId": source_id, "signals": count}
        for source_id, count in sorted(by_source_id.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    return {
        "signalCount": len(stored_signals),
        "averageSourceQuality": round(sum(float(signal.sourceQuality) for signal in stored_signals) / len(stored_signals), 3),
        "bySourceType": by_source_type,
        "byEvidenceType": by_evidence_type,
        "topSources": top_sources,
    }


def session_snapshot(session_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for session_key in ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"]:
        rows = session_results.get(session_key) or []
        if not rows:
            continue
        snapshot[session_key] = {
            "leader": compact_session_row(rows[0]),
            "topFive": [compact_session_row(row) for row in rows[:5]],
            "officialRows": sum(1 for row in rows if row.get("is_official")),
        }
    return snapshot


def compact_session_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": row.get("position"),
        "driver": row.get("driver"),
        "constructor": row.get("constructor"),
        "time_or_gap": row.get("time_or_gap"),
        "status": row.get("status"),
        "source": row.get("source"),
        "is_official": bool(row.get("is_official")),
    }


def baseline_top(prediction: dict[str, Any]) -> dict[str, Any]:
    race = prediction.get("race", {})
    return {
        "predictedWinner": race.get("predicted_winner"),
        "topWinProbabilities": race.get("driver_win_probabilities", [])[:6],
        "topConstructorProbabilities": race.get("constructor_win_probabilities", [])[:6],
        "predictedPodium": race.get("predicted_podium", [])[:3],
        "safetyCar": prediction.get("safety_car"),
        "raceVolatility": prediction.get("race_volatility"),
    }


def stage_weighting_guide(stage: str) -> dict[str, Any]:
    if stage in {"after_qualifying", "final_pre_race"}:
        return {
            "priority": [
                "official final grid and qualifying order",
                "sprint result and sprint race pace on sprint weekends",
                "strategy guide, tyre availability, weather, and Safety Car probability",
                "corroborated race pace and team/driver statements",
                "baseline model and championship priors",
            ],
            "note": "Late-session evidence can override season priors when it is official, fresh, and race-relevant.",
        }
    if stage in {"after_sprint", "after_sprint_qualifying"}:
        return {
            "priority": [
                "sprint qualifying and sprint race evidence",
                "official penalties or investigations",
                "practice context with fuel/tyre caveats",
                "baseline model and championship priors",
            ],
            "note": "Sprint weekends produce strong but still context-dependent race evidence.",
        }
    if stage.startswith("after_fp"):
        return {
            "priority": [
                "official practice classification with tyre/fuel caveats",
                "long-run and race-pace signals",
                "weather, reliability, and upgrade evidence",
                "baseline model and championship priors",
            ],
            "note": "Practice signals are useful but should not dominate unless corroborated.",
        }
    if stage == "post_race":
        return {
            "priority": ["official race classification"],
            "note": "Official race results override prediction assumptions.",
        }
    return {
        "priority": [
            "official calendar and standings",
            "historical circuit profile",
            "verified pre-weekend technical/weather signals",
            "baseline model and championship priors",
        ],
        "note": "Before live sessions, keep confidence conservative.",
    }


def analyst_checklist(stage: str) -> list[str]:
    checklist = [
        "Separate observed facts from interpretation before making the forecast.",
        "Make one winner pick even when the probability spread is flat.",
        "Use confidence and chaos/risk labels to express uncertainty instead of refusing a favourite.",
        "Let recent representative weekend evidence override stale priors when justified, and explain the conflict.",
        "Check upgrades/setup claims against observed on-track evidence instead of assuming the claimed gain is real.",
        "Separate one-lap pace, long-run pace, degradation, strategy, reliability, driver execution, and track position.",
    ]
    if stage in {"after_qualifying", "final_pre_race"}:
        checklist.extend(
            [
                "Check whether pole, front row, sprint result, and race pace point to the same team.",
                "Check whether Safety Car, rain, or tyre offset creates a credible upset path.",
            ]
        )
    return checklist


def prediction_from_provider_payload(
    *,
    provider_name: str,
    run_id: str,
    payload: dict[str, Any],
    provider_request_id: str | None,
    model_used: str,
    model_temperature: float | None,
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    win_probabilities = normalize_percent_entries(payload.get("win_probabilities") or [], "driver")
    constructor_probabilities = normalize_percent_entries(payload.get("constructor_win_probabilities") or [], "team")
    podium_probabilities = normalize_optional_percent_entries(payload.get("podium_probabilities") or [], "driver")
    top_driver = win_probabilities[0]["driver"] if win_probabilities else ""
    predicted_winner = str(payload.get("predicted_winner") or top_driver)
    safety_car_probability = bounded_percent(payload.get("safety_car_probability", 0), "safety_car_probability")
    return {
        "provider": provider_name,
        "version": f"{run_id}:{provider_name}:v1",
        "modelUsed": model_used,
        "providerRequestId": provider_request_id,
        "modelTemperature": model_temperature,
        "confidence": bounded_unit(payload.get("confidence", 0.35), "confidence"),
        "sourceQuality": average_source_quality(stored_signals),
        "officialDataAvailability": official_data_availability(session_results),
        "sessionRecency": session_recency(session_results),
        "predicted_winner": predicted_winner,
        "win_probabilities": win_probabilities,
        "constructor_win_probabilities": constructor_probabilities,
        "podium_probabilities": podium_probabilities,
        "top10_probabilities": normalize_optional_percent_entries(payload.get("top10_probabilities") or [], "driver"),
        "dnf_risk": normalize_optional_percent_entries(payload.get("dnf_risk") or [], "driver"),
        "safety_car_probability": safety_car_probability,
        "key_reasons": short_string_list(payload.get("key_reasons")),
        "weak_assumptions": short_string_list(payload.get("weak_assumptions")),
        "analyst_report": analyst_report_from_payload(
            payload.get("analyst_report"),
            predicted_winner=predicted_winner,
            win_probabilities=win_probabilities,
            constructor_probabilities=constructor_probabilities,
            podium_probabilities=podium_probabilities,
            safety_car_probability=safety_car_probability,
            key_reasons=short_string_list(payload.get("key_reasons")),
        ),
        "validation_errors": [],
    }


def baseline_model_prediction(
    *,
    provider_name: str,
    run_id: str,
    baseline_prediction: dict[str, Any],
    stored_signals: list[StoredSignal],
    session_results: dict[str, list[dict[str, Any]]],
    model_used: str,
    provider_request_id: str | None,
    model_temperature: float | None,
) -> dict[str, Any]:
    race = baseline_prediction.get("race", {})
    win_probabilities = normalize_percent_entries(race.get("driver_win_probabilities") or [], "driver")
    constructor_probabilities = normalize_percent_entries(race.get("constructor_win_probabilities") or [], "team")
    podium = app_probability_entries_to_percent(race.get("driver_podium_probabilities") or race.get("predicted_podium") or [], "driver")
    top10 = [
        {"driver": entry["driver"], "probability": round(min(100.0, 45.0 + entry["probability"] * 1.8), 2)}
        for entry in win_probabilities[:10]
        if entry.get("driver")
    ]
    dnf = [{"driver": entry["driver"], "probability": 8.0} for entry in win_probabilities if entry.get("driver")]
    confidence = (baseline_prediction.get("forecast_confidence") or {}).get("score", 0.35)
    top_driver = win_probabilities[0]["driver"] if win_probabilities else str((race.get("predicted_winner") or {}).get("driver") or "TBD")
    return {
        "provider": provider_name,
        "version": f"{run_id}:{provider_name}:baseline:v1",
        "modelUsed": model_used,
        "providerRequestId": provider_request_id,
        "modelTemperature": model_temperature,
        "confidence": bounded_unit(confidence, "confidence"),
        "sourceQuality": average_source_quality(stored_signals),
        "officialDataAvailability": official_data_availability(session_results),
        "sessionRecency": session_recency(session_results),
        "predicted_winner": top_driver,
        "win_probabilities": win_probabilities,
        "constructor_win_probabilities": constructor_probabilities,
        "podium_probabilities": podium,
        "top10_probabilities": top10,
        "dnf_risk": dnf,
        "safety_car_probability": round(float((baseline_prediction.get("safety_car") or {}).get("probability_at_least_one", 0.0)) * 100, 2),
        "key_reasons": ["Deterministic baseline from official structured inputs and current Intel1 rule model."],
        "weak_assumptions": short_string_list(baseline_prediction.get("key_uncertainties")),
        "analyst_report": analyst_report_from_payload(
            {},
            predicted_winner=top_driver,
            win_probabilities=win_probabilities,
            constructor_probabilities=constructor_probabilities,
            podium_probabilities=podium,
            safety_car_probability=round(float((baseline_prediction.get("safety_car") or {}).get("probability_at_least_one", 0.0)) * 100, 2),
            key_reasons=["Deterministic baseline from official structured inputs and current Intel1 rule model."],
        ),
        "validation_errors": [],
    }


def consensus_prediction(
    *,
    run_id: str,
    weekend: WeekendContext,
    predictions: list[dict[str, Any]],
    session_results: dict[str, list[dict[str, Any]]],
    stored_signals: list[StoredSignal],
) -> dict[str, Any]:
    weights = {prediction["provider"]: provider_weight(prediction) for prediction in predictions}
    win_probabilities = weighted_entries(predictions, weights, "win_probabilities", "driver")
    constructor_probabilities = weighted_entries(predictions, weights, "constructor_win_probabilities", "team")

    predicted_winner = win_probabilities[0]["driver"] if win_probabilities else "TBD"
    consensus = {
        "provider": "intel1_consensus",
        "version": f"{run_id}:intel1_consensus:v1",
        "modelUsed": "intel1-consensus-v1",
        "providerRequestId": None,
        "modelTemperature": None,
        "confidence": round(sum(prediction["confidence"] * weights[prediction["provider"]] for prediction in predictions) / max(sum(weights.values()), 0.0001), 3),
        "sourceQuality": average_source_quality(stored_signals),
        "officialDataAvailability": official_data_availability(session_results),
        "sessionRecency": session_recency(session_results),
        "predicted_winner": predicted_winner,
        "win_probabilities": win_probabilities,
        "constructor_win_probabilities": constructor_probabilities,
        "podium_probabilities": weighted_entries(predictions, weights, "podium_probabilities", "driver", require_sum=False),
        "top10_probabilities": weighted_entries(predictions, weights, "top10_probabilities", "driver", require_sum=False),
        "dnf_risk": weighted_entries(predictions, weights, "dnf_risk", "driver", require_sum=False),
        "safety_car_probability": round(weighted_scalar(predictions, weights, "safety_car_probability"), 2),
        "key_reasons": ["Consensus weighted by provider confidence, source quality, official data availability, and session recency."],
        "weak_assumptions": combined_assumptions(predictions),
        "provider_weights": weights,
        "official_overrides": [],
        "disagreement_notes": disagreement_notes(predictions),
        "analyst_report": analyst_report_from_payload(
            {},
            predicted_winner=predicted_winner,
            win_probabilities=win_probabilities,
            constructor_probabilities=constructor_probabilities,
            podium_probabilities=weighted_entries(predictions, weights, "podium_probabilities", "driver", require_sum=False),
            safety_car_probability=round(weighted_scalar(predictions, weights, "safety_car_probability"), 2),
            key_reasons=["Consensus weighted by provider confidence, source quality, official data availability, and session recency."],
        ),
        "validation_errors": [],
        "eventId": weekend.weekend_id,
        "stage": weekend.stage,
    }
    validate_model_prediction(consensus)
    return consensus


def normalize_percent_entries(entries: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    normalized = normalize_optional_percent_entries(entries, label)
    if not normalized:
        raise ValueError(f"{label} probabilities are empty")
    raw = {entry[label]: entry["probability"] for entry in normalized if entry.get(label)}
    values = normalize_percent_map(raw)
    output = [{label: key, "probability": value} for key, value in values.items()]
    output.sort(key=lambda item: item["probability"], reverse=True)
    return output


def normalize_optional_percent_entries(entries: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    output = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = canonical_label_name(entry.get(label) or entry.get("driver") or entry.get("team") or "", label)
        if not name:
            continue
        probability = entry.get("probability", 0.0)
        output.append({label: name, "probability": bounded_percent(probability, f"{label} probability")})
    output.sort(key=lambda item: item["probability"], reverse=True)
    return output


def normalize_percent_map(raw: dict[str, float], precision: int = 2) -> dict[str, float]:
    if not raw:
        return {}
    scale = 100 * (10**precision)
    positive = {key: max(0.0, float(value)) for key, value in raw.items()}
    total = sum(positive.values())
    if total <= 0:
        units = {key: scale // len(positive) for key in positive}
    else:
        scaled = {key: (value / total) * scale for key, value in positive.items()}
        units = {key: int(value) for key, value in scaled.items()}
        remainder = scale - sum(units.values())
        ordered = sorted(scaled, key=lambda key: scaled[key] - units[key], reverse=True)
        for key in ordered[:remainder]:
            units[key] += 1
    return {key: round(value / (10**precision), precision) for key, value in units.items()}


def app_probability_entries_to_percent(entries: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    output = []
    for entry in entries:
        name = canonical_label_name(entry.get(label) or entry.get("driver") or entry.get("team") or "", label)
        if not name:
            continue
        probability = finite_number(entry.get("probability", 0.0), f"{label} app probability")
        if 0 <= probability <= 1:
            probability *= 100
        output.append({label: name, "probability": round(max(0.0, min(100.0, probability)), 2)})
    output.sort(key=lambda item: item["probability"], reverse=True)
    return output


def weighted_entries(
    predictions: list[dict[str, Any]],
    weights: dict[str, float],
    key: str,
    label: str,
    *,
    require_sum: bool = True,
) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for prediction in predictions:
        weight = weights.get(prediction["provider"], 0.0)
        for entry in prediction.get(key, []):
            name = canonical_label_name(entry.get(label), label)
            if not name:
                continue
            totals[name] = totals.get(name, 0.0) + float(entry.get("probability", 0.0)) * weight
    if require_sum:
        values = normalize_percent_map(totals)
        output = [{label: name, "probability": probability} for name, probability in values.items()]
    else:
        total_weight = max(sum(weights.values()), 0.0001)
        output = [{label: name, "probability": round(value / total_weight, 2)} for name, value in totals.items()]
    output.sort(key=lambda item: item["probability"], reverse=True)
    return output


def validate_model_prediction(prediction: dict[str, Any]) -> None:
    validate_percent_group("win_probabilities", prediction["win_probabilities"], "driver", expected_sum=100.0)
    validate_percent_group("constructor_win_probabilities", prediction["constructor_win_probabilities"], "team", expected_sum=100.0)
    top = prediction["win_probabilities"][0]
    if prediction.get("predicted_winner") != top.get("driver"):
        raise ValueError("Predicted winner must match highest win probability")
    for key, label in [("podium_probabilities", "driver"), ("top10_probabilities", "driver"), ("dnf_risk", "driver")]:
        validate_percent_group(key, prediction.get(key, []), label, expected_sum=None)
    bounded_unit(prediction.get("confidence", 0), "confidence")
    bounded_percent(prediction.get("safety_car_probability", 0), "safety_car_probability")


def validate_percent_group(key: str, entries: list[dict[str, Any]], label: str, expected_sum: float | None) -> None:
    if not entries and expected_sum is not None:
        raise ValueError(f"{key} is empty")
    total = 0.0
    for entry in entries:
        if not entry.get(label):
            raise ValueError(f"{key} entry missing {label}")
        probability = bounded_percent(entry.get("probability", 0), f"{key} probability")
        total += probability
    if expected_sum is not None and round(total, 2) != expected_sum:
        raise ValueError(f"{key} must sum to {expected_sum}")


def validate_prediction_prompt(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    forbidden = ("rawtext", "raw_excerpt", "raw_content")
    if any(term in serialized for term in forbidden):
        raise ValueError("Prediction prompt contains raw source text")


def provider_weight(prediction: dict[str, Any]) -> float:
    confidence = float(prediction.get("confidence", 0.0))
    weight = confidence
    weight *= float(prediction.get("sourceQuality", 0.5))
    weight *= float(prediction.get("officialDataAvailability", 0.75))
    weight *= float(prediction.get("sessionRecency", 0.75))
    if confidence < 0.35:
        weight = min(weight, 0.15)
    return round(max(0.01, weight), 4)


def official_data_availability(session_results: dict[str, list[dict[str, Any]]]) -> float:
    official_rows = [row for rows in session_results.values() for row in rows if row.get("is_official")]
    return 1.0 if official_rows else 0.72


def session_recency(session_results: dict[str, list[dict[str, Any]]]) -> float:
    for key, value in [("race", 1.0), ("qualifying", 0.95), ("sprint", 0.9), ("sprint_qualifying", 0.85), ("fp3", 0.78), ("fp2", 0.7), ("fp1", 0.62)]:
        if session_results.get(key):
            return value
    return 0.55


def average_source_quality(signals: list[StoredSignal]) -> float:
    if not signals:
        return 0.72
    return round(sum(signal.sourceQuality for signal in signals) / len(signals), 3)


def disagreement_notes(predictions: list[dict[str, Any]]) -> list[str]:
    if len(predictions) < 2:
        return []
    first, second = predictions[0], predictions[1]
    if first.get("predicted_winner") != second.get("predicted_winner") and first.get("confidence", 0) >= 0.45 and second.get("confidence", 0) >= 0.45:
        return [
            f"{first['provider']} favours {first.get('predicted_winner')}; {second['provider']} favours {second.get('predicted_winner')}."
        ]
    return []


def combined_assumptions(predictions: list[dict[str, Any]]) -> list[str]:
    assumptions: list[str] = []
    for prediction in predictions:
        assumptions.extend(short_string_list(prediction.get("weak_assumptions"))[:3])
    return assumptions[:6]


def weighted_scalar(predictions: list[dict[str, Any]], weights: dict[str, float], key: str) -> float:
    numerator = sum(float(prediction.get(key, 0.0)) * weights[prediction["provider"]] for prediction in predictions)
    return numerator / max(sum(weights.values()), 0.0001)


def canonical_label_name(value: Any, label: str) -> str:
    name = " ".join(str(value or "").split())
    if label != "driver" or not name:
        return name
    key = compact_name_key(name)
    if key in {"kimiantonelli", "andreakimiantonelli", "antonelli"}:
        return "Andrea Kimi Antonelli"
    return name


def compact_name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def bounded_unit(value: Any, label: str) -> float:
    number = finite_number(value, label)
    if number < 0 or number > 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return round(number, 3)


def bounded_percent(value: Any, label: str) -> float:
    number = finite_number(value, label)
    if number < 0 or number > 100:
        raise ValueError(f"{label} must be between 0 and 100")
    return round(number, 2)


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            text = text[:-1].strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            value = match.group(0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def short_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split())[:220] for item in value if str(item).strip()][:8]


def analyst_report_from_payload(
    raw_report: Any,
    *,
    predicted_winner: str,
    win_probabilities: list[dict[str, Any]],
    constructor_probabilities: list[dict[str, Any]],
    podium_probabilities: list[dict[str, Any]],
    safety_car_probability: float,
    key_reasons: list[str],
) -> dict[str, Any]:
    report = raw_report if isinstance(raw_report, dict) else {}
    final_call = report.get("final_call") if isinstance(report.get("final_call"), dict) else {}
    constructor = constructor_probabilities[0]["team"] if constructor_probabilities else "TBD"
    podium = report_string_list(final_call.get("podium"), limit=3) or [
        entry["driver"] for entry in podium_probabilities[:3] if entry.get("driver")
    ]
    if not podium:
        podium = [entry["driver"] for entry in win_probabilities[:3] if entry.get("driver")]
    highest_scoring_team = clean_report_text(final_call.get("highest_scoring_team"), constructor, 80)
    risk_label = clean_report_text(final_call.get("safety_car_risk"), safety_car_label(safety_car_probability), 120)

    return {
        "title": clean_report_text(report.get("title"), "Grand Prix prediction", 120),
        "assumption": clean_report_text(report.get("assumption"), "", 220),
        "final_call": {
            "winner_driver": clean_report_text(final_call.get("winner_driver"), predicted_winner, 80),
            "winner_constructor": clean_report_text(final_call.get("winner_constructor"), constructor, 80),
            "podium": podium[:3],
            "highest_scoring_team": highest_scoring_team,
            "safety_car_risk": risk_label,
            "rain_impact": clean_report_text(final_call.get("rain_impact"), "TBD", 80),
            "chaos_level": clean_report_text(final_call.get("chaos_level"), "TBD", 80),
            "most_likely_upset_winner": clean_report_text(final_call.get("most_likely_upset_winner"), upset_winner(win_probabilities, predicted_winner), 80),
            "dark_horse_podium": report_string_list(final_call.get("dark_horse_podium"), limit=3),
        },
        "narrative": normalized_narrative(report.get("narrative"), key_reasons),
        "strategy": normalized_strategy(report.get("strategy")),
        "biggest_risks": normalized_risks(report.get("biggest_risks")),
        "final_answer": clean_report_text(
            report.get("final_answer"),
            f"Winner: {predicted_winner}. Constructor: {constructor}. Podium: {', '.join(podium[:3]) or 'TBD'}. Safety Car: {risk_label}.",
            500,
        ),
    }


def clean_report_text(value: Any, fallback: str, limit: int) -> str:
    text = " ".join(str(value or fallback or "").split())
    return text[:limit]


def report_string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [item.strip() for item in value.replace("/", ",").split(",")]
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).split())[:90] for item in value if str(item).strip()][:limit]


def normalized_narrative(raw: Any, key_reasons: list[str]) -> list[dict[str, str]]:
    output = []
    if isinstance(raw, list):
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            title = clean_report_text(item.get("title"), "", 60)
            body = clean_report_text(item.get("body"), "", 380)
            if title and body:
                output.append({"title": title, "body": body})
    if output:
        return output
    return [{"title": "Why this pick", "body": reason} for reason in key_reasons[:3]]


def normalized_strategy(raw: Any) -> dict[str, str]:
    strategy = raw if isinstance(raw, dict) else {}
    return {
        "dry": clean_report_text(strategy.get("dry"), "", 260),
        "wet_mixed": clean_report_text(strategy.get("wet_mixed") or strategy.get("wet"), "", 260),
    }


def normalized_risks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    output = []
    for item in raw[:5]:
        if isinstance(item, str):
            risk = clean_report_text(item, "", 140)
            benefits: list[str] = []
        elif isinstance(item, dict):
            risk = clean_report_text(item.get("risk") or item.get("title"), "", 140)
            benefits = report_string_list(item.get("benefits") or item.get("who_benefits") or item.get("beneficiaries"), limit=4)
        else:
            continue
        if risk:
            output.append({"risk": risk, "benefits": benefits})
    return output


def safety_car_label(probability: float) -> str:
    if probability >= 75:
        return f"Very high, around {round(probability)}%"
    if probability >= 55:
        return f"High, around {round(probability)}%"
    if probability >= 35:
        return f"Medium, around {round(probability)}%"
    return f"Low, around {round(probability)}%"


def upset_winner(win_probabilities: list[dict[str, Any]], predicted_winner: str) -> str:
    for entry in win_probabilities:
        driver = entry.get("driver")
        if driver and driver != predicted_winner:
            return driver
    return "TBD"
