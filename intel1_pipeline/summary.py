from __future__ import annotations

import json
from os import getenv
from typing import Any

from .ai import AI_ANALYST_CONTRACT, contains_forbidden_probability_language, openai_available, parse_json_object
from .signals import ExtractedSignal
from .structured_data import WeekendContext
from .time_utils import isoformat, utc_now


DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-5.5"


def build_summary(run_id: str, weekend: WeekendContext, prediction: dict[str, Any], signals: list[ExtractedSignal], skip_ai: bool = False) -> dict[str, Any]:
    if openai_summary_enabled() and openai_available(skip_ai) and prediction.get("material_change_detected", False):
        try:
            return openai_summary(run_id, weekend, prediction, signals)
        except Exception:
            pass
    return deterministic_summary(run_id, weekend, prediction, signals)


def openai_summary_model_name() -> str:
    return getenv("OPENAI_SUMMARY_MODEL") or getenv("OPENAI_MODEL") or DEFAULT_OPENAI_SUMMARY_MODEL


def openai_summary_enabled() -> bool:
    value = getenv("INTEL1_OPENAI_SUMMARY_ENABLED")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def deterministic_summary(run_id: str, weekend: WeekendContext, prediction: dict[str, Any], signals: list[ExtractedSignal]) -> dict[str, Any]:
    race = prediction["race"]
    winner = race["predicted_winner"]
    outlook = race.get("market_outlook") or {}
    confidence = prediction["forecast_confidence"]
    volatility = prediction["race_volatility"]
    headline = f"{weekend.grand_prix_name}: {outlook.get('label') or 'Model outlook'}"
    changed = human_change_summary(prediction.get("material_change_summary"))
    pick_reason = model_pick_reason(prediction, signals)
    outlook_explanation = human_outlook_explanation(prediction, weekend)
    caveats = "This is an evidence-bound heuristic outlook, not betting advice. Small percentage gaps should be read as bands, not exact rankings."
    return {
        "schema_version": "1.1",
        "run_id": run_id,
        "updated_at": isoformat(utc_now()),
        "weekend_id": weekend.weekend_id,
        "stage": weekend.stage,
        "headline": headline,
        "executive_summary": (
            f"{outlook_explanation} "
            f"{pick_reason}"
        ),
        "race_outlook": race_outlook_text(prediction, weekend),
        "sprint_outlook": sprint_text(prediction["sprint"], weekend, prediction.get("session_results")) if prediction["sprint"]["enabled"] else None,
        "podium_outlook": podium_text(race["predicted_podium"]),
        "constructor_outlook": constructor_text(race["predicted_winning_constructor"]),
        "safety_car_note": safety_car_text(prediction["safety_car"]),
        "what_changed": changed,
        "caveats": caveats,
        "sections": [
            {"title": "Current model outlook", "body": inferred_read(prediction)},
            {"title": "What changed", "body": changed},
            {"title": "Key evidence", "body": confirmed_facts(signals)},
            {"title": "Main uncertainty", "body": main_uncertainty(prediction)},
            {"title": "Watch next", "body": watchlist(signals)},
        ],
    }


def openai_summary(run_id: str, weekend: WeekendContext, prediction: dict[str, Any], signals: list[ExtractedSignal]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    signal_context = [
        {
            "signal_type": signal.signal_type,
            "direction": signal.direction,
            "impact_level": signal.impact_level,
            "teams": signal.teams,
            "drivers": signal.drivers,
            "evidence_summary": signal.evidence_summary,
            "evidence_type": signal.evidence_type,
            "corroboration_status": signal.corroboration_status,
            "is_confirmed": signal.is_confirmed,
            "requires_corroboration": signal.requires_corroboration,
            "confidence": signal.confidence,
            "can_shift_probability": signal.can_shift_probability,
            "material_change": signal.material_change,
            "model_relevance": signal.model_relevance,
        }
        for signal in signals[:24]
    ]
    prompt = {
        "task": "Write an evidence-bound F1 intelligence brief from structured data only. Do not browse, assume, add unsupported facts, or create probabilities. Use only the supplied model output and signals.",
        "behaviour_contract": "Follow the Intel1 AI analyst contract. Separate fact, session data, journalist analysis, team/driver statements, rumour, weather, technical observation, and model inference. Avoid hype, betting language, and false precision.",
        "weekend": {
            "weekend_id": weekend.weekend_id,
            "grand_prix_name": weekend.grand_prix_name,
            "circuit_name": weekend.circuit_name,
            "stage": weekend.stage,
            "is_sprint_weekend": weekend.is_sprint_weekend,
        },
        "prediction": {
            "winner": prediction["race"]["predicted_winner"],
            "top_drivers": prediction["race"]["driver_win_probabilities"][:6],
            "top_constructors": prediction["race"]["constructor_win_probabilities"][:5],
            "sprint": prediction["sprint"],
            "safety_car": prediction["safety_car"],
            "confidence": prediction["forecast_confidence"],
            "volatility": prediction["race_volatility"],
            "uncertainties": prediction["key_uncertainties"],
            "material_change_detected": prediction.get("material_change_detected", False),
            "material_change_summary": prediction.get("material_change_summary"),
        },
        "signals": signal_context,
        "output_schema": {
            "headline": "short plain-English headline",
            "executive_summary": "2 sentences: current model outlook and evidence basis",
            "race_outlook": "2 sentences, clearly labelled as model outlook",
            "sprint_outlook": "string or null",
            "podium_outlook": "1 sentence",
            "constructor_outlook": "1 sentence",
            "safety_car_note": "1 sentence, call it heuristic",
            "what_changed": "1-2 sentences",
            "caveats": "1 sentence",
            "sections": [{"title": "Current model outlook|What changed|Key evidence|Main uncertainty|Watch next", "body": "1-2 sentences"}],
        },
    }
    response = client.chat.completions.create(
        model=openai_summary_model_name(),
        messages=[
            {"role": "system", "content": AI_ANALYST_CONTRACT + "\nWrite compact F1 race-weekend intelligence from structured evidence. Return valid JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=float(getenv("OPENAI_SUMMARY_TEMPERATURE", "0.2")),
        response_format={"type": "json_object"},
    )
    payload = parse_json_object(response.choices[0].message.content or "{}")
    fallback = deterministic_summary(run_id, weekend, prediction, signals)
    sections = payload.get("sections") if isinstance(payload.get("sections"), list) else fallback["sections"]
    return {
        **fallback,
        "headline": safe_text(payload.get("headline"), fallback["headline"], 180),
        "executive_summary": safe_text(payload.get("executive_summary"), fallback["executive_summary"], 700),
        "race_outlook": safe_text(payload.get("race_outlook"), fallback["race_outlook"], 700),
        "sprint_outlook": safe_text(payload.get("sprint_outlook"), fallback["sprint_outlook"], 500) if prediction["sprint"]["enabled"] else None,
        "podium_outlook": safe_text(payload.get("podium_outlook"), fallback["podium_outlook"], 500),
        "constructor_outlook": safe_text(payload.get("constructor_outlook"), fallback["constructor_outlook"], 500),
        "safety_car_note": safe_text(payload.get("safety_car_note"), fallback["safety_car_note"], 500),
        "what_changed": safe_text(payload.get("what_changed"), fallback["what_changed"], 600),
        "caveats": safe_text(payload.get("caveats"), fallback["caveats"], 500),
        "sections": normalize_sections(sections, fallback["sections"]),
    }


def podium_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No podium outlook available yet."
    names = ", ".join(entry.get("driver") or "TBD" for entry in entries[:3])
    return f"Current podium band: {names}."


def human_change_summary(value: str | None) -> str:
    if not value or value == "No material source signal changed the model inputs.":
        return "No major swing since the last run."
    return value


def human_outlook_explanation(prediction: dict[str, Any], weekend: WeekendContext) -> str:
    race = prediction["race"]
    outlook = race.get("market_outlook") or {}
    leader = race["predicted_winner"].get("driver", "the model leader")
    probability = float(race["predicted_winner"].get("probability", 0))
    if outlook.get("state") in {"no_clear_favourite", "close_front_group"} or race["predicted_winner"].get("probability", 0) < 0.15:
        if not prediction.get("session_results", {}).get("qualifying"):
            return f"{leader} has the current Intel1 edge for {weekend.grand_prix_name}, but confidence is limited until grid evidence is cleaner."
        return f"{leader} has the current Intel1 edge for {weekend.grand_prix_name} at {probability:.0%}; the front group remains tightly packed."
    return outlook.get("explanation") or f"Intel1 currently has {leader} as the model leader for {weekend.grand_prix_name}."


def race_outlook_text(prediction: dict[str, Any], weekend: WeekendContext) -> str:
    race = prediction["race"]
    confidence = prediction["forecast_confidence"]
    volatility = prediction["race_volatility"]
    if race["predicted_winner"].get("probability", 0) < 0.15:
        return (
            f"{race['predicted_winner']['driver']} is the top current-race pick. "
            f"Confidence is {confidence['level']} and volatility is {volatility['level']}, so treat the edge as fragile."
        )
    return (
        f"Confidence is {confidence['level']} and race volatility is {volatility['level']}. "
        f"{podium_text(race['predicted_podium'])}"
    )


def model_pick_reason(prediction: dict[str, Any], signals: list[ExtractedSignal]) -> str:
    outlook = prediction["race"].get("market_outlook") or {}
    if outlook.get("state") in {"no_clear_favourite", "close_front_group"}:
        return "Treat the leader as a narrow model edge rather than a strong prediction."
    adjustment = next((item for item in prediction.get("score_adjustment_log", []) if item.get("target_name") == prediction["race"]["predicted_winner"]["driver"]), None)
    if adjustment:
        return f"The pick is supported by a {str(adjustment.get('signal_type', 'model')).replace('_', ' ')} adjustment: {adjustment.get('reason', 'model evidence')}."
    material = next((signal for signal in signals if signal.material_change), None)
    if material:
        return f"The latest material signal is {material.signal_type.replace('_', ' ')}: {material.evidence_summary}."
    return "Intel1 is waiting for cleaner race-week evidence before treating any driver as a real favourite."


def constructor_text(entry: dict[str, Any]) -> str:
    return f"Constructor model leader: {entry.get('team') or 'TBD'} at {round(entry.get('probability', 0) * 100)} percent."


def safety_car_text(safety_car: dict[str, Any]) -> str:
    return (
        f"{str(safety_car['risk_level']).capitalize()} interruption risk: about "
        f"{round(safety_car['probability_at_least_one'] * 100)} percent chance of at least one Safety Car or major race interruption."
    )


def sprint_text(sprint: dict[str, Any], weekend: WeekendContext, session_results: dict[str, Any] | None = None) -> str:
    if not sprint.get("enabled"):
        return ""
    sprint_grid = (session_results or {}).get("sprint_qualifying") or []
    if sprint_grid:
        front = ", ".join(f"P{row.get('position')} {row.get('driver')}" for row in sprint_grid[:3])
        return f"Sprint grid is set: {front}. The Sprint read starts from that order, then adjusts for pace, tyres, weather, and race volatility."
    leader = sprint.get("predicted_winner") or {}
    driver = leader.get("driver") or "the current model leader"
    outlook = sprint.get("market_outlook") or {}
    outlook_explanation = outlook.get("explanation") or f"The Sprint model currently has {driver} as the narrow leader."
    return (
        f"{weekend.grand_prix_name} is a Sprint weekend, so Intel1 also produces a Sprint outlook. "
        f"{outlook_explanation} "
        "Sprint-format evidence remains sensitive to traffic, tyre offsets, and risk management."
    )


def confirmed_facts(signals: list[ExtractedSignal]) -> str:
    confirmed = [signal.evidence_summary for signal in signals if signal.is_confirmed and "API available" not in signal.evidence_summary]
    return " ".join(confirmed[:3]) if confirmed else "No official fact signal has materially changed the model yet."


def inferred_read(prediction: dict[str, Any]) -> str:
    confidence = prediction["forecast_confidence"]
    if prediction["race"]["predicted_winner"].get("probability", 0) < 0.15:
        return "This is a close-pack read, not a confident winner call."
    return f"Model confidence is {confidence['level']}; small gaps should be treated as a close group."


def watchlist(signals: list[ExtractedSignal]) -> str:
    watch = [signal.evidence_summary for signal in signals if signal.requires_corroboration]
    return " ".join(watch[:3]) if watch else "Watch FIA documents, final grid, weather, and post-session long-run context."


def main_uncertainty(prediction: dict[str, Any]) -> str:
    uncertainties = prediction.get("key_uncertainties") or []
    return uncertainties[0] if uncertainties else "Fuel loads, tyre usage, traffic, and session programmes can distort the current read."


def safe_text(raw: Any, fallback: str | None, limit: int) -> str:
    value = str(raw or fallback or "").strip()
    if not value or contains_forbidden_probability_language(value):
        value = str(fallback or "").strip()
    return value[:limit]


def normalize_sections(raw_sections: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(raw_sections, list):
        return fallback
    sections = []
    for section in raw_sections[:5]:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        body = str(section.get("body") or "").strip()
        if contains_forbidden_probability_language(title) or contains_forbidden_probability_language(body):
            continue
        if title and body:
            sections.append({"title": title[:80], "body": body[:700]})
    return sections or fallback
