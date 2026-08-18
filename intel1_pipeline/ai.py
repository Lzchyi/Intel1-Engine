from __future__ import annotations

import json
import os
import uuid
from dataclasses import fields
from typing import Any

from .signals import (
    ExtractedSignal,
    deterministic_extract_signals,
    normalize_evidence_type,
    normalize_signal_controls,
    relevance_targets,
    severity_for,
)
from .source_items import SourceItem


DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_EXTRACT_MODEL = "gpt-5.5"
MAX_AI_ITEMS = int(os.getenv("OPENAI_MAX_SOURCE_ITEMS", "28"))

AI_ANALYST_CONTRACT = """
You are Intel1's Formula 1 race intelligence analyst.
Act as a professional F1 journalist, race analyst, strategy observer, and evidence-bound forecasting assistant.
You must be skeptical, concise, technically aware, and clear about uncertainty.
You do not create probabilities directly. The prediction engine owns probability calculation.
Your job is to extract structured signals, classify evidence quality, identify material changes, and explain model movement from provided data only.
Always separate confirmed facts, observed session data, journalist analysis, team/driver statements, rumours, weather forecasts, technical observations, and model inference.
Do not overreact to practice times. Consider fuel loads, tyre compounds, run plans, traffic, track evolution, red/yellow flags, weather, circuit overtaking difficulty, sprint format, parc ferme state, penalties, FIA documents, and reliability context.
Treat FIA documents and official classifications as higher authority than media interpretation. Treat rumours cautiously unless corroborated.
Avoid hype, fan language, betting language, and false precision. Never say a driver will win.
If evidence is weak, say so. If sources conflict, mark the conflict and lower confidence.
"""

FORBIDDEN_PROBABILITY_TERMS = ("bet", "value pick", "lock", "sure win", "guaranteed", "will win", "nailed on")


def openai_model_name() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def openai_extract_model_name() -> str:
    return os.getenv("OPENAI_EXTRACT_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_EXTRACT_MODEL


def openai_available(skip_ai: bool) -> bool:
    return not skip_ai and bool(os.getenv("OPENAI_API_KEY"))


def extract_signals(
    *,
    items: list[SourceItem],
    weekend_id: str,
    run_id: str,
    stage: str,
    skip_ai: bool,
    cached_signals_by_hash: dict[str, list[dict[str, Any]]] | None = None,
) -> list[ExtractedSignal]:
    cached, uncached_items = cached_signals_for_items(items, weekend_id, run_id, stage, cached_signals_by_hash or {})
    if not uncached_items:
        return cached
    if not openai_available(skip_ai):
        return cached + deterministic_extract_signals(uncached_items, weekend_id, run_id, stage)
    try:
        ai_signals = openai_extract(uncached_items, weekend_id, run_id, stage)
    except Exception:
        return cached + deterministic_extract_signals(uncached_items, weekend_id, run_id, stage)
    return cached + (ai_signals or deterministic_extract_signals(uncached_items, weekend_id, run_id, stage))


def openai_extract(items: list[SourceItem], weekend_id: str, run_id: str, stage: str) -> list[ExtractedSignal]:
    from openai import OpenAI

    client = OpenAI()
    candidate_items = select_ai_items(items)
    compact_items = [
        {
            "source_item_id": item.source_item_id,
            "source_id": item.source_id,
            "source_name": item.source_name,
            "source_tier": item.source_tier,
            "reliability_weight": item.reliability_weight,
            "title": item.title,
            "url": item.url,
            "excerpt": (item.raw_content or item.raw_excerpt)[:1400],
            "language": item.language,
        }
        for item in candidate_items[:MAX_AI_ITEMS]
    ]
    prompt = {
        "task": "Extract evidence-bound F1 race-weekend intelligence signals. Do not invent facts, drivers, teams, or probabilities. Prefer no signal over a weak unsupported signal.",
        "behaviour_contract": "Follow the Intel1 AI analyst contract exactly: evidence-bound, skeptical, no hype, no betting language, no direct probability invention, practice-session caveats, rumours low weight unless corroborated.",
        "weekend_id": weekend_id,
        "stage": stage,
        "evidence_types": [
            "official_fact",
            "session_data",
            "journalist_analysis",
            "team_statement",
            "driver_statement",
            "rumour",
            "model_inference",
            "weather_forecast",
            "technical_observation",
        ],
        "corroboration_statuses": ["single_source", "multi_source", "officially_confirmed", "contradicted", "unclear"],
        "confidence_rules": {
            "0.90-1.00": "official confirmed fact",
            "0.75-0.89": "strong credible report or clean session signal",
            "0.55-0.74": "plausible but incomplete signal",
            "0.35-0.54": "weak or context-limited signal",
            "0.00-0.34": "rumour or low confidence; do not shift probability",
        },
        "impact_rules": {
            "high": "confirmed grid/final-grid/penalty/component/reliability/FIA/weather participation-level changes only",
            "medium": "credible race pace, tyre degradation, upgrade, setup, reliability, or sector-specific evidence",
            "low": "FP1 pace, generic optimism, single-source interpretation, or unconfirmed paddock read",
        },
        "allowed_signal_types": [
            "single_lap_pace_positive",
            "single_lap_pace_negative",
            "race_pace_positive",
            "race_pace_negative",
            "tyre_degradation_positive",
            "tyre_degradation_negative",
            "wet_weather_strength_positive",
            "wet_weather_strength_negative",
            "reliability_concern",
            "power_unit_concern",
            "cooling_concern",
            "brake_or_suspension_concern",
            "confirmed_grid_penalty",
            "grid_penalty_risk",
            "pending_investigation",
            "parc_ferme_change_risk",
            "pit_lane_start_risk",
            "component_change_notice",
            "strategy_volatility_increase",
            "weather_volatility",
            "safety_car_risk_increase",
            "upgrade_positive",
            "upgrade_unclear",
            "upgrade_negative",
            "session_data_low_representativeness",
            "traffic_or_flags_compromised_lap",
            "long_run_data_contaminated",
            "floor_or_bodywork_damage",
            "setup_experiment",
            "strategic_tyre_offset",
            "track_specific_suitability_positive",
            "track_specific_suitability_negative",
            "driver_error",
            "car_limitation",
        ],
        "items": compact_items,
        "output": "Return JSON only: {\"signals\": [{source_item_id, teams, drivers, signal_type, direction, impact_level, confidence, source_tier, evidence_summary, evidence_type, corroboration_status, requires_corroboration, can_shift_probability, should_surface_in_app, material_change, model_relevance}]}",
    }
    response = client.chat.completions.create(
        model=openai_extract_model_name(),
        messages=[
            {"role": "system", "content": AI_ANALYST_CONTRACT + "\nReturn valid JSON only. Mark rumours as rumours and never upgrade them into facts."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    payload = parse_json_object(content)
    by_id = {item.source_item_id: item for item in candidate_items}
    signals: list[ExtractedSignal] = []
    for raw in payload.get("signals", []):
        item = by_id.get(raw.get("source_item_id"))
        if not item:
            continue
        signal_type = str(raw.get("signal_type", "")).strip()
        if not signal_type:
            continue
        evidence_type = normalize_evidence_type(raw.get("evidence_type"), item)
        is_confirmed = item.source_tier == "A" and evidence_type == "official_fact"
        confidence = float(raw.get("confidence", 0.55))
        impact, confidence, corroboration_status, requires_corroboration, can_shift_probability, material_change = normalize_signal_controls(
            signal_type=signal_type,
            source_tier=item.source_tier,
            evidence_type=evidence_type,
            requested_impact_level=str(raw.get("impact_level", "medium")),
            requested_confidence=confidence,
            requested_corroboration_status=str(raw.get("corroboration_status") or ""),
            requested_requires_corroboration=bool(raw.get("requires_corroboration", item.source_tier != "A")),
            requested_can_shift_probability=bool(raw.get("can_shift_probability", True)),
        )
        summary = clean_evidence_summary(str(raw.get("evidence_summary") or item.title))
        if contains_forbidden_probability_language(summary):
            summary = item.title
        signals.append(
            ExtractedSignal(
                signal_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{item.source_item_id}:{signal_type}:ai")),
                weekend_id=weekend_id,
                run_id=run_id,
                session_context=stage,
                source_item_id=item.source_item_id,
                source_content_hash=item.content_hash,
                source_id=item.source_id,
                source_name=item.source_name,
                source_tier=item.source_tier,
                source_reliability_weight=item.reliability_weight,
                source_url=item.url,
                source_published_at=item.published_at,
                teams=list(raw.get("teams") or []),
                drivers=list(raw.get("drivers") or []),
                signal_type=signal_type,
                direction=str(raw.get("direction", "mixed")),
                impact_level=impact,
                confidence=confidence,
                evidence_summary=summary[:280],
                model_relevance=list(raw.get("model_relevance") or relevance_targets(signal_type)),
                is_confirmed=is_confirmed,
                requires_corroboration=requires_corroboration,
                evidence_type=evidence_type,
                corroboration_status=corroboration_status,
                contradicting_signal_ids=[],
                prediction_impact_targets=relevance_targets(signal_type),
                severity_score=severity_for(impact),
                expiry_stage=None,
                can_shift_probability=can_shift_probability,
                should_surface_in_app=bool(raw.get("should_surface_in_app", True)),
                material_change=material_change,
                raw_evidence_excerpt=None,
                event_category="ai_extracted",
                linked_document_type="fia_document" if item.source_id.startswith("fia_") else None,
            )
        )
    return signals


def cached_signals_for_items(
    items: list[SourceItem],
    weekend_id: str,
    run_id: str,
    stage: str,
    cached_signals_by_hash: dict[str, list[dict[str, Any]]],
) -> tuple[list[ExtractedSignal], list[SourceItem]]:
    cached: list[ExtractedSignal] = []
    uncached: list[SourceItem] = []
    signal_fields = {field.name for field in fields(ExtractedSignal)}
    for item in items:
        if item.connector_type == "api" and item.title.endswith("API available"):
            continue
        if item.content_hash not in cached_signals_by_hash:
            uncached.append(item)
            continue
        for raw in cached_signals_by_hash[item.content_hash]:
            signal_type = str(raw.get("signal_type", "")).strip()
            if not signal_type:
                continue
            payload = {key: value for key, value in raw.items() if key in signal_fields}
            payload.update(
                {
                    "signal_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{item.source_item_id}:{signal_type}:cache")),
                    "weekend_id": weekend_id,
                    "run_id": run_id,
                    "session_context": stage,
                    "source_item_id": item.source_item_id,
                    "source_content_hash": item.content_hash,
                    "source_id": item.source_id,
                    "source_name": item.source_name,
                    "source_tier": item.source_tier,
                    "source_reliability_weight": item.reliability_weight,
                    "source_url": item.url,
                    "source_published_at": item.published_at,
                }
            )
            try:
                cached.append(ExtractedSignal(**payload))
            except TypeError:
                uncached.append(item)
                break
    return cached, uncached


def build_extraction_cache(items: list[SourceItem], signals: list[ExtractedSignal]) -> dict[str, list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = {item.content_hash: [] for item in items}
    for signal in signals:
        by_hash.setdefault(signal.source_content_hash, []).append(
            {
                key: value
                for key, value in signal.to_dict().items()
                if key
                not in {
                    "signal_id",
                    "weekend_id",
                    "run_id",
                    "session_context",
                    "source_item_id",
                    "source_id",
                    "source_name",
                    "source_tier",
                    "source_reliability_weight",
                    "source_url",
                    "source_published_at",
                }
            }
        )
    return by_hash


def clean_evidence_summary(value: str) -> str:
    return " ".join(value.split())


def contains_forbidden_probability_language(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in FORBIDDEN_PROBABILITY_TERMS)


def select_ai_items(items: list[SourceItem]) -> list[SourceItem]:
    high_signal_terms = (
        "penalty",
        "summons",
        "stewards",
        "classification",
        "grid",
        "upgrade",
        "floor",
        "long run",
        "race pace",
        "degradation",
        "tyre",
        "strategy",
        "need to know",
        "safety car",
        "vsc",
        "sprint",
        "qualifying",
        "pole",
        "front row",
        "front-row",
        "power unit",
        "engine",
        "reliability",
        "rain",
        "weather",
        "parc ferme",
        "investigation",
        "damage",
    )
    scored: list[tuple[int, SourceItem]] = []
    for index, item in enumerate(items):
        text = f"{item.title} {item.raw_excerpt}".lower()
        score = 4 if item.source_tier == "A" else 1
        score += sum(1 for term in high_signal_terms if term in text)
        if item.weekend_relevance_status == "irrelevant":
            score -= 4
        scored.append((score * 1000 - index, item))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if _ > -2000]


def parse_json_object(content: str) -> dict[str, Any]:
    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = trimmed.strip("`")
        if trimmed.startswith("json"):
            trimmed = trimmed[4:].strip()
    try:
        payload = json.loads(trimmed)
    except json.JSONDecodeError:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(trimmed[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response was not a JSON object")
    return payload
