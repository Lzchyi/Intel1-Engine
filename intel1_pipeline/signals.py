from __future__ import annotations

import uuid
from dataclasses import dataclass

from .source_items import SourceItem


EVIDENCE_TYPES = {
    "official_fact",
    "session_data",
    "journalist_analysis",
    "team_statement",
    "driver_statement",
    "rumour",
    "model_inference",
    "weather_forecast",
    "technical_observation",
}

CORROBORATION_STATUSES = {
    "single_source",
    "multi_source",
    "officially_confirmed",
    "contradicted",
    "unclear",
}

HIGH_IMPACT_SIGNAL_TYPES = {
    "confirmed_grid_penalty",
    "pit_lane_start_risk",
    "component_change_notice",
    "power_unit_concern",
    "reliability_concern",
    "brake_or_suspension_concern",
    "confirmed_final_grid",
    "weather_volatility",
}

MATERIAL_SIGNAL_TYPES = HIGH_IMPACT_SIGNAL_TYPES | {
    "race_pace_positive",
    "race_pace_negative",
    "tyre_degradation_positive",
    "tyre_degradation_negative",
    "single_lap_pace_positive",
    "single_lap_pace_negative",
    "pending_investigation",
    "parc_ferme_change_risk",
    "floor_or_bodywork_damage",
    "upgrade_positive",
    "upgrade_negative",
}

SIGNAL_RULES: list[tuple[str, str, str, list[str]]] = [
    ("confirmed_grid_penalty", "negative", "high", ["grid penalty", "penalty", "back of the grid"]),
    ("pending_investigation", "negative", "medium", ["investigation", "summons", "stewards"]),
    ("reliability_concern", "negative", "medium", ["reliability", "failure", "stopped", "issue", "problem"]),
    ("power_unit_concern", "negative", "medium", ["power unit", "engine", "pu component"]),
    ("upgrade_positive", "positive", "medium", ["upgrade", "new floor", "package"]),
    ("tyre_degradation_negative", "negative", "medium", ["degradation", "graining", "tyre wear"]),
    ("weather_volatility", "mixed", "medium", ["rain", "wet", "weather", "storm"]),
    ("safety_car_risk_increase", "mixed", "low", ["street circuit", "barrier", "safety car"]),
    ("race_pace_positive", "positive", "medium", ["long run", "race pace"]),
    ("single_lap_pace_positive", "positive", "low", ["pole", "qualifying pace", "single-lap"]),
    ("floor_or_bodywork_damage", "negative", "medium", ["floor damage", "bodywork damage", "crash damage", "rebuild"]),
    ("setup_experiment", "mixed", "low", ["setup experiment", "experimental setup", "set-up experiment"]),
    ("strategic_tyre_offset", "mixed", "medium", ["tyre offset", "tire offset", "saved a set", "new tyre advantage"]),
    ("track_specific_suitability_positive", "positive", "medium", ["suited to this circuit", "track should suit", "strong in slow corners"]),
    ("track_specific_suitability_negative", "negative", "medium", ["not suited to this circuit", "weak in slow corners", "draggy"]),
    ("traffic_or_flags_compromised_lap", "mixed", "low", ["traffic", "yellow flag", "red flag", "compromised lap"]),
    ("long_run_data_contaminated", "mixed", "low", ["traffic on long run", "red flag interrupted", "unrepresentative long run"]),
]


@dataclass
class ExtractedSignal:
    signal_id: str
    weekend_id: str
    run_id: str
    session_context: str
    source_item_id: str
    source_content_hash: str
    source_id: str
    source_name: str
    source_tier: str
    source_reliability_weight: float
    source_url: str
    source_published_at: str | None
    teams: list[str]
    drivers: list[str]
    signal_type: str
    direction: str
    impact_level: str
    confidence: float
    evidence_summary: str
    model_relevance: list[str]
    is_confirmed: bool
    requires_corroboration: bool
    evidence_type: str
    corroboration_status: str
    contradicting_signal_ids: list[str]
    prediction_impact_targets: list[str]
    severity_score: float
    expiry_stage: str | None
    can_shift_probability: bool
    should_surface_in_app: bool
    material_change: bool
    raw_evidence_excerpt: str | None
    event_category: str
    linked_document_type: str | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def deterministic_extract_signals(items: list[SourceItem], weekend_id: str, run_id: str, stage: str) -> list[ExtractedSignal]:
    signals: list[ExtractedSignal] = []
    for item in items:
        if item.connector_type == "api" and item.title.endswith("API available"):
            continue
        if not is_official_deterministic_source(item):
            continue
        text = f"{item.title} {item.raw_excerpt}".lower()
        for signal_type, direction, impact, terms in SIGNAL_RULES:
            if not any(term in text for term in terms):
                continue
            evidence_type = evidence_type_for(item)
            if evidence_type == "rumour":
                continue
            confirmed = item.source_tier == "A" and evidence_type == "official_fact"
            impact, confidence, corroboration_status, requires_corroboration, can_shift_probability, material_change = normalize_signal_controls(
                signal_type=signal_type,
                source_tier=item.source_tier,
                evidence_type=evidence_type,
                requested_impact_level=impact,
                requested_confidence=min(0.9, 0.45 + item.reliability_weight * 0.35),
                requested_corroboration_status="officially_confirmed" if confirmed else "single_source",
                requested_requires_corroboration=item.source_tier != "A",
                requested_can_shift_probability=True,
            )
            signals.append(
                ExtractedSignal(
                    signal_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{item.source_item_id}:{signal_type}")),
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
                    teams=[],
                    drivers=[],
                    signal_type=signal_type,
                    direction=direction,
                    impact_level=impact,
                    confidence=confidence,
                    evidence_summary=item.title,
                    model_relevance=relevance_targets(signal_type),
                    is_confirmed=confirmed,
                    requires_corroboration=requires_corroboration,
                    evidence_type=evidence_type,
                    corroboration_status=corroboration_status,
                    contradicting_signal_ids=[],
                    prediction_impact_targets=relevance_targets(signal_type),
                    severity_score=severity_for(impact),
                    expiry_stage=expiry_for(signal_type),
                    can_shift_probability=can_shift_probability,
                    should_surface_in_app=True,
                    material_change=material_change,
                    raw_evidence_excerpt=None,
                    event_category=category_for(signal_type),
                    linked_document_type=document_type_for(item),
                )
            )
            break
    return signals


def is_official_deterministic_source(item: SourceItem) -> bool:
    if item.source_tier != "A":
        return False
    return item.source_type in {"official_fia", "official_f1", "weather"}


def evidence_type_for(item: SourceItem) -> str:
    if item.source_id.startswith("fia_"):
        return "official_fact"
    if item.source_tier == "A":
        return "official_fact"
    text = f"{item.title} {item.raw_excerpt}".lower()
    if "rumour" in text or "rumor" in text:
        return "rumour"
    if "weather" in text or "rain" in text or "wet" in text:
        return "weather_forecast"
    if "technical" in item.source_name.lower() or "floor" in text or "upgrade" in text:
        return "technical_observation"
    if "said" in text or "quote" in text:
        return "team_statement"
    return "journalist_analysis"


def normalize_evidence_type(value: str | None, item: SourceItem) -> str:
    evidence_type = (value or evidence_type_for(item)).strip().lower()
    return evidence_type if evidence_type in EVIDENCE_TYPES else evidence_type_for(item)


def normalize_corroboration_status(value: str | None, source_tier: str, evidence_type: str) -> str:
    if source_tier == "A" and evidence_type == "official_fact":
        return "officially_confirmed"
    status = (value or "single_source").strip().lower()
    return status if status in CORROBORATION_STATUSES else "single_source"


def normalize_signal_controls(
    *,
    signal_type: str,
    source_tier: str,
    evidence_type: str,
    requested_impact_level: str,
    requested_confidence: float,
    requested_corroboration_status: str,
    requested_requires_corroboration: bool,
    requested_can_shift_probability: bool,
) -> tuple[str, float, str, bool, bool, bool]:
    confidence = max(0.0, min(1.0, requested_confidence))
    impact_level = requested_impact_level if requested_impact_level in {"low", "medium", "high"} else "medium"
    corroboration_status = normalize_corroboration_status(requested_corroboration_status, source_tier, evidence_type)
    requires_corroboration = requested_requires_corroboration or corroboration_status in {"single_source", "unclear", "contradicted"}
    can_shift_probability = requested_can_shift_probability

    if evidence_type == "rumour":
        confidence = min(confidence, 0.34)
        impact_level = "low"
        requires_corroboration = True
        can_shift_probability = False
        corroboration_status = "single_source" if corroboration_status == "officially_confirmed" else corroboration_status
    elif source_tier == "A" and evidence_type == "official_fact":
        confidence = max(confidence, 0.9)
        requires_corroboration = False
        corroboration_status = "officially_confirmed"
    elif corroboration_status == "single_source" and impact_level == "high":
        impact_level = "medium"
    elif source_tier in {"C", "D"} and impact_level == "high":
        impact_level = "medium"

    if impact_level == "high" and signal_type not in HIGH_IMPACT_SIGNAL_TYPES:
        impact_level = "medium"
    if evidence_type in {"team_statement", "driver_statement"} and impact_level == "high":
        impact_level = "medium"

    material_change = material_change_for(
        signal_type=signal_type,
        impact_level=impact_level,
        confidence=confidence,
        evidence_type=evidence_type,
        corroboration_status=corroboration_status,
        can_shift_probability=can_shift_probability,
    )
    return impact_level, confidence, corroboration_status, requires_corroboration, can_shift_probability, material_change


def material_change_for(
    *,
    signal_type: str,
    impact_level: str,
    confidence: float,
    evidence_type: str,
    corroboration_status: str,
    can_shift_probability: bool,
) -> bool:
    if evidence_type == "rumour" or corroboration_status == "contradicted":
        return False
    if signal_type not in MATERIAL_SIGNAL_TYPES:
        return False
    if impact_level == "high" and confidence >= 0.55:
        return True
    if corroboration_status == "officially_confirmed" and confidence >= 0.75:
        return True
    if can_shift_probability and impact_level == "medium" and confidence >= 0.75 and corroboration_status in {"multi_source", "officially_confirmed"}:
        return True
    return False


def relevance_targets(signal_type: str) -> list[str]:
    if "safety_car" in signal_type or "weather" in signal_type:
        return ["safety_car_probability", "race_volatility"]
    if "grid" in signal_type or "pace" in signal_type or "upgrade" in signal_type:
        return ["race_win_probability", "podium_probability", "constructor_win_probability"]
    return ["race_win_probability"]


def severity_for(impact: str) -> float:
    return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(impact, 0.4)


def expiry_for(signal_type: str) -> str | None:
    if "qualifying" in signal_type or "single_lap" in signal_type:
        return "after_qualifying"
    if "sprint" in signal_type:
        return "after_sprint"
    return None


def category_for(signal_type: str) -> str:
    if "penalty" in signal_type or "investigation" in signal_type:
        return "official_document"
    if "pace" in signal_type or "tyre" in signal_type:
        return "performance"
    if "weather" in signal_type or "safety_car" in signal_type:
        return "conditions"
    if "reliability" in signal_type or "power_unit" in signal_type:
        return "reliability"
    return "intelligence"


def document_type_for(item: SourceItem) -> str | None:
    if item.source_id.startswith("fia_"):
        lowered = item.title.lower()
        if "summons" in lowered:
            return "summons"
        if "decision" in lowered:
            return "stewards_decision"
        if "classification" in lowered:
            return "classification"
        if "grid" in lowered:
            return "starting_grid"
        return "fia_document"
    return None
