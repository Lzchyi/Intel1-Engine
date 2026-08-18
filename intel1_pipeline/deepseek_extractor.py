from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .ai import cached_signals_for_items, select_ai_items
from .providers import AIProvider, DeepSeekProvider, ProviderUnavailable
from .signal_store import (
    PROMPT_VERSION,
    ExtractionError,
    StoredSignal,
    build_source_batch_id,
    extraction_error,
    signal_to_stored_signal,
    signals_to_stored_signals,
    validate_stored_signal,
)
from .signals import ExtractedSignal, deterministic_extract_signals, normalize_signal_controls, relevance_targets, severity_for
from .source_items import SourceItem


HYBRID_EXTRACTOR_PROMPT = """
You are Intel1's DeepSeek extraction layer.
Extract structured Formula 1 intelligence only from the source items supplied by the backend.
Do not browse, infer missing facts, or create probabilities.
Reject unsupported claims by omitting them.
Every signal must cite one supplied source_item_id and evidenceUrl.
Summaries must be short paraphrases, not quotes.
Return JSON only.
"""


@dataclass
class HybridExtractionResult:
    signals: list[ExtractedSignal]
    stored_signals: list[StoredSignal]
    extraction_errors: list[ExtractionError]


class DeepSeekSignalExtractor:
    def __init__(self, provider: AIProvider | None = None) -> None:
        self.provider = provider or DeepSeekProvider()

    def extract(self, *, items: list[SourceItem], weekend_id: str, run_id: str, stage: str) -> HybridExtractionResult:
        candidate_items = select_ai_items(items)
        source_batch_id = build_source_batch_id(run_id, candidate_items)
        response = self.provider.complete_json(
            system_prompt=HYBRID_EXTRACTOR_PROMPT,
            user_payload=deepseek_extraction_payload(candidate_items, weekend_id, stage, source_batch_id),
        )
        signals: list[ExtractedSignal] = []
        stored: list[StoredSignal] = []
        by_id = {item.source_item_id: item for item in candidate_items}
        raw_signals = response.payload.get("signals")
        if not isinstance(raw_signals, list):
            raise ValueError("DeepSeek extraction response missing signals list")

        for raw in raw_signals:
            signal = extracted_signal_from_deepseek(
                raw,
                by_id=by_id,
                weekend_id=weekend_id,
                run_id=run_id,
                stage=stage,
            )
            signals.append(signal)
            stored.append(
                validate_stored_signal(
                    signal_to_stored_signal(
                        signal,
                        source_batch_id=source_batch_id,
                        model_used=response.model_used,
                        provider_request_id=response.provider_request_id,
                        model_temperature=response.model_temperature,
                    )
                )
            )
        return HybridExtractionResult(signals=signals, stored_signals=stored, extraction_errors=[])


def extract_hybrid_signals(
    *,
    items: list[SourceItem],
    weekend_id: str,
    run_id: str,
    stage: str,
    skip_ai: bool,
    cached_signals_by_hash: dict[str, list[dict[str, Any]]] | None = None,
    provider: AIProvider | None = None,
) -> HybridExtractionResult:
    cached, uncached_items = cached_signals_for_items(items, weekend_id, run_id, stage, cached_signals_by_hash or {})
    if not uncached_items:
        return HybridExtractionResult(
            signals=cached,
            stored_signals=signals_to_stored_signals(cached, run_id=run_id),
            extraction_errors=[],
        )

    if skip_ai:
        official_signals = deterministic_extract_signals(uncached_items, weekend_id, run_id, stage)
        signals = cached + official_signals
        return HybridExtractionResult(
            signals=signals,
            stored_signals=signals_to_stored_signals(signals, run_id=run_id),
            extraction_errors=[],
        )

    extractor = DeepSeekSignalExtractor(provider=provider)
    source_batch_id = build_source_batch_id(run_id, uncached_items)
    try:
        result = extractor.extract(items=uncached_items, weekend_id=weekend_id, run_id=run_id, stage=stage)
        return HybridExtractionResult(
            signals=cached + result.signals,
            stored_signals=signals_to_stored_signals(cached, run_id=run_id) + result.stored_signals,
            extraction_errors=result.extraction_errors,
        )
    except ProviderUnavailable as error:
        return extraction_failure_result(
            cached=cached,
            fallback_items=uncached_items,
            weekend_id=weekend_id,
            run_id=run_id,
            stage=stage,
            source_batch_id=source_batch_id,
            error_type="provider_unavailable",
            message=str(error),
            model_used=getattr(extractor.provider, "model_name", "deepseek"),
        )
    except Exception as error:
        return extraction_failure_result(
            cached=cached,
            fallback_items=uncached_items,
            weekend_id=weekend_id,
            run_id=run_id,
            stage=stage,
            source_batch_id=source_batch_id,
            error_type="validation_or_provider_error",
            message=f"{type(error).__name__}: {error}",
            model_used=getattr(extractor.provider, "model_name", "deepseek"),
        )


def extraction_failure_result(
    *,
    cached: list[ExtractedSignal],
    fallback_items: list[SourceItem],
    weekend_id: str,
    run_id: str,
    stage: str,
    source_batch_id: str,
    error_type: str,
    message: str,
    model_used: str,
) -> HybridExtractionResult:
    official_signals = deterministic_extract_signals(fallback_items, weekend_id, run_id, stage)
    signals = cached + official_signals
    return HybridExtractionResult(
        signals=signals,
        stored_signals=signals_to_stored_signals(signals, run_id=run_id),
        extraction_errors=[
            extraction_error(
                event_id=weekend_id,
                session_type=stage,
                provider="deepseek",
                model_used=model_used,
                source_batch_id=source_batch_id,
                error_type=error_type,
                message=message,
            )
        ],
    )


def deepseek_extraction_payload(items: list[SourceItem], weekend_id: str, stage: str, source_batch_id: str) -> dict[str, Any]:
    return {
        "task": "Extract F1 intelligence signals from supplied source items only.",
        "promptVersion": PROMPT_VERSION,
        "eventId": weekend_id,
        "sessionType": stage,
        "sourceBatchId": source_batch_id,
        "allowedEvidenceTypes": ["official", "reported", "rumour", "sentiment", "derived"],
        "sourceItems": [
            {
                "source_item_id": item.source_item_id,
                "sourceId": item.source_id,
                "sourceType": item.source_type,
                "sourceQuality": item.reliability_weight,
                "evidenceUrl": item.url,
                "title": item.title,
                "author": None,
                "fetchedAt": item.fetched_at,
                "publishedAt": item.published_at,
                "inputHash": item.content_hash,
                "rawText": item.raw_excerpt[:900],
            }
            for item in items
        ],
        "outputSchema": {
            "signals": [
                {
                    "source_item_id": "string",
                    "signalType": "string",
                    "target": "driver/team/field",
                    "drivers": ["string"],
                    "teams": ["string"],
                    "summary": "short paraphrase",
                    "strength": "number 0-1",
                    "confidence": "number 0-1",
                    "evidenceType": "official|reported|rumour|sentiment|derived",
                    "canShiftProbability": "boolean",
                }
            ]
        },
    }


def extracted_signal_from_deepseek(
    raw: Any,
    *,
    by_id: dict[str, SourceItem],
    weekend_id: str,
    run_id: str,
    stage: str,
) -> ExtractedSignal:
    if not isinstance(raw, dict):
        raise ValueError("Signal entry must be an object")
    source_item_id = str(raw.get("source_item_id") or raw.get("sourceItemId") or "")
    item = by_id.get(source_item_id)
    if item is None:
        raise ValueError(f"Signal references unknown source_item_id: {source_item_id}")
    signal_type = str(raw.get("signalType") or raw.get("signal_type") or "").strip()
    target = str(raw.get("target") or "field").strip()
    if not signal_type or not target:
        raise ValueError("Signal missing signalType or target")
    evidence_type = compact_to_legacy_evidence_type(str(raw.get("evidenceType") or raw.get("evidence_type") or "reported"))
    strength = bounded_float(raw.get("strength", 0.45), "strength")
    confidence = bounded_float(raw.get("confidence", 0.5), "confidence")
    source_quality = bounded_float(raw.get("sourceQuality", item.reliability_weight), "sourceQuality")
    if abs(source_quality - item.reliability_weight) > 0.4:
        raise ValueError("sourceQuality is inconsistent with configured source quality")

    drivers = string_list(raw.get("drivers") or raw.get("driver_mentions"))
    teams = string_list(raw.get("teams") or raw.get("team_mentions"))
    if target != "field" and not drivers and not teams:
        drivers = [target]

    impact_level = "high" if strength >= 0.75 else "medium" if strength >= 0.42 else "low"
    can_shift = bool(raw.get("canShiftProbability", evidence_type not in {"rumour", "model_inference"}))
    impact, confidence, corroboration_status, requires_corroboration, can_shift, material_change = normalize_signal_controls(
        signal_type=signal_type,
        source_tier=item.source_tier,
        evidence_type=evidence_type,
        requested_impact_level=impact_level,
        requested_confidence=confidence,
        requested_corroboration_status="officially_confirmed" if evidence_type == "official_fact" and item.source_tier == "A" else "single_source",
        requested_requires_corroboration=item.source_tier != "A",
        requested_can_shift_probability=can_shift,
    )
    summary = " ".join(str(raw.get("summary") or raw.get("evidence_summary") or item.title).split())[:220]
    if not summary:
        raise ValueError("Signal summary is empty")

    return ExtractedSignal(
        signal_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{source_item_id}:{signal_type}:deepseek:{target}")),
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
        teams=teams,
        drivers=drivers,
        signal_type=signal_type,
        direction=str(raw.get("direction") or "mixed"),
        impact_level=impact,
        confidence=confidence,
        evidence_summary=summary,
        model_relevance=string_list(raw.get("model_relevance")) or relevance_targets(signal_type),
        is_confirmed=item.source_tier == "A" and evidence_type == "official_fact",
        requires_corroboration=requires_corroboration,
        evidence_type=evidence_type,
        corroboration_status=corroboration_status,
        contradicting_signal_ids=[],
        prediction_impact_targets=drivers + teams if drivers or teams else relevance_targets(signal_type),
        severity_score=severity_for(impact),
        expiry_stage=None,
        can_shift_probability=can_shift,
        should_surface_in_app=True,
        material_change=material_change,
        raw_evidence_excerpt=None,
        event_category="deepseek_extracted",
        linked_document_type="fia_document" if item.source_type == "official_fia" else None,
    )


def compact_to_legacy_evidence_type(value: str) -> str:
    evidence_type = value.strip().lower()
    if evidence_type == "official":
        return "official_fact"
    if evidence_type == "rumour":
        return "rumour"
    if evidence_type in {"sentiment", "derived"}:
        return "model_inference"
    return "journalist_analysis"


def bounded_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if number < 0 or number > 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
