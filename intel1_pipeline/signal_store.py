from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from .signals import ExtractedSignal
from .source_items import SourceItem
from .time_utils import isoformat, utc_now


EVIDENCE_TYPES = {"official", "reported", "rumour", "sentiment", "derived"}
SOCIAL_SOURCE_TYPES = {"reddit", "x"}
PROMPT_VERSION = "hybrid-signals-v1"


@dataclass(frozen=True)
class StoredSignal:
    id: str
    eventId: str
    sessionType: str
    sourceType: str
    sourceId: str
    evidenceUrl: str
    evidenceType: str
    signalType: str
    target: str
    summary: str
    strength: float
    confidence: float
    sourceQuality: float
    createdAt: str
    modelUsed: str
    promptVersion: str
    inputHash: str
    sourceBatchId: str
    providerRequestId: str | None
    modelTemperature: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ExtractionError:
    id: str
    eventId: str
    sessionType: str
    provider: str
    modelUsed: str
    sourceBatchId: str
    errorType: str
    message: str
    createdAt: str
    providerRequestId: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_source_batch_id(run_id: str, items: list[SourceItem]) -> str:
    hashes = "-".join(sorted(item.content_hash[:12] for item in items[:40]))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{hashes}"))


def signal_to_stored_signal(
    signal: ExtractedSignal,
    *,
    source_batch_id: str,
    model_used: str,
    provider_request_id: str | None,
    model_temperature: float | None,
) -> StoredSignal:
    target = (signal.drivers or signal.teams or ["field"])[0]
    return validate_stored_signal(
        StoredSignal(
            id=signal.signal_id,
            eventId=signal.weekend_id,
            sessionType=signal.session_context,
            sourceType=signal_source_type(signal),
            sourceId=signal.source_id,
            evidenceUrl=signal.source_url,
            evidenceType=compact_evidence_type(signal),
            signalType=signal.signal_type,
            target=target,
            summary=safe_summary(signal.evidence_summary),
            strength=signal.severity_score,
            confidence=signal.confidence,
            sourceQuality=signal.source_reliability_weight,
            createdAt=isoformat(utc_now()),
            modelUsed=model_used,
            promptVersion=PROMPT_VERSION,
            inputHash=signal.source_content_hash,
            sourceBatchId=source_batch_id,
            providerRequestId=provider_request_id,
            modelTemperature=model_temperature,
        )
    )


def signals_to_stored_signals(signals: list[ExtractedSignal], *, run_id: str) -> list[StoredSignal]:
    batch_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:stored-signals"))
    return [
        signal_to_stored_signal(
            signal,
            source_batch_id=batch_id,
            model_used="legacy-or-deterministic",
            provider_request_id=None,
            model_temperature=None,
        )
        for signal in signals
    ]


def validate_stored_signal(signal: StoredSignal) -> StoredSignal:
    if signal.evidenceType not in EVIDENCE_TYPES:
        raise ValueError(f"Unsupported evidenceType: {signal.evidenceType}")
    for label, value in {
        "strength": signal.strength,
        "confidence": signal.confidence,
        "sourceQuality": signal.sourceQuality,
    }.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 or value > 1:
            raise ValueError(f"{label} must be finite and between 0 and 1")
    if not signal.signalType or not signal.target or not signal.evidenceUrl:
        raise ValueError("StoredSignal requires signalType, target, and evidenceUrl")
    if len(signal.summary) > 240:
        raise ValueError("StoredSignal summary must be short")
    return signal


def extraction_error(
    *,
    event_id: str,
    session_type: str,
    provider: str,
    model_used: str,
    source_batch_id: str,
    error_type: str,
    message: str,
    provider_request_id: str | None = None,
) -> ExtractionError:
    return ExtractionError(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{event_id}:{session_type}:{provider}:{source_batch_id}:{error_type}:{message[:80]}")),
        eventId=event_id,
        sessionType=session_type,
        provider=provider,
        modelUsed=model_used,
        sourceBatchId=source_batch_id,
        errorType=error_type,
        message=safe_summary(message, limit=180),
        createdAt=isoformat(utc_now()),
        providerRequestId=provider_request_id,
    )


def compact_evidence_type(signal: ExtractedSignal) -> str:
    if signal.evidence_type == "rumour":
        return "rumour"
    if signal.evidence_type in {"model_inference", "weather_forecast", "session_data"}:
        return "derived"
    if signal.source_tier == "A" and signal.evidence_type == "official_fact":
        return "official"
    return "reported"


def signal_source_type(signal: ExtractedSignal) -> str:
    source_id = signal.source_id.lower()
    if "reddit" in source_id:
        return "reddit"
    if "twitter" in source_id or source_id.endswith("_x"):
        return "x"
    if source_id.startswith("fia_"):
        return "official_fia"
    if signal.source_tier == "A":
        return "official_f1"
    return "trusted_news"


def safe_summary(value: str, limit: int = 220) -> str:
    return " ".join(value.split())[:limit].strip()
