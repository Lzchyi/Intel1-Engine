from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .providers import AIProvider, DeepSeekProvider
from .source_items import SourceItem
from .structured_data import WeekendContext, safe_int, safe_optional_int, session_result_row


SESSION_RESULT_KEYS = ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"]

SESSION_RESULT_PROMPT = """
You are Intel1's DeepSeek session classification extractor.
Use only the supplied source items.
Extract official/latest Formula 1 session classifications only when the source text explicitly lists positions.
Do not browse, infer missing rows, reorder by opinion, or invent times.
Return strict JSON only.
"""

SESSION_RESULT_TERMS = (
    "classification",
    "classified",
    "qualifying result",
    "qualifying results",
    "sprint qualifying",
    "sprint shootout",
    "starting grid",
    "final grid",
    "race result",
    "race results",
    "session result",
    "session results",
)


@dataclass
class DeepSeekSessionResultExtractor:
    provider: AIProvider

    def extract(
        self,
        *,
        items: list[SourceItem],
        weekend: WeekendContext,
        missing_sessions: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        response = self.provider.complete_json(
            system_prompt=SESSION_RESULT_PROMPT,
            user_payload=session_result_payload(items, weekend, missing_sessions),
        )
        return normalize_session_result_response(response.payload, items, missing_sessions)


def supplement_session_results_with_deepseek(
    *,
    items: list[SourceItem],
    weekend: WeekendContext,
    session_results: dict[str, list[dict[str, Any]]],
    skip_ai: bool,
    provider: AIProvider | None = None,
) -> dict[str, list[dict[str, Any]]]:
    patched = {key: list(session_results.get(key) or []) for key in SESSION_RESULT_KEYS}
    missing_sessions = [key for key in expected_completed_session_keys(weekend) if not patched.get(key)]
    candidate_items = session_result_candidate_items(items, missing_sessions)
    if skip_ai or not missing_sessions or not candidate_items:
        return patched

    extractor = DeepSeekSessionResultExtractor(provider or DeepSeekProvider())
    try:
        extracted = extractor.extract(items=candidate_items, weekend=weekend, missing_sessions=missing_sessions)
    except Exception:
        return patched

    for key in missing_sessions:
        rows = extracted.get(key) or []
        if rows:
            patched[key] = rows
    return patched


def expected_completed_session_keys(weekend: WeekendContext) -> list[str]:
    by_stage = {
        "after_fp1": ["fp1"],
        "after_fp2": ["fp1", "fp2"],
        "after_fp3": ["fp1", "fp2", "fp3"],
        "after_sprint_qualifying": ["fp1", "sprint_qualifying"],
        "after_sprint": ["fp1", "sprint_qualifying", "sprint"],
        "after_qualifying": ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying"],
        "final_pre_race": ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying"],
        "post_race": ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"],
    }
    keys = by_stage.get(weekend.stage, [])
    if not weekend.is_sprint_weekend:
        keys = [key for key in keys if key not in {"sprint_qualifying", "sprint"}]
    scheduled = {item.get("session") for item in weekend.session_schedule if isinstance(item, dict)}
    if scheduled:
        keys = [key for key in keys if key in scheduled]
    return [key for key in keys if key in SESSION_RESULT_KEYS]


def session_result_candidate_items(items: list[SourceItem], missing_sessions: list[str]) -> list[SourceItem]:
    wanted = set(missing_sessions)
    candidates: list[SourceItem] = []
    for item in items:
        if item.fetch_status != "success":
            continue
        if item.connector_type == "api" and item.title.endswith("API available"):
            continue
        text = f"{item.title} {item.raw_excerpt} {item.raw_content or ''}".lower()
        if not any(term in text for term in SESSION_RESULT_TERMS):
            continue
        if "sprint qualifying" in text or "sprint shootout" in text:
            if "sprint_qualifying" not in wanted:
                continue
        elif "sprint" in text and "sprint" not in wanted and "sprint_qualifying" not in wanted:
            continue
        elif "qualifying" in text and "qualifying" not in wanted:
            continue
        elif "race" in text and "race" not in wanted:
            continue
        candidates.append(item)
    return candidates[:10]


def session_result_payload(items: list[SourceItem], weekend: WeekendContext, missing_sessions: list[str]) -> dict[str, Any]:
    return {
        "task": "Extract missing F1 session classifications from supplied source items only.",
        "eventId": weekend.weekend_id,
        "grandPrix": weekend.grand_prix_name,
        "circuit": weekend.circuit_name,
        "stage": weekend.stage,
        "missingSessions": missing_sessions,
        "acceptedSessionKeys": SESSION_RESULT_KEYS,
        "sourceItems": [
            {
                "source_item_id": item.source_item_id,
                "sourceName": item.source_name,
                "sourceTier": item.source_tier,
                "evidenceUrl": item.url,
                "title": item.title,
                "text": (item.raw_content or item.raw_excerpt)[:5000],
            }
            for item in items
        ],
        "outputSchema": {
            "session_results": {
                "sprint_qualifying": [
                    {
                        "position": "integer",
                        "driver": "string",
                        "constructor": "string",
                        "time_or_gap": "string",
                        "laps": "integer or null",
                        "status": "string",
                        "source_item_id": "string",
                    }
                ],
                "qualifying": "same row shape",
                "race": "same row shape",
            }
        },
    }


def normalize_session_result_response(
    payload: dict[str, Any],
    items: list[SourceItem],
    missing_sessions: list[str],
) -> dict[str, list[dict[str, Any]]]:
    raw_sessions = payload.get("session_results") or payload.get("sessions") or {}
    if not isinstance(raw_sessions, dict):
        return {}
    by_id = {item.source_item_id: item for item in items}
    fallback_item = items[0] if items else None
    output: dict[str, list[dict[str, Any]]] = {}
    for key in missing_sessions:
        raw_rows = raw_sessions.get(key) or []
        if not isinstance(raw_rows, list):
            continue
        rows = [
            row
            for row in (normalize_deepseek_session_row(raw, by_id, fallback_item) for raw in raw_rows)
            if row is not None
        ]
        deduped = dedupe_session_rows(rows)
        if deduped:
            output[key] = deduped
    return output


def normalize_deepseek_session_row(
    raw: Any,
    by_id: dict[str, SourceItem],
    fallback_item: SourceItem | None,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    position = safe_int(raw.get("position"))
    driver = " ".join(str(raw.get("driver") or "").split())
    if position <= 0 or not driver:
        return None
    source_item_id = str(raw.get("source_item_id") or raw.get("sourceItemId") or "")
    item = by_id.get(source_item_id) or fallback_item
    constructor = " ".join(str(raw.get("constructor") or raw.get("team") or "Unknown").split()) or "Unknown"
    time_or_gap = " ".join(str(raw.get("time_or_gap") or raw.get("time") or raw.get("gap") or "").split())
    status = " ".join(str(raw.get("status") or "classified").split())
    return session_result_row(
        position=position,
        driver=driver,
        constructor=constructor,
        time_or_gap=time_or_gap,
        laps=safe_optional_int(raw.get("laps")),
        status=status,
        source=item.source_name if item else "DeepSeek",
        is_official=item.source_tier == "A" if item else False,
    )


def dedupe_session_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["position"]):
        key = (row["position"], row["driver"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output
