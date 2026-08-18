from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    id: str
    name: str
    tier: str
    enabled: bool
    connector_type: str
    reliability_weight: float
    url: str | None = None
    base_url: str | None = None
    role: str | None = None
    language: str = "en"
    source_type: str | None = None
    fallback_page_url: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceConfig":
        return cls(
            id=payload["id"],
            name=payload["name"],
            tier=payload.get("tier", "C"),
            enabled=bool(payload.get("enabled", False)),
            connector_type=payload.get("connector_type", "html_index_page"),
            reliability_weight=float(payload.get("reliability_weight", 0.5)),
            url=payload.get("url"),
            base_url=payload.get("base_url"),
            role=payload.get("role"),
            language=payload.get("language", "en"),
            source_type=payload.get("source_type"),
            fallback_page_url=payload.get("fallback_page_url"),
        )


def load_source_registry(path: Path) -> list[SourceConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SourceConfig.from_dict(item) for item in payload.get("sources", [])]
