from __future__ import annotations

import hashlib
import html
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import urljoin

from .config import SourceConfig
from .http import FetchError, fetch_text
from .time_utils import isoformat, utc_now


@dataclass
class SourceItem:
    source_item_id: str
    source_id: str
    source_name: str
    source_tier: str
    source_type: str
    reliability_weight: float
    connector_type: str
    title: str
    url: str
    canonical_url: str
    published_at: str | None
    updated_at: str | None
    fetched_at: str
    raw_excerpt: str
    raw_content: str | None
    language: str
    weekend_relevance_status: str
    fetch_status: str
    content_hash: str
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fetch_source_items(sources: Iterable[SourceConfig], max_items_per_source: int = 8) -> tuple[list[SourceItem], list[SourceItem]]:
    successful: list[SourceItem] = []
    failed: list[SourceItem] = []
    for source in sources:
        if not source.enabled or source.language.lower() != "en":
            continue
        try:
            items = fetch_items_for_source(source, max_items=max_items_per_source)
            successful.extend(items)
        except FetchError as error:
            failed.append(failure_item(source, str(error)))
        except Exception as error:  # defensive source isolation
            failed.append(failure_item(source, f"{type(error).__name__}: {error}"))
    return dedupe_items(successful), failed


def fetch_items_for_source(source: SourceConfig, max_items: int) -> list[SourceItem]:
    if source.connector_type == "rss" and source.url:
        return parse_rss(source, fetch_text(source.url), max_items=max_items)
    if source.connector_type in {"html_article", "html_article_page"} and source.url:
        return [parse_html_article(source, fetch_text(source.url))]
    if source.url:
        return parse_html_index(source, fetch_text(source.url), max_items=max_items)
    if source.base_url:
        return [
            make_item(
                source=source,
                title=f"{source.name} API available",
                url=source.base_url,
                raw_excerpt=f"Structured source role: {source.role or 'structured data'}",
                fetch_status="success",
            )
        ]
    return []


def parse_rss(source: SourceConfig, body: str, max_items: int) -> list[SourceItem]:
    root = ET.fromstring(body)
    items = root.findall(".//item")
    parsed: list[SourceItem] = []
    for item in items[:max_items]:
        title = text_of(item, "title") or source.name
        link = text_of(item, "link") or source.url or ""
        description = strip_markup(text_of(item, "description") or "")
        published = normalize_datetime(text_of(item, "pubDate"))
        linked_content = fetch_linked_weekend_article_text(title, link)
        excerpt = linked_content[:1000] if linked_content else description[:600]
        parsed.append(
            make_item(
                source=source,
                title=html.unescape(title).strip(),
                url=link.strip(),
                raw_excerpt=excerpt,
                raw_content=linked_content,
                published_at=published,
                fetch_status="success",
            )
        )
    return parsed


def parse_html_index(source: SourceConfig, body: str, max_items: int) -> list[SourceItem]:
    title = extract_title(body) or source.name
    links = extract_links(source.url or "", body)
    description = extract_description(body) or title
    if not links:
        return [
            make_item(
                source=source,
                title=title,
                url=source.url or "",
                raw_excerpt=description,
                fetch_status="partial",
            )
        ]
    parsed: list[SourceItem] = []
    for link_title, link_url in links[:max_items]:
        linked_content = fetch_linked_weekend_article_text(link_title, link_url)
        excerpt = linked_content[:1000] if linked_content else description or link_title
        parsed.append(
            make_item(
                source=source,
                title=link_title,
                url=link_url,
                raw_excerpt=excerpt,
                raw_content=linked_content,
                fetch_status="success",
            )
        )
    return parsed


def parse_html_article(source: SourceConfig, body: str) -> SourceItem:
    title = extract_title(body) or source.name
    description = extract_description(body) or title
    article_text = strip_markup(body)
    excerpt = article_text[:1000] if article_text else description
    return make_item(
        source=source,
        title=title,
        url=source.url or "",
        raw_excerpt=excerpt,
        raw_content=article_text,
        fetch_status="success",
    )


def make_item(
    *,
    source: SourceConfig,
    title: str,
    url: str,
    raw_excerpt: str,
    fetch_status: str,
    published_at: str | None = None,
    raw_content: str | None = None,
) -> SourceItem:
    normalized_title = " ".join(title.split())
    normalized_excerpt = " ".join(raw_excerpt.split())
    normalized_content = " ".join(raw_content.split()) if raw_content else None
    canonical_url = canonicalize_url(url)
    content_hash = stable_hash("|".join([normalized_title, canonical_url, normalized_excerpt]))
    fetched_at = isoformat(utc_now())
    return SourceItem(
        source_item_id=stable_hash("|".join([source.id, canonical_url, normalized_title]))[:24],
        source_id=source.id,
        source_name=source.name,
        source_tier=source.tier,
        source_type=source_type_for(source),
        reliability_weight=source.reliability_weight,
        connector_type=source.connector_type,
        title=normalized_title or source.name,
        url=url,
        canonical_url=canonical_url,
        published_at=published_at,
        updated_at=None,
        fetched_at=fetched_at,
        raw_excerpt=normalized_excerpt[:1000],
        raw_content=normalized_content[:8000] if normalized_content else None,
        language=source.language,
        weekend_relevance_status="unknown",
        fetch_status=fetch_status,
        content_hash=content_hash,
    )


def failure_item(source: SourceConfig, reason: str) -> SourceItem:
    now = isoformat(utc_now())
    return SourceItem(
        source_item_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source.id}:{now}:failed")),
        source_id=source.id,
        source_name=source.name,
        source_tier=source.tier,
        source_type=source_type_for(source),
        reliability_weight=source.reliability_weight,
        connector_type=source.connector_type,
        title=f"{source.name} fetch failed",
        url=source.url or source.base_url or "",
        canonical_url=source.url or source.base_url or "",
        published_at=None,
        updated_at=None,
        fetched_at=now,
        raw_excerpt="",
        raw_content=None,
        language=source.language,
        weekend_relevance_status="unknown",
        fetch_status="failed",
        content_hash=stable_hash(reason),
        failure_reason=reason[:500],
    )


def dedupe_items(items: list[SourceItem]) -> list[SourceItem]:
    seen: set[str] = set()
    deduped: list[SourceItem] = []
    for item in items:
        key = item.content_hash
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def text_of(item: ET.Element, tag: str) -> str | None:
    value = item.findtext(tag)
    return value.strip() if value else None


def extract_title(body: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
    return strip_markup(match.group(1)) if match else None


def extract_description(body: str) -> str | None:
    match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        body,
        re.IGNORECASE,
    )
    return html.unescape(match.group(1)).strip() if match else None


def extract_links(base_url: str, body: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for href, title in re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, re.IGNORECASE | re.DOTALL):
        clean_title = strip_markup(title)
        if not clean_title or len(clean_title) < 8:
            continue
        if not is_f1_relevant(clean_title + " " + href):
            continue
        links.append((clean_title[:180], urljoin(base_url, href)))
    return links


def strip_markup(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(re.sub(r"\s+", " ", value)).strip()


def is_f1_relevant(text: str) -> bool:
    lowered = text.lower()
    terms = ["f1", "formula 1", "grand prix", "fia", "qualifying", "sprint", "pirelli", "steward"]
    return any(term in lowered for term in terms)


def fetch_linked_weekend_article_text(title: str, url: str) -> str | None:
    if not is_weekend_article_relevant(title + " " + url):
        return None
    try:
        return strip_markup(fetch_text(url, timeout=10))[:8000]
    except FetchError:
        return None


def fetch_linked_session_result_text(title: str, url: str) -> str | None:
    return fetch_linked_weekend_article_text(title, url)


def is_weekend_article_relevant(text: str) -> bool:
    if is_session_result_relevant(text):
        return True
    lowered = text.lower()
    terms = [
        "fp1",
        "fp2",
        "fp3",
        "practice",
        "qualifying",
        "sprint",
        "pole",
        "front row",
        "front-row",
        "race pace",
        "long run",
        "strategy guide",
        "strategy",
        "need to know",
        "what the teams said",
        "tyre",
        "tire",
        "rain",
        "weather",
        "safety car",
        "vsc",
        "hazard",
        "podium",
        "wins",
        "win",
        "pace",
    ]
    return any(term in lowered for term in terms)


def is_session_result_relevant(text: str) -> bool:
    lowered = text.lower()
    terms = [
        "classification",
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
    ]
    return any(term in lowered for term in terms)


def source_type_for(source: SourceConfig) -> str:
    configured = (source.source_type or "").strip().lower()
    allowed = {"official_fia", "official_f1", "team_official", "trusted_news", "reddit", "x", "weather"}
    if configured in allowed:
        return configured
    text = f"{source.id} {source.name} {source.role or ''}".lower()
    if "reddit" in text:
        return "reddit"
    if "twitter" in text or text.endswith(" x") or " x_" in text:
        return "x"
    if "weather" in text:
        return "weather"
    if "fia" in text:
        return "official_fia"
    if source.tier == "A":
        return "official_f1"
    return "trusted_news"


def canonicalize_url(url: str) -> str:
    return url.split("#", 1)[0].strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return None
