from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class ProviderJSONResponse:
    payload: dict[str, Any]
    provider_request_id: str
    model_used: str
    model_temperature: float | None


class AIProvider(Protocol):
    provider_name: str
    model_name: str
    model_temperature: float | None

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> ProviderJSONResponse:
        ...


class OpenAIProvider:
    provider_name = "chatgpt"

    def __init__(self, *, model_name: str | None = None, model_temperature: float = 0.0) -> None:
        self.model_name = model_name or os.getenv("OPENAI_PREDICTION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.5"
        self.model_temperature = model_temperature

    @property
    def available(self) -> bool:
        return openai_prediction_enabled() and bool(os.getenv("OPENAI_API_KEY"))

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> ProviderJSONResponse:
        if not openai_prediction_enabled():
            raise ProviderUnavailable("OpenAI prediction provider disabled for this run")
        if not self.available:
            raise ProviderUnavailable("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        client = OpenAI()
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        sent_temperature: float | None = None
        if self.model_temperature is not None and openai_model_supports_temperature(self.model_name):
            request["temperature"] = self.model_temperature
            sent_temperature = self.model_temperature
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or "{}"
        return ProviderJSONResponse(
            payload=parse_json_object(content),
            provider_request_id=str(getattr(response, "id", "") or uuid.uuid4()),
            model_used=self.model_name,
            model_temperature=sent_temperature,
        )


class DeepSeekProvider:
    provider_name = "deepseek"

    def __init__(self, *, model_name: str | None = None, model_temperature: float = 0.0) -> None:
        self.model_name = model_name or deepseek_model_name()
        self.model_temperature = model_temperature

    @property
    def available(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> ProviderJSONResponse:
        if not self.available:
            raise ProviderUnavailable("DEEPSEEK_API_KEY is not configured")
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=self.model_temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return ProviderJSONResponse(
            payload=parse_json_object(content),
            provider_request_id=str(getattr(response, "id", "") or uuid.uuid4()),
            model_used=self.model_name,
            model_temperature=self.model_temperature,
        )


class ProviderUnavailable(RuntimeError):
    pass


def deepseek_model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL


def openai_prediction_enabled() -> bool:
    return env_flag("INTEL1_OPENAI_PREDICTION_ENABLED", default=True)


def env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def openai_model_supports_temperature(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    default_only_prefixes = ("gpt-5", "o1", "o3", "o4")
    return not normalized.startswith(default_only_prefixes)


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
        raise ValueError("Provider response was not a JSON object")
    return payload
