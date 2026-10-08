from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from agentic_radiogen.data.http_json import RemoteApiError, post_json


# Strong free-tier defaults (no paid key required on provider free tiers).
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Tried in order if the preferred model returns 404 for new API keys.
GEMINI_MODEL_FALLBACKS = (
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class LlmConfig:
    provider: str  # gemini | groq | off
    model: str
    api_key: str | None = None


def resolve_llm_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    enabled: bool | None = None,
) -> LlmConfig:
    """Pick the strongest available free provider from env / args.

    Preference: Gemini 3.8 Flash (free AI Studio tier) → Groq gpt-oss-120b → off.
    """
    if enabled is False:
        return LlmConfig(provider="off", model="", api_key=None)

    env_provider = (provider or os.environ.get("AGENTIC_RADIOGEN_LLM") or "").strip().lower()
    # --llm / enabled=True overrides AGENTIC_RADIOGEN_LLM=off.
    if enabled is not True and env_provider in {"off", "none", "rules", "0", "false"}:
        return LlmConfig(provider="off", model="", api_key=None)

    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
        or ""
    ).strip()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()

    if env_provider in {"gemini", "google"}:
        return LlmConfig(
            provider="gemini",
            model=(model or os.environ.get("AGENTIC_RADIOGEN_LLM_MODEL") or DEFAULT_GEMINI_MODEL),
            api_key=gemini_key or None,
        )
    if env_provider == "groq":
        return LlmConfig(
            provider="groq",
            model=(model or os.environ.get("AGENTIC_RADIOGEN_LLM_MODEL") or DEFAULT_GROQ_MODEL),
            api_key=groq_key or None,
        )

    # Auto: strongest free option with a key present.
    if gemini_key or enabled is True:
        return LlmConfig(
            provider="gemini",
            model=(model or os.environ.get("AGENTIC_RADIOGEN_LLM_MODEL") or DEFAULT_GEMINI_MODEL),
            api_key=gemini_key or None,
        )
    if groq_key:
        return LlmConfig(
            provider="groq",
            model=(model or os.environ.get("AGENTIC_RADIOGEN_LLM_MODEL") or DEFAULT_GROQ_MODEL),
            api_key=groq_key or None,
        )
    if enabled is True:
        # Explicitly requested but no key — still declare preferred provider for errors.
        return LlmConfig(
            provider="gemini",
            model=(model or DEFAULT_GEMINI_MODEL),
            api_key=None,
        )
    return LlmConfig(provider="off", model="", api_key=None)


class LlmClient:
    """Minimal free-tier chat client (Gemini + Groq). No paid SDK required."""

    def __init__(
        self,
        config: LlmConfig | None = None,
        *,
        poster: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or resolve_llm_config()
        self._post = poster or post_json

    @property
    def available(self) -> bool:
        return self.config.provider != "off" and bool(self.config.api_key)

    def complete_json(self, *, system: str, user: str, timeout: int | None = None) -> dict[str, Any]:
        if self.config.provider == "off":
            raise RuntimeError("LLM provider is off")
        if not self.config.api_key:
            raise RuntimeError(
                f"No API key for LLM provider '{self.config.provider}'. "
                "Set GEMINI_API_KEY (Google AI Studio, free) or GROQ_API_KEY."
            )
        # Keep this short: Orchestrator falls back to rules on failure. Long waits
        # look like a hang (especially on WSL).
        if timeout is None:
            timeout = int(os.environ.get("AGENTIC_RADIOGEN_LLM_TIMEOUT") or 25)
        if self.config.provider == "gemini":
            text = self._gemini_text(system=system, user=user, timeout=timeout)
        elif self.config.provider == "groq":
            text = self._groq_text(system=system, user=user, timeout=timeout)
        else:
            raise RuntimeError(f"Unknown LLM provider '{self.config.provider}'")
        return _parse_json_object(text)

    def _gemini_text(self, *, system: str, user: str, timeout: int) -> str:
        models: list[str] = []
        for name in (self.config.model, *GEMINI_MODEL_FALLBACKS):
            if name and name not in models:
                models.append(name)
        last_error: Exception | None = None
        for model in models:
            try:
                data = self._gemini_generate(
                    model=model,
                    system=system,
                    user=user,
                    timeout=timeout,
                    json_mime=True,
                )
            except RemoteApiError as exc:
                last_error = exc
                msg = str(exc).lower()
                # Timeouts: do not cascade into more slow attempts; surface once.
                if "timed out" in msg or "timeout" in msg:
                    raise RemoteApiError(_redact_secrets(str(exc))) from exc
                # Retry without JSON mime, then try next model on 404.
                if "404" not in msg:
                    try:
                        data = self._gemini_generate(
                            model=model,
                            system=system,
                            user=user,
                            timeout=timeout,
                            json_mime=False,
                        )
                    except RemoteApiError as exc2:
                        last_error = exc2
                        msg2 = str(exc2).lower()
                        if "404" in msg2 or "timed out" in msg2 or "timeout" in msg2:
                            if "404" in msg2:
                                continue
                            raise RemoteApiError(_redact_secrets(str(exc2))) from exc2
                        raise RemoteApiError(_redact_secrets(str(exc2))) from exc2
                else:
                    continue
            # Remember which model actually worked (for logs / backend tag).
            if model != self.config.model:
                object.__setattr__(
                    self,
                    "config",
                    LlmConfig(
                        provider=self.config.provider,
                        model=model,
                        api_key=self.config.api_key,
                    ),
                )
            return _gemini_response_text(data)
        raise RemoteApiError(_redact_secrets(str(last_error or "Gemini request failed")))

    def _gemini_generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout: int,
        json_mime: bool,
    ) -> Any:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        generation: dict[str, Any] = {"temperature": 0.1}
        if json_mime:
            generation["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        try:
            # Prefer header auth so timeouts/errors do not echo the API key in the URL.
            # One try only — Orchestrator falls back to rules instead of hanging.
            return self._post(
                url,
                payload,
                timeout=timeout,
                headers={"x-goog-api-key": str(self.config.api_key)},
                retries=1,
            )
        except RemoteApiError as exc:
            raise RemoteApiError(_redact_secrets(str(exc))) from exc

    def _groq_text(self, *, system: str, user: str, timeout: int) -> str:
        data = self._post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                "model": self.config.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            retries=1,
        )
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise RemoteApiError(f"Empty Groq response: {data!r}")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if not content:
            raise RemoteApiError(f"Groq response missing content: {data!r}")
        return str(content)


def _gemini_response_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise RemoteApiError(f"Unexpected Gemini response: {data!r}")
    candidates = data.get("candidates") or []
    if not candidates:
        raise RemoteApiError(f"Gemini returned no candidates: {data!r}")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RemoteApiError(f"Gemini response missing text: {data!r}")
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"LLM JSON must be an object, got {type(data).__name__}")
    return data


def _redact_secrets(text: str) -> str:
    """Strip API keys from error strings before they hit logs."""
    out = re.sub(r"([?&]key=)[^&\s]+", r"\1***", text, flags=re.IGNORECASE)
    out = re.sub(r"(Bearer\s+)\S+", r"\1***", out, flags=re.IGNORECASE)
    return out
