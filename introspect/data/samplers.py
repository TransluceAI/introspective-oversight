"""One text-in, text-out sampler over Gemini, Anthropic, or OpenRouter.

Used for LLM judges during training and for the generator model in data generation. Every
backend is a plain HTTPS call, so there are no provider SDKs to install. API keys come from
``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``, or ``OPENROUTER_API_KEY``.

    async with httpx.AsyncClient(timeout=120) as client:
        sample = TextSampler(client, "anthropic", "claude-haiku-4-5", max_tokens=1024)
        text = await sample("Say hi")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

import httpx

Backend = Literal["gemini", "anthropic", "openrouter"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]

API_KEY_ENV: dict[Backend, str] = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_RETRYABLE_STATUS = {408, 409, 425, 429}


class PermanentAPIError(RuntimeError):
    """A request that will never succeed (bad key, unknown model, bad request)."""


class TextSampler:
    """Sample one response per prompt, retrying transient failures up to 5 times."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        backend: Backend,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning: ReasoningEffort | None = None,  # Gemini thinking level / OpenRouter effort
        max_in_flight: int = 128,
    ) -> None:
        if backend == "anthropic" and reasoning is not None:
            raise ValueError("reasoning is not supported for the anthropic backend")
        api_key = os.environ.get(API_KEY_ENV[backend])
        if not api_key:
            raise ValueError(f"{API_KEY_ENV[backend]} must be set for backend={backend}")
        self.backend = backend
        self.model = model
        self._client = client
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning = reasoning
        self._semaphore = asyncio.Semaphore(max_in_flight)

    def _request(self, prompt: str, system: str | None) -> tuple[str, dict[str, str], dict[str, Any]]:
        if self.backend == "gemini":
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self._temperature,
                    "maxOutputTokens": self._max_tokens,
                },
            }
            if self._reasoning is not None:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self._reasoning}
            if system is not None:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model.removeprefix('models/')}:generateContent"
            )
            return url, {"x-goog-api-key": self._api_key}, body
        if self.backend == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system is not None:
                body["system"] = system
            headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
            return "https://api.anthropic.com/v1/messages", headers, body
        messages = [{"role": "system", "content": system}] if system is not None else []
        body = {
            "model": self.model,
            "messages": [*messages, {"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning is not None:
            body["reasoning"] = {"effort": self._reasoning, "exclude": True}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return "https://openrouter.ai/api/v1/chat/completions", headers, body

    def _text(self, payload: dict[str, Any]) -> str:
        if self.backend == "gemini":
            candidates = payload.get("candidates") or [{}]
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        elif self.backend == "anthropic":
            text = "".join(b.get("text", "") for b in payload.get("content", []) if b["type"] == "text")
        else:
            choices = payload.get("choices") or [{}]
            text = choices[0].get("message", {}).get("content") or ""
        if not text.strip():
            raise ValueError(f"{self.backend} returned no text")
        return text.strip()

    async def _sample_once(self, prompt: str, system: str | None) -> str:
        url, headers, body = self._request(prompt, system)
        async with self._semaphore:
            response = await self._client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return self._text(response.json())

    async def __call__(self, prompt: str, system: str | None = None) -> str:
        for attempt in range(5):
            try:
                return await self._sample_once(prompt, system)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status not in _RETRYABLE_STATUS and not 500 <= status < 600:
                    raise PermanentAPIError(f"{self.backend} {self.model}: HTTP {status}") from error
                if attempt == 4:
                    raise
            except (httpx.TransportError, TimeoutError, ValueError):
                if attempt == 4:
                    raise
            await asyncio.sleep(min(2**attempt, 16))
        raise AssertionError("unreachable")
