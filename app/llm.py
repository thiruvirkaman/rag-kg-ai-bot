"""OpenAI-compatible LLM client wrapper.

Works with OpenAI, Ollama Cloud, LM Studio, or any OpenAI-compatible endpoint
configured via ``llm_base_url`` / ``llm_api_key`` in settings.
"""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .console import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper over the OpenAI SDK with retry + JSON helpers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        model = model or self.settings.llm_chat_model
        temperature = (
            self.settings.llm_temperature if temperature is None else temperature
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("LLM chat call failed: %s", exc)
            raise

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Call LLM and parse the response as JSON.

        Requests ``json_object`` response format when supported, and falls
        back to extracting a ```json``` fenced block or first ``{...}`` slice
        if the server ignores the format hint.
        """
        raw = self.chat(messages, model=model, temperature=temperature,
                        json_mode=True)
        data = _parse_json_loose(raw)
        if data is None:
            # second pass without json_mode in case server rejected it
            log.warning("JSON parse failed with json_mode; retrying plain.")
            raw = self.chat(messages, model=model, temperature=temperature)
            data = _parse_json_loose(raw)
        if data is None:
            raise LLMError(f"Could not parse JSON from LLM response: {raw[:300]}")
        return data


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # ```json fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def make_client(settings: Settings) -> LLMClient:
    return LLMClient(settings)
