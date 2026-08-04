"""Embeddings client (OpenAI-compatible)."""
from __future__ import annotations

from collections.abc import Sequence

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


class EmbeddingsClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        base = settings.embed_base_url or settings.llm_base_url
        key = settings.embed_api_key or settings.llm_api_key
        self._client = OpenAI(base_url=base, api_key=key, timeout=60.0)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        resp = self._client.embeddings.create(
            model=self.settings.embed_model,
            input=texts,
            dimensions=self.settings.embed_dimensions,
        )
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_client(settings: Settings) -> EmbeddingsClient:
    return EmbeddingsClient(settings)
