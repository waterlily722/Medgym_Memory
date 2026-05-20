from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingClient:
    """
    Minimal OpenAI-compatible embedding client.

    Expected endpoint:
      {base_url}/embeddings

    Returns a list of float vectors, one per input text.
    Falls back to None (no embedding) on any error, so the retriever can
    degrade gracefully to token-based cosine similarity.
    """

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: int = 30
    dimensions: int = 0  # 0 = use model default

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("MEMORY_EMBEDDING_MODEL", "")
        self.base_url = self.base_url or os.getenv("MEMORY_EMBEDDING_BASE_URL", "")
        self.api_key = self.api_key or os.getenv("MEMORY_EMBEDDING_API_KEY", "")
        self.base_url = self.base_url.rstrip("/")

    def available(self) -> bool:
        return bool(self.model and self.base_url)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None if unavailable or on error."""
        if not self.available() or not texts:
            logger.warning(
                "EmbeddingClient not available — model=%r base_url=%r",
                self.model, self.base_url,
            )
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions

        request = urllib.request.Request(
            url=f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("EmbeddingClient HTTP error: %s", exc)
            return None

        try:
            data = json.loads(raw)
            embeddings = data.get("data", [])
            # Sort by index to maintain order
            embeddings.sort(key=lambda item: int(item.get("index", 0)))
            vectors = [item["embedding"] for item in embeddings]
            logger.debug(
                "Embedded %d texts with model=%s (dim=%d)",
                len(texts), self.model, len(vectors[0]) if vectors else 0,
            )
            return vectors
        except Exception as exc:
            logger.warning("EmbeddingClient parse error: %s", exc)
            return None

    def embed_one(self, text: str) -> list[float] | None:
        """Convenience: embed a single text."""
        result = self.embed([text])
        if result:
            return result[0]
        return None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
