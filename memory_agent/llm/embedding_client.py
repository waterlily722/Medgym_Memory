from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingClient:
    """
    Minimal OpenAI-compatible embedding client.

    Expected endpoint:
      {base_url}/embeddings

    Returns a list of float vectors, one per input text. Returns None on request
    errors; embedding-mode callers treat that as a fatal error.
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

    def embed(
        self,
        texts: list[str],
        input_type: str | None = None,
    ) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None if unavailable or on error."""
        if not self.available():
            logger.warning(
                "EmbeddingClient not available — model=%r base_url=%r",
                self.model, self.base_url,
            )
            return None
        if not texts:
            logger.debug("EmbeddingClient received an empty text batch")
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._prepare_texts(texts, input_type=input_type),
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
            if self._is_local_base_url():
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                response_ctx = opener.open(request, timeout=self.timeout)
            else:
                response_ctx = urllib.request.urlopen(request, timeout=self.timeout)
            with response_ctx as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            logger.warning(
                "EmbeddingClient HTTP error: %s; url=%s model=%s body=%s",
                exc,
                request.full_url,
                self.model,
                body[:500],
            )
            return None
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

    def embed_one(
        self,
        text: str,
        input_type: str | None = None,
    ) -> list[float] | None:
        """Convenience: embed a single text."""
        result = self.embed([text], input_type=input_type)
        if result:
            return result[0]
        return None

    def embed_query(self, text: str) -> list[float] | None:
        """Embed a retrieval query."""
        return self.embed_one(text, input_type="query")

    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        """Embed stored memory/document texts."""
        return self.embed(texts, input_type="passage")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_texts(
        self,
        texts: list[str],
        input_type: str | None = None,
    ) -> list[str]:
        if "e5" not in (self.model or "").lower() or not input_type:
            return texts

        prefix = "passage" if input_type in {"passage", "document"} else "query"
        prepared: list[str] = []
        for text in texts:
            stripped = str(text)
            if stripped.lower().startswith(("query:", "passage:")):
                prepared.append(stripped)
            else:
                prepared.append(f"{prefix}: {stripped}")
        return prepared

    def _is_local_base_url(self) -> bool:
        parsed = urlparse(self.base_url or "")
        return parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}
