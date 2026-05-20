from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class RetrievalHit(SerializableMixin):
    memory_id: str
    memory_type: str
    content: dict[str, Any]
    score: float = 0.0


@dataclass
class MemoryRetrievalResult(SerializableMixin):
    positive_experience_hits: list[RetrievalHit] = field(default_factory=list)
    negative_experience_hits: list[RetrievalHit] = field(default_factory=list)
    skill_hits: list[RetrievalHit] = field(default_factory=list)
    knowledge_hits: list[RetrievalHit] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            return cls()

        def hits(items):
            return [RetrievalHit.from_dict(item) for item in items or []]

        return cls(
            positive_experience_hits=hits(data.get("positive_experience_hits")),
            negative_experience_hits=hits(data.get("negative_experience_hits")),
            skill_hits=hits(data.get("skill_hits")),
            knowledge_hits=hits(data.get("knowledge_hits")),
        )