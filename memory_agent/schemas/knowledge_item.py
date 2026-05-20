from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class KnowledgeItem(SerializableMixin):
    memory_id: str
    memory_type: str = "knowledge"
    content: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.5
