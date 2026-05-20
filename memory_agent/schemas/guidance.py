from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryGuidance(SerializableMixin):
    selected_memories: list[dict] = field(default_factory=list)
