from __future__ import annotations

from pathlib import Path

from ..schemas import SkillCard
from .base_store import JsonMemoryStore


class SkillMemoryStore(JsonMemoryStore):
    def __init__(self, root_dir: str | Path, strict_json: bool = True) -> None:
        super().__init__(
            root_dir=root_dir,
            filename="skill_memory.jsonl",
            item_factory=SkillCard.from_dict,
            strict_json=strict_json,
        )