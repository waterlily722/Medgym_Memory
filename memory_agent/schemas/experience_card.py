from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from .common import OutcomeType, SerializableMixin


@dataclass
class ExperienceCard(SerializableMixin):
    memory_id: str
    memory_type: str = "experience"

    # Main semi-structured medical diagnostic insight.
    # Suggested content: clinical pattern, key positive/negative evidence,
    # discriminative clue, lab/imaging/exam finding, missed or misleading evidence,
    # and diagnostic implication.
    text: str = ""

    # safety / retrieval
    outcome_type: str = OutcomeType.POSITIVE.value

    # lightweight confidence / provenance
    confidence: float = 0.5
    support_count: int = 1
    source: dict[str, list[str]] = field(default_factory=dict)

    source_episode_ids: InitVar[list[str] | None] = None
    source_case_ids: InitVar[list[str] | None] = None
    source_turn_ids: InitVar[list[str] | None] = None

    def __post_init__(
        self,
        source_episode_ids: list[str] | None,
        source_case_ids: list[str] | None,
        source_turn_ids: list[str] | None,
    ) -> None:
        legacy_sources = {
            "episode_ids": source_episode_ids or [],
            "case_ids": source_case_ids or [],
            "turn_ids": source_turn_ids or [],
        }
        if not isinstance(self.source, dict):
            self.source = {}
        for key, values in legacy_sources.items():
            if self.source.get(key):
                continue
            normalized = [str(item) for item in values if str(item)]
            if normalized:
                self.source[key] = normalized

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            raise ValueError(f"{cls.__name__}.from_dict received None")
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expected dict, got {type(data)}")

        values = dict(data)
        raw_source = values.get("source")
        source = dict(raw_source) if isinstance(raw_source, dict) else {}
        legacy_sources = {
            "episode_ids": values.get("source_episode_ids") or [],
            "case_ids": values.get("source_case_ids") or [],
            "turn_ids": values.get("source_turn_ids") or [],
        }
        for key, legacy_values in legacy_sources.items():
            if source.get(key):
                continue
            source[key] = [str(item) for item in legacy_values if str(item)]
        values["source"] = source

        return super().from_dict(values)
