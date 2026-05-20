from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import SerializableMixin


@dataclass
class EpisodeFeedback(SerializableMixin):
    episode_id: str
    case_id: str = ""

    success: bool = False
    total_reward: float = 0.0

    final_diagnosis: str = ""
    gold_diagnosis: str = ""

    summary: str = ""


@dataclass
class DistilledEpisode(SerializableMixin):
    episode_id: str
    case_id: str = ""

    turn_records: list[dict[str, Any]] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)

    summary: str = ""