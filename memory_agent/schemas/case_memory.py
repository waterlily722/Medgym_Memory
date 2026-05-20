from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class CaseMemory(SerializableMixin):
    case_id: str
    turn_id: int = 0
    chief_complaint: str = ""
    diagnosis_goal: str = ""
    efficient_turn_information: list[str] = field(default_factory=list)
    prior_information_summary: str = ""
