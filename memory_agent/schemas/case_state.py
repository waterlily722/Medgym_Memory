from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class CaseState(SerializableMixin):
    case_id: str
    turn_id: int = 0
    chief_complaint: str = ""
    current_turn: list[dict] = field(default_factory=list)
    acquired_information: list[dict] = field(default_factory=list)
