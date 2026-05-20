from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .applicability import ApplicabilityResult
from .case_state import CaseState
from .common import SerializableMixin
from .guidance import MemoryGuidance
from .memory_query import MemoryQuery
from .retrieval import MemoryRetrievalResult


@dataclass
class TurnRecord(SerializableMixin):
    """Snapshot of one agent turn with full memory pipeline state.

    All memory-pipeline fields use their typed schema objects.
    ``to_dict()`` / ``from_dict()`` (inherited from SerializableMixin) handle
    recursive serialization automatically.
    """
    episode_id: str = ""
    case_id: str = ""
    turn_id: int = 0

    case_state: Optional[CaseState] = None
    memory_query: Optional[MemoryQuery] = None
    retrieval_result: Optional[MemoryRetrievalResult] = None
    applicability_result: Optional[ApplicabilityResult] = None
    memory_guidance: Optional[MemoryGuidance] = None

    # Clean doctor/environment interaction used for offline experience mining.
    # This intentionally excludes retrieval/applicability/debug payloads.
    clinical_turn: dict[str, Any] = field(default_factory=dict)

    selected_action: dict[str, Any] = field(default_factory=dict)
    memory_debug: dict[str, Any] = field(default_factory=dict)

    env_observation: dict[str, Any] | str = field(default_factory=dict)
    env_info: dict[str, Any] = field(default_factory=dict)

    reward: float = 0.0
    done: bool = False
