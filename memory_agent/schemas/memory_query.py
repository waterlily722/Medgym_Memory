from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    """Retrieval query for memory search.

    ``query_text`` is used for experience memory retrieval — describes known
    clinical facts for matching similar past diagnostic experiences.

    ``skill_query_text`` is used for skill memory retrieval — describes the
    current decision point and action context for matching workflow guidance.

    Construction:
    - Rule mode: concatenates :class:`CaseMemory` fields into one text.
    - LLM mode: first extracts CaseMemory from CaseState, then asks the LLM to
      produce concise retrieval queries from CaseMemory.

    See :func:`online.query_builder.build_memory_query` for details.
    """
    case_id: str
    turn_id: int = 0
    query_text: str = ""
    skill_query_text: str = ""
