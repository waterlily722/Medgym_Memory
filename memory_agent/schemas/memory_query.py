from __future__ import annotations

from dataclasses import dataclass, field

from .common import SerializableMixin


@dataclass
class MemoryQuery(SerializableMixin):
    """Retrieval query for memory search.

    ``query_text`` is a flat text string built from CaseMemory for embedding /
    cosine-similarity matching.

    - Chief complaint
    - Current-turn information
    - LLM summary of prior exposed information
    - Candidate next actions

    Construction:
    - Rule mode: concatenates :class:`CaseMemory` fields into one text.
    - LLM mode: first extracts CaseMemory from CaseState, then asks the LLM to
      produce a concise retrieval query from CaseMemory.

    See :func:`online.query_builder.build_memory_query` for details.
    """
    case_id: str
    turn_id: int = 0
    query_text: str = ""
