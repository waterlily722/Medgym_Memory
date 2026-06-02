from __future__ import annotations

from typing import Any

from ..schemas import (
    ApplicabilityResult,
    MemoryGuidance,
    MemoryRetrievalResult,
    RetrievalHit,
)


HIT_GROUPS = (
    "positive_experience_hits",
    "negative_experience_hits",
    "skill_hits",
    "knowledge_hits",
)


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _memory_content_view(hit: RetrievalHit) -> dict[str, Any]:
    content = hit.content or {}
    memory_type = hit.memory_type
    if memory_type == "experience":
        return {
            "experience": content.get("text") or "",
            "outcome_type": content.get("outcome_type") or "",
            "tags": content.get("tags") or [],
        }
    if memory_type == "skill":
        return {
            "skill": content.get("skill_text") or "",
            "procedure": content.get("procedure") or "",
            "tags": content.get("tags") or [],
        }
    if memory_type == "knowledge":
        return {
            "content": content.get("content") or "",
            "source": content.get("source") or "",
            "confidence": content.get("confidence") or "",
            "tags": content.get("tags") or [],
        }
    return dict(content)


def _index_retrieval_hits(
    retrieval_result: MemoryRetrievalResult | None,
) -> dict[str, RetrievalHit]:
    if retrieval_result is None:
        return {}
    by_id: dict[str, RetrievalHit] = {}
    for group in HIT_GROUPS:
        for hit in getattr(retrieval_result, group, []) or []:
            if hit.memory_id and hit.memory_id not in by_id:
                by_id[hit.memory_id] = hit
    return by_id


def build_memory_guidance(
    applicability_result: ApplicabilityResult,
    retrieval_result: MemoryRetrievalResult | None = None,
) -> MemoryGuidance:
    """Convert applicability into doctor-facing memory references only.

    Applicability remains the internal selection step. The doctor agent receives
    selected memory content, not selection reasons or prescriptive action labels.
    """
    hit_index = _index_retrieval_hits(retrieval_result)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for assessment in applicability_result.memory_assessments:
        if assessment.decision != "apply" or not assessment.memory_id:
            continue
        if assessment.memory_id in seen:
            continue
        seen.add(assessment.memory_id)
        hit = hit_index.get(assessment.memory_id)
        if hit is None:
            selected.append(
                {
                    "memory_id": assessment.memory_id,
                    "memory_type": assessment.memory_type,
                    "content": {},
                }
            )
            continue
        selected.append(
            {
                "memory_id": hit.memory_id,
                "memory_type": hit.memory_type,
                "score": hit.score,
                "content": _memory_content_view(hit),
            }
        )

    return MemoryGuidance(selected_memories=selected)


def guidance_to_text(guidance: MemoryGuidance) -> str:
    from ..utils.config import GUIDANCE_CONFIG

    max_chars = GUIDANCE_CONFIG.get("max_guidance_chars", 800)
    max_memories = GUIDANCE_CONFIG.get("max_memories_in_guidance", 2)

    if not guidance.selected_memories:
        return ""

    lessons = []
    for memory in guidance.selected_memories[:max_memories]:
        mem_type = memory.get("memory_type", "")
        # Direct text: experience cards have "text", skill cards have "skill_text"
        lesson = (
            memory.get("text")
            or memory.get("skill_text")
            or ""
        ).strip()

        # Fallback: extract from nested content dict (legacy format)
        if not lesson:
            content = memory.get("content") or {}
            parts = []
            for key in ("experience", "situation", "action", "outcome", "boundary",
                        "skill", "procedure", "content"):
                values = _as_text_list(content.get(key))
                if values:
                    parts.append(" ".join(values))
            lesson = " ".join(parts).strip()

        if lesson:
            prefix = "Experience" if mem_type == "experience" else "Skill"
            lessons.append(f"[{prefix}] {lesson}")

    if not lessons:
        return ""

    result = "[Past diagnostic experience — non-ground-truth references]\n" + "\n".join(
        f"- {l}" for l in lessons
    )

    # Truncate to max_chars, keeping complete lines
    if len(result) > max_chars:
        lines = result.split("\n")
        kept = []
        total = 0
        for line in lines:
            if total + len(line) + 1 <= max_chars:
                kept.append(line)
                total += len(line) + 1
            else:
                remaining = max_chars - total - 3
                if remaining > 20:
                    kept.append(line[:remaining] + "...")
                break
        result = "\n".join(kept)

    return result
