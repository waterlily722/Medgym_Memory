from __future__ import annotations

import logging
import os
from typing import Any

from ..llm import LLMClient, experience_merge_prompt, parse_validate_repair

logger = logging.getLogger(__name__)
from ..llm.schemas import EXPERIENCE_MERGE_SCHEMA
from ..schemas import ExperienceCard
from ..utils.config import MERGE_CONFIG
from ..utils.scoring import bm25_similarity, cosine_similarity as token_cosine


def _threshold(name: str, default: float) -> float:
    try:
        return float(MERGE_CONFIG.get(name, default))
    except Exception:
        return default


def _same_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    return left.outcome_type == right.outcome_type


def _same_trigger(left: ExperienceCard, right: ExperienceCard) -> bool:
    situation_threshold = _threshold("semantic_threshold", 0.80)
    if _merge_scoring_mode() != "fielded_bm25":
        similarity_score = token_cosine(left.text, right.text)
    else:
        similarity_score = bm25_similarity(left.text, right.text)

    return similarity_score >= situation_threshold


def _merge_scoring_mode() -> str:
    return str(
        os.environ.get("MEDGYM_MERGE_SCORING")
        or MERGE_CONFIG.get("candidate_scoring")
        or "cosine"
    ).strip().lower()


def _can_merge(left: ExperienceCard, right: ExperienceCard) -> bool:
    return (
        _same_trigger(left, right)
        and _same_direction(left, right)
    )


def _unique(values: list[Any]) -> list[Any]:
    seen = set()
    output = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _choose_longer(left: str, right: str) -> str:
    left = left or ""
    right = right or ""
    return left if len(left) >= len(right) else right


def merge_experience(base: ExperienceCard, incoming: ExperienceCard) -> ExperienceCard:
    """
    Merge two experiences with same trigger and same outcome direction.
    Keep base memory_id stable.
    """
    merged = ExperienceCard.from_dict(base.to_dict())

    merged.support_count = max(1, base.support_count) + max(1, incoming.support_count)

    merged.text = _choose_longer(base.text, incoming.text)

    merged.confidence = max(base.confidence, incoming.confidence)

    merged.source = {
        "episode_ids": _unique(
            list((base.source or {}).get("episode_ids") or [])
            + list((incoming.source or {}).get("episode_ids") or [])
        ),
        "case_ids": _unique(
            list((base.source or {}).get("case_ids") or [])
            + list((incoming.source or {}).get("case_ids") or [])
        ),
        "turn_ids": _unique(
            list((base.source or {}).get("turn_ids") or [])
            + list((incoming.source or {}).get("turn_ids") or [])
        ),
    }

    return merged


def decide_merge_rule(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
) -> dict[str, Any]:
    for existing in similar_existing:
        if _can_merge(existing, new_experience):
            merged = merge_experience(existing, new_experience)
            return {
                "merge_decision": "merge",
                "target_memory_ids": [existing.memory_id],
                "reason": "same situation/action/outcome direction",
                "merged_experience": merged.to_dict(),
            }

    return {
        "merge_decision": "insert_new",
        "target_memory_ids": [],
        "reason": "no compatible existing memory",
        "merged_experience": new_experience.to_dict(),
    }


def decide_merge_llm(
    new_experience: ExperienceCard,
    similar_existing: list[ExperienceCard],
    llm_client: LLMClient,
) -> dict[str, Any]:
    fallback = decide_merge_rule(new_experience, similar_existing)

    if not similar_existing:
        return fallback

    if not llm_client.available():
        return fallback

    payload = {
        "new_experience": new_experience.to_dict(),
        "similar_existing": [item.to_dict() for item in similar_existing],
        "rule_decision": fallback,
        "instruction": (
            "Decide whether to merge the new experience into one retrieved "
            "candidate or insert it as a separate new memory. "
            "Never merge opposite outcomes or incompatible boundaries; choose "
            "insert_new when uncertain."
        ),
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(experience_merge_prompt(payload), max_tokens=1200),
        EXPERIENCE_MERGE_SCHEMA,
        fallback,
    )

    decision = str(parsed.get("merge_decision") or "insert_new")
    if decision not in {"insert_new", "merge"}:
        logger.warning(
            "LLM merge returned invalid decision=%r; falling back to rule", decision
        )
        return fallback

    if decision == "merge" and not isinstance(parsed.get("merged_experience"), dict):
        logger.warning("LLM merge decided 'merge' but merged_experience is not dict; fallback")
        return fallback

    if decision == "merge":
        candidate_ids = {item.memory_id for item in similar_existing}
        target_ids = [str(item) for item in parsed.get("target_memory_ids") or []]
        target_id = next((mid for mid in target_ids if mid in candidate_ids), "")
        merged = parsed.get("merged_experience") or {}
        merged_id = str(merged.get("memory_id") or "")
        if not target_id or merged_id != target_id:
            logger.warning(
                "LLM merge did not preserve a retrieved candidate memory_id; fallback"
            )
            return fallback

    return parsed
