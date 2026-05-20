from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..memory_store import ExperienceMemoryStore, SkillMemoryStore

logger = logging.getLogger(__name__)
from ..schemas import DistilledEpisode, ExperienceCard
from ..schemas import OutcomeType
from ..utils.config import MEMORY_ROOT_DIRNAME, MERGE_CONFIG
from ..utils.scoring import bm25_scores, cosine_similarity as token_cosine, tag_overlap_score
from .experience_extractor import extract_experiences
from .experience_merger import decide_merge_llm, decide_merge_rule
from .skill_extractor import extract_skills_from_distilled_episode


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else Path(MEMORY_ROOT_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _similar_existing(
    experience: ExperienceCard,
    existing: list[ExperienceCard],
    limit: int | None = None,
) -> list[ExperienceCard]:
    if limit is None:
        limit = int(MERGE_CONFIG.get("candidate_top_k", 20) or 20)
    candidates = [
        item for item in existing
        if _same_outcome_direction(experience, item)
    ]
    if not candidates:
        return []

    scoring_mode = _merge_scoring_mode()
    if scoring_mode != "fielded_bm25":
        scored = [
            (token_cosine(_experience_similarity_text(experience), _experience_similarity_text(item)), item)
            for item in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[:limit] if score > 0.0]

    text_scores = _normalize_scores(
        bm25_scores(
            _experience_similarity_text(experience),
            [_experience_similarity_text(item) for item in candidates],
        )
    )

    scored: list[tuple[float, ExperienceCard]] = []
    for idx, item in enumerate(candidates):
        score = (
            0.90 * text_scores[idx]
            + 0.10 * tag_overlap_score(experience.tags, item.tags)
        )
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0.0]


def _merge_scoring_mode() -> str:
    return str(
        os.environ.get("MEDGYM_MERGE_SCORING")
        or MERGE_CONFIG.get("candidate_scoring")
        or "cosine"
    ).strip().lower()


def _experience_similarity_text(experience: ExperienceCard) -> str:
    return "\n".join(
        [
            experience.text or "",
            experience.outcome_type or "",
            " ".join(str(tag) for tag in (experience.tags or [])),
        ]
    )


def _positive_outcome(card: ExperienceCard) -> bool:
    return card.outcome_type == OutcomeType.POSITIVE.value


def _same_outcome_direction(left: ExperienceCard, right: ExperienceCard) -> bool:
    return _positive_outcome(left) == _positive_outcome(right)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0.0:
        return [0.0 for _ in scores]
    return [max(0.0, score / max_score) for score in scores]


def write_memory_from_distilled_episode(
    distilled_episode: DistilledEpisode | dict[str, Any],
    root_dir: str | None = None,
    experience_extraction_mode: str = "llm",
    experience_merge_mode: str = "rule",
    skill_extraction_mode: str = "llm",
    llm_client=None,
) -> dict[str, Any]:
    """
    Write ExperienceCards and episode-level SkillCards from a distilled episode.

    Important:
      - Experience extraction defaults to LLM.
      - If no LLM is available, extraction returns [] and nothing is written.
      - Skill extraction only runs for successful episodes.
    """
    distilled = (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )

    root = _root(root_dir)
    store = ExperienceMemoryStore(root)

    existing = store.list_all()
    extracted = extract_experiences(
        distilled,
        mode=experience_extraction_mode,
        llm_client=llm_client,
    )

    written_ids: list[str] = []
    merged_count = 0
    inserted_count = 0

    for experience in extracted:
        candidates = _similar_existing(experience, existing)

        if experience_merge_mode == "llm" and llm_client is not None:
            decision = decide_merge_llm(experience, candidates, llm_client)
        else:
            decision = decide_merge_rule(experience, candidates)

        merge_decision = str(decision.get("merge_decision") or "insert_new")

        if merge_decision == "merge":
            merged_payload = decision.get("merged_experience") or experience.to_dict()
            merged = ExperienceCard.from_dict(merged_payload)
            store.upsert(merged)
            written_ids.append(merged.memory_id)
            merged_count += 1
            existing = store.list_all()
            continue

        store.upsert(experience)
        written_ids.append(experience.memory_id)
        inserted_count += 1
        existing.append(experience)

    result = {
        "episode_id": distilled.episode_id,
        "merge_scoring_mode": _merge_scoring_mode(),
        "written_experience_ids": written_ids,
        "extracted_count": len(extracted),
        "merged_count": merged_count,
        "inserted_count": inserted_count,
        "experience_store_count": len(store.list_all()),
    }

    skill_store = SkillMemoryStore(root)
    skills = extract_skills_from_distilled_episode(
        distilled,
        mode=skill_extraction_mode,
        llm_client=llm_client,
    )
    written_skill_ids: list[str] = []
    for skill in skills:
        skill_store.upsert(skill)
        written_skill_ids.append(skill.memory_id)

    result.update(
        {
            "written_skill_ids": written_skill_ids,
            "skill_extracted_count": len(skills),
            "skill_store_count": len(skill_store.list_all()),
        }
    )

    logger.info(
        "Memory write done — episode=%s exp_extracted=%d merged=%d inserted=%d "
        "exp_total=%d skill_extracted=%d skill_total=%d",
        distilled.episode_id, len(extracted), merged_count,
        inserted_count, result["experience_store_count"],
        len(skills), result["skill_store_count"],
    )
    return result
