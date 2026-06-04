from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from ..llm import EmbeddingClient
from ..memory_store import ExperienceMemoryStore, SkillMemoryStore

logger = logging.getLogger(__name__)
from ..schemas import DistilledEpisode, ExperienceCard
from ..schemas import OutcomeType
from ..utils.config import MEMORY_ROOT_DIRNAME, MERGE_CONFIG
from ..utils.scoring import bm25_scores, cosine_similarity as token_cosine
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
    embedding_client: EmbeddingClient | None = None,
) -> tuple[list[ExperienceCard], dict[str, float]]:
    if limit is None:
        limit = int(MERGE_CONFIG.get("candidate_top_k", 5) or 5)
    min_score = float(MERGE_CONFIG.get("candidate_min_score", 0.70) or 0.70)
    candidates = [
        item for item in existing
        if _same_outcome_direction(experience, item)
    ]
    if not candidates:
        return [], {}

    scoring_mode = _merge_scoring_mode()
    if scoring_mode == "embedding":
        if embedding_client is None or not embedding_client.available():
            raise RuntimeError(
                "Embedding merge requested but embedding client is unavailable"
            )
        query_vector = embedding_client.embed_query(_experience_similarity_text(experience))
        document_vectors = embedding_client.embed_documents(
            [_experience_similarity_text(item) for item in candidates]
        )
        if (
            not query_vector
            or document_vectors is None
            or len(document_vectors) != len(candidates)
            or any(not vector for vector in document_vectors)
        ):
            raise RuntimeError(
                "Embedding merge candidate encoding failed: "
                f"expected {len(candidates)} document vectors, "
                f"received {len(document_vectors) if document_vectors else 0}"
            )
        scored = [
            (_embedding_cosine(query_vector, vector), item)
            for vector, item in zip(document_vectors, candidates, strict=False)
        ]
    elif scoring_mode != "fielded_bm25":
        scored = [
            (token_cosine(_experience_similarity_text(experience), _experience_similarity_text(item)), item)
            for item in candidates
        ]
    else:
        text_scores = _normalize_scores(
            bm25_scores(
                _experience_similarity_text(experience),
                [_experience_similarity_text(item) for item in candidates],
            )
        )
        scored = [
            (text_scores[idx], item)
            for idx, item in enumerate(candidates)
        ]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [(score, item) for score, item in scored if score > min_score][:limit]
    return (
        [item for _, item in selected],
        {item.memory_id: score for score, item in selected},
    )


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
        ]
    )


def _embedding_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        raise RuntimeError(
            "Embedding merge cosine requires non-empty vectors with equal dimensions"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(value * value for value in vec_a))
    norm_b = math.sqrt(sum(value * value for value in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise RuntimeError("Embedding merge cosine received a zero-norm vector")
    return dot / (norm_a * norm_b)


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
    embedding_client: EmbeddingClient | None = None,
    strict: bool = True,
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
        strict=strict,
    )

    written_ids: list[str] = []
    merged_count = 0
    inserted_count = 0

    for experience in extracted:
        candidates, candidate_scores = _similar_existing(
            experience,
            existing,
            embedding_client=embedding_client,
        )

        if experience_merge_mode == "llm":
            decision = decide_merge_llm(
                experience,
                candidates,
                llm_client,
            )
        else:
            decision = decide_merge_rule(
                experience,
                candidates,
                candidate_scores=candidate_scores,
            )

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
        strict=strict,
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
