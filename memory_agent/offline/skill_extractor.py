from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from ..llm import LLMClient, parse_validate_repair, skill_extraction_prompt
from ..llm.schemas import SKILL_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, SkillCard
from ..utils.config import LLM_CONFIG
from .experience_extractor import _latest_case_memory, build_clinical_episode_trace

logger = logging.getLogger(__name__)

MAX_SKILLS_PER_EPISODE = 2
MAX_SKILL_EXTRACTION_OUTPUT_TOKENS = int(
    LLM_CONFIG.get("skill_extraction_max_output_tokens", 1600)
)

EXACT_AGE_RE = re.compile(
    r"\b(?:aged\s*\d{1,3}|\d{1,3}\s*[- ]?(?:year[- ]old|years? old|yo|y/o))\b",
    re.IGNORECASE,
)
BARE_AGE_RE = re.compile(r"\bage\s*\d{1,3}\b", re.IGNORECASE)
INCIDENTAL_DEMOGRAPHIC_RE = re.compile(
    r"\b(?:male|female|man|woman|boy|girl)\s+(?:patient|adult)\b",
    re.IGNORECASE,
)


def _as_distilled(distilled_episode: DistilledEpisode | dict[str, Any]) -> DistilledEpisode:
    return (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fallback_skill_id() -> str:
    return f"skill_{uuid.uuid4().hex[:12]}"


def _normalize_procedure(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    procedure: list[dict[str, str]] = []
    for item in value[:5]:
        if isinstance(item, dict):
            action_type = str(item.get("action_type") or "").upper()
            action_label = str(item.get("action_label") or item.get("label") or "")
        else:
            action_type = ""
            action_label = str(item)
        if action_type or action_label:
            procedure.append({"action_type": action_type, "action_label": action_label})
    return procedure


def _sanitize_skill_text(text: str) -> str:
    text = EXACT_AGE_RE.sub("screening-age adult", text)
    text = BARE_AGE_RE.sub("screening-age adult", text)
    text = INCIDENTAL_DEMOGRAPHIC_RE.sub("patient", text)
    text = re.sub(r"\b(\d{1,3})\s*[- ]?year[- ]old\b", "screening-age adult", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpatients?\s+screening-age adult\b", "screening-age adults", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s+screening-age adult patient\b", "a screening-age adult", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def _sanitize_procedure(procedure: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for step in procedure:
        sanitized.append(
            {
                "action_type": str(step.get("action_type") or "").upper(),
                "action_label": _sanitize_skill_text(str(step.get("action_label") or "")),
            }
        )
    return sanitized


def _sanitize_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    for tag in tags:
        normalized = str(tag).strip()
        if not normalized:
            continue
        if re.search(r"\d", normalized):
            continue
        if normalized.lower() in {"male", "female", "man", "woman"}:
            continue
        cleaned.append(normalized)
    return cleaned


def _turn_value_signals(clinical_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for turn in clinical_trace:
        if not isinstance(turn, dict):
            continue
        signals.append(
            {
                "turn_id": turn.get("turn_id"),
                "tool_name": turn.get("tool_name") or "",
                "doctor_action_type": turn.get("doctor_action_type") or "",
                "reward": _safe_float(turn.get("reward"), 0.0),
                "conf_before": _safe_float(turn.get("conf_before"), 0.0),
                "conf_after": _safe_float(turn.get("conf_after"), 0.0),
                "delta": _safe_float(turn.get("delta"), 0.0),
                "turn_reward": _safe_float(turn.get("turn_reward"), 0.0),
                "importance": _safe_float(turn.get("importance"), 0.0),
            }
        )
    return signals


def _skill_from_raw(raw: dict[str, Any], distilled: DistilledEpisode) -> SkillCard | None:
    if not isinstance(raw, dict):
        return None

    skill_text = _sanitize_skill_text(str(raw.get("skill_text") or "").strip())
    if not skill_text:
        return None

    procedure = _normalize_procedure(raw.get("procedure"))
    if not procedure:
        return None

    tags = _sanitize_tags([str(item) for item in raw.get("tags") or [] if str(item).strip()])
    raw_source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    source = {
        "episode_ids": [str(item) for item in raw_source.get("episode_ids") or [] if str(item)],
        "case_ids": [str(item) for item in raw_source.get("case_ids") or [] if str(item)],
        "turn_ids": [str(item) for item in raw_source.get("turn_ids") or [] if str(item)],
    }
    if distilled.episode_id and distilled.episode_id not in source["episode_ids"]:
        source["episode_ids"].append(distilled.episode_id)
    if distilled.case_id and distilled.case_id not in source["case_ids"]:
        source["case_ids"].append(distilled.case_id)

    skill = SkillCard(
        memory_id=str(raw.get("memory_id") or _fallback_skill_id()),
        memory_type="skill",
        skill_text=skill_text,
        procedure=_sanitize_procedure(procedure),
        tags=tags,
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        support_count=max(1, int(_safe_float(raw.get("support_count"), 1))),
        source=source,
    )

    if not skill.skill_text or not skill.procedure:
        return None
    return skill


def extract_skills_from_distilled_episode(
    distilled_episode: DistilledEpisode | dict[str, Any],
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> list[SkillCard]:
    distilled = _as_distilled(distilled_episode)
    feedback = distilled.feedback if isinstance(distilled.feedback, dict) else {}
    if not bool(feedback.get("success", False)):
        return []

    if mode != "llm" or llm_client is None or not llm_client.available():
        logger.warning(
            "Skill extraction skipped — llm_client available=%s, mode=%s",
            llm_client.available() if llm_client else False,
            mode,
        )
        return []

    clinical_trace = build_clinical_episode_trace(distilled.turn_records)
    if not clinical_trace:
        return []

    payload = {
        "episode_id": distilled.episode_id,
        "case_id": distilled.case_id,
        "episode_outcome": {
            "success": True,
            "final_diagnosis": feedback.get("final_diagnosis") or "",
            "gold_diagnosis": feedback.get("gold_diagnosis") or "",
            "total_reward": feedback.get("total_reward") or 0.0,
            "summary": feedback.get("summary") or "",
        },
        "final_case_memory": _latest_case_memory(distilled.turn_records),
        "turn_value_signals": _turn_value_signals(clinical_trace),
        "clinical_episode_trace": clinical_trace,
        "max_skills": MAX_SKILLS_PER_EPISODE,
    }

    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(
            skill_extraction_prompt(payload),
            max_tokens=MAX_SKILL_EXTRACTION_OUTPUT_TOKENS,
        ),
        SKILL_EXTRACTION_SCHEMA,
        {"skills": []},
    )

    skills: list[SkillCard] = []
    for raw in (parsed.get("skills") or [])[:MAX_SKILLS_PER_EPISODE]:
        skill = _skill_from_raw(raw, distilled)
        if skill is not None:
            skills.append(skill)

    logger.info(
        "Extracted %d skill cards from successful episode %s",
        len(skills),
        distilled.episode_id,
    )
    return skills
