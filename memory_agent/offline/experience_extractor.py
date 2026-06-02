from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from ..llm import LLMClient, experience_extraction_prompt, parse_validate_repair
from ..llm.schemas import EXPERIENCE_EXTRACTION_SCHEMA
from ..schemas import DistilledEpisode, ExperienceCard, OutcomeType
from ..utils.config import LLM_CONFIG

logger = logging.getLogger(__name__)

MAX_EPISODE_TURNS_FOR_EXTRACTION = int(
    LLM_CONFIG.get("experience_extraction_max_turns", 15)
)
MAX_EXPERIENCES_PER_EPISODE = 3
MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS = int(
    LLM_CONFIG.get("experience_extraction_max_output_tokens", 1200)
)
MAX_PROMPT_TEXT_CHARS = int(
    LLM_CONFIG.get("experience_extraction_max_text_chars", 1800)
)


def _as_distilled(distilled_episode: DistilledEpisode | dict[str, Any]) -> DistilledEpisode:
    return (
        distilled_episode
        if isinstance(distilled_episode, DistilledEpisode)
        else DistilledEpisode.from_dict(distilled_episode)
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _selected_action(turn_record: dict[str, Any]) -> dict[str, Any]:
    action = turn_record.get("selected_action") or {}
    return action if isinstance(action, dict) else {}


def select_episode_turns(
    turn_records: list[dict[str, Any]],
    limit: int = MAX_EPISODE_TURNS_FOR_EXTRACTION,
) -> list[dict[str, Any]]:
    """Use the complete episode context, capped only to protect prompt length."""
    return list(turn_records or [])[-limit:]


def _clip_text(value: Any, max_chars: int = MAX_PROMPT_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _compact_case_state(case_state: Any) -> dict[str, Any]:
    state = case_state if isinstance(case_state, dict) else {}
    acquired = state.get("acquired_information") or []
    compact_acquired: list[dict[str, Any]] = []
    for item in acquired[-3:]:
        if not isinstance(item, dict):
            continue
        compact_acquired.append(
            {
                "turn_id": item.get("turn_id"),
                "source_path": item.get("source_path"),
                "content": _clip_text(item.get("content"), 300),
            }
        )
    return {
        "chief_complaint": _clip_text(state.get("chief_complaint"), 300),
        "acquired_information_recent": compact_acquired,
    }


def _latest_case_memory(turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the final CaseMemory snapshot produced during the episode."""
    for turn in reversed(turn_records or []):
        if not isinstance(turn, dict):
            continue
        debug = turn.get("memory_debug") if isinstance(turn.get("memory_debug"), dict) else {}
        query_debug = debug.get("query_builder") if isinstance(debug.get("query_builder"), dict) else {}
        case_memory = query_debug.get("final_case_memory")
        if isinstance(case_memory, dict) and case_memory:
            return {
                "case_id": case_memory.get("case_id"),
                "turn_id": case_memory.get("turn_id"),
                "chief_complaint": _clip_text(case_memory.get("chief_complaint"), 400),
                "diagnostic_strategy": _clip_text(case_memory.get("diagnostic_strategy"), 700),
                "efficient_turn_information": [
                    _clip_text(item, 500)
                    for item in (case_memory.get("efficient_turn_information") or [])[:12]
                ],
                "ineffective_turn_information": [
                    _clip_text(item, 500)
                    for item in (case_memory.get("ineffective_turn_information") or [])[:12]
                ],
                "prior_information_summary": _clip_text(
                    case_memory.get("prior_information_summary"), 1200
                ),
            }
    return {}


def _compact_memory_query(query: Any) -> dict[str, Any]:
    q = query if isinstance(query, dict) else {}
    return {
        "query_text": _clip_text(q.get("query_text"), 400),
        "query_facets": q.get("query_facets") or {},
    }


def _compact_memory_guidance(guidance: Any) -> dict[str, Any]:
    g = guidance if isinstance(guidance, dict) else {}
    selected = []
    for item in (g.get("selected_memories") or [])[:2]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        selected.append(
            {
                "memory_id": item.get("memory_id"),
                "memory_type": item.get("memory_type"),
                "score": item.get("score"),
                "content": {
                    key: _clip_text(content.get(key), 300)
                    for key in (
                        "situation",
                        "action",
                        "outcome",
                        "boundary",
                        "skill",
                        "procedure",
                        "content",
                    )
                    if content.get(key)
                },
            }
        )
    return {"selected_memories": selected}


def _compact_observation(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"memory_guidance", "memory_guidance_structured"}:
                continue
            if isinstance(item, (dict, list)):
                compact[key] = _clip_text(item, 500)
            else:
                compact[key] = _clip_text(item, 500)
        return compact
    if isinstance(value, list):
        return [_clip_text(item, 400) for item in value[-3:]]
    return _clip_text(value, 600)


def _compact_env_info(value: Any) -> dict[str, Any]:
    info = value if isinstance(value, dict) else {}
    keep_keys = (
        "response",
        "metadata",
        "turn_type",
        "tool_name",
        "diagnosis",
        "final_response",
        "case_id",
    )
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in info:
            compact[key] = _clip_text(info.get(key), 500)
    return compact


def _compact_selected_action(action: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "action_type",
        "tool_name",
        "name",
        "arguments",
        "final_response",
    ):
        if key not in action:
            continue
        value = action.get(key)
        if isinstance(value, (dict, list)):
            compact[key] = _clip_text(value, 400)
        else:
            compact[key] = value if isinstance(value, bool) else _clip_text(value, 400)
    if not compact:
        compact["raw"] = _clip_text(action, 500)
    return compact


def _compact_clinical_turn(turn: dict[str, Any]) -> dict[str, Any]:
    clinical = turn.get("clinical_turn") if isinstance(turn.get("clinical_turn"), dict) else {}
    if clinical:
        turn_importance = clinical.get("turn_importance") or {}
        return {
            "turn_id": clinical.get("turn_id") or turn.get("turn_id", 0),
            "doctor_action_type": clinical.get("doctor_action_type") or "",
            "tool_name": clinical.get("tool_name") or "",
            "arguments": clinical.get("arguments") or {},
            "patient_or_tool_response": _compact_observation(
                clinical.get("patient_or_tool_response")
            ),
            "env_info": _compact_env_info(clinical.get("env_info")),
            "reward": _safe_float(clinical.get("reward"), _safe_float(turn.get("reward"), 0.0)),
            "conf_before": _safe_float(clinical.get("conf_before"), 0.0),
            "conf_after": _safe_float(clinical.get("conf_after"), 0.0),
            "delta": _safe_float(clinical.get("delta"), 0.0),
            "turn_reward": _safe_float(clinical.get("turn_reward"), 0.0),
            "importance": _safe_float(clinical.get("importance"), 0.0),
            "turn_importance": {
                "method": turn_importance.get("method"),
                "tool_name": turn_importance.get("tool_name"),
                "conf_before": _safe_float(turn_importance.get("conf_before"), 0.0),
                "conf_after": _safe_float(turn_importance.get("conf_after"), 0.0),
                "delta": _safe_float(turn_importance.get("delta"), 0.0),
                "turn_reward": _safe_float(turn_importance.get("turn_reward"), 0.0),
                "importance": _safe_float(turn_importance.get("importance"), 0.0),
                "answer_excerpt": _clip_text(turn_importance.get("answer_excerpt"), 300),
            },
            "done": bool(clinical.get("done", turn.get("done", False))),
        }

    action = _selected_action(turn)
    raw = action.get("raw") or action.get("action_label") or action
    return {
        "turn_id": turn.get("turn_id", 0),
        "doctor_action_type": action.get("action_type") or "",
        "tool_name": action.get("tool_name") or action.get("name") or "",
        "arguments": action.get("arguments") or {},
        "doctor_action": _clip_text(raw, 500),
        "patient_or_tool_response": _compact_observation(turn.get("env_observation")),
        "env_info": _compact_env_info(turn.get("env_info")),
        "reward": _safe_float(turn.get("reward"), 0.0),
        "done": bool(turn.get("done", False)),
    }


def build_clinical_episode_trace(
    turn_records: list[dict[str, Any]],
    limit: int = MAX_EPISODE_TURNS_FOR_EXTRACTION,
) -> list[dict[str, Any]]:
    return [_compact_clinical_turn(turn) for turn in select_episode_turns(turn_records, limit)]


TURN_ID_RE = re.compile(r"\bturn_(\d+)\b")


def _turn_records_by_id(
    turn_records: list[dict[str, Any]],
    limit: int = MAX_EPISODE_TURNS_FOR_EXTRACTION,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for turn in select_episode_turns(turn_records, limit):
        compact = _compact_clinical_turn(turn)
        try:
            turn_id = int(compact.get("turn_id") or 0)
        except Exception:
            continue
        if turn_id:
            indexed[turn_id] = compact
    return indexed


def _records_by_turn_id(records: list[Any]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for record in records or []:
        text = str(record or "").strip()
        if not text:
            continue
        for match in TURN_ID_RE.finditer(text):
            turn_id = int(match.group(1))
            grouped.setdefault(turn_id, []).append(_clip_text(text, 500))
    return grouped


def _clean_diagnostic_trajectory(
    turn_records: list[dict[str, Any]],
    final_case_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only CaseMemory-effective turns with reward and environment context."""
    effective_by_turn = _records_by_turn_id(
        final_case_memory.get("efficient_turn_information") or []
    )
    turn_by_id = _turn_records_by_id(turn_records)
    trajectory: list[dict[str, Any]] = []
    for turn_id in sorted(effective_by_turn):
        turn = turn_by_id.get(turn_id)
        if not turn:
            continue
        trajectory.append(
            {
                "turn_id": turn_id,
                "doctor_action_type": turn.get("doctor_action_type") or "",
                "tool_name": turn.get("tool_name") or "",
                "arguments": turn.get("arguments") or {},
                "patient_or_tool_response": turn.get("patient_or_tool_response"),
                "env_info": turn.get("env_info") or {},
                "reward": _safe_float(turn.get("reward"), 0.0),
                "turn_reward": _safe_float(turn.get("turn_reward"), 0.0),
                "importance": _safe_float(turn.get("importance"), 0.0),
                "evidence_records": effective_by_turn.get(turn_id) or [],
            }
        )
    return trajectory


def _ineffective_interactions(final_case_memory: dict[str, Any]) -> list[str]:
    return [
        _clip_text(item, 500)
        for item in (final_case_memory.get("ineffective_turn_information") or [])[:12]
        if str(item or "").strip()
    ]


def _fallback_memory_id() -> str:
    return f"exp_{uuid.uuid4().hex[:12]}"


def _card_from_raw(
    raw: dict[str, Any],
    distilled: DistilledEpisode,
) -> ExperienceCard | None:
    if not isinstance(raw, dict):
        return None

    text = str(raw.get("text") or "").strip()
    if not text:
        return None

    feedback = distilled.feedback if isinstance(distilled.feedback, dict) else {}
    episode_success = bool(feedback.get("success", False))
    outcome_type = (
        OutcomeType.POSITIVE.value
        if episode_success
        else OutcomeType.NEGATIVE.value
    )

    raw_source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    source_case_ids = [str(item) for item in raw_source.get("case_ids") or [] if str(item)]
    source_episode_ids = [str(item) for item in raw_source.get("episode_ids") or [] if str(item)]
    source_turn_ids = []
    for item in raw_source.get("turn_ids") or []:
        try:
            source_turn_ids.append(str(int(item)))
        except Exception:
            pass

    card = ExperienceCard(
        memory_id=str(raw.get("memory_id") or _fallback_memory_id()),
        memory_type="experience",
        text=text,
        outcome_type=outcome_type,
        confidence=max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        support_count=max(1, int(_safe_float(raw.get("support_count"), 1))),
        source={
            "episode_ids": source_episode_ids,
            "case_ids": source_case_ids,
            "turn_ids": source_turn_ids,
        },
    )

    card.source.setdefault("episode_ids", [])
    card.source.setdefault("case_ids", [])
    card.source.setdefault("turn_ids", [])
    if distilled.episode_id and distilled.episode_id not in card.source["episode_ids"]:
        card.source["episode_ids"].append(distilled.episode_id)
    if distilled.case_id and distilled.case_id not in card.source["case_ids"]:
        card.source["case_ids"].append(distilled.case_id)

    if not card.text:
        return None
    return card


def extract_experiences(
    distilled_episode: DistilledEpisode | dict[str, Any],
    mode: str = "llm",
    llm_client: LLMClient | None = None,
) -> list[ExperienceCard]:
    """
    Extract ExperienceCards via LLM from the full episode context.

    Rule extraction is intentionally disabled because experience text should be
    generated by clinical reasoning rather than string templates.
    """
    distilled = _as_distilled(distilled_episode)
    if mode != "llm" or llm_client is None or not llm_client.available():
        logger.warning(
            "Experience extraction skipped — llm_client available=%s, mode=%s",
            llm_client.available() if llm_client else False,
            mode,
        )
        return []

    final_case_memory = _latest_case_memory(distilled.turn_records)
    diagnostic_trajectory = _clean_diagnostic_trajectory(
        distilled.turn_records,
        final_case_memory,
    )
    if not diagnostic_trajectory:
        logger.info(
            "No effective diagnostic trajectory in episode %s — skipping extraction",
            distilled.episode_id,
        )
        return []

    feedback = distilled.feedback if isinstance(distilled.feedback, dict) else {}
    episode_success = bool(feedback.get("success", False))
    payload = {
        "episode_id": distilled.episode_id,
        "case_id": distilled.case_id,
        "episode_outcome": {
            "success": episode_success,
            "final_diagnosis": feedback.get("final_diagnosis") or "",
            "gold_diagnosis": feedback.get("gold_diagnosis") or "",
            "total_reward": feedback.get("total_reward") or 0.0,
            "summary": feedback.get("summary") or "",
        },
        "case_context": {
            "chief_complaint": final_case_memory.get("chief_complaint") or "",
            "diagnostic_strategy": final_case_memory.get("diagnostic_strategy") or "",
            "prior_information_summary": final_case_memory.get("prior_information_summary") or "",
        },
        "diagnostic_trajectory": diagnostic_trajectory,
        "ineffective_interactions": _ineffective_interactions(final_case_memory),
        "max_experiences": MAX_EXPERIENCES_PER_EPISODE,
    }
    prompt = experience_extraction_prompt(payload)
    logger.info(
        "Experience extraction prompt size for episode %s: %d chars, %d turns, max_output_tokens=%d",
        distilled.episode_id,
        len(prompt),
        len(diagnostic_trajectory),
        MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS,
    )
    parsed, _, _ = parse_validate_repair(
        llm_client.generate_json(
            prompt,
            max_tokens=MAX_EXPERIENCE_EXTRACTION_OUTPUT_TOKENS,
        ),
        EXPERIENCE_EXTRACTION_SCHEMA,
        {"experiences": []},
    )

    cards: list[ExperienceCard] = []
    for raw in (parsed.get("experiences") or [])[:MAX_EXPERIENCES_PER_EPISODE]:
        card = _card_from_raw(raw, distilled)
        if card is not None:
            cards.append(card)

    logger.info(
        "Extracted %d experience cards from episode %s (episode turns used: %d)",
        len(cards), distilled.episode_id, len(diagnostic_trajectory),
    )
    return cards
