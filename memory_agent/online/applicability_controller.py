from __future__ import annotations

import logging

from ..llm import LLMClient, applicability_prompt, parse_validate_repair

logger = logging.getLogger(__name__)
from ..llm.schemas import APPLICABILITY_SCHEMA
from ..schemas import (
    ActionAssessment,
    ApplicabilityResult,
    CaseState,
    MemoryApplicabilityAssessment,
    MemoryQuery,
    MemoryRetrievalResult,
    OutcomeType,
    RetrievalHit,
)
from ..utils.config import APPLICABILITY_CONFIG, MEMORY_ACTION_CONFIG

DEFAULT_ACTIONS = list(MEMORY_ACTION_CONFIG["default_actions"])
FINALIZE_ACTION = str(MEMORY_ACTION_CONFIG["finalize_action"])


def _all_hits(retrieval_result: MemoryRetrievalResult) -> list[RetrievalHit]:
    return (
        retrieval_result.positive_experience_hits
        + retrieval_result.negative_experience_hits
        + retrieval_result.skill_hits
        + retrieval_result.knowledge_hits
    )


def _outcome_type(hit: RetrievalHit) -> str:
    return str(hit.content.get("outcome_type") or "").lower()


_NEGATIVE_OUTCOMES = {OutcomeType.NEGATIVE.value}
_POSITIVE_OUTCOMES = {OutcomeType.POSITIVE.value}


def _is_negative_experience(hit: RetrievalHit) -> bool:
    tags = {str(tag).lower() for tag in (hit.content.get("tags") or [])}
    return hit.memory_type == "experience" and (
        "negative" in tags or _outcome_type(hit) in _NEGATIVE_OUTCOMES
    )


def _infer_action_from_steps(steps: object) -> str:
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, dict):
            return str(first.get("action_type") or "").upper()
    return ""


def _infer_action_from_text(text: str) -> str:
    upper = text.upper()
    for action in DEFAULT_ACTIONS:
        if action in upper:
            return action
    return ""


def _infer_action_from_memory(hit: RetrievalHit) -> str:
    if hit.memory_type == "experience":
        return _infer_action_from_text(str(hit.content.get("text") or ""))
    if hit.memory_type == "skill":
        action = _infer_action_from_steps(hit.content.get("procedure") or [])
        return action or _infer_action_from_text(str(hit.content.get("skill_text") or ""))
    return ""


def _skill_apply_threshold() -> float:
    return float(APPLICABILITY_CONFIG.get("skill_apply_score", 0.40))


def _reusable_memory_threshold() -> float:
    return float(APPLICABILITY_CONFIG.get("reusable_memory_score", 0.30))


def _rule_memory_assessment(hit: RetrievalHit) -> MemoryApplicabilityAssessment:
    action = _infer_action_from_memory(hit)
    reusable = hit.score >= _reusable_memory_threshold()

    if not reusable:
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="ignore",
            reason="Retrieved memory score is below the reusable-memory threshold.",
            action_bias={},
            blocked_actions=[],
        )

    if _is_negative_experience(hit):
        bias = {action: -0.35} if action else {}
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="apply",
            reason="Retrieved negative experience is reusable as a caution for the same local decision point.",
            action_bias=bias,
            blocked_actions=[],
        )

    if hit.memory_type == "experience":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="apply",
            reason="Retrieved positive experience is reusable for the current local decision point.",
            action_bias={action: 0.30} if action else {},
            blocked_actions=[],
        )

    if hit.memory_type == "skill":
        return MemoryApplicabilityAssessment(
            memory_id=hit.memory_id,
            memory_type=hit.memory_type,
            decision="apply",
            reason="Retrieved skill is reusable because its clinical situation and action timing match the current step.",
            action_bias={action: 0.25} if action else {},
            blocked_actions=[],
        )

    return MemoryApplicabilityAssessment(
        memory_id=hit.memory_id,
        memory_type=hit.memory_type,
        decision="apply",
        reason="Retrieved knowledge is reusable background for the current decision point.",
        action_bias={},
        blocked_actions=[],
    )


def _postprocess_llm_assessment(
    hit: RetrievalHit,
    assessment: MemoryApplicabilityAssessment,
) -> MemoryApplicabilityAssessment:
    if assessment.decision not in {"apply", "ignore"}:
        return _rule_memory_assessment(hit)

    assessment.memory_id = hit.memory_id
    assessment.memory_type = hit.memory_type

    if assessment.decision == "apply" and hit.score < _reusable_memory_threshold():
        assessment.decision = "ignore"
        assessment.action_bias = {}
        assessment.blocked_actions = []
        assessment.reason = (
            assessment.reason
            or "Memory score is below reusable threshold; ignore."
        )

    if hit.memory_type == "skill" and assessment.decision == "apply" and hit.score < _skill_apply_threshold():
        assessment.decision = "ignore"
        assessment.action_bias = {}
        assessment.reason = assessment.reason or "Skill score is below reusable skill threshold."

    if assessment.decision == "ignore":
        assessment.action_bias = {}
        assessment.blocked_actions = []

    assessment.blocked_actions = []
    assessment.action_bias = {
        str(key).upper(): float(value)
        for key, value in (assessment.action_bias or {}).items()
        if str(key).upper() in DEFAULT_ACTIONS
    }
    return assessment


def _llm_memory_assessment(
    hit: RetrievalHit,
    case_state: CaseState,
    memory_query: MemoryQuery,
    llm_client: LLMClient,
    debug: dict[str, object] | None = None,
    strict: bool = True,
) -> MemoryApplicabilityAssessment:
    rule_assessment = _rule_memory_assessment(hit)
    payload = {
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieved_memory": {
            "memory_id": hit.memory_id,
            "memory_type": hit.memory_type,
            "score": hit.score,
            "content": hit.content,
        },
        "allowed_decisions": ["apply", "ignore"],
        "allowed_actions": DEFAULT_ACTIONS,
        "instruction": (
            "Select whether this memory is reusable for the current decision point. "
            "Return apply only when trigger, boundary, evidence state, and action timing match. "
            "Return ignore for partial matches, diagnosis-only overlap, or unavailable evidence."
        ),
    }
    prompt = applicability_prompt(payload)
    if debug is not None:
        debug["mode"] = "llm"
        debug["hit"] = payload["retrieved_memory"]
        debug["rule_assessment"] = rule_assessment.to_dict()
        debug["llm_available"] = llm_client.available()
        debug["payload"] = payload
        debug["prompt"] = prompt
    if not llm_client.available():
        message = f"Applicability LLM mode requested but memory LLM is unavailable for memory {hit.memory_id}"
        if strict:
            raise RuntimeError(message)
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = "llm_unavailable"
            debug["final_assessment"] = rule_assessment.to_dict()
        return rule_assessment

    raw_output = llm_client.generate_json(prompt, max_tokens=900)
    raw_empty = not str(raw_output or "").strip() or str(raw_output or "").strip() == "{}"
    parsed, ok, errors = parse_validate_repair(
        raw_output,
        APPLICABILITY_SCHEMA,
        rule_assessment.to_dict(),
    )
    if debug is not None:
        debug["raw_output"] = raw_output
        debug["parsed_output"] = parsed
        debug["validation_ok"] = ok
        debug["validation_errors"] = errors
    if raw_empty or not ok:
        message = (
            f"Applicability LLM output invalid for memory {hit.memory_id}: "
            f"errors={errors}, raw_output={raw_output!r}"
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    try:
        assessment = MemoryApplicabilityAssessment.from_dict(parsed)
    except Exception as exc:
        message = f"LLM applicability assessment parse failed for memory {hit.memory_id}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        logger.warning(message)
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = f"parse_exception: {exc}"
            debug["final_assessment"] = rule_assessment.to_dict()
        return rule_assessment
    result = _postprocess_llm_assessment(hit, assessment)
    if debug is not None:
        debug["used_fallback"] = False
        debug["final_assessment"] = result.to_dict()
    return result


def _hard_block_actions(case_state: CaseState) -> tuple[list[str], str]:
    return [], ""


def _aggregate_action_assessments(
    memory_assessments: list[MemoryApplicabilityAssessment],
    hard_blocked_actions: list[str],
) -> list[ActionAssessment]:
    action_map: dict[str, ActionAssessment] = {}

    def ensure(action: str) -> ActionAssessment:
        action = str(action).upper()
        if action not in action_map:
            action_map[action] = ActionAssessment(action_type=action)
        return action_map[action]

    for action in hard_blocked_actions:
        item = ensure(action)
        item.blocked = True
        item.reason = str(APPLICABILITY_CONFIG.get("hard_block_reason", "Configured safety rule blocked this action."))

    for assessment in memory_assessments:
        if assessment.decision == "ignore":
            continue

        for action, delta in assessment.action_bias.items():
            action = str(action).upper()
            if not action:
                continue
            item = ensure(action)
            try:
                item.score_delta += float(delta)
            except Exception:
                pass
            if assessment.decision == "apply" and delta > 0:
                item.supporting_memory_ids.append(assessment.memory_id)
            if delta < 0:
                item.warning_memory_ids.append(assessment.memory_id)

    for item in action_map.values():
        item.supporting_memory_ids = list(dict.fromkeys(item.supporting_memory_ids))
        item.warning_memory_ids = list(dict.fromkeys(item.warning_memory_ids))
        if not item.reason:
            if item.score_delta > 0:
                item.reason = "Supported by retrieved memory."
            elif item.score_delta < 0:
                item.reason = "Discouraged by retrieved memory."
            else:
                item.reason = "Neutral."

    return sorted(action_map.values(), key=lambda item: item.action_type)


def apply_applicability_control(
    case_state: CaseState,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, object] | None = None,
    strict: bool = True,
) -> ApplicabilityResult:
    memory_assessments: list[MemoryApplicabilityAssessment] = []
    if debug is not None:
        debug["mode"] = mode
        debug["case_state"] = case_state.to_dict()
        debug["memory_query"] = memory_query.to_dict()
        debug["retrieval_result"] = retrieval_result.to_dict()
        debug["hit_assessments"] = []

    for hit in _all_hits(retrieval_result):
        hit_debug: dict[str, object] = {}
        if mode in {"llm", "hybrid"} and llm_client is not None:
            assessment = _llm_memory_assessment(
                hit,
                case_state,
                memory_query,
                llm_client,
                debug=hit_debug if debug is not None else None,
                strict=strict,
            )
        elif mode in {"llm", "hybrid"} and strict:
            raise RuntimeError("Applicability LLM mode requested but llm_client is None")
        else:
            assessment = _rule_memory_assessment(hit)
            if debug is not None:
                hit_debug["mode"] = "rule"
                hit_debug["hit"] = hit.to_dict()
                hit_debug["final_assessment"] = assessment.to_dict()
        memory_assessments.append(assessment)
        if debug is not None:
            debug["hit_assessments"].append(hit_debug)

    if APPLICABILITY_CONFIG.get("enable_rule_action_guards", False):
        hard_blocked, risk_warning = _hard_block_actions(case_state)
    else:
        hard_blocked, risk_warning = [], ""
    action_assessments = _aggregate_action_assessments(memory_assessments, hard_blocked)
    if debug is not None:
        debug["hard_blocked_actions"] = hard_blocked
        debug["risk_warning"] = risk_warning

    result = ApplicabilityResult(
        memory_assessments=memory_assessments,
        action_assessments=action_assessments,
        hard_blocked_actions=hard_blocked,
        risk_warning=risk_warning,
    )
    if debug is not None:
        debug["final_applicability_result"] = result.to_dict()
    return result
