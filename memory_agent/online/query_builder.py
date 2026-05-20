from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient, parse_validate_repair, query_builder_prompt
from ..llm.schemas import QUERY_BUILDER_SCHEMA
from ..schemas import CaseMemory, CaseState, MemoryQuery


def _join(values: list[Any], limit: int = 6) -> str:
    cleaned = [str(value).strip() for value in values[:limit] if str(value).strip()]
    return "; ".join(cleaned)


def _record_to_text(item: dict[str, Any]) -> str:
    turn_id = item.get("turn_id")
    source_path = str(item.get("source_path") or "").strip()
    content = str(item.get("content") or "").strip()
    if not content:
        return ""
    if source_path == "initial":
        prefix = "initial_information"
    else:
        prefix = f"turn_{turn_id}" if turn_id else ""
        if source_path.startswith("interaction."):
            tool_name = source_path.split(".", 1)[1]
            prefix = f"{prefix} {tool_name}_interaction".strip()
        elif source_path.startswith("tool_response."):
            prefix = f"{prefix} tool_response".strip()
        elif source_path == "patient_response":
            prefix = f"{prefix} patient_response".strip()
        elif source_path and source_path not in {"observation", "question"}:
            prefix = f"{prefix} observed_result".strip()
    return f"{prefix}: {content}" if prefix else content


def _initial_information(case_state: CaseState) -> list[str]:
    chief = str(case_state.chief_complaint or "").strip()
    if not chief:
        return []
    return [f"initial_information: chief_complaint: {chief}"]


def _turn_information(case_state: CaseState, turn_id: int | None = None) -> list[str]:
    if turn_id is None and getattr(case_state, "current_turn", None):
        return [
            text for text in (
                _record_to_text(item)
                for item in (case_state.current_turn or [])
                if isinstance(item, dict)
            )
            if text
        ]
    target_turn = case_state.turn_id if turn_id is None else turn_id
    out: list[str] = []
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if item.get("turn_id") == target_turn:
            text = _record_to_text(item)
            if text:
                out.append(text)
    return out


LOW_INFORMATION_PATTERNS = (
    "no information available",
    "not available",
    "not documented",
    "not found",
    "no result",
    "no results",
    "unknown",
)

UNKNOWN_RESPONSE_PATTERNS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "not sure",
    "i wasn't told",
    "i was not told",
    "i haven't been told",
    "i have not been told",
    "as far as i know, i haven't had any problems in that area",
    "as far as i know, i have not had any problems in that area",
)

CLINICAL_SIGNAL_PATTERNS = (
    "pain",
    "cramp",
    "diarrhea",
    "constipat",
    "mucus",
    "blood",
    "fever",
    "chills",
    "nausea",
    "vomit",
    "weight",
    "appetite",
    "tired",
    "fatigue",
    "dizzy",
    "lightheaded",
    "syncope",
    "lost consciousness",
    "short of breath",
    "oxygen",
    "cough",
    "urinary",
    "stool",
    "breathing",
    "swelling",
    "weakness",
    "numbness",
)


def _is_low_information_text(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True
    if len(normalized) <= 8:
        return True
    if "available clinical presentation" in normalized:
        return True
    if normalized.startswith("assess and diagnose"):
        return True
    if "no knowledge base available" in normalized and "no result" in normalized:
        return True
    if normalized in LOW_INFORMATION_PATTERNS:
        return True
    if any(normalized == pattern for pattern in UNKNOWN_RESPONSE_PATTERNS):
        return True
    has_clinical_signal = any(pattern in normalized for pattern in CLINICAL_SIGNAL_PATTERNS)
    if not has_clinical_signal and len(normalized) <= 220 and any(
        normalized.startswith(pattern) for pattern in UNKNOWN_RESPONSE_PATTERNS
    ):
        return True
    return False


def _effective_candidate_records(case_state: CaseState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(case_state.acquired_information or []):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path") or "")
        if source_path == "initial":
            continue
        content = str(item.get("content") or "").strip()
        text = _record_to_text(item)
        if not content or not text:
            continue
        candidates.append(
            {
                "candidate_id": f"c{idx}",
                "turn_id": item.get("turn_id"),
                "source_path": source_path,
                "text": text,
                "rule_low_information": _is_low_information_text(content),
            }
        )
    return candidates


def _turn_effectiveness_prompt(payload: dict[str, Any]) -> str:
    return f"""
You judge whether observed doctor-patient/tool turns should enter a clinical evidence ledger.

Return exactly one valid JSON object with these fields:
- effective_turn_ids: list of candidate_id strings to keep
- ineffective_turn_ids: list of candidate_id strings to exclude

Only classify candidates. Do not rewrite, summarize, deduplicate, or invent content.

Keep a candidate effective when it contains clinically useful positive or negative evidence,
including symptoms, duration, severity, absence of key symptoms, risk factors, exam/test
results, or retrieved medical knowledge that helps the diagnostic process.

Mark ineffective only when it is generic, unavailable, unknown without clinical signal,
off-topic, or a no-result/no-knowledge-base tool response.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _llm_effective_candidate_ids(
    case_state: CaseState,
    llm_client: LLMClient,
    *,
    debug: dict[str, Any] | None = None,
) -> set[str] | None:
    candidates = _effective_candidate_records(case_state)
    if not candidates or not llm_client.available():
        return None

    prompt = _turn_effectiveness_prompt(
        {
            "case_id": case_state.case_id,
            "chief_complaint": case_state.chief_complaint,
            "candidates": candidates,
        }
    )
    fallback = {
        "effective_turn_ids": [
            item["candidate_id"] for item in candidates if not item["rule_low_information"]
        ],
        "ineffective_turn_ids": [
            item["candidate_id"] for item in candidates if item["rule_low_information"]
        ],
    }
    raw_output = llm_client.generate_json(prompt, max_tokens=700)
    parsed, ok, errors = parse_validate_repair(
        raw_output,
        {
            "required": ["effective_turn_ids", "ineffective_turn_ids"],
            "list_fields": ["effective_turn_ids", "ineffective_turn_ids"],
            "dict_fields": [],
        },
        fallback,
    )
    valid_ids = {str(item["candidate_id"]) for item in candidates}
    effective_ids = {str(value) for value in parsed.get("effective_turn_ids", [])}
    effective_ids &= valid_ids
    if not effective_ids and fallback["effective_turn_ids"]:
        effective_ids = set(fallback["effective_turn_ids"])

    if debug is not None:
        debug["turn_effectiveness_mode"] = "llm"
        debug["turn_effectiveness_candidates"] = candidates
        debug["turn_effectiveness_prompt"] = prompt
        debug["turn_effectiveness_raw_output"] = raw_output
        debug["turn_effectiveness_parsed_output"] = parsed
        debug["turn_effectiveness_validation_ok"] = ok
        debug["turn_effectiveness_validation_errors"] = errors
        debug["turn_effectiveness_effective_ids"] = sorted(effective_ids)
    return effective_ids


def _efficient_turn_information(case_state: CaseState, limit: int = 200) -> list[str]:
    """Rule-built evidence ledger, not an LLM summary.

    It contains the initially exposed chief complaint plus every complete
    effective interaction record currently stored in CaseState. Low-information
    turns are excluded, but clinically meaningful negative evidence is kept.
    """
    out: list[str] = _initial_information(case_state)
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_path") or "") != "initial":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        text = _record_to_text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_path") or "") == "initial":
            continue
        content = str(item.get("content") or "").strip()
        if _is_low_information_text(content):
            continue
        text = _record_to_text(item)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _efficient_turn_information_from_ids(
    case_state: CaseState,
    effective_ids: set[str] | None,
    limit: int = 200,
) -> list[str]:
    if effective_ids is None:
        return _efficient_turn_information(case_state, limit=limit)

    out: list[str] = _initial_information(case_state)
    candidates = _effective_candidate_records(case_state)
    for item in candidates:
        if item["candidate_id"] not in effective_ids:
            continue
        text = str(item.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def inefficient_turn_information(case_state: CaseState, limit: int = 200) -> list[str]:
    items: list[str] = []
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_path") or "") == "initial":
            continue
        content = str(item.get("content") or "").strip()
        if not _is_low_information_text(content):
            continue
        text = _record_to_text(item)
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _inefficient_turn_summary(case_state: CaseState, limit: int = 8) -> str:
    items = inefficient_turn_information(case_state, limit=limit)
    if not items:
        return ""
    return "\n".join(
        f"{item} [low-information/no-result]" for item in items
    )


def _prior_information_text(case_state: CaseState, limit: int = 8) -> str:
    lines: list[str] = []
    for item in case_state.acquired_information or []:
        if not isinstance(item, dict):
            continue
        if item.get("turn_id") == case_state.turn_id:
            continue
        if str(item.get("source_path") or "") == "initial":
            continue
        content = str(item.get("content") or "").strip()
        if content and not _is_low_information_text(content):
            lines.append(content)
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _diagnosis_goal_rule(case_state: CaseState) -> str:
    efficient_count = len(_efficient_turn_information(case_state, limit=200))
    inefficient_count = len(
        [
            item for item in case_state.acquired_information or []
            if isinstance(item, dict)
            and str(item.get("source_path") or "") != "initial"
            and _is_low_information_text(str(item.get("content") or ""))
        ]
    )
    goal = (
        "Maintain a focused differential diagnosis, preserve early important "
        "positive/negative evidence, and choose the next action that resolves "
        "the largest remaining diagnostic uncertainty."
    )
    progress = f"Evidence-bearing turns captured: {efficient_count}."
    if inefficient_count:
        progress += (
            f" Low-information/no-result turns already attempted: {inefficient_count}; "
            "avoid repeating the same query and switch to targeted exam/tool evidence or reasoning."
        )
    return f"{goal} {progress}"


def build_case_memory_rule(case_state: CaseState) -> CaseMemory:
    prior_summary_parts = []
    prior_text = _prior_information_text(case_state)
    inefficient_summary = _inefficient_turn_summary(case_state)
    if prior_text:
        prior_summary_parts.append(prior_text)
    if inefficient_summary:
        prior_summary_parts.append("Ineffective/no-result interactions:\n" + inefficient_summary)
    return CaseMemory(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        chief_complaint=case_state.chief_complaint,
        diagnosis_goal=_diagnosis_goal_rule(case_state),
        efficient_turn_information=_efficient_turn_information(case_state),
        prior_information_summary="\n\n".join(prior_summary_parts),
    )


def build_case_memory_llm(
    case_state: CaseState,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseMemory:
    effective_ids = _llm_effective_candidate_ids(case_state, llm_client, debug=debug)
    rule_memory = build_case_memory_rule(case_state)
    classified_memory = CaseMemory(
        case_id=rule_memory.case_id,
        turn_id=rule_memory.turn_id,
        chief_complaint=rule_memory.chief_complaint,
        diagnosis_goal=rule_memory.diagnosis_goal,
        efficient_turn_information=_efficient_turn_information_from_ids(
            case_state,
            effective_ids,
        ),
        prior_information_summary=rule_memory.prior_information_summary,
    )
    if debug is not None:
        debug["case_memory_mode"] = "llm_turn_classification_only"
        debug["case_state"] = case_state.to_dict()
        debug["rule_case_memory"] = rule_memory.to_dict()
        debug["classified_case_memory"] = classified_memory.to_dict()
        debug["case_memory_llm_available"] = llm_client.available()
        debug["case_memory_used_fallback"] = effective_ids is None
        if effective_ids is None:
            debug["case_memory_fallback_reason"] = "llm_unavailable_or_no_candidates"
        debug["final_case_memory"] = classified_memory.to_dict()
    return classified_memory


def build_case_memory(
    case_state: CaseState,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseMemory:
    if mode == "llm" and llm_client is not None:
        return build_case_memory_llm(case_state, llm_client, debug=debug, strict=strict)
    if mode == "llm" and strict:
        raise RuntimeError("CaseMemory LLM mode requested but llm_client is None")
    result = build_case_memory_rule(case_state)
    if debug is not None:
        debug["case_memory_mode"] = "rule"
        debug["case_state"] = case_state.to_dict()
        debug["final_case_memory"] = result.to_dict()
    return result


def _case_memory_to_dict(case_memory: CaseMemory | dict[str, Any]) -> dict[str, Any]:
    if isinstance(case_memory, CaseMemory):
        return case_memory.to_dict()
    return dict(case_memory or {})


def _build_memory_query_rule_from_case_memory(
    case_memory: CaseMemory | dict[str, Any],
) -> MemoryQuery:
    cm = _case_memory_to_dict(case_memory)
    sections: list[str] = []
    for field in ["chief_complaint", "diagnosis_goal", "efficient_turn_information", "prior_information_summary"]:
        value = cm.get(field)
        if isinstance(value, list):
            text = _join(value, limit=12)
        else:
            text = str(value or "").strip()
        if text:
            sections.append(f"{field}: {text}")

    query_text = "\n".join(sections).strip()
    if not query_text:
        raise RuntimeError(
            f"Cannot build memory query for case_id={cm.get('case_id')!r} "
            f"turn_id={cm.get('turn_id')}: CaseMemory contains no queryable information"
        )

    return MemoryQuery(
        case_id=str(cm.get("case_id") or ""),
        turn_id=int(cm.get("turn_id") or 0),
        query_text=query_text,
    )


def _action_to_text(action: Any) -> str:
    if isinstance(action, dict):
        action_type = str(action.get("action_type") or action.get("tool") or "").strip()
        action_label = str(action.get("action_label") or action.get("label") or "").strip()
        if action_type and action_label:
            return f"{action_type}: {action_label}"
        return action_type or action_label
    return str(action).strip()


def build_memory_query_rule(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
) -> MemoryQuery:
    """
    Build a natural-language retrieval query from existing CaseState fields only.
    The MemoryQuery schema stays minimal: case_id, turn_id, query_text.
    """
    return _build_memory_query_rule_from_case_memory(
        build_case_memory_rule(case_state),
    )


def build_memory_query_llm(
    case_state: CaseState,
    candidate_actions: list[Any] | None,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> MemoryQuery:
    case_memory = build_case_memory(
        case_state,
        mode="llm",
        llm_client=llm_client,
        debug=debug,
        strict=strict,
    )
    rule_query = _build_memory_query_rule_from_case_memory(case_memory)
    payload = {
        "case_memory": case_memory.to_dict(),
        "instruction": (
            "Create one concise retrieval query for memory search. "
            "Use only case_memory from the input. "
            "Focus on the current turn plus information already exposed to the doctor agent. "
            "Do not restate the full historical dialogue. "
            "Do not infer new diagnoses, missing information, uncertainty, or risk labels. "
            "The query_text should naturally include the clinical situation, newly exposed facts, "
            "available/reviewed modalities, and useful next-action needs when present. "
            "Return JSON with only query_text."
        ),
    }
    prompt = query_builder_prompt(payload)
    if debug is not None:
        debug["mode"] = "llm"
        debug["case_memory"] = case_memory.to_dict()
        debug["rule_query"] = rule_query.to_dict()
        debug["llm_available"] = llm_client.available()
        debug["payload"] = payload
        debug["prompt"] = prompt
    if not llm_client.available():
        message = "Memory query LLM mode requested but memory LLM is unavailable"
        if strict:
            raise RuntimeError(message)
        if debug is not None:
            debug["used_fallback"] = True
            debug["fallback_reason"] = "llm_unavailable"
            debug["final_query"] = rule_query.to_dict()
        return rule_query

    raw_output = llm_client.generate_json(prompt, max_tokens=800)
    raw_empty = not str(raw_output or "").strip() or str(raw_output or "").strip() == "{}"
    parsed, ok, errors = parse_validate_repair(
        raw_output,
        QUERY_BUILDER_SCHEMA,
        {"query_text": rule_query.query_text},
    )
    query_text = str(parsed.get("query_text") or "").strip()
    if raw_empty or not ok or not query_text:
        message = (
            f"Memory query LLM output invalid for case_id={case_state.case_id!r} "
            f"turn_id={case_state.turn_id}: errors={errors}, raw_output={raw_output!r}"
        )
        if strict and not raw_empty:
            raise RuntimeError(message)
        query_text = rule_query.query_text
    result = MemoryQuery(
        case_id=case_state.case_id,
        turn_id=case_state.turn_id,
        query_text=query_text,
    )
    if debug is not None:
        debug["raw_output"] = raw_output
        debug["parsed_output"] = parsed
        debug["validation_ok"] = ok
        debug["validation_errors"] = errors
        debug["used_fallback"] = query_text == rule_query.query_text and (raw_empty or not ok)
        debug["final_query"] = result.to_dict()
    return result


def build_memory_query(
    case_state: CaseState,
    candidate_actions: list[Any] | None = None,
    mode: str = "rule",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> MemoryQuery:
    if mode == "llm" and llm_client is not None:
        return build_memory_query_llm(case_state, candidate_actions, llm_client, debug=debug, strict=strict)
    if mode == "llm" and strict:
        raise RuntimeError("Memory query LLM mode requested but llm_client is None")
    result = build_memory_query_rule(case_state, candidate_actions)
    if debug is not None:
        debug["mode"] = "rule"
        debug["case_memory"] = build_case_memory_rule(case_state).to_dict()
        debug["final_query"] = result.to_dict()
    return result
