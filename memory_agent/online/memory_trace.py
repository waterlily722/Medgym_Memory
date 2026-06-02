from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..schemas import (
    ApplicabilityResult,
    MemoryGuidance,
    MemoryQuery,
    MemoryRetrievalResult,
)
from ..utils.config import TRACE_CONFIG


TRACE_SCHEMA = "memory_trace.memory_only.v4"
HIT_GROUPS = (
    "positive_experience_hits",
    "negative_experience_hits",
    "skill_hits",
    "knowledge_hits",
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _cfg(key: str, default: Any = None) -> Any:
    return TRACE_CONFIG.get(key, default)


def _to_dict(x: Any) -> Any:
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, dict):
        return {str(k): _to_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_dict(v) for v in x]
    return x


def _clip(x: Any, max_chars: int | None = None) -> Any:
    if x is None or isinstance(x, (int, float, bool)):
        return x
    text = str(x)
    limit = max_chars or int(_cfg("max_text_chars", 600))
    if limit <= 0:
        return text
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _get(d: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    return (d or {}).get(key, default)


# -----------------------------------------------------------------------------
# Compact views
# -----------------------------------------------------------------------------


def _case_memory_view(case_memory: dict[str, Any] | None) -> dict[str, Any]:
    """Only keep fields needed to understand why query/guidance changed."""
    compact = case_memory or {}
    return {
        "chief_complaint": _clip(compact.get("chief_complaint")),
        "diagnostic_strategy": _clip(compact.get("diagnostic_strategy")),
        "efficient_turn_information": [
            _clip(x) for x in compact.get("efficient_turn_information") or []
        ],
        "ineffective_turn_information": [
            _clip(x) for x in compact.get("ineffective_turn_information") or []
        ],
        "prior_information_summary": _clip(compact.get("prior_information_summary")),
    }


TURN_ID_RE = re.compile(r"\bturn_(\d+)\b")


def _records_by_turn_id(records: list[Any]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for record in records or []:
        text = str(record or "").strip()
        if not text:
            continue
        for match in TURN_ID_RE.finditer(text):
            turn_id = int(match.group(1))
            grouped.setdefault(turn_id, []).append(_clip(text, 800))
    return grouped


def _latest_case_memory_from_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    for turn in reversed(turns or []):
        case_memory = turn.get("case_memory")
        if isinstance(case_memory, dict) and case_memory:
            return case_memory
    return {}


def _turn_id_value(turn: dict[str, Any]) -> int:
    try:
        return int(turn.get("turn_id") or 0)
    except Exception:
        return 0


def _diagnostic_trajectory_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_memory = _latest_case_memory_from_turns(turns)
    effective_by_turn = _records_by_turn_id(
        case_memory.get("efficient_turn_information") or []
    )
    turns_by_id = {_turn_id_value(turn): turn for turn in turns or [] if _turn_id_value(turn)}
    trajectory: list[dict[str, Any]] = []
    for turn_id in sorted(effective_by_turn):
        turn = turns_by_id.get(turn_id)
        if not turn:
            continue
        clinical = turn.get("clinical_turn") or {}
        trajectory.append(
            {
                "turn_id": turn_id,
                "doctor_action_type": clinical.get("doctor_action_type") or "",
                "tool_name": clinical.get("tool_name") or "",
                "arguments": clinical.get("arguments") or {},
                "patient_or_tool_response": clinical.get("patient_or_tool_response"),
                "env_info": clinical.get("env_info") or {},
                "reward": clinical.get("reward"),
                "turn_reward": clinical.get("turn_reward"),
                "importance": clinical.get("importance"),
                "evidence_records": effective_by_turn.get(turn_id) or [],
            }
        )
    return trajectory


def _ineffective_interactions_from_turns(turns: list[dict[str, Any]]) -> list[Any]:
    case_memory = _latest_case_memory_from_turns(turns)
    return [
        _clip(item, 800)
        for item in case_memory.get("ineffective_turn_information") or []
    ]


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
    if len(normalized) <= 220 and any(
        normalized.startswith(pattern) for pattern in UNKNOWN_RESPONSE_PATTERNS
    ):
        return True
    return False


def _case_record_to_text(item: dict[str, Any]) -> str:
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
    return f"{prefix}: {content}" if prefix else content


def _inefficient_turn_information(case_state: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    for item in (case_state or {}).get("acquired_information") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_path") or "") == "initial":
            continue
        content = str(item.get("content") or "").strip()
        if not _is_low_information_text(content):
            continue
        text = _case_record_to_text(item)
        if text:
            out.append(_clip(text))
    return out


def _hit_view(hit: dict[str, Any] | None) -> dict[str, Any]:
    """Memory hit summary; avoid dumping full content/source payloads."""
    h = hit or {}
    c = h.get("content") or {}
    return {
        "id": h.get("memory_id") or c.get("memory_id"),
        "type": h.get("memory_type") or c.get("memory_type"),
        "score": h.get("score"),
        "outcome": c.get("outcome_type"),
        "text": _clip(c.get("text")),
        "tags": c.get("tags") or [],
    }


def _retrieval_view(result: dict[str, Any] | None) -> dict[str, Any]:
    r = result or {}
    max_hits = int(_cfg("max_hits_per_type", 3))
    out: dict[str, Any] = {"counts": {}, "top_hits": {}}

    for group in HIT_GROUPS:
        hits = r.get(group) or []
        name = group.replace("_hits", "")
        out["counts"][name] = len(hits)
        out["top_hits"][name] = [_hit_view(hit) for hit in hits[:max_hits]]

    return out


def _applicability_view(app: dict[str, Any] | None) -> dict[str, Any]:
    app = app or {}
    return {
        "risk_warning": app.get("risk_warning") or "",
        "hard_blocked_actions": app.get("hard_blocked_actions") or [],
        "memory_assessments": app.get("memory_assessments") or [],
        "action_assessments": app.get("action_assessments") or [],
    }


def _maybe_llm_io(debug: dict[str, Any] | None) -> dict[str, Any]:
    """Prompt/raw output are the main knobs for prompt/schema debugging."""
    if not _cfg("include_llm_io", False):
        return {}

    d = debug or {}
    out = {"raw_output": d.get("raw_output")}

    if _cfg("include_prompt_payload", False):
        out["prompt"] = d.get("prompt")
        out["payload"] = d.get("payload")

    return out


def _builder_view(debug: dict[str, Any] | None, *, kind: str) -> dict[str, Any]:
    """Shared compact view for case/query/applicability LLM builders."""
    d = debug or {}
    out = {
        "mode": d.get("mode"),
        "llm_available": d.get("llm_available"),
    }

    if kind == "case":
        out["validation_ok"] = d.get("validation_ok")
        out["validation_errors"] = d.get("validation_errors") or []
        out["rule_updated"] = _case_view(d.get("rule_updated_case_state"))
    elif kind == "query":
        out["rule_query"] = d.get("rule_query")
        out["parsed_output"] = d.get("parsed_output")
    elif kind == "hit_assessment":
        out["parsed_output"] = d.get("parsed_output")
        out["final_assessment"] = d.get("final_assessment")

    out.update(_maybe_llm_io(d))
    return out


def _hit_assessment_view(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    return {
        "hit": _hit_view(item.get("hit")),
        **_builder_view(item, kind="hit_assessment"),
    }


def _selection_guidance_view(
    app: dict[str, Any] | None,
    guidance: dict[str, Any] | None,
    app_debug: dict[str, Any] | None,
) -> dict[str, Any]:
    app = app or {}
    guidance = guidance or {}
    assessments = app.get("memory_assessments") or []
    applied = [
        item.get("memory_id")
        for item in assessments
        if item.get("decision") == "apply" and item.get("memory_id")
    ]
    return {
        "mode": (app_debug or {}).get("mode"),
        "selected_memory_ids": applied,
        "selected_memories": guidance.get("selected_memories") or [],
    }


def _clinical_turn_view(clinical: dict[str, Any] | None) -> dict[str, Any]:
    c = clinical or {}
    if not c:
        return {}
    return {
        "tool_name": c.get("tool_name") or "",
        "doctor_action_type": c.get("doctor_action_type") or "",
        "arguments": c.get("arguments") or {},
        "patient_or_tool_response": _clip(c.get("patient_or_tool_response")),
        "env_info": _clip(c.get("env_info")),
        "reward": c.get("reward"),
        "done": c.get("done"),
        "conf_before": c.get("conf_before"),
        "conf_after": c.get("conf_after"),
        "delta": c.get("delta"),
        "turn_reward": c.get("turn_reward"),
        "importance": c.get("importance"),
    }


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------


def build_trace_payload(
    case_state: Any,
    memory_query: MemoryQuery,
    retrieval_result: MemoryRetrievalResult,
    applicability_result: ApplicabilityResult,
    memory_guidance: MemoryGuidance,
    selected_action: dict[str, Any] | None = None,
    memory_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compatibility adapter for existing call sites.

    It intentionally returns a plain turn record; append_memory_turn_trace() decides
    how much to keep.
    """
    return {
        "case_id": case_state.case_id,
        "turn_id": case_state.turn_id,
        "case_state": case_state.to_dict(),
        "memory_query": memory_query.to_dict(),
        "retrieval_result": retrieval_result.to_dict(),
        "applicability_result": applicability_result.to_dict(),
        "memory_guidance": memory_guidance.to_dict(),
        "selected_action": selected_action or {},
        "memory_debug": memory_debug or {},
        "blocked_actions": [],
    }


def compact_turn_record(turn_record: dict[str, Any] | Any) -> dict[str, Any]:
    """
    Minimal readable trace for debugging memory behavior.

    The trace answers five questions:
    1. What did the case schema look like?
    2. What query did memory build?
    3. What did retrieval return?
    4. What did applicability block/warn about?
    5. What guidance was injected into the agent?
    """
    r = _to_dict(turn_record)
    d = r.get("memory_debug") or {}

    query_dbg = d.get("query_builder") or {}
    retrieval_dbg = d.get("retrieval") or {}
    app_dbg = d.get("applicability") or {}
    guidance_dbg = d.get("guidance") or {}

    retrieval_result = retrieval_dbg.get("result") or r.get("retrieval_result") or {}
    final_app = app_dbg.get("final_applicability_result") or r.get("applicability_result") or {}
    final_guidance = guidance_dbg.get("structured") or r.get("memory_guidance") or {}

    out = {
        "case_id": r.get("case_id"),
        "turn_id": r.get("turn_id"),
        "case_memory": _case_memory_view(query_dbg.get("final_case_memory") or query_dbg.get("case_memory")),
        "inefficient_turn_information": _inefficient_turn_information(r.get("case_state")),
        "query": query_dbg.get("final_query") or r.get("memory_query") or {},
        "retrieval": _retrieval_view(retrieval_result),
        "selection_guidance": _selection_guidance_view(final_app, final_guidance, app_dbg),
    }
    if retrieval_dbg.get("retrieval_mode"):
        out["retrieval"]["mode"] = retrieval_dbg.get("retrieval_mode")
    if "embedding_available" in retrieval_dbg:
        out["retrieval"]["embedding_available"] = retrieval_dbg.get("embedding_available")

    clinical_turn = _clinical_turn_view(r.get("clinical_turn"))
    if clinical_turn:
        out["clinical_turn"] = clinical_turn

    if _cfg("include_applicability_debug", False):
        out["applicability_debug"] = {
            **_applicability_view(final_app),
            "mode": app_dbg.get("mode"),
            "hit_assessments": [
                _hit_assessment_view(item)
                for item in app_dbg.get("hit_assessments") or []
            ],
        }

    if r.get("selected_action"):
        out["selected_action"] = r.get("selected_action")

    # Optional large payloads for prompt/injection debugging.
    if _cfg("include_observation_payload", False):
        out["observation"] = d.get("input_observation")
        out["guidance_injection"] = d.get("guidance_injection")

    # Optional raw fallback for rare deep debugging. Keep off by default.
    if _cfg("include_full_turn", False):
        out["raw_turn"] = r

    return out


# -----------------------------------------------------------------------------
# Public writers
# -----------------------------------------------------------------------------


def append_memory_turn_trace(trace_root: str | Path, turn_record: dict[str, Any] | Any) -> Path:
    """Upsert one turn into <trace_root>/<case_id>.json."""
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)

    record = compact_turn_record(turn_record)
    case_id = str(record.get("case_id") or "unknown_case")
    path = root / f"{case_id}.json"

    payload = _read_json(path)
    if payload.get("schema") not in (None, TRACE_SCHEMA, "memory_trace.memory_only.v3"):
        payload = {}

    payload.setdefault("schema", TRACE_SCHEMA)
    payload.setdefault("case_id", case_id)

    turns = payload.setdefault("turns", [])
    _upsert_turn(turns, record)
    turns.sort(key=lambda x: int(x.get("turn_id") or 0))
    payload["diagnostic_trajectory"] = _diagnostic_trajectory_from_turns(turns)
    payload["ineffective_interactions"] = _ineffective_interactions_from_turns(turns)

    _write_json_atomic(path, payload)
    return path


def append_memory_trace(trace_root: str | Path, payload: dict[str, Any]) -> Path:
    """
    Backward-compatible alias.

    The old JSONL snapshot was removed because it duplicates the readable trace.
    Reintroduce JSONL only if an offline evaluator consumes line-delimited turns.
    """
    return append_memory_turn_trace(trace_root, payload)


# -----------------------------------------------------------------------------
# File IO
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _upsert_turn(turns: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = record.get("turn_id")
    for i, old in enumerate(turns):
        if old.get("turn_id") == key:
            turns[i] = record
            return
    turns.append(record)
