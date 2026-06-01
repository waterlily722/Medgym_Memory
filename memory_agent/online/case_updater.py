from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient
from ..schemas import CaseState

CASE_STATE_FIELDS = [
    "case_id",
    "turn_id",
    "chief_complaint",
    "current_turn",
    "acquired_information",
]

def _nested_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _unwrap_osce_examination(ehr: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ehr, dict):
        return {}
    if isinstance(ehr.get("OSCE_Examination"), dict):
        return ehr["OSCE_Examination"]
    if isinstance(ehr.get("osce_examination"), dict):
        return ehr["osce_examination"]
    return ehr


VISIBLE_OBSERVATION_KEYS = {
    "question",
    "tool_outputs",
    "memory_guidance",
    "memory_guidance_structured",
}


def _doctor_visible_observation(payload: Any) -> Any:
    """Keep only information that is actually exposed to the doctor agent."""
    if not isinstance(payload, dict):
        return payload
    if "question" in payload:
        return {"question": payload.get("question")}
    if "tool_outputs" in payload:
        return {"tool_outputs": payload.get("tool_outputs") or {}}
    return {
        key: value
        for key, value in payload.items()
        if key in VISIBLE_OBSERVATION_KEYS and value not in (None, "", [], {})
    }


def _collect_information_records(payload: Any, path: str = "observation") -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    payload = _doctor_visible_observation(payload)
    if isinstance(payload, dict):
        if "question" in payload:
            text = str(payload.get("question") or "").strip()
            return [{"source_path": "initial", "content": text}] if text else []
        if "tool_outputs" in payload and isinstance(payload.get("tool_outputs"), dict):
            for tool_call_id, output in (payload.get("tool_outputs") or {}).items():
                text = _compact_text(output).strip()
                if text:
                    records.append(
                        {
                            "source_path": f"tool_response.{tool_call_id}",
                            "content": text,
                        }
                    )
            return records
        for key, value in payload.items():
            if key not in VISIBLE_OBSERVATION_KEYS:
                continue
            text = _compact_text(value).strip()
            if text:
                records.append({"source_path": str(key), "content": text})
    elif payload is not None:
        text = str(payload).strip()
        if text:
            records.append({"source_path": path, "content": text})
    return records


def _compact_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            parts.append(f"{key}={_compact_text(item)}")
        return "; ".join(parts)
    if isinstance(value, list):
        return "; ".join(_compact_text(item) for item in value if item not in (None, "", [], {}))
    return str(value).strip()


def _tool_name_from_action(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return str(
        action.get("tool_name")
        or action.get("name")
        or action.get("action_type")
        or ""
    ).strip()


def _doctor_action_text(action: Any) -> str:
    if not isinstance(action, dict):
        return _compact_text(action)
    tool_name = _tool_name_from_action(action)
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        raw = action.get("raw")
        if isinstance(raw, dict):
            arguments = raw.get("arguments")
        elif isinstance(raw, str) and raw.strip().startswith("<tool_call>"):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    action_type = str(action.get("action_type") or "").strip()
    label = tool_name or action_type or "action"
    args_text = _compact_text(arguments)
    return f"{label}({args_text})" if args_text else label


def _visible_response_text(response: Any) -> str:
    if isinstance(response, dict):
        if "question" in response:
            return str(response.get("question") or "").strip()
        if "tool_outputs" in response and isinstance(response.get("tool_outputs"), dict):
            return "; ".join(
                str(value).strip()
                for value in (response.get("tool_outputs") or {}).values()
                if str(value).strip()
            )
    return _compact_text(response).strip()


def _complete_turn_record(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    if "doctor_action" not in payload or "environment_response" not in payload:
        return None

    action = payload.get("doctor_action")
    response = _doctor_visible_observation(payload.get("environment_response"))
    tool_name = _tool_name_from_action(action)
    action_text = _doctor_action_text(action)
    response_text = _visible_response_text(response)
    if not response_text:
        return None

    content_parts = []
    if action_text:
        content_parts.append(f"doctor_action: {action_text}")
    content_parts.append(f"response: {response_text}")
    return {
        "source_path": f"interaction.{tool_name}" if tool_name else "interaction",
        "content": "\n".join(content_parts),
    }


def _case_id_from_bundle(bundle: Any) -> str:
    if isinstance(bundle, dict):
        return str(bundle.get("case_id") or bundle.get("id") or bundle.get("uid") or "")
    return str(
        getattr(bundle, "case_id", "")
        or getattr(bundle, "id", "")
        or getattr(bundle, "uid", "")
        or ""
    )


def _ehr_from_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, dict):
        ehr = (
            bundle.get("ehr")
            or bundle.get("EHR")
            or _nested_get(bundle, ["medenv_case_bundle", "ehr"])
            or _nested_get(bundle, ["case_bundle", "ehr"])
            or bundle
        )
    else:
        ehr = getattr(bundle, "ehr", None) or getattr(bundle, "EHR", None) or {}
    if isinstance(ehr, dict) and isinstance(ehr.get("ehr"), dict):
        ehr = ehr["ehr"]
    return ehr if isinstance(ehr, dict) else {}


def _chief_complaint_from_ehr(ehr: dict[str, Any]) -> str:
    osce = _unwrap_osce_examination(ehr)
    history = (
        _nested_get(osce, ["Patient_Actor", "History"], {})
        or osce.get("History")
        or ehr.get("History")
        or {}
    )
    symptoms = (
        _nested_get(osce, ["Patient_Actor", "Symptoms"], {})
        or osce.get("Symptoms")
        or ehr.get("Symptoms")
        or {}
    )

    chief = ""
    if isinstance(symptoms, dict):
        chief = str(symptoms.get("Chief_Complaint") or "")
    if not chief and isinstance(history, dict):
        chief = str(history.get("Chief_Complaint") or "")

    return chief.strip()


_CLINICAL_KEYWORDS = (
    "pain", "ache", "swelling", "bleeding", "fever", "cough", "shortness",
    "breath", "nausea", "vomiting", "diarrhea", "constipation", "fatigue",
    "dizziness", "headache", "rash", "numbness", "weakness", "weight",
    "urinary", "chest", "abdominal", "back", "joint", "hip", "knee",
    "fracture", "wound", "infection", "mass", "tumor", "cancer",
    "diabetes", "hypertension", "pneumonia", "asthma", "arthritis",
    "heart", "lung", "kidney", "liver", "thyroid", "colon", "breast",
    "screening", "surgery", "chemotherapy", "dialysis", "pregnancy",
)


def _is_valid_chief_complaint(cc: str) -> bool:
    """Reject demographics-only or redacted chief complaints."""
    if not cc or not cc.strip():
        return False
    cc = cc.strip()
    lower = cc.lower()
    # Redacted text (starts with "___")
    if lower.startswith("___"):
        return False
    # Pure demographics: "I am a XX-year-old X patient..."
    if lower.startswith("i am ") and "year-old" in lower:
        # Only reject if there are NO clinical keywords after demographics
        if not any(kw in lower for kw in _CLINICAL_KEYWORDS):
            return False
    # Generic non-informative phrases
    generic_phrases = (
        "i haven't been feeling well recently",
        "i have not been feeling well recently",
        "i don't feel well",
    )
    if lower.rstrip(". ") in generic_phrases:
        return False
    return True


def _chief_complaint_from_visible_question(bundle: Any) -> str:
    if not isinstance(bundle, dict):
        return ""
    question = str(bundle.get("question") or "").strip()
    if not question:
        return ""
    marker = "my main issue is:"
    lower = question.lower()
    if marker not in lower:
        # No marker — the full question is likely demographics-only
        # Validate before returning
        if _is_valid_chief_complaint(question):
            return question
        return ""
    chief = question[lower.find(marker) + len(marker):].strip()
    if "available clinical presentation" in chief.lower():
        return ""
    if chief.lower().startswith("assess and diagnose"):
        return ""
    chief = chief.strip(" .")
    if not _is_valid_chief_complaint(chief):
        return ""
    return chief


def _chief_complaint_from_hpi(ehr: dict[str, Any]) -> str:
    """Extract presenting complaint from HPI (History of Present Illness)."""
    history = ehr.get("History") or {}
    if not isinstance(history, dict):
        return ""
    hpi = str(history.get("HPI") or "").strip()
    if not hpi or len(hpi) < 10:
        return ""
    # HPI is often a long narrative; take first sentence as chief complaint
    first_sentence = hpi.split(".")[0].strip()
    if len(first_sentence) > 200:
        first_sentence = first_sentence[:200].rsplit(" ", 1)[0]
    return first_sentence


def _chief_complaint_from_observation(bundle: Any) -> str:
    """Derive presenting complaint from the initial observation text shown to doctor.

    The observation typically contains 'Patient presents with ...' or similar
    language that encodes the chief complaint even when the structured EHR
    Chief_Complaint field is None.
    """
    if not isinstance(bundle, dict):
        return ""
    # The initial observation may be stored under different keys
    obs = (
        str(bundle.get("observation") or "").strip()
        or str(bundle.get("question") or "").strip()
    )
    if not obs:
        return ""

    # Try common patterns in the observation text
    import re
    for pattern in [
        r"(?i)patient (?:presents with|reports|complains of)[:\s]+([^.]{5,200})",
        r"(?i)chief complaint[:\s]+([^.]{5,200})",
        r"(?i)presenting (?:problem|complaint|issue)[:\s]+([^.]{5,200})",
        r"(?i)my main issue is[:\s]+([^.]{5,200})",
    ]:
        m = re.search(pattern, obs)
        if m:
            cc = m.group(1).strip()
            if "available clinical presentation" not in cc.lower():
                return cc

    # If no pattern matches, use the full text (trimmed) as a last resort
    # but only if it's not a generic instruction and not demographics-only
    generic_patterns = [
        "assess and diagnose",
        "available clinical presentation",
        "you are a doctor",
    ]
    lower = obs.lower()
    if any(p in lower for p in generic_patterns):
        return ""
    candidate = obs[:200]
    if not _is_valid_chief_complaint(candidate):
        return ""
    return candidate


def init_case_state(bundle: Any, no_cxr: bool = False) -> CaseState:
    ehr = _ehr_from_bundle(bundle)
    case_id = _case_id_from_bundle(bundle)

    # Try multiple sources for chief_complaint in priority order
    chief_complaint = _chief_complaint_from_ehr(ehr)
    if not chief_complaint:
        chief_complaint = _chief_complaint_from_hpi(ehr)
    if not chief_complaint:
        chief_complaint = _chief_complaint_from_visible_question(bundle)
    if not chief_complaint:
        chief_complaint = _chief_complaint_from_observation(bundle)

    return CaseState(
        case_id=case_id,
        turn_id=0,
        chief_complaint=chief_complaint,
        current_turn=[],
        acquired_information=[],
    )


def _collect_initial_info_text(bundle: Any) -> str:
    """Collect all available initial info text from the observation bundle."""
    parts: list[str] = []
    if isinstance(bundle, dict):
        question = str(bundle.get("question") or "").strip()
        if question:
            parts.append(f"Patient question: {question}")
        context = bundle.get("context", {})
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(context, dict):
            ehr = context.get("ehr", context)
            if isinstance(ehr, dict):
                # Objective_for_Doctor often contains a summary
                obj = str(ehr.get("Objective_for_Doctor") or "").strip()
                if obj and not obj.lower().startswith("assess and diagnose"):
                    parts.append(f"Objective: {obj}")
                # HPI
                history = ehr.get("History") or {}
                if isinstance(history, dict):
                    hpi = str(history.get("HPI") or "").strip()
                    if hpi and hpi != "None" and len(hpi) > 10:
                        parts.append(f"HPI: {hpi[:500]}")
                # Patient info
                pi = ehr.get("Patient_info") or {}
                if isinstance(pi, dict):
                    parts.append(
                        f"Patient: {pi.get('age', '')}yo {pi.get('gender', '')}"
                    )
    return "\n".join(parts)


_CHIEF_COMPLAINT_PROMPT = """You are extracting the chief complaint from clinical information.
The chief complaint should be a concise statement of the patient's main symptom or problem.

Given the initial clinical information below, extract the chief complaint.
- Focus on the main symptom, problem, or reason for the visit.
- Do NOT include demographics alone (age, gender) without a clinical problem.
- Keep it under 50 words.
- If no clinical information is available, return empty string.

Return JSON: {{"chief_complaint": "..."}}

Initial information:
{info}"""


def build_chief_complaint_with_llm(
    case_state: CaseState,
    bundle: Any,
    llm_client: LLMClient,
) -> str:
    """Use LLM to construct chief_complaint when rule-based extraction failed."""
    if not llm_client.available():
        return ""
    info = _collect_initial_info_text(bundle)
    if not info or len(info.strip()) < 20:
        return ""
    prompt = _CHIEF_COMPLAINT_PROMPT.format(info=info)
    try:
        raw = llm_client.generate_json(prompt, max_tokens=200)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        cc = str(parsed.get("chief_complaint") or "").strip()
        if _is_valid_chief_complaint(cc):
            return cc
    except Exception:
        pass
    return ""


def _dedupe_records(values: list[Any], limit: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values or []:
        if not isinstance(value, dict):
            continue
        source_path = str(value.get("source_path") or "").strip()
        content = str(value.get("content") or "").strip()
        turn_id = str(value.get("turn_id") or "").strip()
        if not content:
            continue
        key = (turn_id, source_path, content)
        if key in seen:
            continue
        seen.add(key)
        item = dict(value)
        item["source_path"] = source_path
        item["content"] = content
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def update_case_state_rule(prev_case_state: CaseState, observation: Any) -> CaseState:
    """
    Deterministic observation ledger.

    CaseState records the complete information already exposed to the doctor
    agent by the environment or tools. It does not summarize, truncate, or infer.
    """
    state = CaseState.from_dict(prev_case_state.to_dict())
    previous_turn_id = state.turn_id
    state.turn_id += 1

    complete_record = _complete_turn_record(observation)
    records = [complete_record] if complete_record is not None else _collect_information_records(observation)
    current_turn_records: list[dict[str, Any]] = []
    for record in records:
        text = record["content"]
        content = text.strip()
        if not content:
            continue
        item = {
            "turn_id": state.turn_id,
            "source_path": (
                "patient_response"
                if previous_turn_id > 0 and record.get("source_path") == "initial"
                else record.get("source_path", "")
            ),
            "content": content,
        }
        state.acquired_information.append(item)
        current_turn_records.append(item)

    state.current_turn = _dedupe_records(current_turn_records, 50)
    state.acquired_information = _dedupe_records(state.acquired_information, 200)
    return state


def _truncate_payload(value: Any, max_chars: int = 5000) -> Any:
    if isinstance(value, dict):
        return {str(key): _truncate_payload(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_payload(item, max_chars=max_chars) for item in value[:50]]
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return value


def update_case_state_llm(
    prev_case_state: CaseState,
    observation: Any,
    llm_client: LLMClient,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseState:
    result = update_case_state_rule(prev_case_state, observation)
    if debug is not None:
        debug["mode"] = "observed"
        debug["llm_skipped"] = True
        debug["skip_reason"] = "CaseState is an observed-information ledger; no LLM extraction is used"
        debug["previous_case_state"] = prev_case_state.to_dict()
        debug["observation"] = _truncate_payload(observation)
        debug["final_case_state"] = result.to_dict()
    return result


def update_case_state(
    prev_case_state: CaseState,
    observation: Any,
    mode: str = "observed",
    llm_client: LLMClient | None = None,
    debug: dict[str, Any] | None = None,
    strict: bool = True,
) -> CaseState:
    result = update_case_state_rule(prev_case_state, observation)
    if debug is not None:
        debug["mode"] = "observed"
        debug["requested_mode"] = mode
        debug["llm_skipped"] = True
        debug["previous_case_state"] = prev_case_state.to_dict()
        debug["observation"] = _truncate_payload(observation)
        debug["final_case_state"] = result.to_dict()
    return result
