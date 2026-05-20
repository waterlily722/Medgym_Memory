# rllm/tools/med_tool/exam_result_tool.py
"""
request_exam 工具：医生请求某项检查结果。
1) 用模型判断 EHR 中是否包含该检查结果；
2) 若包含则从 EHR 抽取并返回给医生；
3) 若不包含则从 knowledge base 给出大致回答。
参考 dialog_tool 的 patient_answer / llm_context_answerable 逻辑。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.tools.tool_base import Tool, ToolOutput

from rllm.tools.med_tool.dialog_tool import (
    strip_think,
    to_pretty_text,
    get_ehr_section,
    OpenAICompatChatModel,
    _normalize_openai_base_url,
    load_json_file,
)

logger = logging.getLogger(__name__)

# 可提供结果的 EHR 区块（检查相关）。科室列表可后续扩展，这里先与 ROUTE_LABELS 对齐
EXAM_EHR_SECTIONS = [
    "Test_Results-Labs",
    "Test_Results-Microbiology",
    "Test_Results-Imaging",
    "CXR",
]
# 可选：科室/检查类型说明（先空着，后续按需填写）
AVAILABLE_EXAM_DEPARTMENTS: List[str] = []


# ---------- Prompts ----------
EXAM_SECTION_ROUTER_PROMPT = (
    "You are a medical record router.\n"
    "The doctor requested exam/check result: \"{exam_type}\".\n"
    "Which of the following EHR sections contains results for this exam? "
    "Sections: " + ", ".join(EXAM_EHR_SECTIONS) + ".\n\n"
    "Output requirements:\n"
    "- Output exactly ONE section name from the list above, or the word NONE if no section contains this exam type.\n"
    "Do NOT output any explanation."
)

EHR_CONTAINS_EXAM_PROMPT = (
    "You are an answerability judge for medical exam results.\n"
    "Given the doctor's requested exam type and a piece of EHR text, decide whether the text contains "
    "actual results for that exam (e.g. values, findings, impressions).\n"
    "Output ONLY 'YES' or 'NO'. Do NOT output any explanation."
)

EXTRACT_RESULT_FOR_DOCTOR_PROMPT = (
    "You are a clinical assistant. The doctor requested the following exam result: \"{exam_type}\".\n"
    "Below is the relevant patient record. Extract and summarize the result in a clear, concise way "
    "for the doctor (e.g. key values, findings, impression). Use plain medical language. Do not invent data."
)

KB_APPROXIMATE_EXAM_PROMPT = (
    "You are a clinical assistant. The doctor requested exam result: \"{exam_type}\".\n"
    "This exam result is NOT present in the patient's EHR. Below is medical knowledge about the patient's condition.\n"
    "Provide a brief, approximate answer for the doctor (e.g. what one might typically expect in such a condition), "
    "clearly stating that this is not from the actual record. Do not invent specific numeric values."
)


def _has_content(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x.strip())
    if isinstance(x, list):
        return len(x) > 0
    if isinstance(x, dict):
        return bool(x)
    return True


def _ensure_context(context: Any) -> Dict[str, Any]:
    if context is None:
        return {}
    if isinstance(context, dict):
        return context
    return {"_raw": context}


def _load_ehr_from_context(case_id: str, context: Dict[str, Any], case_dir: str = "") -> Optional[Dict[str, Any]]:
    ehr = context.get("ehr")
    if isinstance(ehr, dict):
        if "ehr" in ehr and isinstance(ehr.get("ehr"), dict):
            return ehr["ehr"]
        return ehr
    ehr_path = context.get("ehr_path")
    if isinstance(ehr_path, str) and ehr_path:
        try:
            obj = load_json_file(ehr_path)
            if isinstance(obj, dict) and "ehr" in obj and isinstance(obj.get("ehr"), dict):
                return obj["ehr"]
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    cid = (case_id or "").strip()
    if not cid or not case_dir:
        return None
    base = Path(case_dir)
    for cand in [base / cid, base / f"{cid}.json"]:
        if cand.exists() and cand.is_file():
            try:
                obj = load_json_file(str(cand))
                if isinstance(obj, dict) and "ehr" in obj and isinstance(obj.get("ehr"), dict):
                    return obj["ehr"]
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _load_knowledge_from_context(case_id: str, context: Dict[str, Any], case_dir: str = "") -> Any:
    kb = context.get("knowbase")
    if kb is not None:
        if isinstance(kb, dict) and "knowbase" in kb:
            return kb.get("knowbase")
        return kb
    cid = (case_id or "").strip()
    if not cid or not case_dir:
        return None
    base = Path(case_dir)
    for cand in [base / cid, base / f"{cid}.json"]:
        if cand.exists() and cand.is_file():
            try:
                obj = load_json_file(str(cand))
                if isinstance(obj, dict):
                    return obj.get("knowledge")
            except Exception:
                pass
    return None


def _llm_yes_no(chat_model: Any, system: str, user: str) -> bool:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = chat_model.chat(msgs, temperature=0.0, max_tokens=64)
    out = (out or "").strip().upper()
    return out.startswith("YES")


def _llm_one_line(chat_model: Any, system: str, user: str, max_tokens: int = 512) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = chat_model.chat(msgs, temperature=0.2, max_tokens=max_tokens)
    return strip_think(out or "").strip()


class ExamResultTool(Tool):
    """
    医生请求检查结果：先判断 EHR 是否包含该检查，包含则从 EHR 返回，否则从 knowledge base 给出大致回答。
    """

    def __init__(
        self,
        name: str = "request_exam",
        description: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ):
        self._base_url = _normalize_openai_base_url(
            base_url or os.getenv("RLLM_PATIENT_BASE_URL", "http://localhost:30001")
        )
        self._api_key = (api_key or os.getenv("RLLM_PATIENT_API_KEY", "None") or "None").strip()
        self._model = (model or os.getenv("RLLM_PATIENT_MODEL", "") or "").strip()
        self._timeout = timeout

        desc = (
            description
            or "Request exam/test results for the patient. "
            "Specify which exam or check you want (e.g. labs, CXR, imaging, blood test, microbiology). "
            + (
                f"Available departments/exam types: {', '.join(AVAILABLE_EXAM_DEPARTMENTS)}."
                if AVAILABLE_EXAM_DEPARTMENTS
                else "Available departments/exam types: (e.g. labs, imaging, CXR, microbiology — specify as needed)."
            )
            + " Returns result from the record if present, otherwise an approximate answer from knowledge base."
        )
        super().__init__(name=name, description=desc)

    @property
    def json(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exam_type": {
                            "type": "string",
                            "description": "The exam or check to request (e.g. labs, CXR, imaging, blood test, microbiology).",
                        },
                        "case_id": {"type": "string", "description": "Optional case identifier.", "default": ""},
                        "context": {"type": "object", "description": "Optional context (injected by env).", "default": {}},
                    },
                    "required": ["exam_type"],
                },
            },
        }

    def _get_llm(self) -> Optional[OpenAICompatChatModel]:
        if not self._model:
            return None
        return OpenAICompatChatModel(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            timeout=max(30, self._timeout),
        )

    def forward(
        self,
        exam_type: str,
        case_id: str = "",
        context: dict | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        if "exam_type" in kwargs and not exam_type:
            exam_type = str(kwargs.pop("exam_type", ""))
        exam_type = (exam_type or "").strip()
        context = _ensure_context(context or kwargs.get("context"))

        if not exam_type:
            return ToolOutput(
                name=self.name,
                error="Missing required argument: exam_type (e.g. labs, CXR, imaging).",
                metadata={"mode": "error"},
            )

        case_dir = context.get("case_dir") or os.getenv("RLLM_CASE_DIR", "")
        ehr = _load_ehr_from_context(case_id, context, case_dir)
        knowledge_obj = _load_knowledge_from_context(case_id, context, case_dir)
        llm = self._get_llm()

        # ---------- 1) 判断 EHR 中是否包含该检查结果 ----------
        section_containing: Optional[str] = None
        section_content: Any = None

        if llm and isinstance(ehr, dict):
            router_sys = EXAM_SECTION_ROUTER_PROMPT.format(exam_type=exam_type)
            router_user = "Output exactly one section name or NONE."
            section_raw = _llm_one_line(llm, router_sys, router_user, max_tokens=64)
            section_raw = (section_raw or "").strip()
            if section_raw and section_raw.upper() != "NONE" and section_raw in EXAM_EHR_SECTIONS:
                section_containing = section_raw
                section_content = get_ehr_section(ehr, section_containing)

            if section_content is not None and _has_content(section_content):
                section_text = to_pretty_text(section_content, max_chars=30000)
                ehr_has_result = _llm_yes_no(
                    llm,
                    EHR_CONTAINS_EXAM_PROMPT,
                    f"Doctor requested exam: \"{exam_type}\".\n\nEHR section ({section_containing}):\n{section_text}",
                )
                if not ehr_has_result:
                    section_containing = None
                    section_content = None

        # ---------- 2) 若包含则从 EHR 抽取结果返回给医生 ----------
        if section_containing and _has_content(section_content) and llm:
            section_text = to_pretty_text(section_content, max_chars=30000)
            sys_prompt = EXTRACT_RESULT_FOR_DOCTOR_PROMPT.format(exam_type=exam_type)
            result_text = _llm_one_line(llm, sys_prompt, section_text, max_tokens=1024)
            return ToolOutput(
                name=self.name,
                output={
                    "exam_type": exam_type,
                    "source": "ehr",
                    "section": section_containing,
                    "result": result_text,
                },
                metadata={"mode": "ehr"},
            )

        # ---------- 3) 若不包含则从 knowledge base 给出大致回答 ----------
        if knowledge_obj is not None and llm:
            kb_text = to_pretty_text(knowledge_obj, max_chars=20000)
            sys_prompt = KB_APPROXIMATE_EXAM_PROMPT.format(exam_type=exam_type)
            result_text = _llm_one_line(llm, sys_prompt, kb_text, max_tokens=512)
            return ToolOutput(
                name=self.name,
                output={
                    "exam_type": exam_type,
                    "source": "knowledge_base",
                    "result": result_text,
                },
                metadata={"mode": "kb"},
            )

        # 无 EHR 或无 LLM 或无 KB 时的兜底
        fallback = (
            f"No result for \"{exam_type}\" found in the record and no knowledge base available for an approximate answer."
        )
        return ToolOutput(
            name=self.name,
            output={
                "exam_type": exam_type,
                "source": "none",
                "result": fallback,
            },
            metadata={"mode": "none"},
        )
