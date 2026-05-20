# ./rllm/rllm/tools/med_tool/dialog_tool.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)

def _now_ms() -> int:
    return int(time.time() * 1000)


def _post_json(url: str, payload: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    """
    Custom RPC helper (NOT OpenAI/vLLM). For OpenAI/vLLM, use OpenAICompatChatModel.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"RPC HTTPError {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"RPC request failed: {type(e).__name__}: {e}") from e

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

def strip_think(text: str) -> str:
    if not text:
        return ""
    s = _THINK_RE.sub("", text)
    s = re.sub(r"</?think\b[^>]*>", "", s, flags=re.IGNORECASE)
    return s.strip()

ROUTE_LABELS = [
    "History",
    "Physical_Examination_Findings",
    "Test_Results-Labs",
    "Test_Results-Microbiology",
    "Test_Results-Imaging",
    "CXR",
    "Medrecon",
]


def load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_pretty_text(obj: Any, max_chars: int = 200_000) -> str:
    """Make EHR/KB objects readable; cap size to avoid extremely long prompts."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...(truncated)"
    return text


def get_ehr_section(ehr: Dict[str, Any], route_label: str) -> Any:
    return ehr.get(route_label)


class ChatModel(Protocol):
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        ...


def _looks_like_vllm_openai_endpoint(url: str) -> bool:
    u = (url or "").rstrip("/")
    return (
        u.endswith("/v1")
        or u.endswith("/v1/chat/completions")
        or u.endswith("/chat/completions")
        or u.endswith("/v1/completions")
        or u.endswith("/completions")
    )


def _normalize_openai_base_url(u: str) -> str:
    """
    Accept:
      - http://host:port
      - http://host:port/v1
      - http://host:port/v1/chat/completions
      - http://host:port/chat/completions
    Normalize to:
      - http://host:port/v1
    """
    u = (u or "").strip().rstrip("/")
    if not u:
        return u

    if u.endswith("/v1/chat/completions"):
        return u[: -len("/chat/completions")]
    if u.endswith("/chat/completions"):
        return u[: -len("/chat/completions")]
    if u.endswith("/v1"):
        return u
    return u + "/v1"


class OpenAICompatChatModel:
    """
    OpenAI-compatible chat client for vLLM OpenAI server.
    base_url example: http://localhost:30001/v1
    It will call: {base_url}/chat/completions

    IMPORTANT:
    - model must be the served model name (e.g. --served-model-name patient_agent),
      not a local filesystem path.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30):
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        for k in ["top_p", "top_k", "stop", "presence_penalty", "frequency_penalty", "seed"]:
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"OpenAICompat HTTPError {e.code}: {body}\n"
                f"Request URL: {url}\n"
                f"Request model: {self.model}\n"
                f"Payload keys: {list(payload.keys())}\n"
            ) from e
        except Exception as e:
            raise RuntimeError(f"OpenAICompat request failed: {type(e).__name__}: {e}") from e

        try:
            content = obj["choices"][0]["message"]["content"]
            return strip_think(str(content))
        except Exception:
            return strip_think(raw)


ROUTER_SYSTEM_PROMPT = (
    "Given a doctor's question, you MUST choose exactly ONE most relevant section from:\n"
    f"[{', '.join(ROUTE_LABELS)}]\n\n"
    "Output requirements:\n"
    "- Output ONLY the section name, exactly matching one option above\n"
    "- Do NOT output any explanation or extra tokens"
)


def route_question(model: ChatModel, doctor_question: str) -> str:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Doctor's question is:\n{doctor_question}"},
    ]

    raw = strip_think(model.chat(messages, temperature=0.0, max_tokens=8192)).strip()
    s = raw.splitlines()[-1].strip() if raw else ""
    if s not in ROUTE_LABELS:
        s = "Symptoms"
    return s


PATIENT_SYSTEM_PROMPT = (
    "You are the PATIENT in a clinical encounter.\n"
    "A doctor will ask you questions. You must respond as the patient speaking to the doctor in FIRST PERSON (e.g., 'I', 'my').\n\n"
    "Grounding rules:\n"
    "- Use ONLY the information explicitly present in the provided EHR context and (if present) the medical info.\n"
    "- Do NOT invent or guess symptoms, timelines, test results, medications, diagnoses, or treatments.\n"
    "- If the information is not in the context, say you do not know / you were not told, and specify what is missing.\n"
    "- Do NOT mention 'EHR', 'knowbase', 'context', or that you are an AI.\n\n"
    "Style rules:\n"
    "- Sound like a real patient: plain language, concise, and focused on what you experienced.\n"
    "- If asked about labs/imaging/diagnosis/medications, answer only if documented; otherwise say you are not sure or were not told.\n"
    "- Respond in English.\n"
    "- Do not output your reasoning; only provide the answer."
)

PATIENT_SYSTEM_PROMPT_KB = (
    "You are the PATIENT in a clinical encounter.\n"
    "A doctor will ask you questions. You must respond as the patient speaking to the doctor in FIRST PERSON (e.g., 'I', 'my').\n\n"
    "Important:\n"
    "- The provided medical info describes my current condition and its diagnostic rationale.\n"
    "- This implies I may have the symptoms and typical test abnormalities described in that medical info.\n"
    "- You may answer the doctor's question using symptoms, typical test findings, and diagnostic basis in the medical info.\n"
    "- Do NOT directly tell the doctor the diagnosis name or explicitly state the final diagnosis.\n\n"
    "Grounding rules:\n"
    "- Use ONLY the information explicitly present in the provided medical info.\n"
    "- Do NOT mention 'EHR', 'knowbase', 'context', or that you are an AI.\n\n"
    "Style rules:\n"
    "- Sound like a real patient: plain language, concise, and focused.\n"
    "- Respond in English.\n"
    "- Do not output your reasoning; only provide the answer."
)

# -----------------------------
# ✅ Split EHR/KB answerability prompts (align with simplified script)
# -----------------------------
EHR_CONTEXT_ANSWERABLE_PROMPT = (
    "You are an answerability judge for medical QA.\n"
    "Given a doctor's question and a piece of medical text, decide whether the text contains enough information to answer the question directly.\n"
    "Output ONLY 'YES' or 'NO'.\n"
    "- YES: the text contains relevant facts to answer the question.\n"
    "- NO: the text does not contain enough information.\n"
    "Do NOT output any explanation."
)

KB_CONTEXT_ANSWERABLE_PROMPT = (
    "You are an answerability judge for medical QA.\n"
    "The provided medical text describes the patient's confirmed condition and its typical diagnostic findings.\n"
    "Decide whether the text is sufficient to answer the doctor's question in a patient-appropriate way.\n\n"
    "Answerability standard:\n"
    "- Output YES if the text supports a qualitative answer (e.g., high/low/normal, likely increased/decreased), "
    "even if exact numeric results are not provided.\n"
    "- Output NO only if the text provides no basis to answer (no relevant findings, symptoms, tests, or patterns).\n\n"
    "Output ONLY 'YES' or 'NO'. Do NOT output any explanation."
)


def llm_context_answerable(
    judge_model: ChatModel,
    question: str,
    context: str,
    source: str,
    mode: str = "EHR",  # "EHR" or "KB"
) -> bool:
    system_prompt = EHR_CONTEXT_ANSWERABLE_PROMPT if (mode or "EHR").upper() == "EHR" else KB_CONTEXT_ANSWERABLE_PROMPT
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Source: {source}\n\nDoctor question:\n{question}\n\nText:\n{context or '(empty)'}"},
    ]

    out = judge_model.chat(msgs, temperature=0.0, max_tokens=2048).upper()
    out = out.splitlines()[-1].strip() if out else ""
    return out.startswith("YES")


def normal_in_that_aspect_answer() -> str:
    return (
        "As far as I know, I haven't had any problems in that area, "
        "and I haven't been told of any abnormal findings related to it."
    )


def patient_answer(
    llm: Optional[ChatModel],
    doctor_question: str,
    ehr: Dict[str, Any],
    knowledge_obj: Any, 
    max_answer_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    debug: Dict[str, Any] = {}

    if llm is None:
        return normal_in_that_aspect_answer(), debug

    route = route_question(llm, doctor_question)
    debug["route"] = route

    section_obj = get_ehr_section(ehr, route)
    debug["ehr_section_missing"] = section_obj is None

    ehr_answerable = llm_context_answerable(
        llm, doctor_question, section_obj, source=f"EHR:{route}", mode="EHR"
    )
    # print(f"###### section_obj ######:\n{section_obj}")
    # print(f"###### ehr_answerable ######:\n{ehr_answerable}")
    debug["ehr_answerable"] = ehr_answerable

    if ehr_answerable:
        messages = [
            {"role": "system", "content": PATIENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Doctor question:\n{doctor_question}\n\n"
                    f"Patient record ({route}):\n{section_obj or '(empty)'}"
                ),
            },
        ]
        ans = llm.chat(messages, temperature=0.2, max_tokens=max_answer_tokens)
        debug["answer_source"] = "EHR"
        return ans, debug

    if knowledge_obj is None:
        debug["answer_source"] = "NORMAL_NO_KB"
        return normal_in_that_aspect_answer(), debug

    # knowledge_obj = flatten_knowledge_list(knowledge_obj)

    kb_text = to_pretty_text(knowledge_obj)
    # print(f"###### kb_text ######:\n{kb_text}")
    # exit()
    debug["kb_present"] = True

    kb_answerable = llm_context_answerable(
        llm, doctor_question, kb_text, source="KB:knowledge", mode="KB"
    )
    debug["kb_answerable"] = kb_answerable

    if kb_answerable:
        messages2 = [
            {"role": "system", "content": PATIENT_SYSTEM_PROMPT_KB},
            {
                "role": "user",
                "content": (
                    f"Doctor question:\n{doctor_question}\n\n"
                    f"Medical info (DO NOT tell the doctor the diagnosis name):\n{kb_text}"
                ),
            },
        ]
        ans = llm.chat(messages2, temperature=0.2, max_tokens=max_answer_tokens)
        debug["answer_source"] = "KB"
        return ans, debug

    debug["answer_source"] = "NORMAL_KB_NOT_ANSWERABLE"
    return normal_in_that_aspect_answer(), debug



def _ensure_dialogue_context(context: Any) -> Dict[str, Any]:
    """
    Normalize tool context into a dict and ensure context['dialogue'] is a list.
    """
    if context is None:
        context = {}
    elif isinstance(context, list):
        context = {"dialogue": context}
    elif not isinstance(context, dict):
        context = {"_raw_context": context}

    dlg = context.get("dialogue")
    if not isinstance(dlg, list):
        context["dialogue"] = []
    return context


def _append_dialogue_turn(context: Any, doctor_q: str, patient_a: str, max_turns: int) -> Dict[str, Any]:
    """
    Append (doctor, patient) messages into context['dialogue'] safely.
    Return the normalized context dict.
    """
    context = _ensure_dialogue_context(context)

    dq = strip_think(doctor_q)
    pa = strip_think(patient_a)

    context["dialogue"].append({"role": "doctor", "content": dq})
    context["dialogue"].append({"role": "patient", "content": pa})

    if max_turns and max_turns > 0:
        keep = max_turns * 2
        if len(context["dialogue"]) > keep:
            context["dialogue"] = context["dialogue"][-keep:]

    return context


class DialogTool(Tool):
    """
    ask_patient tool.

    - Custom RPC mode (rare): expects a custom endpoint (NOT OpenAI/vLLM).
      payload: {tool, ts, question, case_id, context, extra}

    - vLLM/OpenAI mode: uses EHR/KB + vLLM OpenAI-compatible chat api:
        POST {RLLM_PATIENT_BASE_URL}/chat/completions
        body must include {"model": served-model-name, "messages": [...]}

    Key envs for vLLM mode:
      RLLM_PATIENT_BASE_URL  e.g. http://127.0.0.1:30001 OR http://127.0.0.1:30001/v1
      RLLM_PATIENT_MODEL     e.g. patient_agent   (must match --served-model-name)
      RLLM_PATIENT_API_KEY   default None
    """

    def __init__(
        self,
        name: str = "ask_patient",
        description: str = "Ask patient a question and get a response (patient simulator).",
        rpc_url: str | None = None,
        timeout: int = 50,
        answer_map: dict[str, str] | None = None,
        default_answer: str = "I’m not sure. Could you ask a more specific question?",
        case_dir: str | None = None,
        knowbase_json: str | None = None,
        max_history_turns: int = 30,
        patient_max_answer_tokens: int = 8192,
    ):
        self.timeout = timeout
        self.answer_map = answer_map or {}
        self.default_answer = default_answer

        self.case_dir = case_dir or os.getenv("RLLM_CASE_DIR", "")
        self.knowbase_json = knowbase_json or os.getenv("RLLM_KNOWBASE_JSON", "")
        self.max_history_turns = max_history_turns
        self.patient_max_answer_tokens = patient_max_answer_tokens

        # vLLM/OpenAI patient-agent config
        self.patient_base_url = _normalize_openai_base_url(os.getenv("RLLM_PATIENT_BASE_URL", "http://localhost:30001"))
        self.patient_api_key = (os.getenv("RLLM_PATIENT_API_KEY", "None") or "None").strip()
        self.patient_model_id = (os.getenv("RLLM_PATIENT_MODEL", "") or "").strip()

        # Custom RPC config (avoid mixing with vLLM endpoints)
        raw_rpc = (rpc_url or os.getenv("RLLM_DIALOG_RPC", "")).strip()
        if raw_rpc and _looks_like_vllm_openai_endpoint(raw_rpc):
            logger.warning(
                "RLLM_DIALOG_RPC=%s looks like vLLM/OpenAI endpoint. "
                "Disable custom RPC mode to avoid 400 missing-messages.",
                raw_rpc,
            )
            self.rpc_url = ""
        else:
            self.rpc_url = raw_rpc

        self._ehr_cache: Dict[str, Dict[str, Any]] = {}

        super().__init__(name=name, description=description)

    @property
    def json(self) -> dict[str, Any]:
        # ✅ 改回 required=["question"]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to ask the patient."},
                        "case_id": {"type": "string", "description": "Optional case identifier.", "default": ""},
                        "context": {"type": "object", "description": "Optional dialogue context.", "default": {}},
                    },
                    "required": ["question"],
                },
            },
        }

    def _get_llm(self) -> Optional[ChatModel]:
        if not self.patient_model_id:
            return None
        return OpenAICompatChatModel(
            base_url=self.patient_base_url,
            api_key=self.patient_api_key,
            model=self.patient_model_id,
            timeout=max(30, self.timeout),
        )

    def _load_case_ehr(self, case_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1) context direct
        ehr = context.get("ehr")
        if isinstance(ehr, dict):
            if "ehr" in ehr and isinstance(ehr.get("ehr"), dict):
                return ehr["ehr"]
            return ehr

        ehr_path = context.get("ehr_path")
        if isinstance(ehr_path, str) and ehr_path:
            try:
                obj = load_json_file(ehr_path)
                if isinstance(obj, dict):
                    if "ehr" in obj and isinstance(obj.get("ehr"), dict):
                        return obj["ehr"]
                    return obj
            except Exception:
                pass

        # 3) cache + case_dir (optional)
        cid = (case_id or "").strip()
        if not cid:
            return None
        if cid in self._ehr_cache:
            return self._ehr_cache[cid]

        if not self.case_dir:
            return None
        base = Path(self.case_dir)
        cand = base / cid
        if not cand.exists() and not cid.endswith(".json"):
            cand2 = base / f"{cid}.json"
            if cand2.exists():
                cand = cand2

        if cand.exists() and cand.is_file():
            try:
                obj = load_json_file(cand)
                if isinstance(obj, dict):
                    if "ehr" in obj and isinstance(obj.get("ehr"), dict):
                        obj = obj["ehr"]
                    self._ehr_cache[cid] = obj
                    return obj
            except Exception:
                return None
        return None

    def _load_case_knowledge(self, case_id: str, context: Dict[str, Any]) -> Any:
        """
        Knowledge priority (similar to _load_case_ehr):
        1) context["knowledge"] direct
        2) context["knowledge_path"] load file
        3) case_dir + case_id file: try to load and read obj["knowledge"]
        knowledge is typically a list[dict], but we keep it as-is.
        """

        kb = context.get("knowbase")
        # print(f"###### kb ######:\n{kb}")

        if kb is not None:
            if isinstance(kb, dict) and "knowbase" in kb:
                return kb.get("knowbase")
            return kb


        cid = (case_id or "").strip()
        if not cid or not self.case_dir:
            return None

        base = Path(self.case_dir)
        cand = base / cid
        if not cand.exists() and not cid.endswith(".json"):
            cand2 = base / f"{cid}.json"
            if cand2.exists():
                cand = cand2

        if cand.exists() and cand.is_file():
            try:
                obj = load_json_file(cand)
                if isinstance(obj, dict):
                    return obj.get("knowledge")

            except Exception:
                return None

        return None

    def forward(self, question: str, case_id: str = "", context: dict | None = None, **kwargs) -> ToolOutput:
        if (not question) and ("question" in kwargs):
            question = str(kwargs.pop("question"))

        context = _ensure_dialogue_context(context)

        if not question:
            return ToolOutput(
                name=self.name,
                error="Missing required argument: question",
                metadata={"mode": "error"},
            )

        if self.rpc_url:
            payload = {
                "tool": self.name,
                "ts": _now_ms(),
                "question": question,
                "case_id": case_id,
                "context": context,
                "extra": kwargs or {},
            }
            try:
                resp = _post_json(self.rpc_url, payload, timeout=self.timeout)
                ans = strip_think(str(resp.get("answer", ""))).strip() or self.default_answer
                context = _append_dialogue_turn(context, question, ans, max_turns=self.max_history_turns)
                resp["answer"] = ans
                resp["context"] = context
                return ToolOutput(name=self.name, output=resp, metadata={"mode": "rpc"})
            except Exception as e:
                return ToolOutput(name=self.name, error=str(e), metadata={"mode": "rpc"})

        if question in self.answer_map:
            ans = strip_think(self.answer_map[question]).strip()
            context = _append_dialogue_turn(context, question, ans, max_turns=self.max_history_turns)
            return ToolOutput(name=self.name, output={"answer": ans, "context": context}, metadata={"mode": "local_map"})
        if "__default__" in self.answer_map:
            ans = strip_think(self.answer_map["__default__"]).strip()
            context = _append_dialogue_turn(context, question, ans, max_turns=self.max_history_turns)
            return ToolOutput(name=self.name, output={"answer": ans, "context": context}, metadata={"mode": "local_map"})

        ehr = self._load_case_ehr(case_id=case_id, context=context)
        if not isinstance(ehr, dict):
            ans = self.default_answer
            context = _append_dialogue_turn(context, question, ans, max_turns=self.max_history_turns)
            return ToolOutput(
                name=self.name,
                output={
                    "answer": ans,
                    "context": context,
                    "debug": {
                        "reason": "missing_ehr",
                        "case_id": case_id,
                        "context_keys": sorted(list(context.keys())),
                        "patient_base_url": self.patient_base_url,
                        "patient_model": self.patient_model_id,
                    },
                },
                metadata={"mode": "local_patient_agent"},
            )

        knowledge_obj = self._load_case_knowledge(case_id=case_id, context=context)
        # print(f"###### knowledge_obj ######:\n{knowledge_obj}")

        llm = self._get_llm()

        try:
            ans, dbg = patient_answer(
                llm=llm,
                doctor_question=question,
                ehr=ehr,
                knowledge_obj=knowledge_obj,
                max_answer_tokens=self.patient_max_answer_tokens,
            )
            # print(f"##### doctor_question #####\n{question}")
            # print(f"##### patient_answer #####\n{ans}")
            # print(dbg)
            dbg.setdefault("patient_base_url", self.patient_base_url)
            dbg.setdefault("patient_model", self.patient_model_id)

        except Exception as e:
            # print(f"##### TRY PATIENT_ANSWER Failed #####\n{e}")
            # exit()
            ans = self.default_answer
            dbg = {
                "error": f"{type(e).__name__}: {e}",
                "patient_base_url": self.patient_base_url,
                "patient_model": self.patient_model_id,
            }

        ans = strip_think(ans).strip() or self.default_answer
        context = _append_dialogue_turn(context, question, ans, max_turns=self.max_history_turns)

        return ToolOutput(
            name=self.name,
            output={"answer": ans, "context": context, "debug": dbg},
            metadata={"mode": "local_patient_agent"},
        )