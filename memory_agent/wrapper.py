from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from rllm.rewards.med_diagnosis_reward import (
    _confidence_heuristic,
    result_reward,
    result_reward_judge,
)

logger = logging.getLogger(__name__)

try:
    from rllm.agents import ToolAgent as _BaseAgent
except Exception:  # pragma: no cover - used in lightweight unit tests
    class _BaseAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.tools = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt", "")
            self.parser_name = kwargs.get("parser_name", "")

        def update_from_env(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def update_from_model(self, *args: Any, **kwargs: Any) -> Any:
            return None

from .llm import EmbeddingClient, LLMClient
from .offline import distill_from_trajectory, write_memory_from_distilled_episode
from .online import (
    append_memory_turn_trace,
    append_memory_trace,
    apply_applicability_control,
    build_memory_guidance,
    build_memory_query,
    build_trace_payload,
    guidance_to_text,
    init_case_state,
    build_chief_complaint_with_llm,
    retrieve_multi_memory,
    update_case_state,
)
from .schemas import (
    ApplicabilityResult,
    CaseState,
    EpisodeFeedback,
    MemoryGuidance,
    MemoryQuery,
    MemoryRetrievalResult,    OutcomeType,    TurnRecord,
)
from .utils.config import MEMORY_ROOT_DIRNAME, RETRIEVAL_CONFIG
from .utils.config import MEMORY_ACTION_CONFIG

ACTION_MAP = dict(MEMORY_ACTION_CONFIG["tool_to_action"])
DEFAULT_ACTIONS = list(MEMORY_ACTION_CONFIG["default_actions"])
FINALIZE_ACTION = str(MEMORY_ACTION_CONFIG["finalize_action"])

BOXED_DIAGNOSIS_RE = re.compile(r"\\box(?:ed)?\{(.+?)\}", re.S)
TRACE_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _extract_boxed_diagnosis(text: str) -> str:
    match = BOXED_DIAGNOSIS_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _safe_trace_tag(value: str) -> str:
    tag = TRACE_TAG_RE.sub("_", str(value or "").strip()).strip("._-")
    return tag[:80]


class MemoryWrappedMedicalAgent(_BaseAgent):
    """
    Memory wrapper for a MedEnv doctor agent.

    Online path:
        observation -> update CaseState -> build MemoryQuery -> retrieve memory
        -> applicability control -> MemoryGuidance -> inject into base agent.

    Offline path:
        done episode -> DistilledEpisode -> LLM experience extraction
        -> merge/write ExperienceStore.

    The base doctor agent remains responsible for action generation.
    """

    # ------------------------------------------------------------------
    # CXR multi-modal support (adapted from MedicalAgent)
    # ------------------------------------------------------------------
    @staticmethod
    def _cxr_tool_output_to_multimodal(
        tool_output_str: str,
    ) -> tuple[str | None, list[dict] | None]:
        """
        Parse CXR tool JSON: if it contains base64 images, return
        (tool_summary, multimodal_content_list). Otherwise (None, None).
        """
        try:
            data = json.loads(tool_output_str)
        except (json.JSONDecodeError, TypeError):
            return None, None
        if not isinstance(data, dict):
            return None, None
        images = data.get("images") or []
        if not isinstance(images, list) or not images:
            return None, None
        has_b64 = any(
            isinstance(img, dict) and img.get("image_base64") for img in images
        )
        if not has_b64:
            return None, None

        summary = data.get("summary") or ""
        view_list: list[str] = []
        content_parts: list[dict] = [{"type": "text", "text": summary}]

        for img in images:
            if not isinstance(img, dict):
                continue
            b64 = img.get("image_base64")
            vp = img.get("view_position", "unknown")
            view_list.append(vp)
            if b64:
                mime = (img.get("mime") or "image/jpeg").strip().lower()
                url = f"data:{mime};base64,{b64}" if "image/" in mime else f"data:image/jpeg;base64,{b64}"
                content_parts.append({"type": "image_url", "image_url": {"url": url}})

        tool_summary = (
            summary
            or f"CXR returned {len(images)} image(s). Views: {', '.join(view_list)}. Images attached below."
        )
        return tool_summary, content_parts

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Format environment observation with CXR multi-modal support."""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    tool_summary, user_parts = self._cxr_tool_output_to_multimodal(
                        tool_output_str
                    )
                    if tool_summary is not None and user_parts is not None:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_summary,
                                "tool_call_id": tool_call_id,
                            }
                        )
                        messages.append({"role": "user", "content": user_parts})
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_output_str,
                                "tool_call_id": tool_call_id,
                            }
                        )
            else:
                messages.append(
                    {"role": "user", "content": json.dumps(obs, ensure_ascii=False)}
                )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs is not None:
            messages.append({"role": "user", "content": str(obs)})
        return messages

    def __init__(
        self,
        *args: Any,
        case_update_mode: str = "observed",
        query_builder_mode: str = "llm",
        applicability_mode: str = "llm",
        experience_extraction_mode: str = "llm",
        experience_merge_mode: str = "llm",
        memory_top_k: int = 5,
        log_memory_trace: bool = False,
        inject_case_memory: bool = False,
        trace_tag: str = "",
        disable_memory: bool = False,
        enable_memory: bool | None = None,
        disable_experience_memory: bool = False,
        disable_skill_memory: bool = False,
        disable_knowledge_memory: bool = False,
        disable_memory_write: bool = False,
        memory_root: str | None = None,
        memory_llm_model: str = "",
        memory_llm_base_url: str = "",
        memory_llm_api_key: str = "",
        memory_embedding_model: str = "",
        memory_embedding_base_url: str = "",
        memory_embedding_api_key: str = "",
        strict_memory_errors: bool = True,
        no_cxr: bool = False,
        **kwargs: Any,
    ) -> None:
        # Support both enable_memory (from run script) and disable_memory (internal)
        if enable_memory is not None:
            disable_memory = not enable_memory
        self.memory_root = memory_root or MEMORY_ROOT_DIRNAME
        self.case_update_mode = case_update_mode
        self.query_builder_mode = query_builder_mode
        self.applicability_mode = applicability_mode
        self.experience_extraction_mode = experience_extraction_mode
        self.experience_merge_mode = experience_merge_mode
        # memory_top_k is reserved for future top-k filtering at the wrapper
        # level. Currently per-memory-type top-k is configured in RETRIEVAL_CONFIG.
        self.memory_top_k = memory_top_k
        self.log_memory_trace = log_memory_trace
        self.inject_case_memory = inject_case_memory
        effective_trace_tag = _safe_trace_tag(trace_tag)
        if not effective_trace_tag and inject_case_memory:
            effective_trace_tag = "inject_case_memory"
        trace_dirname = f"trace_{effective_trace_tag}" if effective_trace_tag else "trace"
        self.trace_root = str(Path(self.memory_root) / trace_dirname)
        self.trace_tag = effective_trace_tag
        self.disable_memory = disable_memory
        self.disable_experience_memory = disable_experience_memory
        self.disable_skill_memory = disable_skill_memory
        self.disable_knowledge_memory = disable_knowledge_memory
        self.disable_memory_write = disable_memory_write
        self.strict_memory_errors = strict_memory_errors
        self.no_cxr = no_cxr
        self.memory_llm = LLMClient(
            model=memory_llm_model,
            base_url=memory_llm_base_url,
            api_key=memory_llm_api_key,
        )
        self.memory_embedding = EmbeddingClient(
            model=memory_embedding_model,
            base_url=memory_embedding_base_url,
            api_key=memory_embedding_api_key,
        )

        self.episode_id = f"pending_episode_{uuid.uuid4().hex[:10]}"
        self.case_state: CaseState | None = None
        self.latest_query: MemoryQuery | None = None
        self.latest_retrieval: MemoryRetrievalResult | None = None
        self.latest_applicability: ApplicabilityResult | None = None
        self.latest_guidance: MemoryGuidance | None = None
        self.latest_memory_debug: dict[str, Any] = {}
        self.pending_selected_action: dict[str, Any] = {}
        self.turn_records: list[TurnRecord] = []
        self.episode_finalized = False
        # Track which memory_ids have been injected this episode to avoid repetition
        self._injected_memory_ids: dict[str, int] = {}
        # Cached original task bundle (set during first update_from_env call)
        self._case_bundle: dict[str, Any] = {}

        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Some unit-test environments use a minimal fallback base class.
            super().__init__()
            self.tools = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt", "")
            self.parser_name = kwargs.get("parser_name", "")

    # ------------------------------------------------------------------
    # Base agent bridge
    # ------------------------------------------------------------------
    def _call_base(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(super(MemoryWrappedMedicalAgent, self), method_name, None)
        if method is None:
            return None
        try:
            return method(*args, **kwargs)
        except TypeError:
            if args:
                try:
                    return method(args[0])
                except TypeError:
                    return None
            return None

    # ------------------------------------------------------------------
    # Online path
    # ------------------------------------------------------------------
    def update_from_env(
        self,
        observation: Any = None,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if observation is None and "obs" in kwargs:
            observation = kwargs["obs"]
        info = info or kwargs.get("info") or {}
        action_for_case_state = dict(self.pending_selected_action or {})

        self._finalize_pending_turn(
            env_observation=observation,
            env_info=info,
            reward=reward,
            done=done,
        )

        if done:
            self._finalize_episode_if_needed(reward=reward, info=info)
            return self._call_base(
                "update_from_env",
                observation,
                reward=reward,
                done=done,
                info=info,
                **kwargs,
            )

        memory_debug: dict[str, Any] = {
            "episode_id": self.episode_id,
            "input_observation": self._safe_payload(observation),
            "env_info_before_memory": self._safe_payload(info),
            "memory_enabled": not self.disable_memory,
        }

        if self.case_state is None:
            bundle = self._case_bundle_from(observation, info)
            self._case_bundle = bundle if isinstance(bundle, dict) else {}
            initial_case_state = init_case_state(bundle, no_cxr=self.no_cxr)
            # If chief_complaint is empty, try LLM construction from initial info
            if not initial_case_state.chief_complaint:
                llm_cc = build_chief_complaint_with_llm(
                    initial_case_state, bundle, self.memory_llm,
                )
                if llm_cc:
                    initial_case_state.chief_complaint = llm_cc
            if self.strict_memory_errors and not initial_case_state.case_id:
                raise RuntimeError("Memory CaseState initialization failed: missing case_id")
            self.episode_id = self._episode_id_from_bundle(self._case_bundle, initial_case_state.case_id)
            memory_debug["episode_id"] = self.episode_id
            case_state_debug: dict[str, Any] = {}
            case_observation = self._case_state_observation(
                observation,
                selected_action=action_for_case_state,
            )
            self.case_state = update_case_state(
                initial_case_state,
                case_observation,
                mode=self.case_update_mode,
                llm_client=self.memory_llm,
                debug=case_state_debug,
                strict=self.strict_memory_errors,
            )
            memory_debug["case_state_update"] = {
                "mode": "init_then_current_turn_update",
                "bundle": self._safe_payload(bundle),
                "initial_case_state": initial_case_state.to_dict(),
                "current_turn_update": case_state_debug,
                "final_case_state": self.case_state.to_dict(),
            }
        else:
            case_state_debug: dict[str, Any] = {}
            case_observation = self._case_state_observation(
                observation,
                selected_action=action_for_case_state,
            )
            self.case_state = update_case_state(
                self.case_state,
                case_observation,
                mode=self.case_update_mode,
                llm_client=self.memory_llm,
                debug=case_state_debug,
                strict=self.strict_memory_errors,
            )
            memory_debug["case_state_update"] = case_state_debug

        if not self.disable_memory:
            self._run_memory_pipeline(memory_debug=memory_debug)

        enriched_observation = self._inject_guidance(observation)
        memory_debug["guidance_injection"] = {
            "guidance_text": guidance_to_text(self.latest_guidance) if self.latest_guidance else "",
            "original_observation": self._safe_payload(observation),
            "enriched_observation": self._safe_payload(enriched_observation),
        }
        self.latest_memory_debug = memory_debug

        if self.log_memory_trace and self._has_memory_snapshot():
            payload = build_trace_payload(
                case_state=self.case_state,
                memory_query=self.latest_query,
                retrieval_result=self.latest_retrieval,
                applicability_result=self.latest_applicability,
                memory_guidance=self.latest_guidance,
                selected_action={},
                memory_debug=self.latest_memory_debug,
            )
            append_memory_trace(self.trace_root, payload)

        return self._call_base(
            "update_from_env",
            enriched_observation,
            reward=reward,
            done=done,
            info=info,
            **kwargs,
        )

    def _case_state_observation(
        self,
        observation: Any,
        selected_action: dict[str, Any] | None = None,
    ) -> Any:
        selected_action = selected_action or self.pending_selected_action
        if not selected_action:
            return observation
        doctor_action = dict(selected_action)
        parsed_tool = self._parse_tool_call_text(doctor_action.get("raw"))
        if parsed_tool:
            doctor_action["tool_name"] = parsed_tool.get("tool_name") or doctor_action.get("tool_name") or ""
            doctor_action["arguments"] = parsed_tool.get("arguments") or {}
        return {
            "doctor_action": doctor_action,
            "environment_response": observation,
        }

    def update_from_model(self, model_output: Any = None, **kwargs: Any) -> Any:
        if model_output is None and "output" in kwargs:
            model_output = kwargs["output"]

        parsed_action = self._parse_selected_action(model_output)
        self.pending_selected_action = parsed_action

        base_result = self._call_base("update_from_model", model_output, **kwargs)
        return base_result if base_result is not None else model_output

    def _current_turn_is_ineffective(self) -> bool:
        """Check whether every record in the current turn is low-information.

        Uses the same _is_low_information_text predicate that the CaseMemory
        pipeline uses internally, so the classification is consistent.
        """
        if self.case_state is None:
            return False
        from .online.query_builder import _is_low_information_text
        current_turn = self.case_state.current_turn or []
        if not current_turn:
            return True
        for record in current_turn:
            if not isinstance(record, dict):
                continue
            content = str(record.get("content") or "").strip()
            if content and not _is_low_information_text(content):
                return False
        return True

    def _run_memory_pipeline(self, memory_debug: dict[str, Any] | None = None) -> None:
        if self.case_state is None:
            return

        candidate_actions = self._candidate_actions()
        if memory_debug is not None:
            memory_debug["candidate_actions"] = candidate_actions
            memory_debug["memory_llm"] = {
                "available": self.memory_llm.available(),
                "model": self.memory_llm.model,
                "base_url": self.memory_llm.base_url,
            }
            memory_debug["query_builder"] = {}

        # ── Ineffective-turn gate ────────────────────────────────────
        # When all current-turn records are low-information, skip the
        # expensive case_memory LLM + query_builder LLM calls. Reuse
        # the previous query and retrieval. Applicability still runs
        # because case_state.current_turn carries the ineffective
        # interaction, so the LLM can see "last action returned no
        # useful evidence" and discourage repeating the same action.
        can_reuse_previous = (
            self._current_turn_is_ineffective()
            and self.latest_query is not None
            and self.latest_retrieval is not None
            and self.case_state.turn_id > 1
        )

        if can_reuse_previous:
            if memory_debug is not None:
                memory_debug["query_builder"]["mode"] = "reused_previous"
                memory_debug["query_builder"]["reason"] = (
                    "current turn is entirely low-information; "
                    "reusing previous query and retrieval"
                )
                memory_debug["query_builder"]["reused_query"] = self.latest_query.to_dict()
            logger.debug(
                "Ineffective-turn gate active for episode %s turn %d — "
                "reusing previous query/retrieval",
                self.episode_id,
                self.case_state.turn_id,
            )
        else:
            self.latest_query = build_memory_query(
                case_state=self.case_state,
                candidate_actions=candidate_actions,
                mode=self.query_builder_mode,
                llm_client=self.memory_llm,
                debug=memory_debug.get("query_builder") if memory_debug is not None else None,
                strict=self.strict_memory_errors,
            )
            self.latest_retrieval = retrieve_multi_memory(
                memory_query=self.latest_query,
                root_dir=self.memory_root,
                disable_experience_memory=self.disable_experience_memory,
                disable_skill_memory=self.disable_skill_memory,
                disable_knowledge_memory=self.disable_knowledge_memory,
                embedding_client=self.memory_embedding if self.memory_embedding.available() else None,
            )
            if memory_debug is not None:
                memory_debug["retrieval"] = {
                    "memory_root": self.memory_root,
                    "disable_experience_memory": self.disable_experience_memory,
                    "disable_skill_memory": self.disable_skill_memory,
                    "disable_knowledge_memory": self.disable_knowledge_memory,
                    "embedding_available": self.memory_embedding.available(),
                    "retrieval_mode": (
                        "embedding"
                        if self.memory_embedding.available()
                        else str(
                            os.environ.get("MEDGYM_RETRIEVAL_FALLBACK_SCORING")
                            or RETRIEVAL_CONFIG.get("fallback_scoring")
                            or "cosine"
                        )
                    ),
                    "result": self.latest_retrieval.to_dict(),
                }
                memory_debug["applicability"] = {}

        # Always re-run applicability — case_state may have changed even
        # if the query/retrieval are reused.
        if memory_debug is not None and "applicability" not in memory_debug:
            memory_debug["applicability"] = {}
        self.latest_applicability = apply_applicability_control(
            case_state=self.case_state,
            memory_query=self.latest_query,
            retrieval_result=self.latest_retrieval,
            mode=self.applicability_mode,
            llm_client=self.memory_llm,
            debug=memory_debug.get("applicability") if memory_debug is not None else None,
            strict=self.strict_memory_errors,
        )
        self.latest_guidance = build_memory_guidance(
            self.latest_applicability,
            self.latest_retrieval,
        )
        # Filter out memories that have already been injected too many times this episode
        self.latest_guidance = self._deduplicate_guidance(self.latest_guidance)
        if memory_debug is not None:
            memory_debug["guidance"] = {
                "structured": self.latest_guidance.to_dict(),
                "text": guidance_to_text(self.latest_guidance),
            }

    def _deduplicate_guidance(self, guidance: MemoryGuidance) -> MemoryGuidance:
        """Filter out memories already injected too many times this episode."""
        from .utils.config import GUIDANCE_CONFIG

        max_inject = GUIDANCE_CONFIG.get("max_repeated_injections_per_memory", 1)
        filtered = []
        for mem in guidance.selected_memories:
            mid = mem.get("memory_id", "")
            count = self._injected_memory_ids.get(mid, 0)
            if count < max_inject:
                filtered.append(mem)
                self._injected_memory_ids[mid] = count + 1
        return MemoryGuidance(selected_memories=filtered)

    def _inject_guidance(self, observation: Any) -> Any:
        if self.disable_memory:
            return observation
        guidance_text = guidance_to_text(self.latest_guidance) if self.latest_guidance else ""
        case_memory_text = self._case_memory_for_doctor_text() if self.inject_case_memory else ""
        if not guidance_text and not case_memory_text:
            return observation
        if isinstance(observation, dict):
            enriched = dict(observation)
            if guidance_text:
                enriched["memory_guidance"] = guidance_text
                enriched["memory_guidance_structured"] = self.latest_guidance.to_dict()
            if case_memory_text:
                enriched["case_memory"] = case_memory_text
            return enriched
        if isinstance(observation, str):
            blocks = [observation]
            if case_memory_text:
                blocks.append("[Case Memory]\n" + case_memory_text)
            if guidance_text:
                blocks.append("[Memory Guidance]\n" + guidance_text)
            return "\n\n".join(blocks)
        return observation

    def _case_memory_for_doctor_text(self) -> str:
        if self.latest_query is None:
            return ""
        query_debug = (self.latest_memory_debug or {}).get("query_builder") or {}
        case_memory = (
            query_debug.get("final_case_memory")
            or query_debug.get("classified_case_memory")
            or query_debug.get("case_memory")
            or {}
        )
        if not isinstance(case_memory, dict):
            return ""
        lines: list[str] = []
        chief = str(case_memory.get("chief_complaint") or "").strip()
        if chief:
            lines.append(f"Presenting problem: {chief}")
        goal = str(case_memory.get("diagnostic_strategy") or case_memory.get("diagnosis_goal") or "").strip()
        if goal:
            lines.append(f"Diagnostic strategy: {goal}")
        efficient = [
            str(item).strip()
            for item in (case_memory.get("efficient_turn_information") or [])[-6:]
            if str(item).strip()
        ]
        if efficient:
            lines.append("Evidence so far:\n" + "\n".join(f"- {item}" for item in efficient))
        prior = str(case_memory.get("prior_information_summary") or "").strip()
        if prior:
            lines.append(f"Prior summary: {prior}")
        if self.latest_query.query_text:
            lines.append(f"Memory retrieval intent: {self.latest_query.query_text}")
        return "\n".join(lines)

    def _candidate_actions(self) -> list[str]:
        raw_tools = getattr(self, "tools", None) or []
        if hasattr(raw_tools, "tools"):
            tools = getattr(raw_tools, "tools") or []
        elif isinstance(raw_tools, dict):
            tools = raw_tools.keys()
        else:
            tools = raw_tools
        actions: list[str] = []
        for tool in tools:
            name = str(tool)
            action = ACTION_MAP.get(name)
            if action:
                actions.append(action)
        if not actions:
            if self.strict_memory_errors:
                raise RuntimeError("Cannot derive candidate memory actions: no registered tools matched MEMORY_ACTION_CONFIG")
            actions = list(DEFAULT_ACTIONS)
        for action in MEMORY_ACTION_CONFIG.get("always_include_actions", []):
            if action not in actions:
                actions.append(str(action))
        return list(dict.fromkeys(actions))

    # ------------------------------------------------------------------
    # Turn records
    # ------------------------------------------------------------------
    def _finalize_pending_turn(
        self,
        env_observation: Any,
        env_info: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        if not self.pending_selected_action:
            return
        if self.case_state is None:
            return

        record = TurnRecord(
            episode_id=self.episode_id,
            case_id=self.case_state.case_id,
            turn_id=self.case_state.turn_id,
            case_state=self.case_state,
            memory_query=self.latest_query,
            retrieval_result=self.latest_retrieval,
            applicability_result=self.latest_applicability,
            memory_guidance=self.latest_guidance,
            clinical_turn=self._build_clinical_turn(
                self.pending_selected_action,
                env_observation,
                env_info,
                reward,
                done,
            ),
            selected_action=dict(self.pending_selected_action),
            memory_debug=dict(self.latest_memory_debug or {}),
            env_observation=self._safe_payload(env_observation),
            env_info=self._safe_payload(env_info),
            reward=float(reward or 0.0),
            done=bool(done),
        )
        self.turn_records.append(record)
        if self.log_memory_trace:
            append_memory_turn_trace(self.trace_root, record.to_dict())
        self.pending_selected_action = {}

    def _parse_tool_call_text(self, value: Any) -> dict[str, Any]:
        text = str(value or "").strip()
        if not text:
            return {}
        start = text.find("<tool_call>")
        end = text.find("</tool_call>")
        if start >= 0 and end > start:
            text = text[start + len("<tool_call>"):end].strip()
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        args = parsed.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}
        return {
            "tool_name": str(parsed.get("name") or ""),
            "arguments": args,
        }

    def _extract_final_diagnosis_from_action(self, action: dict[str, Any]) -> str:
        parsed = self._parse_tool_call_text(action.get("raw"))
        args = parsed.get("arguments") if isinstance(parsed, dict) else {}
        if not isinstance(args, dict):
            args = {}

        for key in ("diagnosis", "final_diagnosis"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("final_response", "result", "response"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                boxed = _extract_boxed_diagnosis(value)
                return boxed or value.strip()

        raw = str(action.get("raw") or action.get("action_label") or "")
        return _extract_boxed_diagnosis(raw) or raw.strip()

    def _is_final_diagnosis_correct_by_reward(
        self,
        final_diagnosis: str,
        gold_diagnosis: str,
        info: dict[str, Any],
    ) -> bool:
        if not gold_diagnosis:
            return False

        judge_model_name = str(
            info.get("judge_model_name")
            or self._case_bundle.get("judge_model_name")
            or ""
        ).strip()
        judge_base_url = str(
            info.get("judge_base_url")
            or self._case_bundle.get("judge_base_url")
            or ""
        ).strip()
        judge_api_key = (
            info.get("judge_api_key")
            or self._case_bundle.get("judge_api_key")
            or self._case_bundle.get("api_key")
            or "None"
        )

        if judge_model_name and judge_base_url:
            _, metadata = result_reward_judge(
                f"The final diagnosis is: \\boxed{{{final_diagnosis}}}.",
                gold_diagnosis,
                judge_model_name=judge_model_name,
                judge_base_url=judge_base_url,
                api_key=judge_api_key,
            )
            return bool(metadata.get("judge_consistent", False))

        _, metadata = result_reward(
            f"The final diagnosis is: \\boxed{{{final_diagnosis}}}.",
            gold_diagnosis,
        )
        return bool(metadata.get("ground_truth_contained", False))

    def _build_clinical_turn(
        self,
        selected_action: dict[str, Any],
        env_observation: Any,
        env_info: dict[str, Any],
        reward: float,
        done: bool,
    ) -> dict[str, Any]:
        action = dict(selected_action or {})
        parsed_tool = self._parse_tool_call_text(action.get("raw"))
        tool_name = parsed_tool.get("tool_name") or str(
            action.get("tool_name") or action.get("name") or ""
        )
        arguments = parsed_tool.get("arguments") or action.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"raw": str(arguments)}

        observation = self._safe_payload(env_observation, max_chars=1200)
        info = self._safe_payload(env_info, max_chars=1200)
        turn_importance = self._compute_turn_importance(
            tool_name=tool_name,
            arguments=arguments,
            observation=observation,
            env_info=env_info,
            reward=reward,
        )
        return {
            "turn_id": self.case_state.turn_id if self.case_state else 0,
            "doctor_action_type": action.get("action_type") or "",
            "tool_name": tool_name,
            "arguments": arguments,
            "patient_or_tool_response": observation,
            "env_info": info,
            "reward": float(reward or 0.0),
            "done": bool(done),
            "conf_before": turn_importance.get("conf_before", 0.0),
            "conf_after": turn_importance.get("conf_after", 0.0),
            "delta": turn_importance.get("delta", 0.0),
            "turn_reward": turn_importance.get("turn_reward", 0.0),
            "importance": turn_importance.get("importance", 0.0),
            "turn_importance": turn_importance,
        }

    # ------------------------------------------------------------------
    # Offline write
    # ------------------------------------------------------------------
    def _finalize_episode_if_needed(self, reward: float, info: dict[str, Any]) -> None:
        if self.episode_finalized:
            return
        if self.case_state is None:
            logger.warning(
                "Episode %s done but case_state is None — skipping offline write",
                self.episode_id,
            )
            return
        if self.disable_memory_write:
            self.episode_finalized = True
            return

        # ── Extract fields from env info dict ─────────────────────────
        # Total reward from trajectory (sum of step rewards) or fallback
        total_reward = float(
            sum(record.reward for record in self.turn_records) or reward or 0.0
        )

        # Final diagnosis: extract from last turn record's selected_action
        final_diagnosis = ""
        if self.turn_records:
            last_action = self.turn_records[-1].selected_action or {}
            final_diagnosis = self._extract_final_diagnosis_from_action(last_action)

        gold_diagnosis = self._gold_diagnosis_from_context(info)

        # Keep memory extraction success identical to diagnose_acc.py summary:
        # use med_diagnosis_reward.py result reward logic, including judge mode
        # when judge_model_name + judge_base_url are configured.
        is_correct = self._is_final_diagnosis_correct_by_reward(
            final_diagnosis,
            gold_diagnosis,
            info,
        )

        summary = str(
            info.get("summary")
            or (info.get("metadata") or {}).get("result_reward_reason")
            or ""
        )

        feedback = EpisodeFeedback(
            episode_id=self.episode_id,
            case_id=self.case_state.case_id,
            success=is_correct,
            total_reward=total_reward,
            final_diagnosis=final_diagnosis,
            gold_diagnosis=gold_diagnosis,
            summary=summary,
        )
        trajectory = {
            "info": {
                "memory_agent": {
                    "turn_records": [record.to_dict() for record in self.turn_records]
                }
            }
        }
        distilled = distill_from_trajectory(trajectory, feedback)
        write_memory_from_distilled_episode(
            distilled,
            root_dir=self.memory_root,
            experience_extraction_mode=self.experience_extraction_mode,
            experience_merge_mode=self.experience_merge_mode,
            llm_client=self.memory_llm,
        )
        self.episode_finalized = True

    # ------------------------------------------------------------------
    # Parsing / safety
    # ------------------------------------------------------------------
    def _parse_selected_action(self, model_output: Any) -> dict[str, Any]:
        if isinstance(model_output, dict):
            action_type = (
                model_output.get("action_type")
                or model_output.get("tool")
                or model_output.get("name")
                or model_output.get("action")
                or ""
            )
            action_label = (
                model_output.get("action_label")
                or model_output.get("argument")
                or model_output.get("args")
                or model_output.get("content")
                or action_type
                or ""
            )
            return {
                "action_type": self._normalize_action_type(str(action_type)),
                "action_label": str(action_label),
                "raw": model_output,
            }

        text = str(model_output or "")
        return {
            "action_type": self._infer_action_type_from_text(text),
            "action_label": text[:240],
            "raw": text,
        }

    def _normalize_action_type(self, action: str) -> str:
        action = action.strip()
        if not action:
            return ""
        upper = action.upper()
        if upper in DEFAULT_ACTIONS:
            return upper
        mapped = ACTION_MAP.get(action.lower())
        if mapped:
            return mapped
        return upper

    def _infer_action_type_from_text(self, text: str) -> str:
        lowered = text.lower()
        if "diagnosis" in lowered or "final" in lowered:
            return FINALIZE_ACTION
        if "cxr" in lowered or "x-ray" in lowered or "image" in lowered:
            return "REVIEW_IMAGE"
        if "lab" in lowered or "test" in lowered or "exam" in lowered:
            return "REQUEST_LAB"
        if "ask" in lowered or "patient" in lowered or "question" in lowered:
            return "ASK"
        return "UPDATE_HYPOTHESIS"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _case_bundle_from(self, observation: Any, info: dict[str, Any]) -> Any:
        return info.get("case") or info.get("task") or info.get("case_bundle") or observation or {}

    def _episode_id_from_bundle(self, bundle: dict[str, Any], case_id: str) -> str:
        case_text = str(case_id or bundle.get("case_id") or "unknown_case").strip()
        repeat = bundle.get("_repeat_id")
        if repeat is None:
            repeat = bundle.get("repeat_id")
        if repeat is None:
            repeat = bundle.get("repeat")
        if repeat is not None and str(repeat).strip() != "":
            return f"case_{case_text}_repeat_{repeat}"
        return f"case_{case_text}"

    def _safe_payload(self, value: Any, max_chars: int = 4000) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._safe_payload(item, max_chars=max_chars)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._safe_payload(item, max_chars=max_chars) for item in value[:50]]
        text = str(value)
        if len(text) > max_chars:
            return text[:max_chars] + "...[truncated]"
        return value

    def _gold_diagnosis_from_context(self, info: dict[str, Any] | None = None) -> str:
        info = info or {}
        bundle_ehr = (
            self._case_bundle.get("ehr")
            or (self._case_bundle.get("context") or {}).get("ehr")
            or (self._case_bundle.get("medenv_case_bundle") or {}).get("ehr")
            or {}
        )
        if isinstance(bundle_ehr, dict):
            gold_diagnosis = str(
                bundle_ehr.get("Correct_Diagnosis")
                or bundle_ehr.get("Principal_Diagnosis")
                or bundle_ehr.get("Final_Result")
                or ""
            ).strip()
            if gold_diagnosis:
                return gold_diagnosis
        return str(
            info.get("gold_diagnosis")
            or info.get("ground_truth")
            or self._case_bundle.get("ground_truth")
            or self._case_bundle.get("gold_diagnosis")
            or ""
        ).strip()

    def _response_text(self, value: Any, max_chars: int = 1200) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            if "question" in value:
                return self._response_text(value.get("question"), max_chars=max_chars)
            if "tool_outputs" in value and isinstance(value.get("tool_outputs"), dict):
                return "\n".join(
                    self._response_text(item, max_chars=max_chars)
                    for item in value.get("tool_outputs", {}).values()
                ).strip()
            return json.dumps(value, ensure_ascii=False)[:max_chars]
        if isinstance(value, list):
            return "\n".join(self._response_text(item, max_chars=max_chars) for item in value[-5:]).strip()
        text = str(value).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "...[truncated]"
        return text

    def _dialogue_text_from_turn_records(self) -> str:
        parts: list[str] = []
        for record in self.turn_records:
            clinical = record.clinical_turn or {}
            tool_name = str(clinical.get("tool_name") or "").strip()
            arguments = clinical.get("arguments") or {}
            response = self._response_text(clinical.get("patient_or_tool_response"))
            if isinstance(arguments, dict):
                question = str(arguments.get("question") or "").strip()
                args_text = json.dumps(arguments, ensure_ascii=False)
            else:
                question = ""
                args_text = str(arguments)

            if question:
                parts.append(f"doctor[{tool_name}]: {question}")
            elif tool_name:
                parts.append(f"doctor[{tool_name}]: {args_text}")
            if response:
                parts.append(f"response[{tool_name}]: {response}")
        return "\n".join(parts)

    def _compute_turn_importance(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        observation: Any,
        env_info: dict[str, Any],
        reward: float,
    ) -> dict[str, Any]:
        gold_diagnosis = self._gold_diagnosis_from_context(env_info)
        dialogue_before = self._dialogue_text_from_turn_records()
        answer = self._response_text(observation)
        question = ""
        if isinstance(arguments, dict):
            question = str(arguments.get("question") or "").strip()

        conf_before = _confidence_heuristic(dialogue_before, gold_diagnosis)
        context_after = f"{dialogue_before}\n{answer}".strip()
        conf_after = _confidence_heuristic(context_after, gold_diagnosis)
        delta = conf_after - conf_before
        turn_reward = max(-0.1, min(0.2, delta * 0.2))
        return {
            "method": "med_diagnosis_reward_confidence_delta",
            "applies_to": "all_turns",
            "available": bool(gold_diagnosis and answer),
            "tool_name": tool_name,
            "question": question,
            "answer_excerpt": answer[:500],
            "gold_diagnosis": gold_diagnosis,
            "conf_before": conf_before,
            "conf_after": conf_after,
            "delta": delta,
            "turn_reward": turn_reward,
            "importance": abs(delta),
            "env_reward": float(reward or 0.0),
        }

    def _has_memory_snapshot(self) -> bool:
        return (
            self.case_state is not None
            and self.latest_query is not None
            and self.latest_retrieval is not None
            and self.latest_applicability is not None
            and self.latest_guidance is not None
        )

    def reset_memory(self) -> None:
        self.episode_id = f"pending_episode_{uuid.uuid4().hex[:10]}"
        self.case_state = None
        self.latest_query = None
        self.latest_retrieval = None
        self.latest_applicability = None
        self.latest_guidance = None
        self.pending_selected_action = {}
        self.turn_records = []
        self.episode_finalized = False
        self._injected_memory_ids = {}

    def reset(self) -> Any:
        self.reset_memory()
        return self._call_base("reset")
