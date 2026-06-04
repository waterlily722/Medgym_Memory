import time
import re
import logging
import json
from dataclasses import dataclass
from typing import Any, Optional

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class FinalizeAtMaxStepsResult:
    final_response: str
    termination_reason: str
    llm_time: float
    total_time: float
    response_token_len: int
    response_tokens: Any = None
    response_masks: Any = None
    tool_output: ToolOutput | None = None
    debug: dict[str, Any] | None = None


class DiagnosisTool(Tool):
    NAME = "diagnosis"
    DESCRIPTION = (
        "Submit the final diagnosis and terminate the episode. "
        "You MUST provide `final_response` as exactly one line: "
        "The final diagnosis is: \\\\boxed{<diagnosis>}."
    )

    _LINE_RE = re.compile(
        r"The final diagnosis is:\s*\\box(?:ed)?\{(.+?)\}\s*\.?\s*$",
        re.IGNORECASE,
    )
    _BOX_RE = re.compile(r"\\box(?:ed)?\{(.+?)\}")

    def __init__(self, name: str | None = None, description: str | None = None, **_ignored_kwargs):
        super().__init__(name=name or self.NAME, description=description or self.DESCRIPTION)

        self.tokenizer = None
        self.chat_parser = None
        self.get_model_response = None
        self.convert_messages_to_tokens_and_masks = None

        self.enforce_max_prompt_length = False
        self.max_prompt_length = None
        self.max_response_length = None
        self.trajectory_timeout = None

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_response": {
                            "type": "string",
                            "description": "MUST be exactly one line: The final diagnosis is: \\\\boxed{xxx}.",
                        },
                        "done": {
                            "type": "boolean",
                            "description": "Whether to terminate the episode (default true).",
                        },
                        "print_result": {
                            "type": "boolean",
                            "description": "Whether to print parsed diagnosis to stdout.",
                        },
                    },
                    "required": ["final_response"],
                },
            },
        }


    @classmethod
    def extract_boxed_diagnosis(cls, text: str) -> Optional[str]:
        """从 'The final diagnosis is: \\\\boxed{xxx}.' 中提取 xxx；若无该行则从任意 \\boxed{xxx} 提取。"""
        if not text:
            return None
        text = text.split("</think>")[-1].strip()

        m = cls._LINE_RE.search(text)
        if m:
            return m.group(1).strip()
        m = cls._BOX_RE.search(text)
        if m:
            return m.group(1).strip()
        return None

    def forward(self, final_response: str | None = None, done: bool = True, print_result: bool = True, **kwargs) -> ToolOutput:
        try:
            diag = self.extract_boxed_diagnosis(final_response or "")
            if not diag:
                return ToolOutput(
                    name=self.name,
                    error="Missing/invalid final_response. Expect exactly: The final diagnosis is: \\\\boxed{xxx}.",
                )

            if print_result:
                print(f"[DiagnosisTool] diagnosis = {diag}")

            return ToolOutput(
                name=self.name,
                output={"diagnosis": diag, "done": bool(done)},
                metadata={"parsed_from": "final_response"},
            )
        except Exception as e:
            logger.exception("DiagnosisTool.forward failed: %s", e)
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")


    def bind(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    async def afinalize_at_max_steps(
        self,
        agent,
        application_id: str,
        kwargs: dict[str, Any] | None = None,
        episode_steps: list[dict[str, Any]] | None = None,
        mode: str = "Text",
        response_token_len: int = 0,
        response_tokens: Any = None,
        response_masks: Any = None,
        llm_time: float = 0.0,
        total_time: float = 0.0,
    ) -> FinalizeAtMaxStepsResult:

        kwargs = kwargs or {}
        episode_steps = episode_steps or []
        dbg: dict[str, Any] = {"stage": "afinalize_at_max_steps"}

        last_response = ""
        if episode_steps and isinstance(episode_steps[-1], dict):
            last_response = str(episode_steps[-1].get("response") or "")
        dbg["has_last_response"] = bool(last_response)

        diag = self.extract_boxed_diagnosis(last_response) if last_response else None
        dbg["parsed_from_last_response"] = bool(diag)

        final_response = last_response

        if not diag:
            can_finalize = callable(getattr(self, "get_model_response", None))
            dbg["has_get_model_response"] = bool(can_finalize)

            if can_finalize:
                try:
                    prompt_messages = list(getattr(agent, "chat_completions", []) or [])
                except Exception:
                    prompt_messages = []

                prompt_messages = prompt_messages + [
                    {
                        "role": "user",
                        "content": (
                            "We are at the last step. "
                            "Output ONE line only in the exact format:\n"
                            "The final diagnosis is: \\boxed{<final diagnosis>}."
                        ),
                    }
                ]

                finalize_kwargs = dict(kwargs)
                finalize_kwargs["temperature"] = 0.0
                finalize_kwargs["max_tokens"] = min(int(finalize_kwargs.get("max_tokens", 256)), 512)
                finalize_kwargs["tools"] = [self.json]
                finalize_kwargs["tool_choice"] = "required"

                t0 = time.time()
                model_out = await self.get_model_response(prompt_messages, application_id, **finalize_kwargs)
                dt = time.time() - t0
                llm_time += dt
                total_time += dt

                if hasattr(model_out, "text"):
                    final_response = str(model_out.text or "")
                else:
                    final_response = str(model_out or "")

                for call in getattr(model_out, "tool_calls", None) or []:
                    function = getattr(call, "function", None)
                    name = str(getattr(function, "name", "") or "")
                    raw_arguments = getattr(function, "arguments", "{}") or "{}"
                    if name != self.name:
                        continue
                    try:
                        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    except Exception:
                        arguments = {}
                    if isinstance(arguments, dict):
                        final_response = str(arguments.get("final_response") or "")
                    break

                dbg["finalize_extra_llm_call"] = True
                dbg["finalize_extra_llm_dt"] = dt

                diag = self.extract_boxed_diagnosis(final_response)
                dbg["parsed_from_finalize_call"] = bool(diag)
            else:
                dbg["finalize_extra_llm_call"] = False

        else:
            dbg["finalize_extra_llm_call"] = False

        if not diag:
            raise RuntimeError(
                "Max-step diagnosis finalization failed to produce a valid diagnosis tool call"
            )
        dbg["fallback_diagnosis_used"] = False

        tool_out = self.forward(diagnosis=diag, final_response=final_response, done=True, print_result=True)

        term = "MAX_STEPS_DIAGNOSIS_TOOL_CALLED" if not tool_out.error else "MAX_STEPS_DIAGNOSIS_TOOL_ERROR"

        return FinalizeAtMaxStepsResult(
            final_response=final_response,
            termination_reason=term,
            llm_time=llm_time,
            total_time=total_time,
            response_token_len=response_token_len,
            response_tokens=response_tokens,
            response_masks=response_masks,
            tool_output=tool_out,
            debug=dbg,
        )
