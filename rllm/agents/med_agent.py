from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import Any, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT
from rllm.parser import ToolParser, get_tool_parser
from rllm.tools.mcp_tool import MCPTool
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool
from rllm.agents.tool_agent import ToolAgent

logger = logging.getLogger(__name__)


def _cxr_tool_output_to_multimodal(tool_output_str: str) -> tuple[str | None, list[dict] | None]:
    """
    解析 CXR 工具返回的 JSON：若包含 base64 图片，则生成 (tool 短摘要, user 多模态 content 列表)。
    若无法解析或非 CXR 带图输出，返回 (None, None)，调用方应使用原始字符串作为 tool content。

    Returns:
        (tool_content_summary, user_content_parts) 或 (None, None)
        - user_content_parts 为 OpenAI 多模态格式: [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}, ...]
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
    # 至少有一张图带 image_base64 才当作 CXR 带图结果
    has_b64 = any(isinstance(img, dict) and img.get("image_base64") for img in images)
    if not has_b64:
        return None, None

    summary = data.get("summary") or ""
    view_list = []
    content_parts: list[dict] = [{"type": "text", "text": summary}]

    for i, img in enumerate(images):
        if not isinstance(img, dict):
            continue
        b64 = img.get("image_base64")
        vp = img.get("view_position", "unknown")
        view_list.append(vp)
        if b64:
            mime = (img.get("mime") or "image/jpeg").strip().lower()
            if "jpeg" in mime or "jpg" in mime:
                url = f"data:image/jpeg;base64,{b64}"
            elif "png" in mime:
                url = f"data:image/png;base64,{b64}"
            else:
                url = f"data:image/jpeg;base64,{b64}"
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

    tool_summary = summary or f"CXR returned {len(images)} image(s). Views: {', '.join(view_list)}. Images attached below."
    return tool_summary, content_parts


class MedicalAgent(BaseAgent):
    """
    A tool agent that can use tools AND do multi-turn dialogue.

    Key behavior change (vs old):
    - If no tool call is parsed, by default we return the raw text as the action (dialogue turn),
      instead of forcing a 'finish' tool call.
    """

    def __init__(
        self,
        system_prompt: str = TOOL_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        auto_finish_on_no_tool: bool = False,
        append_assistant_message: bool = True,
    ):
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.system_prompt = system_prompt
        self.auto_finish_on_no_tool = auto_finish_on_no_tool
        self.append_assistant_message = append_assistant_message

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        # Tools prompt (schema) injected into system message
        self.tools_prompt = self.tool_parser.get_tool_prompt(
            json.dumps(self.tools.json, indent=2)
        )

        # State
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Format environment observation into chat messages."""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                # treated as user turn (e.g., patient reply)
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                # treated as tool messages；CXR 带图输出会再跟一条 user 多模态消息，让 vision 模型能看图
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    tool_summary, user_content_parts = _cxr_tool_output_to_multimodal(tool_output_str)
                    if tool_summary is not None and user_content_parts is not None:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_summary,
                                "tool_call_id": tool_call_id,
                            }
                        )
                        messages.append({"role": "user", "content": user_content_parts})
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_output_str,
                                "tool_call_id": tool_call_id,
                            }
                        )
            else:
                messages.append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs is not None:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update agent state based on env feedback."""
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

        # backfill last step's reward/done/info
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info

    def _build_finish_tool_call(self, assistant_content: str) -> list[dict]:
        return [
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": {"response": assistant_content},
                },
            }
        ]

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Parse model response into:
        - tool calls (list[dict]) if parsable
        - otherwise a dialogue action (str) by default
        """
        assistant_content = response
        tool_calls_dict: list[dict] = []
        parse_error: Optional[str] = None

        # Try parse tool calls
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"
            tool_calls_dict = []

        # Append assistant message to history (keep conversational context)
        if self.append_assistant_message:
            self.messages.append({"role": "assistant", "content": assistant_content})

        # If tool calls exist, normalize arguments to JSON string (some env/tool layer expects str)
        if tool_calls_dict:
            for call in tool_calls_dict:
                fn = call.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args, ensure_ascii=False)
            action_to_env: Any = tool_calls_dict
        else:
            # No tool calls parsed
            if parse_error:
                logger.warning(f"ToolAgent: failed to parse tool calls, fallback. {parse_error}")

            if self.auto_finish_on_no_tool:
                # old behavior: end episode
                action_to_env = self._build_finish_tool_call(assistant_content)
            else:
                # NEW behavior: treat as dialogue turn (ask patient / reasoning to continue)
                action_to_env = assistant_content

        # Record trajectory step
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_to_env,                 # can be list[dict] OR str
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        # print(Action(action=action_to_env))

        return Action(action=action_to_env)

    def reset(self):
        """Reset for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self.system_prompt + self.tools_prompt}
        ]
        self.current_observation = None

    @property
    def chat_completions(self) -> list[dict[str, Any]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory


class MCPToolAgent(ToolAgent):
    """
    A ToolAgent variant that uses MCPTool registry / map.

    Fixes the original code bug:
    - original default `tool_map=list[MCPTool]` is a typing annotation, not a runtime dict.
    """

    def __init__(
        self,
        system_prompt: str = TOOL_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        tool_map: Any = None,
        auto_finish_on_no_tool: bool = False,
        append_assistant_message: bool = True,
    ):
        # Normalize tool_map into a dict-like structure of tools
        # Accept:
        # - dict[str, MCPTool] or dict[str, type[MCPTool]]
        # - list[MCPTool] or list[type[MCPTool]]
        if tool_map is None:
            # 如果 MCPTool 内部有注册表，你可以在这里改成 MCPTool.registry 之类
            # 这里先默认空
            normalized_tools_json = []
        else:
            if isinstance(tool_map, dict):
                values = list(tool_map.values())
            elif isinstance(tool_map, list):
                values = tool_map
            else:
                raise TypeError("tool_map must be a dict or a list for MCPToolAgent")

            normalized_tools_json = []
            for t in values:
                # t can be instance or class; both should expose `.json` in your implementation
                try:
                    normalized_tools_json.append(t.json)  # type: ignore[attr-defined]
                except Exception:
                    # fallback: if it's a class, try instantiate without args (if possible)
                    try:
                        inst = t()  # type: ignore[call-arg]
                        normalized_tools_json.append(inst.json)
                    except Exception as e:
                        raise TypeError(f"Invalid MCPTool entry: {t}. Error: {e}") from e

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        tool_parser = parser_class()
        tools_prompt = tool_parser.get_tool_prompt(json.dumps(normalized_tools_json, indent=2, ensure_ascii=False))


        self.system_prompt = system_prompt
        self.auto_finish_on_no_tool = auto_finish_on_no_tool
        self.append_assistant_message = append_assistant_message
        self.tool_parser = tool_parser
        self.tools_prompt = tools_prompt

        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()
