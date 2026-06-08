import asyncio
import base64
import logging
import os
from io import BytesIO

import openai
import httpx
from PIL import Image
from urllib.parse import urlparse

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str = "", tokenizer=None, max_prompt_length: int = 4096, max_response_length: int = 4096, max_model_length: int | None = None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), sampling_params: dict | None = None, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, use_chat_completions: bool | None = None, **kwargs):
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = self.sampling_params.pop("reasoning_effort", "medium")

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = None

        if use_chat_completions is None:
            use_chat_completions = self.tokenizer is None
        self._use_chat_completions = use_chat_completions
        if self.tokenizer is None and not self._use_chat_completions:
            raise ValueError("OpenAIEngine requires a tokenizer unless use_chat_completions=True.")
        if self.tokenizer is None:
            # In this case, we cannot enforce max prompt length or dynamically adjust max_tokens <= max_response_length if needed
            print("No tokenizer provided to OpenAIEngine, will use the chat completions endpoint.")

        parsed_base_url = urlparse(base_url or "")
        is_local_base_url = parsed_base_url.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}
        http_client = httpx.AsyncClient(trust_env=False) if is_local_base_url else None
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _convert_messages_to_openai_format(self, messages: list[dict], strip_tool_messages: bool = False) -> list[dict]:
        """Convert messages from rllm format to OpenAI multimodal format.
        - If message has 'images' (PIL list), build content list from them.
        - If message['content'] is already a list (e.g. CXR tool → user message with image_url parts), pass through.
        - XML tool-call agents store environment outputs as role=tool, but chat
          APIs require role=tool only after native OpenAI tool_calls. Convert
          those tool outputs back into user-visible XML text for compatibility.
        - When strip_tool_messages=True, always convert role=tool to role=user
          (used when tools are intentionally omitted, e.g. tool_choice="none").
        """
        convert_tool_messages = strip_tool_messages or not self.tools
        converted_messages = []
        for message in messages:
            if message.get("role") == "tool" and convert_tool_messages:
                content = message.get("content") or ""
                converted_messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{content}\n</tool_response>",
                    }
                )
                continue

            if "images" in message and message["images"]:
                content = [{"type": "text", "text": message.get("content") or ""}]
                for img in message["images"]:
                    base64_image = self._pil_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

                converted_messages.append({"role": message["role"], "content": content})
            else:
                converted_messages.append(message)

        return converted_messages

    @staticmethod
    def _validate_native_tool_history(messages: list[dict]) -> None:
        """Require every assistant tool call to be answered by adjacent tool messages."""
        for index, message in enumerate(messages):
            tool_calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not tool_calls:
                continue

            expected_ids = [str(call.get("id") or "") for call in tool_calls]
            if any(not call_id for call_id in expected_ids):
                raise RuntimeError(
                    f"Assistant tool_calls message at index {index} contains a missing tool_call_id"
                )

            received_ids: list[str] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                received_ids.append(str(messages[cursor].get("tool_call_id") or ""))
                cursor += 1

            if sorted(received_ids) != sorted(expected_ids):
                next_role = messages[index + 1].get("role") if index + 1 < len(messages) else "<end>"
                raise RuntimeError(
                    "Invalid native tool-call history before API request: "
                    f"assistant message index={index}, expected tool_call_ids={expected_ids}, "
                    f"adjacent received tool_call_ids={received_ids}, next_role={next_role!r}"
                )

    def _prepare_max_tokens_param(self, sampling_params: dict, prompt_length: int = None) -> dict:
        """Prepare max tokens parameter for API call (supports O3's max_completion_tokens)."""
        if "max_completion_tokens" in sampling_params:
            return {"max_completion_tokens": sampling_params.pop("max_completion_tokens")}

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        # Adjust for prompt length if provided (completion method needs this)
        if prompt_length and self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining
                print(f"Warning: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return {"max_tokens": max_tokens}

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("enforce_max_prompt_length", None)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        request_tools = sampling_params.pop("tools", self.tools)
        request_tool_choice = sampling_params.pop(
            "tool_choice",
            "required" if request_tools else None,
        )

        # When tool_choice="none", strip tools from the request entirely.
        # Some providers (e.g. DeepSeek) ignore tool_choice="none" and still
        # call tools if tools are present, returning non-text output (DSML).
        # Removing tools forces a pure text completion.  We must also convert
        # any role=tool history messages to role=user so the API accepts them.
        no_tools_mode = request_tool_choice == "none"
        if no_tools_mode:
            request_tools = []
            request_tool_choice = None

        create_params = self._prepare_max_tokens_param(sampling_params)
        converted_messages = self._convert_messages_to_openai_format(
            messages, strip_tool_messages=no_tools_mode
        )
        if request_tools:
            self._validate_native_tool_history(converted_messages)

        retries = self.api_retries
        while retries > 0:
            try:
                tool_params = (
                    {"tools": request_tools, "tool_choice": request_tool_choice}
                    if request_tools
                    else {}
                )
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=converted_messages,
                    timeout=3600,
                    **tool_params,
                    **create_params,
                    **sampling_params,
                )

                content = response.choices[0].message.content
                reasoning = response.choices[0].message.reasoning if hasattr(response.choices[0].message, "reasoning") and isinstance(response.choices[0].message.reasoning, str) else ""
                tool_calls = response.choices[0].message.tool_calls if hasattr(response.choices[0].message, "tool_calls") and isinstance(response.choices[0].message.tool_calls, list) else []

                # Build text with reasoning if available, otherwise use content
                if reasoning:
                    text = f"{THOUGHT_DELIMITER_START}\n{reasoning}\n{THOUGHT_DELIMITER_END}\n\n{content or ''}"
                else:
                    text = content or ""

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                return ModelOutput(
                    text=text or "",
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    prompt_ids=[],
                    completion_ids=[],
                    logprobs=[],
                    prompt_logprobs=[],
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                    request_messages=converted_messages,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except openai.APIStatusError as e:
                status_code = int(getattr(e, "status_code", 0) or 0)
                if 400 <= status_code < 500 and status_code not in {408, 409, 429}:
                    raise RuntimeError(
                        f"Non-retryable chat-completion error ({status_code}): {e}"
                    ) from e
                retries -= 1
                if retries == 0:
                    raise Exception(f"API error after retries: {e}") from e
                print(f"Retryable API error: {e}, retrying...")
                await asyncio.sleep(1)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def completion(self, prompt: str | list[int], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        # completion 端点不支持 tool_choice / tools，静默丢弃
        sampling_params.pop("tool_choice", None)
        sampling_params.pop("tools", None)

        if isinstance(prompt, list):
            prompt_ids = prompt
        else:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        create_params = self._prepare_max_tokens_param(sampling_params, prompt_length)
        sampling_params.update(create_params)

        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.completions.create(model=self.model, prompt=prompt, **sampling_params)
                text = response.choices[0].text
                try:
                    completion_ids = response.choices[0].token_ids
                    assert completion_ids is not None
                except Exception:
                    completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

                parsed_output = self.chat_parser.parse_completion(completion_ids)

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                try:
                    assert response.choices[0].logprobs is not None
                    logprobs = response.choices[0].logprobs.token_logprobs
                except Exception:
                    logprobs = []

                try:
                    assert response.choices[0].prompt_logprobs is not None
                    prompt_logprobs: list[float] = [None]
                    for tid, lp in zip(prompt_ids[1:], response.choices[0].prompt_logprobs[1:], strict=False):
                        prompt_logprobs.append(float(lp[str(tid)]["logprob"]))
                except Exception:
                    prompt_logprobs = []

                return ModelOutput(
                    text=text,
                    content=parsed_output["content"],
                    reasoning=parsed_output["reasoning"],
                    tool_calls=parsed_output["tool_calls"],
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                    request_prompt=prompt if isinstance(prompt, str) else None,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        if self._use_chat_completions:
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            if accumulate_reasoning:
                raise ValueError("Accumulate reasoning is not supported for chat completions endpoint.")
            return await self.chat_completion(messages, **kwargs)
        else:
            tools = kwargs.pop("tools", self.tools)
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning, reasoning_effort=reasoning_effort)
            return await self.completion(prompt, **kwargs)
