from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
import uuid
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


MODEL = None
TOKENIZER = None
SERVED_MODEL_NAME = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATION_LOCK = threading.Lock()


def _message_text(messages: list[dict[str, Any]]) -> str:
    if TOKENIZER is not None and getattr(TOKENIZER, "chat_template", None):
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _is_tool_call_request(messages: list[dict[str, Any]]) -> bool:
    text = "\n".join(_content_text(msg.get("content")) for msg in messages)
    return "<tools>" in text and "tool_call" in text


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fence(text)
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(candidate[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fence(text)
    decoder = json.JSONDecoder()

    for start in range(len(candidate)):
        if candidate[start] != "{":
            continue
        try:
            data, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    return None


def _adapt_tool_call_text(messages: list[dict[str, Any]], text: str) -> str:
    """Normalize Hulu-Med's bare tool JSON to the Qwen XML tool-call format.

    The MedGym agent still uses the same qwen parser and environment logic.
    This only adapts the local Hulu-Med OpenAI-compatible server output.
    Memory LLM JSON requests do not contain the tools prompt and are left as
    raw JSON.
    """
    if not _is_tool_call_request(messages):
        return text

    fixed = (text or "").strip()
    for wrong_close_tag in ("</tool_file>", "</tool_request>", "</tool_calls>"):
        fixed = fixed.replace(wrong_close_tag, "</tool_call>")
    if "<tool_call>" in fixed and "</tool_call>" in fixed:
        return fixed

    data = _extract_json_object(fixed)
    if not isinstance(data, dict):
        data = _extract_first_json_object(fixed)
        if not isinstance(data, dict):
            return fixed

    name = data.get("name")
    arguments = data.get("arguments")
    if isinstance(name, str) and isinstance(arguments, dict):
        payload = {"name": name, "arguments": arguments}
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"

    return fixed


def _ensure_local_package(package_name: str, package_dir: Path) -> None:
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package


def _load_local_module(name: str, path: Path, package_dir: Path):
    package_name = name.rsplit(".", 1)[0]
    _ensure_local_package(package_name, package_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    prompt = _message_text(messages)
    max_tokens = int(payload.get("max_tokens") or 1024)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    top_p = float(payload.get("top_p", 1.0) or 1.0)

    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "eos_token_id": TOKENIZER.eos_token_id,
        "pad_token_id": TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    with GENERATION_LOCK, torch.inference_mode():
        output_ids = MODEL.generate(**inputs, **gen_kwargs)
    # Hulu-Med's custom generate() converts input_ids to inputs_embeds before
    # calling the parent generator. In that path Transformers may return only
    # newly generated ids instead of prompt+completion ids, so slice only when
    # the prompt is actually present in the returned sequence.
    returned_prompt = (
        output_ids.shape[-1] > prompt_tokens
        and torch.equal(output_ids[0, :prompt_tokens].cpu(), inputs["input_ids"][0].cpu())
    )
    if returned_prompt:
        completion_ids = output_ids[0, prompt_tokens:]
    else:
        completion_ids = output_ids[0]
    text = TOKENIZER.decode(completion_ids, skip_special_tokens=True).strip()
    text = _adapt_tool_call_text(messages, text)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": prompt_tokens + int(completion_ids.shape[-1]),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SERVED_MODEL_NAME,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(_generate(payload))
        except BrokenPipeError:
            return
        except Exception as exc:
            try:
                self._send_json(
                    {
                        "error": {
                            "message": f"{type(exc).__name__}: {exc}",
                            "type": "server_error",
                        }
                    },
                    status=500,
                )
            except BrokenPipeError:
                return

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HulumedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--served_model_name", default="hulumed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    global MODEL, TOKENIZER, SERVED_MODEL_NAME, DEVICE
    SERVED_MODEL_NAME = args.served_model_name
    model_path = Path(args.model_path).resolve()
    sys.path.insert(0, str(model_path))

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    TOKENIZER = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    # Avoid Transformers dynamic-module cache. Hulu-Med ships local modeling
    # files with sibling imports; direct local import keeps everything offline.
    config_module = _load_local_module(
        "hulumed_local.configuration_hulumed_qwen2",
        model_path / "configuration_hulumed_qwen2.py",
        model_path,
    )
    modeling_module = _load_local_module(
        "hulumed_local.modeling_hulumed_qwen2",
        model_path / "modeling_hulumed_qwen2.py",
        model_path,
    )
    config_cls = getattr(config_module, "HulumedQwen2Config")
    model_cls = getattr(modeling_module, "HulumedQwen2ForCausalLM")
    config = config_cls.from_pretrained(str(model_path), local_files_only=True)
    MODEL = model_cls.from_pretrained(
        str(model_path),
        local_files_only=True,
        config=config,
        torch_dtype=dtype,
    ).to(DEVICE)
    MODEL.eval()

    server = HulumedHTTPServer((args.host, args.port), Handler)
    print(f"Serving {SERVED_MODEL_NAME} on http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
