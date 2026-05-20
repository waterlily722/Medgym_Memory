from __future__ import annotations

import json
import os
import time
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput


def _now_ms() -> int:
    return int(time.time() * 1000)


def _post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
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
        raise RuntimeError(f"RPC HTTPError {e.code}: {body[:300]}") from e
    except Exception as e:
        raise RuntimeError(f"RPC request failed: {type(e).__name__}: {e}") from e


class SegmentTool(Tool):
    """
    医学影像分割工具（器官/病灶 mask）。
    默认走 RPC：从环境变量 RLLM_SEGMENT_RPC 读取 endpoint。
    """

    def __init__(
        self,
        name: str = "segment",
        description: str = "Segment target structures in medical images and return masks.",
        rpc_url: str | None = None,
        timeout: int = 120,
    ):
        self.rpc_url = rpc_url or os.getenv("RLLM_SEGMENT_RPC", "")
        self.timeout = timeout
        super().__init__(name=name, description=description)

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
                        "target": {"type": "string", "description": "Segmentation target, e.g. lung, liver, tumor."},
                        "image_path": {"type": "string", "description": "Local path to image file (optional)."},
                        "image_id": {"type": "string", "description": "Image identifier resolvable by backend (optional)."},
                        "modality": {"type": "string", "description": "CT/MR/XR/US etc.", "default": ""},
                        "threshold": {"type": "number", "description": "Optional prob threshold.", "default": 0.5},
                        "extra": {"type": "object", "description": "Extra backend params (optional).", "default": {}},
                    },
                    "required": ["target"],
                },
            },
        }

    def forward(
        self,
        target: str,
        image_path: str | None = None,
        image_id: str | None = None,
        modality: str = "",
        threshold: float = 0.5,
        extra: dict | None = None,
        **kwargs,
    ) -> ToolOutput:
        extra = {} if extra is None else extra

        if not self.rpc_url:
            return ToolOutput(
                name=self.name,
                error="SegmentTool rpc_url is not set. Set env RLLM_SEGMENT_RPC or pass rpc_url in constructor.",
                metadata={"mode": "rpc"},
            )

        payload = {
            "tool": self.name,
            "ts": _now_ms(),
            "target": target,
            "image_path": image_path,
            "image_id": image_id,
            "modality": modality,
            "threshold": float(threshold),
            "extra": {**extra, **(kwargs or {})},
        }

        try:
            resp = _post_json(self.rpc_url, payload, timeout=self.timeout)
            return ToolOutput(name=self.name, output=resp, metadata={"mode": "rpc"})
        except Exception as e:
            return ToolOutput(name=self.name, error=str(e), metadata={"mode": "rpc"})
