# rllm/tools/med_tool/cxr_grounding_tool.py
"""
cxr_grounding 工具：在对 CXR 图像理解不到位时，医生可指定需要标注的部位文本，
调用 Grounding DINO API 对当前 CXR 做检测，返回带 grounding 框的结果。
依赖：先启动 grounding_dino_server（如 python -m rllm.scripts.grounding_dino_server）。
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.tools.tool_base import Tool, ToolOutput

from rllm.tools.med_tool.cxr_tool import _read_file_b64

logger = logging.getLogger(__name__)


def load_json_file(path: str | Path) -> Any:
    import json as _json
    with open(path, "r", encoding="utf-8") as f:
        return _json.load(f)


def _post_grounding_api(
    api_url: str,
    image_base64: str,
    text_prompts: List[str],
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    timeout: int = 60,
) -> Dict[str, Any]:
    """POST /ground，返回 { detections, ... } 或 { error }。"""
    payload = {
        "image_base64": image_base64,
        "text_prompts": text_prompts,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
    }
    try:
        import urllib.request
        import urllib.error
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "detections": []}


def _get_first_cxr_image_path(
    ehr: Dict[str, Any],
    case_dir: str,
    study_index: int = -1,
) -> Optional[str]:
    """从 ehr["CXR"] 中取第一张（或指定 study）的一张图路径；支持 jpg_path_abs 或 case_dir + jpg_path。"""
    cxr_list = ehr.get("CXR") if isinstance(ehr, dict) else None
    if not isinstance(cxr_list, list) or not cxr_list:
        return None
    idx = int(study_index)
    if idx < 0:
        idx = len(cxr_list) - 1
    if idx >= len(cxr_list):
        return None
    study = cxr_list[idx] if isinstance(cxr_list[idx], dict) else {}
    dicoms = study.get("dicoms") or []
    if not dicoms or not isinstance(dicoms[0], dict):
        return None
    d = dicoms[0]
    abs_path = d.get("jpg_path_abs")
    if isinstance(abs_path, str) and abs_path:
        if Path(abs_path).exists():
            return abs_path
    rel_path = d.get("jpg_path") or ""
    if not rel_path:
        return None
    full = Path(case_dir) / rel_path
    if full.exists() and full.is_file():
        return str(full)
    return None


class CXRGroundingTool(Tool):
    """
    对当前 case 的 CXR 图像调用 Grounding DINO API，用医生指定的文本做目标检测，返回带框结果。
    """

    def __init__(
        self,
        name: str = "cxr_grounding",
        description: str | None = None,
        api_url: str | None = None,
        case_dir: str | None = None,
        timeout: int = 60,
        max_image_bytes: int = 25 * 1024 * 1024,
    ):
        self.api_url = (api_url or os.getenv("RLLM_GROUNDING_API_URL", "http://127.0.0.1:30050/ground")).strip().rstrip("/")
        if "/ground" not in self.api_url:
            self.api_url = f"{self.api_url}/ground"
        self.case_dir = case_dir or os.getenv("RLLM_CASE_DIR", "")
        self.timeout = timeout
        self.max_image_bytes = max_image_bytes
        self._ehr_cache: Dict[str, Dict[str, Any]] = {}

        desc = (
            description
            or "When CXR image understanding is unclear, use this to detect regions by text (e.g. heart, lung, opacity). "
            "Specify text_prompts for the body parts or findings to localize; returns bounding boxes and scores."
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
                        "text_prompts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Body parts or findings to detect (e.g. heart, lung, opacity, cardiomediastinal silhouette).",
                        },
                        "case_id": {"type": "string", "description": "Optional case identifier.", "default": ""},
                        "context": {"type": "object", "description": "Injected by env; contains ehr, case_dir.", "default": {}},
                        "study_index": {
                            "type": "integer",
                            "description": "Which CXR study. Default -1 = latest.",
                            "default": -1,
                        },
                        "box_threshold": {"type": "number", "description": "Detection score threshold.", "default": 0.4},
                        "text_threshold": {"type": "number", "description": "Text matching threshold.", "default": 0.3},
                    },
                    "required": ["text_prompts"],
                },
            },
        }

    def _load_case_ehr(self, case_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ehr = (context or {}).get("ehr")
        if isinstance(ehr, dict):
            if "ehr" in ehr and isinstance(ehr.get("ehr"), dict):
                return ehr["ehr"]
            return ehr
        ehr_path = (context or {}).get("ehr_path")
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
        if not cid or cid in self._ehr_cache:
            return self._ehr_cache.get(cid) if cid else None
        if not self.case_dir:
            return None
        base = Path(self.case_dir)
        for cand in [base / cid, base / f"{cid}.json"]:
            if cand.exists() and cand.is_file():
                try:
                    obj = load_json_file(cand)
                    if isinstance(obj, dict) and "ehr" in obj and isinstance(obj.get("ehr"), dict):
                        self._ehr_cache[cid] = obj["ehr"]
                        return self._ehr_cache[cid]
                except Exception:
                    pass
        return None

    def forward(
        self,
        text_prompts: List[str] | None = None,
        case_id: str = "",
        context: dict | None = None,
        study_index: int = -1,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        **kwargs: Any,
    ) -> ToolOutput:
        if text_prompts is None and "text_prompts" in kwargs:
            text_prompts = kwargs.pop("text_prompts")
        text_prompts = text_prompts or []
        if isinstance(text_prompts, str):
            text_prompts = [s.strip() for s in text_prompts.replace(",", " ").split() if s.strip()]
        context = context or kwargs.get("context") or {}
        case_dir = context.get("case_dir") or self.case_dir

        if not text_prompts:
            return ToolOutput(
                name=self.name,
                error="Missing text_prompts (e.g. [\"heart\", \"lung\"]).",
                metadata={"mode": "cxr_grounding"},
            )

        ehr = self._load_case_ehr(case_id=case_id, context=context)
        if not isinstance(ehr, dict):
            return ToolOutput(
                name=self.name,
                error="missing_ehr",
                metadata={"case_id": case_id, "mode": "cxr_grounding"},
            )

        image_path = _get_first_cxr_image_path(ehr, case_dir, study_index)
        if not image_path:
            return ToolOutput(
                name=self.name,
                error="No CXR image path found in ehr.CXR for this case.",
                metadata={"case_id": case_id, "mode": "cxr_grounding"},
            )

        try:
            image_b64 = _read_file_b64(image_path, max_bytes=self.max_image_bytes)
        except Exception as e:
            return ToolOutput(
                name=self.name,
                error=f"Failed to read CXR image: {e}",
                metadata={"image_path": image_path, "mode": "cxr_grounding"},
            )

        resp = _post_grounding_api(
            api_url=self.api_url,
            image_base64=image_b64,
            text_prompts=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            timeout=self.timeout,
        )

        if resp.get("error"):
            return ToolOutput(
                name=self.name,
                error=resp.get("error", "API error"),
                output=resp,
                metadata={"mode": "cxr_grounding"},
            )

        detections = resp.get("detections") or []
        summary = f"Detected {len(detections)} region(s) for prompts: {', '.join(text_prompts)}."
        if detections:
            summary += " " + "; ".join(
                f"{d.get('label', '?')} (score={d.get('score', 0):.2f}) at {d.get('box', [])}"
                for d in detections[:10]
            )
            if len(detections) > 10:
                summary += f" ... and {len(detections) - 10} more."

        return ToolOutput(
            name=self.name,
            output={
                "summary": summary,
                "text_prompts": text_prompts,
                "num_detections": len(detections),
                "detections": detections,
                "image_path": image_path,
            },
            metadata={"mode": "cxr_grounding"},
        )
