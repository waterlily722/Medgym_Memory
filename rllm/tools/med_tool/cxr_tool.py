# ./rllm/rllm/tools/med_tool/cxr_tool.py
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


def _read_file_b64(p: str | Path, max_bytes: int = 25 * 1024 * 1024) -> str:
    """
    Read binary file and return base64 string.
    """
    fp = Path(p)
    if not fp.exists() or not fp.is_file():
        raise FileNotFoundError(f"Image not found: {fp}")
    data = fp.read_bytes()
    if len(data) > max_bytes:
        raise RuntimeError(f"Image too large ({len(data)} bytes): {fp}")
    return base64.b64encode(data).decode("ascii")


def load_json_file(path: str | Path) -> Any:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CXRTool(Tool):
    """
    cxr tool:
    - Load ehr["CXR"][study_index]["dicoms"]
    - Read each dicom["jpg_path"] and return as base64 + view_position
    """

    def __init__(
        self,
        name: str = "cxr",
        description: str = "Open chest X-ray (CXR) images from ehr.CXR.dicoms and return image(s) with view_position.",
        case_dir: str | None = None,
        max_images: int = 4,
        max_image_bytes: int = 25 * 1024 * 1024,
    ):
        super().__init__(name=name, description=description)
        self.case_dir = case_dir or os.getenv("RLLM_CASE_DIR", "")
        self.max_images = int(max_images)
        self.max_image_bytes = int(max_image_bytes)
        self._ehr_cache: Dict[str, Dict[str, Any]] = {}

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
                        "case_id": {"type": "string", "description": "Optional case identifier.", "default": ""},
                        "context": {"type": "object", "description": "Tool context that may include ehr/ehr_path.", "default": {}},
                        "study_index": {
                            "type": "integer",
                            "description": "Which CXR study to open. Default -1 means latest.",
                            "default": -1,
                        },
                        "view_position": {
                            "type": "string",
                            "description": "Optional filter by view position (e.g., 'PA', 'AP', 'LATERAL'). Case-insensitive substring match.",
                            "default": "",
                        },
                        "return_base64": {
                            "type": "boolean",
                            "description": "If true, return image_base64 for each jpg_path.",
                            "default": True,
                        },
                        "return_paths": {
                            "type": "boolean",
                            "description": "If true, return jpg_path in results.",
                            "default": True,
                        },
                        "max_images": {
                            "type": "integer",
                            "description": "Maximum number of images to return.",
                            "default": 2,
                        },
                    },
                    "required": [],
                },
            },
        }

    def _load_case_ehr(self, case_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1) context direct
        ehr = (context or {}).get("ehr")
        if isinstance(ehr, dict):
            if "ehr" in ehr and isinstance(ehr.get("ehr"), dict):
                return ehr["ehr"]
            return ehr

        # 2) ehr_path
        ehr_path = (context or {}).get("ehr_path")
        if isinstance(ehr_path, str) and ehr_path:
            try:
                obj = load_json_file(ehr_path)
                if isinstance(obj, dict):
                    if "ehr" in obj and isinstance(obj.get("ehr"), dict):
                        return obj["ehr"]
                    return obj
            except Exception:
                pass

        # 3) cache + case_dir
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

    def forward(
        self,
        case_id: str = "",
        context: dict | None = None,
        study_index: int = -1,
        view_position: str = "",
        return_base64: bool = True,
        return_paths: bool = True,
        max_images: int = 2,
        **kwargs,
    ) -> ToolOutput:
        context = context or {}
        # print(context)
        # exit()
        ehr = self._load_case_ehr(case_id=case_id, context=context)
        if not isinstance(ehr, dict):
            return ToolOutput(
                name=self.name,
                error="missing_ehr",
                metadata={"case_id": case_id, "context_keys": sorted(list(context.keys()))},
            )

        cxr_list = ehr.get("CXR")
        if not isinstance(cxr_list, list) or not cxr_list:
            return ToolOutput(
                name=self.name,
                error="missing_cxr_section",
                metadata={"case_id": case_id, "ehr_keys": sorted(list(ehr.keys()))},
            )

        # choose study
        idx = int(study_index)
        if idx < 0:
            idx = len(cxr_list) - 1
        if idx >= len(cxr_list):
            return ToolOutput(
                name=self.name,
                error=f"study_index_out_of_range: {idx} (num_studies={len(cxr_list)})",
                metadata={"case_id": case_id},
            )

        study = cxr_list[idx] if isinstance(cxr_list[idx], dict) else {}
        dicoms = study.get("dicoms", [])
        report_text = study.get("report_text", [])

        if not isinstance(dicoms, list) or not dicoms:
            return ToolOutput(
                name=self.name,
                error="missing_dicoms",
                metadata={"case_id": case_id, "study_index": idx},
            )

        vp_filter = (view_position or "").strip().lower()
        picked: List[dict] = []
        for d in dicoms:
            if not isinstance(d, dict):
                continue
            vp = str(d.get("view_position", "") or "")
            jpg_path = str(d.get("jpg_path", "") or "")
            if not jpg_path:
                continue
            if vp_filter and (vp_filter not in vp.lower()):
                continue
            picked.append({"view_position": vp, "jpg_path": jpg_path})

        if not picked:
            # fallback: no filter match => return first few
            picked = [{"view_position": str(d.get("view_position", "") or ""),
                       "jpg_path": str(d.get("jpg_path", "") or "")}
                      for d in dicoms if isinstance(d, dict) and d.get("jpg_path")]

        out_images: List[dict] = []
        cap = min(int(max_images), self.max_images, len(picked))
        for item in picked[:cap]:
            vp = item["view_position"]
            jpg_path = item["jpg_path"]

            one = {"view_position": vp}
            if return_paths:
                one["jpg_path"] = jpg_path
            if return_base64:
                try:
                    b64 = _read_file_b64(jpg_path, max_bytes=self.max_image_bytes)
                    one["mime"] = "image/jpeg"
                    one["image_base64"] = b64
                except Exception as e:
                    one["image_error"] = f"{type(e).__name__}: {e}"

            out_images.append(one)

        # 给 doctor 的简要说明：明确列出每张图的视图，便于模型理解
        view_list = [img.get("view_position", "unknown") for img in out_images]
        summary = f"CXR images returned: {len(out_images)} image(s) — views: {', '.join(view_list)}."
        rt_str = report_text if isinstance(report_text, str) else (" ".join(str(x) for x in report_text) if isinstance(report_text, (list, tuple)) else str(report_text))
        if rt_str and rt_str.strip():
            summary += f" Report excerpt: {rt_str[:200]}..." if len(rt_str) > 200 else f" Report: {rt_str}"

        return ToolOutput(
            name=self.name,
            output={
                "summary": summary,
                "study_index": idx,
                "num_studies": len(cxr_list),
                "num_dicoms_in_study": len(dicoms),
                "report_text": report_text,
                "images": out_images,
            },
            metadata={"mode": "ehr_cxr"},
        )
