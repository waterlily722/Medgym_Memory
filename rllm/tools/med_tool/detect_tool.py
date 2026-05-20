from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from rllm.tools.tool_base import Tool, ToolOutput


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pick_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize_text_prompt(text: str | List[str]) -> str:
    """
    GroundingDINO requirement:
      - lowercased
      - each query ends with a dot
    Accept:
      - "a cat. a dog."
      - "a cat, a dog"
      - ["a cat", "a dog"]
    """
    if isinstance(text, list):
        parts = [str(x).strip() for x in text if str(x).strip()]
    else:
        s = (text or "").strip()
        if not s:
            return ""
        # split by common separators, keep user dots as separators too
        # e.g. "a, b; c. d" -> ["a","b","c","d"]
        for sep in ["\n", ";", "；", ",", "，", "|"]:
            s = s.replace(sep, ".")
        parts = [p.strip() for p in s.split(".") if p.strip()]

    parts = [p.lower() for p in parts if p]
    if not parts:
        return ""

    # ensure each ends with dot by joining with ". "
    return ". ".join(parts) + "."


class GroundingDinoLocalDetectTool(Tool):
    """
    detect tool (local inference) using GroundingDINO (transformers).

    Model dir default:
      /data/xuxiang/mimic-iv/models/grounding-dino-base

    Input style: keep similar to your RPC DetectTool:
      - image_path / image_id (image_id optional, local tool mainly uses image_path)
      - modality
      - threshold (mapped to box_threshold)
      - extra (can carry prompt/text_threshold/etc)
    """

    def __init__(
        self,
        name: str = "detect",
        description: str = "Detect findings in medical images and return bounding boxes.",
        model_dir: str | None = None,
        device: str = "auto",
        timeout: int = 60,
        default_text: str = "",
        default_text_threshold: float = 0.25,
    ):
        self.timeout = timeout

        self.model_dir = (
            model_dir
            or os.getenv("RLLM_DETECT_MODEL_DIR", "/data/xuxiang/mimic-iv/models/grounding-dino-base")
        )
        self.device = _pick_device(device or os.getenv("RLLM_DETECT_DEVICE", "auto"))

        self.default_text = default_text  # 允许你设置默认检测词表
        self.default_text_threshold = float(default_text_threshold)

        # lazy load cache
        self._processor = None
        self._model = None

        super().__init__(name=name, description=description)

    @property
    def json(self) -> dict[str, Any]:
        # 保持与你给的 DetectTool 结构相近：required=[]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Local path to image file (optional)."},
                        "image_id": {"type": "string", "description": "Image identifier resolvable by backend (optional)."},
                        "modality": {"type": "string", "description": "CT/MR/XR/US etc.", "default": ""},
                        "threshold": {"type": "number", "description": "Score threshold for detections.", "default": 0.3},
                        "extra": {"type": "object", "description": "Extra params (optional).", "default": {}},
                    },
                    "required": [],
                },
            },
        }

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        # 本地目录加载：确保目录内有 config / model 权重等文件
        self._processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self._model.eval()

        # warmup optional: skip to avoid extra latency / memory

    def _load_image(self, image_path: str):
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def forward(
        self,
        image_path: str | None = None,
        image_id: str | None = None,
        modality: str = "",
        threshold: float = 0.3,  # -> box_threshold
        extra: dict | None = None,
        **kwargs,
    ) -> ToolOutput:
        t0 = _now_ms()
        extra = {} if extra is None else dict(extra)

        # 允许 kwargs 覆盖/补充 extra
        if kwargs:
            extra.update(kwargs)

        if not image_path:
            # 本地版本优先 image_path；如你需要 image_id->path 解析，可在这里扩展
            return ToolOutput(
                name=self.name,
                error="Missing image_path for local GroundingDINO detect tool.",
                metadata={"mode": "local_groundingdino"},
            )

        if not os.path.exists(image_path):
            return ToolOutput(
                name=self.name,
                error=f"image_path not found: {image_path}",
                metadata={"mode": "local_groundingdino"},
            )

        # prompt / text：尽量不新增顶层字段，放在 extra 里取
        # 支持：extra["text"] / extra["prompt"] / extra["queries"]
        raw_text = (
            extra.get("text")
            or extra.get("prompt")
            or extra.get("queries")
            or self.default_text
            or ""
        )
        text_prompt = _normalize_text_prompt(raw_text)

        if not text_prompt:
            return ToolOutput(
                name=self.name,
                error="Missing text prompt. Provide extra.text / extra.prompt / extra.queries.",
                metadata={"mode": "local_groundingdino"},
            )

        box_threshold = float(threshold if threshold is not None else 0.3)
        text_threshold = float(extra.get("text_threshold", self.default_text_threshold))
        max_dets = int(extra.get("max_dets", 300))

        try:
            self._lazy_load()

            import torch

            image = self._load_image(image_path)

            inputs = self._processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],  # (h,w)
            )[0]

            # tensors -> python lists
            boxes = results.get("boxes")
            scores = results.get("scores")
            labels = results.get("labels")

            boxes_list = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else (boxes or [])
            scores_list = scores.detach().cpu().tolist() if hasattr(scores, "detach") else (scores or [])
            labels_list = [str(x) for x in labels] if labels is not None else ["object"] * len(scores_list)

            # sort & truncate
            if len(scores_list) > max_dets:
                idx = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)[:max_dets]
                boxes_list = [boxes_list[i] for i in idx]
                scores_list = [scores_list[i] for i in idx]
                labels_list = [labels_list[i] for i in idx]

            resp: Dict[str, Any] = {
                "tool": self.name,
                "ts": t0,
                "latency_ms": _now_ms() - t0,
                "image_path": image_path,
                "image_id": image_id,
                "modality": modality,
                "backend": "groundingdino_transformers",
                "model_dir": self.model_dir,
                "device": self.device,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "prompt": text_prompt,
                "num_dets": len(scores_list),
                "detections": [
                    {
                        "box_xyxy": [float(x) for x in b],  # [x1,y1,x2,y2]
                        "score": float(s),
                        "label": str(l),
                    }
                    for b, s, l in zip(boxes_list, scores_list, labels_list)
                ],
            }

            return ToolOutput(name=self.name, output=resp, metadata={"mode": "local_groundingdino"})

        except Exception as e:
            return ToolOutput(
                name=self.name,
                error=f"{type(e).__name__}: {e}",
                metadata={"mode": "local_groundingdino"},
            )
