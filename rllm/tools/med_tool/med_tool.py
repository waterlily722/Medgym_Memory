from __future__ import annotations

from typing import Any

from rllm.tools.tool_base import Tool

from rllm.tools.med_tool.retrieve_tool import RetrieveTool
from rllm.tools.med_tool.detect_tool import GroundingDinoLocalDetectTool
from rllm.tools.med_tool.segment_tool import SegmentTool
from rllm.tools.med_tool.dialog_tool import DialogTool
from rllm.tools.med_tool.diagnosis_tool import DiagnosisTool
from rllm.tools.med_tool.cxr_tool import CXRTool
from rllm.tools.med_tool.exam_result_tool import ExamResultTool
from rllm.tools.med_tool.cxr_grounding_tool import CXRGroundingTool
from rllm.tools.med_tool.finish_tool import FinishTool



# 尽量兼容不同项目里 tool_registry 的导出方式
try:
    from rllm.tools.registry import tool_registry as _tool_registry  # type: ignore
    from rllm.tools.registry import ToolRegistry  # type: ignore
except Exception:
    from rllm.tools.registry import ToolRegistry  # type: ignore

    _tool_registry = ToolRegistry()  # fallback singleton-ish


DEFAULT_MED_TOOL_CLASSES: dict[str, type[Tool]] = {
    "retrieve": RetrieveTool,
    "detect": GroundingDinoLocalDetectTool,
    "segment": SegmentTool,
    "ask_patient": DialogTool,
    "diagnosis": DiagnosisTool,
    "cxr": CXRTool,
    "request_exam": ExamResultTool,
    "cxr_grounding": CXRGroundingTool,
    "finish": FinishTool,
}


def register_med_tools(
    registry: Any = None,
    tools: dict[str, type[Tool]] | None = None,
    override: bool = False,
):

    reg = registry or _tool_registry
    tools = tools or DEFAULT_MED_TOOL_CLASSES

    for name, cls in tools.items():
        if (not override) and (name in reg):
            continue
        reg.register(name, cls)

    return reg


__all__ = [
    "RetrieveTool",
    "GroundingDinoLocalDetectTool",
    "SegmentTool",
    "DialogTool",
    "DiagnosisTool",
    "CXRTool",
    "ExamResultTool",
    "CXRGroundingTool",
    "FinishTool",
    "DEFAULT_MED_TOOL_CLASSES",
    "register_med_tools",
]
