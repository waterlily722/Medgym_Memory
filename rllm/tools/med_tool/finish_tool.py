# ./rllm/rllm/tools/med_tool/finish_tool.py
from __future__ import annotations

from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput


class FinishTool(Tool):
    """
    Submit final conclusion and end the episode. Used when the agent decides
    not to call other tools (e.g. no CXR needed) but must still respond with a tool call.
    The environment handles this tool specially and does not execute it via tool_runner.
    """

    def __init__(
        self,
        name: str = "finish",
        description: str = "Submit your final conclusion and end the episode. Use this when you decide not to request further tools (e.g. when a chest X-ray is not needed).",
    ):
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
                        "response": {
                            "type": "string",
                            "description": "Your brief conclusion or explanation (e.g. why CXR is not needed).",
                        },
                    },
                    "required": ["response"],
                },
            },
        }

    def forward(self, response: str = "", **kwargs) -> ToolOutput:
        return ToolOutput(name=self.name, output={"response": response or ""})
