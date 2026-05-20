from __future__ import annotations

from rllm.parser.chat_template_parser import ChatTemplateParser
from rllm.parser.tool_parser import QwenToolParser, R1ToolParser, ToolParser

__all__ = [
    "ChatTemplateParser",
    "ToolParser",
    "R1ToolParser",
    "QwenToolParser",
    "get_tool_parser",
]

PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]
