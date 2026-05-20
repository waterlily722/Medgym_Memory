from rllm.tools.registry import ToolRegistry


tool_registry = ToolRegistry()


def register_med_tools(*args, **kwargs):
    from rllm.tools.med_tool.med_tool import register_med_tools as _register_med_tools

    return _register_med_tools(*args, **kwargs)


__all__ = ["ToolRegistry", "tool_registry", "register_med_tools"]
