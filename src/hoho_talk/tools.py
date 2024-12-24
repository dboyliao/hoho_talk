from typing import Callable, Optional

from .data import BlockType

__all__ = ["ToolRegistry"]


class ToolRegistry:
    _TOOLS = {}
    _DELAY_TARGET = object()

    @classmethod
    def register(
        cls,
        tool_name: str,
        description: str,
        parameters: dict,
        target: Optional[Callable] = _DELAY_TARGET,
    ):
        if target is cls._DELAY_TARGET:

            def wrapper(func: Callable):
                cls._TOOLS[tool_name] = {
                    "description": description,
                    "parameters": parameters,
                    "target": func,
                }
                return func

            return wrapper
        cls._TOOLS[tool_name] = {
            "description": description,
            "parameters": parameters,
            "target": target,
        }

    @classmethod
    def get_llm_tool(cls, tool_name: str) -> tuple[dict, Optional[Callable]]:
        if tool_name not in cls._TOOLS:
            raise ValueError(f"Tool {tool_name} not found.")
        tool_data = cls._TOOLS[tool_name]
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_data["description"],
                "parameters": tool_data["parameters"],
            },
        }, tool_data["target"]


ToolRegistry.register(
    "insert_context_block",
    description="Insert a context block into the conversation context.",
    parameters={
        "type": "object",
        "properties": {
            "block_type": {"type": "string", "enum": [k for k in BlockType]},
            "block_content": {"type": "string", "minLength": 1},
        },
        "required": ["block_type", "block_content"],
    },
    target=None,
)
