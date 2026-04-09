"""
Agent base class.

All agents share a common interface: they receive memory and optionally
a tool registry, then expose a .run(**kwargs) method that returns a
structured result dict and updates memory in place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memory.memory import AgentMemory
from tools.tools import ToolRegistry


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Each agent is responsible for a single step in the loop.
    Agents may call tools, update memory, and produce a result dict.

    Parameters
    ----------
    memory : AgentMemory
        Shared memory object (read + write).
    tool_registry : ToolRegistry | None
        Registry of available tools (not all agents need it).
    llm : Any
        LLM backend instance.
    """

    name: str = "base_agent"

    def __init__(
        self,
        memory: AgentMemory,
        tool_registry: ToolRegistry | None = None,
        llm: Any = None,
    ) -> None:
        self.memory = memory
        self.tool_registry = tool_registry
        self.llm = llm

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute agent logic; return a structured result dict."""

    def _use_tool(self, tool_name: str, input_data: Any) -> Any:
        """Convenience wrapper to call a tool and handle missing tools."""
        if self.tool_registry is None:
            raise RuntimeError("No tool registry attached to this agent.")
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")
        return tool.run(input_data)
