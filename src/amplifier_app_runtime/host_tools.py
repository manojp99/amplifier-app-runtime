"""Host-defined tools for transport-agnostic tool registration.

This module provides the infrastructure for host applications to register
custom tools that work across all transports (stdio, HTTP, WebSocket).

Architecture:
- HostToolDefinition: Describes a tool (name, description, parameters, handler)
- HostToolRegistry: Central registry for managing tool definitions
- HostTool: Adapter that implements Amplifier's tool protocol
- ToolContext: Session context passed to tool handlers

Usage:
    from amplifier_app_runtime.host_tools import (
        host_tool_registry,
        HostToolDefinition,
        ToolContext,
        ToolResult,
    )

    async def my_handler(input: dict, context: ToolContext) -> ToolResult:
        result = process(input["query"])
        return ToolResult(success=True, output=result)

    await host_tool_registry.register(HostToolDefinition(
        name="my_tool",
        description="Does something useful",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        },
        handler=my_handler,
    ))
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolScope(str, Enum):
    """When the tool is available."""

    GLOBAL = "global"  # Available to all sessions
    SESSION = "session"  # Created per-session with context


@dataclass
class ToolContext:
    """Context passed to tool handlers.

    Provides information about the session and environment
    where the tool is being executed.
    """

    session_id: str
    cwd: str
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The tool's output (any JSON-serializable value)
        error: Error message if success is False
        metadata: Additional metadata about the execution
    """

    success: bool = True
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for tool handlers - using Any to avoid complex generic typing issues
# The actual signature is: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]
ToolHandler = Any


@dataclass
class HostToolDefinition:
    """Definition of a host-provided tool.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for the LLM
        parameters: JSON Schema for the tool's input parameters
        handler: Async function that implements the tool
        scope: When the tool is available (global or per-session)
        category: Optional category for grouping tools
        requires_approval: Whether to require user approval before execution
        timeout: Optional timeout in seconds for execution
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    scope: ToolScope = ToolScope.GLOBAL
    category: str | None = None
    requires_approval: bool = False
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate the tool definition."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.description:
            raise ValueError("Tool description cannot be empty")
        if not callable(self.handler):
            raise ValueError("Tool handler must be callable")


@runtime_checkable
class AmplifierToolProtocol(Protocol):
    """Protocol that Amplifier expects for tools."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    async def execute(self, input: dict[str, Any]) -> Any: ...


class HostTool:
    """Wrapper that adapts HostToolDefinition to Amplifier's tool protocol.

    This class implements the interface expected by Amplifier's coordinator
    for tool execution.
    """

    def __init__(
        self,
        definition: HostToolDefinition,
        context: ToolContext,
    ) -> None:
        self._definition = definition
        self._context = context

    @property
    def name(self) -> str:
        """Tool name for registration."""
        return self._definition.name

    @property
    def description(self) -> str:
        """Tool description for LLM."""
        return self._definition.description

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return self._definition.parameters

    @property
    def input_schema(self) -> dict[str, Any]:
        """Alias for parameters (Amplifier protocol)."""
        return self._definition.parameters

    @property
    def requires_approval(self) -> bool:
        """Whether this tool requires user approval."""
        return self._definition.requires_approval

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the tool with given input.

        Args:
            input: Tool parameters matching the JSON Schema

        Returns:
            ToolResult with success status and output/error
        """
        try:
            if self._definition.timeout:
                return await asyncio.wait_for(
                    self._definition.handler(input, self._context),
                    timeout=self._definition.timeout,
                )
            return await self._definition.handler(input, self._context)
        except TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self._definition.timeout}s",
            )
        except Exception as e:
            logger.error(f"Host tool '{self.name}' execution error: {e}")
            return ToolResult(success=False, error=str(e))


class HostToolRegistry:
    """Central registry for host-defined tools.

    Thread-safe registry that manages tool definitions and
    creates session-bound tool instances.

    Example:
        registry = HostToolRegistry()
        await registry.register(my_tool_definition)

        # Later, when creating a session:
        tools = registry.create_session_tools(session_id, context)
        for tool in tools:
            await coordinator.mount("tools", tool, name=tool.name)
    """

    def __init__(self) -> None:
        self._tools: dict[str, HostToolDefinition] = {}
        self._lock = asyncio.Lock()

    async def register(self, tool: HostToolDefinition) -> None:
        """Register a host-defined tool.

        Args:
            tool: The tool definition to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        async with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            self._tools[tool.name] = tool
            logger.info(f"Registered host tool: {tool.name}")

    async def register_or_replace(self, tool: HostToolDefinition) -> bool:
        """Register a tool, replacing any existing tool with the same name.

        Args:
            tool: The tool definition to register

        Returns:
            True if an existing tool was replaced, False otherwise
        """
        async with self._lock:
            replaced = tool.name in self._tools
            self._tools[tool.name] = tool
            action = "Replaced" if replaced else "Registered"
            logger.info(f"{action} host tool: {tool.name}")
            return replaced

    async def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister

        Returns:
            True if the tool was unregistered, False if not found
        """
        async with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.info(f"Unregistered host tool: {name}")
                return True
            return False

    def get(self, name: str) -> HostToolDefinition | None:
        """Get a tool definition by name.

        Args:
            name: The tool name to look up

        Returns:
            The tool definition or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[HostToolDefinition]:
        """List all registered tools.

        Returns:
            List of all registered tool definitions
        """
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """List names of all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def create_session_tools(
        self,
        session_id: str,
        context: ToolContext,
    ) -> list[HostTool]:
        """Create tool instances for a session.

        Args:
            session_id: The session ID
            context: Tool context with session information

        Returns:
            List of HostTool instances ready for registration
        """
        tools = []
        for defn in self._tools.values():
            tools.append(HostTool(defn, context))
        return tools

    async def clear(self) -> int:
        """Unregister all tools.

        Returns:
            Number of tools that were unregistered
        """
        async with self._lock:
            count = len(self._tools)
            self._tools.clear()
            logger.info(f"Cleared {count} host tools")
            return count


async def register_host_tools_on_session(
    session: Any,
    registry: HostToolRegistry,
    session_id: str,
    cwd: str,
    environment: dict[str, str] | None = None,
) -> list[str]:
    """Register host-defined tools on an Amplifier session.

    This is the main integration point for adding host tools to a session.

    Args:
        session: AmplifierSession or ManagedSession
        registry: The host tool registry
        session_id: The session ID
        cwd: Working directory for the session
        environment: Optional environment variables

    Returns:
        List of registered tool names
    """
    registered: list[str] = []

    # Get coordinator from session
    coordinator = None
    if hasattr(session, "coordinator"):
        # Direct AmplifierSession
        coordinator = session.coordinator
    elif hasattr(session, "_amplifier_session"):
        # ManagedSession wrapper
        amplifier_session = session._amplifier_session
        if amplifier_session and hasattr(amplifier_session, "coordinator"):
            coordinator = amplifier_session.coordinator

    if coordinator is None:
        logger.warning("Could not find coordinator on session - host tools not registered")
        return registered

    # Create tool context
    context = ToolContext(
        session_id=session_id,
        cwd=cwd,
        environment=environment or {},
    )

    # Create and mount tools
    tools = registry.create_session_tools(session_id, context)

    for tool in tools:
        try:
            await coordinator.mount("tools", tool, name=tool.name)
            registered.append(tool.name)
            logger.info(f"Registered host tool on session {session_id}: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to register host tool {tool.name}: {e}")

    return registered


# Global registry instance for use across the application
host_tool_registry = HostToolRegistry()


# Convenience decorator for defining tools
def host_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    scope: ToolScope = ToolScope.GLOBAL,
    category: str | None = None,
    requires_approval: bool = False,
    timeout: float | None = None,
    registry: HostToolRegistry | None = None,
) -> Callable[[Any], Any]:
    """Decorator for defining host tools.

    Example:
        @host_tool(
            name="my_tool",
            description="Does something",
            parameters={"type": "object", "properties": {...}},
        )
        async def my_tool_handler(input: dict, context: ToolContext) -> ToolResult:
            return ToolResult(success=True, output="done")
    """

    def decorator(func: Any) -> Any:
        definition = HostToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            scope=scope,
            category=category,
            requires_approval=requires_approval,
            timeout=timeout,
        )

        # Register with the specified or global registry
        target_registry = registry or host_tool_registry

        # Use a synchronous approach since decorators can't be async
        # The tool will be registered when the module is loaded
        target_registry._tools[name] = definition
        logger.debug(f"Decorated and registered host tool: {name}")

        return func

    return decorator
