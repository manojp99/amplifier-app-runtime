"""Tests for host-defined tools functionality.

This module tests the host tools feature which allows host applications
to register custom tools that work across all transports.

Test categories:
- Unit tests: Test individual components in isolation
- Integration tests: Test interaction between components
- Advanced tests: Test complex scenarios and edge cases
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_app_runtime.host_tools import (
    HostTool,
    HostToolDefinition,
    HostToolRegistry,
    ToolContext,
    ToolResult,
    ToolScope,
    host_tool,
    host_tool_registry,
    register_host_tools_on_session,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> ToolContext:
    """Create a sample tool context for testing."""
    return ToolContext(
        session_id="test-session-123",
        cwd="/tmp/test",
        environment={"TEST_VAR": "test_value"},
        metadata={"test_key": "test_value"},
    )


@pytest.fixture
def sample_handler() -> Any:
    """Create a sample async handler for testing."""

    async def handler(input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"Processed: {input.get('query', 'no query')}",
            metadata={"session_id": context.session_id},
        )

    return handler


@pytest.fixture
def sample_tool_definition(sample_handler: Any) -> HostToolDefinition:
    """Create a sample tool definition for testing."""
    return HostToolDefinition(
        name="test_tool",
        description="A test tool for unit testing",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to process"},
            },
            "required": ["query"],
        },
        handler=sample_handler,
    )


@pytest.fixture
def fresh_registry() -> HostToolRegistry:
    """Create a fresh registry for each test."""
    return HostToolRegistry()


@pytest.fixture
def populated_registry(
    fresh_registry: HostToolRegistry,
    sample_tool_definition: HostToolDefinition,
) -> HostToolRegistry:
    """Create a registry with a tool already registered."""
    # Use synchronous registration via the internal dict
    # to avoid async fixture issues with pytest-asyncio
    fresh_registry._tools[sample_tool_definition.name] = sample_tool_definition
    return fresh_registry


# =============================================================================
# Unit Tests: ToolContext
# =============================================================================


class TestToolContext:
    """Unit tests for ToolContext dataclass."""

    def test_create_minimal_context(self) -> None:
        """Test creating context with minimal required fields."""
        context = ToolContext(session_id="sess-1", cwd="/tmp")
        assert context.session_id == "sess-1"
        assert context.cwd == "/tmp"
        assert context.environment == {}
        assert context.metadata == {}

    def test_create_full_context(self) -> None:
        """Test creating context with all fields."""
        context = ToolContext(
            session_id="sess-2",
            cwd="/home/user",
            environment={"PATH": "/usr/bin"},
            metadata={"user": "test"},
        )
        assert context.session_id == "sess-2"
        assert context.cwd == "/home/user"
        assert context.environment == {"PATH": "/usr/bin"}
        assert context.metadata == {"user": "test"}


# =============================================================================
# Unit Tests: ToolResult
# =============================================================================


class TestToolResult:
    """Unit tests for ToolResult dataclass."""

    def test_default_result(self) -> None:
        """Test default result values."""
        result = ToolResult()
        assert result.success is True
        assert result.output is None
        assert result.error is None
        assert result.metadata == {}

    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = ToolResult(success=True, output="Hello, world!")
        assert result.success is True
        assert result.output == "Hello, world!"
        assert result.error is None

    def test_error_result(self) -> None:
        """Test creating an error result."""
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.output is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = ToolResult(
            success=True,
            output={"key": "value"},
            metadata={"execution_time": 0.5},
        )
        assert result.metadata == {"execution_time": 0.5}


# =============================================================================
# Unit Tests: HostToolDefinition
# =============================================================================


class TestHostToolDefinition:
    """Unit tests for HostToolDefinition dataclass."""

    def test_create_minimal_definition(self, sample_handler: Any) -> None:
        """Test creating definition with minimal fields."""
        defn = HostToolDefinition(
            name="minimal",
            description="Minimal tool",
            parameters={"type": "object"},
            handler=sample_handler,
        )
        assert defn.name == "minimal"
        assert defn.description == "Minimal tool"
        assert defn.scope == ToolScope.GLOBAL
        assert defn.requires_approval is False
        assert defn.timeout is None

    def test_create_full_definition(self, sample_handler: Any) -> None:
        """Test creating definition with all fields."""
        defn = HostToolDefinition(
            name="full",
            description="Full tool",
            parameters={"type": "object"},
            handler=sample_handler,
            scope=ToolScope.SESSION,
            category="test",
            requires_approval=True,
            timeout=30.0,
        )
        assert defn.name == "full"
        assert defn.scope == ToolScope.SESSION
        assert defn.category == "test"
        assert defn.requires_approval is True
        assert defn.timeout == 30.0

    def test_empty_name_raises(self, sample_handler: Any) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            HostToolDefinition(
                name="",
                description="Test",
                parameters={},
                handler=sample_handler,
            )

    def test_empty_description_raises(self, sample_handler: Any) -> None:
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            HostToolDefinition(
                name="test",
                description="",
                parameters={},
                handler=sample_handler,
            )

    def test_non_callable_handler_raises(self) -> None:
        """Test that non-callable handler raises ValueError."""
        with pytest.raises(ValueError, match="handler must be callable"):
            HostToolDefinition(
                name="test",
                description="Test",
                parameters={},
                handler="not_a_function",  # type: ignore
            )


# =============================================================================
# Unit Tests: HostToolRegistry
# =============================================================================


class TestHostToolRegistry:
    """Unit tests for HostToolRegistry class."""

    @pytest.mark.asyncio
    async def test_register_tool(
        self,
        fresh_registry: HostToolRegistry,
        sample_tool_definition: HostToolDefinition,
    ) -> None:
        """Test registering a tool."""
        await fresh_registry.register(sample_tool_definition)
        assert fresh_registry.count == 1
        assert fresh_registry.get("test_tool") is sample_tool_definition

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(
        self,
        fresh_registry: HostToolRegistry,
        sample_tool_definition: HostToolDefinition,
    ) -> None:
        """Test that registering duplicate tool raises ValueError."""
        await fresh_registry.register(sample_tool_definition)
        with pytest.raises(ValueError, match="already registered"):
            await fresh_registry.register(sample_tool_definition)

    @pytest.mark.asyncio
    async def test_register_or_replace(
        self,
        fresh_registry: HostToolRegistry,
        sample_handler: Any,
    ) -> None:
        """Test register_or_replace functionality."""
        tool1 = HostToolDefinition(
            name="replaceable",
            description="Original",
            parameters={},
            handler=sample_handler,
        )
        tool2 = HostToolDefinition(
            name="replaceable",
            description="Replacement",
            parameters={},
            handler=sample_handler,
        )

        # First registration
        replaced = await fresh_registry.register_or_replace(tool1)
        assert replaced is False
        tool_def = fresh_registry.get("replaceable")
        assert tool_def is not None
        assert tool_def.description == "Original"

        # Second registration (replacement)
        replaced = await fresh_registry.register_or_replace(tool2)
        assert replaced is True
        tool_def = fresh_registry.get("replaceable")
        assert tool_def is not None
        assert tool_def.description == "Replacement"

    @pytest.mark.asyncio
    async def test_unregister_tool(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test unregistering a tool."""
        assert populated_registry.count == 1
        result = await populated_registry.unregister("test_tool")
        assert result is True
        assert populated_registry.count == 0
        assert populated_registry.get("test_tool") is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(
        self,
        fresh_registry: HostToolRegistry,
    ) -> None:
        """Test unregistering a non-existent tool returns False."""
        result = await fresh_registry.unregister("nonexistent")
        assert result is False

    def test_get_tool(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test getting a tool by name."""
        tool = populated_registry.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_get_nonexistent_tool(
        self,
        fresh_registry: HostToolRegistry,
    ) -> None:
        """Test getting a non-existent tool returns None."""
        tool = fresh_registry.get("nonexistent")
        assert tool is None

    def test_list_tools(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test listing all tools."""
        tools = populated_registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_list_names(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test listing tool names."""
        names = populated_registry.list_names()
        assert names == ["test_tool"]

    @pytest.mark.asyncio
    async def test_clear_registry(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test clearing all tools from registry."""
        assert populated_registry.count == 1
        count = await populated_registry.clear()
        assert count == 1
        assert populated_registry.count == 0

    def test_create_session_tools(
        self,
        populated_registry: HostToolRegistry,
        sample_context: ToolContext,
    ) -> None:
        """Test creating session-bound tool instances."""
        tools = populated_registry.create_session_tools(
            session_id="test-sess",
            context=sample_context,
        )
        assert len(tools) == 1
        assert isinstance(tools[0], HostTool)
        assert tools[0].name == "test_tool"


# =============================================================================
# Unit Tests: HostTool
# =============================================================================


class TestHostTool:
    """Unit tests for HostTool adapter class."""

    def test_properties(
        self,
        sample_tool_definition: HostToolDefinition,
        sample_context: ToolContext,
    ) -> None:
        """Test HostTool property access."""
        tool = HostTool(sample_tool_definition, sample_context)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool for unit testing"
        assert "properties" in tool.parameters
        assert tool.input_schema == tool.parameters
        assert tool.requires_approval is False

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        sample_tool_definition: HostToolDefinition,
        sample_context: ToolContext,
    ) -> None:
        """Test successful tool execution."""
        tool = HostTool(sample_tool_definition, sample_context)
        result = await tool.execute({"query": "hello"})
        assert result.success is True
        assert "Processed: hello" in result.output
        assert result.metadata["session_id"] == sample_context.session_id

    @pytest.mark.asyncio
    async def test_execute_with_error(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test tool execution that raises an error."""

        async def failing_handler(input: dict, context: ToolContext) -> ToolResult:
            raise RuntimeError("Intentional error")

        defn = HostToolDefinition(
            name="failing",
            description="Fails on purpose",
            parameters={},
            handler=failing_handler,
        )
        tool = HostTool(defn, sample_context)
        result = await tool.execute({})
        assert result.success is False
        assert result.error is not None
        assert "Intentional error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_timeout(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test tool execution with timeout."""

        async def slow_handler(input: dict, context: ToolContext) -> ToolResult:
            await asyncio.sleep(5)  # Slow operation
            return ToolResult(success=True)

        defn = HostToolDefinition(
            name="slow",
            description="Slow tool",
            parameters={},
            handler=slow_handler,
            timeout=0.1,  # Very short timeout
        )
        tool = HostTool(defn, sample_context)
        result = await tool.execute({})
        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error


# =============================================================================
# Unit Tests: Decorator
# =============================================================================


class TestHostToolDecorator:
    """Unit tests for @host_tool decorator."""

    def test_decorator_registers_tool(self) -> None:
        """Test that decorator registers the tool."""
        test_registry = HostToolRegistry()

        @host_tool(
            name="decorated_tool",
            description="A decorated tool",
            parameters={"type": "object"},
            registry=test_registry,
        )
        async def my_handler(input: dict, context: ToolContext) -> ToolResult:
            return ToolResult(success=True)

        assert test_registry.count == 1
        assert test_registry.get("decorated_tool") is not None

    def test_decorator_preserves_function(self) -> None:
        """Test that decorator returns the original function."""
        test_registry = HostToolRegistry()

        @host_tool(
            name="preserved",
            description="Preserved function",
            parameters={},
            registry=test_registry,
        )
        async def original_function(input: dict, context: ToolContext) -> ToolResult:
            return ToolResult(success=True, output="original")

        # Function should still be callable directly
        assert callable(original_function)


# =============================================================================
# Integration Tests
# =============================================================================


class TestHostToolsIntegration:
    """Integration tests for host tools with session."""

    @pytest.mark.asyncio
    async def test_register_tools_on_session_no_tools(self) -> None:
        """Test registering tools when none are available."""
        registry = HostToolRegistry()
        mock_session = MagicMock()

        registered = await register_host_tools_on_session(
            session=mock_session,
            registry=registry,
            session_id="test",
            cwd="/tmp",
        )
        assert registered == []

    @pytest.mark.asyncio
    async def test_register_tools_on_session_no_coordinator(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test registering tools when session has no coordinator."""
        mock_session = MagicMock(spec=[])  # No coordinator attribute

        registered = await register_host_tools_on_session(
            session=mock_session,
            registry=populated_registry,
            session_id="test",
            cwd="/tmp",
        )
        assert registered == []

    @pytest.mark.asyncio
    async def test_register_tools_on_direct_session(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test registering tools on a direct AmplifierSession-like object."""
        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        mock_session = MagicMock()
        mock_session.coordinator = mock_coordinator

        registered = await register_host_tools_on_session(
            session=mock_session,
            registry=populated_registry,
            session_id="test",
            cwd="/tmp",
        )

        assert len(registered) == 1
        assert "test_tool" in registered
        mock_coordinator.mount.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_tools_on_managed_session(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test registering tools on a ManagedSession wrapper."""
        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator = mock_coordinator

        # Use spec=[] to prevent MagicMock from auto-creating 'coordinator'
        # This forces the code to check _amplifier_session instead
        mock_session = MagicMock(spec=["_amplifier_session"])
        mock_session._amplifier_session = mock_amplifier_session

        registered = await register_host_tools_on_session(
            session=mock_session,
            registry=populated_registry,
            session_id="test",
            cwd="/tmp",
        )

        assert len(registered) == 1
        assert "test_tool" in registered

    @pytest.mark.asyncio
    async def test_register_tools_handles_mount_failure(
        self,
        populated_registry: HostToolRegistry,
    ) -> None:
        """Test that mount failures are handled gracefully."""
        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock(side_effect=RuntimeError("Mount failed"))

        mock_session = MagicMock()
        mock_session.coordinator = mock_coordinator

        # Should not raise, but return empty list
        registered = await register_host_tools_on_session(
            session=mock_session,
            registry=populated_registry,
            session_id="test",
            cwd="/tmp",
        )

        assert registered == []


# =============================================================================
# Advanced Tests
# =============================================================================


class TestHostToolsAdvanced:
    """Advanced tests for complex scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_registration(
        self,
        fresh_registry: HostToolRegistry,
        sample_handler: Any,
    ) -> None:
        """Test concurrent tool registration is thread-safe."""
        tools = [
            HostToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={},
                handler=sample_handler,
            )
            for i in range(10)
        ]

        # Register all tools concurrently
        await asyncio.gather(*[fresh_registry.register(t) for t in tools])

        assert fresh_registry.count == 10
        for i in range(10):
            assert fresh_registry.get(f"tool_{i}") is not None

    @pytest.mark.asyncio
    async def test_concurrent_execution(
        self,
        sample_tool_definition: HostToolDefinition,
        sample_context: ToolContext,
    ) -> None:
        """Test concurrent tool execution."""
        tool = HostTool(sample_tool_definition, sample_context)

        # Execute tool 10 times concurrently
        results = await asyncio.gather(*[tool.execute({"query": f"query_{i}"}) for i in range(10)])

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.success is True
            assert f"query_{i}" in result.output

    @pytest.mark.asyncio
    async def test_tool_with_complex_input(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test tool with complex nested input."""

        async def complex_handler(input: dict, context: ToolContext) -> ToolResult:
            nested_value = input.get("nested", {}).get("deep", {}).get("value")
            array_len = len(input.get("items", []))
            return ToolResult(
                success=True,
                output={"nested_value": nested_value, "array_len": array_len},
            )

        defn = HostToolDefinition(
            name="complex",
            description="Handles complex input",
            parameters={
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {
                            "deep": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            }
                        },
                    },
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            handler=complex_handler,
        )

        tool = HostTool(defn, sample_context)
        result = await tool.execute(
            {
                "nested": {"deep": {"value": "found_it"}},
                "items": ["a", "b", "c"],
            }
        )

        assert result.success is True
        assert result.output["nested_value"] == "found_it"
        assert result.output["array_len"] == 3

    @pytest.mark.asyncio
    async def test_tool_context_isolation(
        self,
        sample_handler: Any,
    ) -> None:
        """Test that tool contexts are isolated between sessions."""
        defn = HostToolDefinition(
            name="context_test",
            description="Tests context isolation",
            parameters={},
            handler=sample_handler,
        )

        context1 = ToolContext(session_id="session-1", cwd="/path/1")
        context2 = ToolContext(session_id="session-2", cwd="/path/2")

        tool1 = HostTool(defn, context1)
        tool2 = HostTool(defn, context2)

        result1 = await tool1.execute({"query": "test"})
        result2 = await tool2.execute({"query": "test"})

        # Results should reflect their respective contexts
        assert result1.metadata["session_id"] == "session-1"
        assert result2.metadata["session_id"] == "session-2"

    @pytest.mark.asyncio
    async def test_multiple_registries(
        self,
        sample_handler: Any,
    ) -> None:
        """Test that multiple registries are independent."""
        registry1 = HostToolRegistry()
        registry2 = HostToolRegistry()

        tool = HostToolDefinition(
            name="shared_name",
            description="Same name, different registries",
            parameters={},
            handler=sample_handler,
        )

        await registry1.register(tool)
        await registry2.register(tool)  # Should not conflict

        assert registry1.count == 1
        assert registry2.count == 1

    @pytest.mark.asyncio
    async def test_tool_returning_various_output_types(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test tools returning different output types."""
        outputs = [
            "string output",
            123,
            123.456,
            True,
            ["list", "of", "items"],
            {"nested": {"object": "value"}},
            None,
        ]

        for expected_output in outputs:
            # Use default argument to capture loop variable by value
            async def handler(
                input: dict,
                context: ToolContext,
                expected: Any = expected_output,
            ) -> ToolResult:
                return ToolResult(success=True, output=expected)

            defn = HostToolDefinition(
                name="type_test",
                description="Type test",
                parameters={},
                handler=handler,
            )

            tool = HostTool(defn, sample_context)
            result = await tool.execute({})

            assert result.success is True
            assert result.output == expected_output

    @pytest.mark.asyncio
    async def test_global_registry_singleton(self) -> None:
        """Test that global registry is a singleton."""
        # The global registry should persist
        original_count = host_tool_registry.count

        async def temp_handler(input: dict, context: ToolContext) -> ToolResult:
            return ToolResult(success=True)

        test_tool = HostToolDefinition(
            name="global_test_unique_12345",
            description="Testing global registry",
            parameters={},
            handler=temp_handler,
        )

        await host_tool_registry.register(test_tool)
        assert host_tool_registry.count == original_count + 1

        # Cleanup
        await host_tool_registry.unregister("global_test_unique_12345")
        assert host_tool_registry.count == original_count

    @pytest.mark.asyncio
    async def test_tool_with_approval_flag(
        self,
        sample_handler: Any,
        sample_context: ToolContext,
    ) -> None:
        """Test tool with requires_approval flag."""
        defn = HostToolDefinition(
            name="sensitive",
            description="Requires approval",
            parameters={},
            handler=sample_handler,
            requires_approval=True,
        )

        tool = HostTool(defn, sample_context)
        assert tool.requires_approval is True

    @pytest.mark.asyncio
    async def test_registry_operations_under_load(
        self,
        sample_handler: Any,
    ) -> None:
        """Test registry operations under concurrent load."""
        registry = HostToolRegistry()

        async def register_and_use(i: int) -> bool:
            tool = HostToolDefinition(
                name=f"load_test_{i}",
                description=f"Load test {i}",
                parameters={},
                handler=sample_handler,
            )
            await registry.register(tool)
            retrieved = registry.get(f"load_test_{i}")
            return retrieved is not None

        # Run 50 concurrent registrations
        results = await asyncio.gather(*[register_and_use(i) for i in range(50)])

        assert all(results)
        assert registry.count == 50

        # Cleanup
        await registry.clear()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestHostToolsEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_handler_returns_wrong_type(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test handling when handler returns wrong type."""

        async def bad_handler(input: dict, context: ToolContext) -> str:  # type: ignore
            return "not a ToolResult"

        defn = HostToolDefinition(
            name="bad_return",
            description="Returns wrong type",
            parameters={},
            handler=bad_handler,  # type: ignore
        )

        tool = HostTool(defn, sample_context)
        # This should return the wrong type - behavior depends on usage
        result = await tool.execute({})
        # The tool wrapper doesn't validate return type, so this passes through
        assert result == "not a ToolResult"

    @pytest.mark.asyncio
    async def test_handler_with_none_input(
        self,
        sample_context: ToolContext,
    ) -> None:
        """Test handler receiving empty input."""

        async def null_handler(input: dict, context: ToolContext) -> ToolResult:
            return ToolResult(success=True, output=len(input))

        defn = HostToolDefinition(
            name="null_test",
            description="Handles null input",
            parameters={},
            handler=null_handler,
        )

        tool = HostTool(defn, sample_context)
        result = await tool.execute({})
        assert result.success is True
        assert result.output == 0

    def test_tool_context_with_unicode(self) -> None:
        """Test tool context with unicode strings."""
        context = ToolContext(
            session_id="测试-セッション-тест",
            cwd="/home/用户/目录",
            environment={"LANG": "日本語"},
            metadata={"emoji": "🎉"},
        )

        assert "测试" in context.session_id
        assert "用户" in context.cwd
        assert context.metadata["emoji"] == "🎉"

    @pytest.mark.asyncio
    async def test_very_long_tool_name(
        self,
        sample_handler: Any,
        fresh_registry: HostToolRegistry,
    ) -> None:
        """Test tool with very long name."""
        long_name = "a" * 1000

        defn = HostToolDefinition(
            name=long_name,
            description="Long name tool",
            parameters={},
            handler=sample_handler,
        )

        await fresh_registry.register(defn)
        assert fresh_registry.get(long_name) is not None

    @pytest.mark.asyncio
    async def test_tool_with_special_characters_in_name(
        self,
        sample_handler: Any,
        fresh_registry: HostToolRegistry,
    ) -> None:
        """Test tool with special characters in name."""
        special_names = [
            "tool_with_underscore",
            "tool-with-dash",
            "tool.with.dots",
            "tool:with:colons",
        ]

        for name in special_names:
            defn = HostToolDefinition(
                name=name,
                description=f"Tool named {name}",
                parameters={},
                handler=sample_handler,
            )
            await fresh_registry.register(defn)
            assert fresh_registry.get(name) is not None
