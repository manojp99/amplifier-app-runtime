# Host-Defined Tools Architecture

> RFC: Transport-agnostic host tool registration for Amplifier Runtime

## Problem Statement

Currently, tools in Amplifier runtime come from two sources:

1. **Bundles**: Tools loaded via amplifier-foundation (e.g., `bash`, `read_file`)
2. **ACP Client Capabilities**: Tools registered when an ACP client connects with specific capabilities (e.g., `ide_terminal`, `ide_read_file`)

There's no general mechanism for the **host application** (the process running the runtime) to define and register custom tools that:
- Work across all transports (stdio, HTTP, WebSocket)
- Can be defined programmatically at runtime startup
- Are automatically available to all sessions
- Support both synchronous and asynchronous execution

## Use Cases

### 1. IDE Integrations
The host IDE wants to provide tools like file editing, terminal access, diagnostics, etc., without implementing full ACP protocol.

### 2. Custom Applications
A host application embedding the runtime wants to expose domain-specific tools:
- Database query tools
- API interaction tools
- Custom file system access
- Integration with proprietary systems

### 3. Plugin Systems
External plugins can register tools at runtime startup without modifying the core runtime.

### 4. Simplified Client Development
Clients that don't implement ACP can still provide host-side capabilities via a simpler tool registration API.

## Design Principles

1. **Transport-agnostic**: Tools work identically regardless of transport
2. **Session-aware**: Tools can access session context when needed
3. **Async-first**: All tools support async execution
4. **Type-safe**: Clear interfaces with full type hints
5. **Dynamic**: Support runtime registration/unregistration
6. **Composable**: Works alongside bundle tools and ACP tools

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Application                          │
│  (IDE, Custom App, CLI wrapper)                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ registers tools via
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HostToolRegistry                              │
│  - register(tool: HostToolDefinition)                           │
│  - unregister(name: str)                                        │
│  - get_tools() -> list[HostToolDefinition]                      │
│  - create_session_tools(session_id, context) -> list[Tool]      │
└─────────────────────────┬───────────────────────────────────────┘
                          │ tools injected into
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SessionManager                                │
│  - On session creation, mounts host tools                       │
│  - Tools available via coordinator.get("tools")                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    All Transports                                │
│  stdio, HTTP, WebSocket, SSE - all get same tools               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Protocol
from enum import Enum

class ToolScope(str, Enum):
    """When the tool is available."""
    GLOBAL = "global"      # Available to all sessions
    SESSION = "session"    # Created per-session with context

@dataclass
class ToolContext:
    """Context passed to tool handlers."""
    session_id: str
    cwd: str
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass  
class ToolResult:
    """Result from tool execution."""
    success: bool = True
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

# Type for tool handlers
ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]

@dataclass
class HostToolDefinition:
    """Definition of a host-provided tool."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: ToolHandler
    scope: ToolScope = ToolScope.GLOBAL
    
    # Optional metadata
    category: str | None = None
    requires_approval: bool = False
    timeout: float | None = None
```

### Registry Implementation

```python
class HostToolRegistry:
    """Central registry for host-defined tools.
    
    Thread-safe registry that manages tool definitions and
    creates session-bound tool instances.
    """
    
    def __init__(self) -> None:
        self._tools: dict[str, HostToolDefinition] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, tool: HostToolDefinition) -> None:
        """Register a host-defined tool."""
        async with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            self._tools[tool.name] = tool
    
    async def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        async with self._lock:
            return self._tools.pop(name, None) is not None
    
    def get(self, name: str) -> HostToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[HostToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def create_session_tools(
        self, 
        session_id: str, 
        context: ToolContext
    ) -> list[HostTool]:
        """Create tool instances for a session."""
        tools = []
        for defn in self._tools.values():
            tools.append(HostTool(defn, context))
        return tools

# Global registry instance
host_tool_registry = HostToolRegistry()
```

### Tool Wrapper for Amplifier

```python
class HostTool:
    """Wrapper that adapts HostToolDefinition to Amplifier's tool protocol.
    
    This class implements the interface expected by Amplifier's coordinator
    for tool execution.
    """
    
    def __init__(
        self, 
        definition: HostToolDefinition, 
        context: ToolContext
    ) -> None:
        self._definition = definition
        self._context = context
    
    @property
    def name(self) -> str:
        return self._definition.name
    
    @property
    def description(self) -> str:
        return self._definition.description
    
    @property
    def parameters(self) -> dict[str, Any]:
        return self._definition.parameters
    
    @property
    def input_schema(self) -> dict[str, Any]:
        """Alias for parameters (Amplifier protocol)."""
        return self._definition.parameters
    
    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the tool with given input."""
        try:
            if self._definition.timeout:
                return await asyncio.wait_for(
                    self._definition.handler(input, self._context),
                    timeout=self._definition.timeout
                )
            return await self._definition.handler(input, self._context)
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self._definition.timeout}s"
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

## Integration Points

### 1. CLI Integration

```bash
# Register tools via config file
amplifier-runtime --host-tools ./my-tools.yaml

# Or via Python entry point
amplifier-runtime --host-tools-module myapp.tools
```

**Config file format (YAML):**
```yaml
tools:
  - name: my_database_query
    description: "Query the application database"
    module: myapp.tools.database
    function: query_handler
    parameters:
      type: object
      properties:
        sql:
          type: string
          description: "SQL query to execute"
      required: [sql]
```

### 2. Programmatic API

```python
from amplifier_app_runtime import RuntimeConfig, create_runtime
from amplifier_app_runtime.host_tools import (
    HostToolRegistry, 
    HostToolDefinition,
    ToolContext,
    ToolResult,
)

# Define a custom tool
async def my_tool_handler(
    input: dict[str, Any], 
    context: ToolContext
) -> ToolResult:
    # Tool implementation
    result = do_something(input["query"], context.cwd)
    return ToolResult(success=True, output=result)

# Register it
registry = HostToolRegistry()
await registry.register(HostToolDefinition(
    name="my_custom_tool",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    },
    handler=my_tool_handler,
))

# Create runtime with registry
config = RuntimeConfig(host_tool_registry=registry)
runtime = create_runtime(config)
```

### 3. Session Creation Hook

In `session.py`, modify `ManagedSession.initialize()`:

```python
async def initialize(self, ...):
    # ... existing initialization ...
    
    # Register host-defined tools
    await self._register_host_tools()

async def _register_host_tools(self) -> None:
    """Register host-defined tools on this session."""
    from .host_tools import host_tool_registry
    
    if not self._amplifier_session:
        return
    
    coordinator = self._amplifier_session.coordinator
    context = ToolContext(
        session_id=self.session_id,
        cwd=self.metadata.cwd,
    )
    
    tools = host_tool_registry.create_session_tools(
        self.session_id, 
        context
    )
    
    for tool in tools:
        try:
            await coordinator.mount("tools", tool, name=tool.name)
            logger.info(f"Registered host tool: {tool.name}")
        except Exception as e:
            logger.warning(f"Failed to register host tool {tool.name}: {e}")
```

## Relationship with ACP Tools

ACP client-side tools (`ide_terminal`, `ide_read_file`, etc.) will continue to work as they do now - they're registered based on ACP client capabilities and require an active ACP connection.

Host-defined tools are **complementary**:
- ACP tools: Require ACP protocol, tied to client capabilities
- Host tools: Transport-agnostic, defined by the host application

A host could choose to:
1. Use only host tools (simple integration)
2. Use only ACP tools (full protocol support)
3. Use both (maximum flexibility)

## Testing Strategy

### Unit Tests

```python
class TestHostToolRegistry:
    async def test_register_tool(self):
        registry = HostToolRegistry()
        tool = HostToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            handler=mock_handler,
        )
        await registry.register(tool)
        assert registry.get("test_tool") is not None
    
    async def test_duplicate_registration_fails(self):
        registry = HostToolRegistry()
        tool = HostToolDefinition(...)
        await registry.register(tool)
        with pytest.raises(ValueError):
            await registry.register(tool)
    
    async def test_create_session_tools(self):
        registry = HostToolRegistry()
        await registry.register(test_tool)
        
        context = ToolContext(session_id="test", cwd="/tmp")
        tools = registry.create_session_tools("test", context)
        
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
```

### Integration Tests

```python
class TestHostToolsIntegration:
    async def test_tool_available_in_session(self):
        """Host tools are available to LLM in session."""
        registry = HostToolRegistry()
        await registry.register(echo_tool)
        
        session = await create_session_with_registry(registry)
        
        # Verify tool is mounted
        tools = session._amplifier_session.coordinator.get("tools")
        tool_names = [t.name for t in tools]
        assert "echo" in tool_names
    
    async def test_tool_execution_via_llm(self):
        """LLM can invoke host-defined tools."""
        # This requires mocking the LLM to call the tool
        ...
    
    async def test_tool_works_across_transports(self):
        """Same tool works via stdio and HTTP."""
        registry = HostToolRegistry()
        await registry.register(test_tool)
        
        # Test via stdio adapter
        stdio_session = await create_stdio_session(registry)
        result1 = await invoke_tool(stdio_session, "test_tool", {})
        
        # Test via HTTP adapter  
        http_session = await create_http_session(registry)
        result2 = await invoke_tool(http_session, "test_tool", {})
        
        assert result1 == result2
```

### E2E Tests

```python
class TestHostToolsE2E:
    async def test_custom_tool_in_real_session(self):
        """End-to-end test with real Amplifier session."""
        # Register a tool that modifies state
        state = {"called": False}
        
        async def stateful_handler(input, context):
            state["called"] = True
            return ToolResult(success=True, output="done")
        
        registry = HostToolRegistry()
        await registry.register(HostToolDefinition(
            name="set_flag",
            description="Sets a flag",
            parameters={"type": "object"},
            handler=stateful_handler,
        ))
        
        # Create real session and execute
        session = await create_real_session(registry)
        
        # Send a prompt that should trigger the tool
        # (requires LLM or mock orchestrator)
        await session.execute("Call the set_flag tool")
        
        assert state["called"] is True
```

## Migration Path

### Phase 1: Core Infrastructure
1. Implement `HostToolDefinition`, `ToolContext`, `ToolResult`
2. Implement `HostToolRegistry`
3. Implement `HostTool` adapter
4. Add integration point in `ManagedSession`

### Phase 2: CLI Support
1. Add `--host-tools` flag for YAML config
2. Add `--host-tools-module` for Python modules
3. Document configuration format

### Phase 3: SDK Support
1. Add programmatic API to SDK
2. Add examples for common patterns
3. Document best practices

### Phase 4: Advanced Features
1. Tool approval flow integration
2. Tool metrics/observability
3. Tool sandboxing options

## Security Considerations

1. **Input Validation**: Tool handlers should validate input against schema
2. **Sandboxing**: Consider optional sandboxing for untrusted tools
3. **Approval Flow**: High-risk tools can require user approval
4. **Audit Logging**: Log all tool invocations with context

## Open Questions

1. Should host tools be able to access other tools? (Tool composition)
2. Should there be a capability negotiation like ACP?
3. How to handle tool versioning?
4. Should tools be able to emit events?

## References

- [ACP Protocol](https://agentclientprotocol.com) - Inspiration for capability model
- [MCP Protocol](https://modelcontextprotocol.io) - Tool definition patterns
- [Amplifier Core Tool Contract](https://github.com/microsoft/amplifier-core/blob/main/docs/contracts/TOOL_CONTRACT.md)
