"""ACP Agent implementation using the official SDK pattern.

This module provides an ACP-compliant agent that wraps Amplifier sessions.
It uses the official ACP Python SDK's Agent interface for proper protocol handling.

Key pattern from SDK examples:
1. Agent stores connection via on_connect()
2. Agent uses conn.session_update() to stream updates
3. run_agent() handles transport setup for stdio
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp import (  # type: ignore[import-untyped]
    PROTOCOL_VERSION,
    Agent,
    Client,
    text_block,
    update_agent_message,
    update_agent_thought,
)
from acp.schema import (  # type: ignore[import-untyped]
    AgentCapabilities,
    AgentPlanUpdate,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    McpCapabilities,
    McpServerStdio,
    NewSessionResponse,
    PlanEntry,
    PromptCapabilities,
    PromptResponse,
    ResourceContentBlock,
    SessionMode,
    SessionModeState,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
)

from .session_discovery import (
    AMPLIFIER_PROJECTS_DIR,
    discover_sessions,
    find_session_directory,
)
from .slash_commands import (
    SlashCommandHandler,
    SlashCommandRegistry,
    create_available_commands_update,
    parse_slash_command,
)

if TYPE_CHECKING:
    from ..session import ManagedSession

logger = logging.getLogger(__name__)

# Default bundle when none specified
DEFAULT_BUNDLE = "foundation"


# Re-export for backward compatibility (functions moved to session_discovery.py)
__all__ = [
    "AmplifierAgent",
    "AmplifierAgentSession",
    "discover_sessions",
    "find_session_directory",
    "AMPLIFIER_PROJECTS_DIR",
]


class AmplifierAgent(Agent):
    """ACP Agent implementation backed by Amplifier sessions.

    This class implements the ACP Agent protocol using the official SDK pattern.
    It manages Amplifier sessions and streams events back to clients via
    the conn.session_update() method.

    Usage:
        # For stdio transport
        from acp import run_agent
        await run_agent(AmplifierAgent())

        # For HTTP/SSE, use with appropriate transport
    """

    def __init__(self) -> None:
        self._conn: Client | None = None
        self._sessions: dict[str, AmplifierAgentSession] = {}
        self._client_capabilities: ClientCapabilities | None = None

    def on_connect(self, conn: Client) -> None:
        """Store the connection for sending updates.

        This is called by the SDK when a client connects.
        The conn object is used to send session updates back to the client.
        """
        self._conn = conn
        logger.info("ACP client connected")

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Handle initialize request from client.

        This negotiates capabilities and prepares the agent for use.
        """
        self._client_capabilities = client_capabilities

        # Handle client_info as dict or object
        client_name = "unknown"
        if client_info:
            if isinstance(client_info, dict):
                client_name = client_info.get("name", "unknown")
            elif hasattr(client_info, "name"):
                client_name = client_info.name

        logger.info(f"ACP initialized: protocol_version={protocol_version}, client={client_name}")

        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentInfo=Implementation(
                name="amplifier-runtime",
                version="0.1.0",
            ),
            agentCapabilities=AgentCapabilities(
                loadSession=True,
                mcpCapabilities=McpCapabilities(http=False, sse=True),
                promptCapabilities=PromptCapabilities(
                    audio=False,  # Not supported by Amplifier kernel
                    embeddedContext=True,  # Supported
                    image=True,  # Supported via context pre-population
                ),
            ),
            authMethods=[],
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a new Amplifier session.

        This creates a real Amplifier session with a bundle and provider.
        The bundle defaults to 'foundation' if not specified.
        """
        # Generate session ID
        session_id = f"acp_{uuid.uuid4().hex[:12]}"

        # Extract bundle from kwargs (ACP extension via field_meta)
        bundle = kwargs.get("field_meta", {}).get("bundle") if kwargs.get("field_meta") else None
        bundle = bundle or DEFAULT_BUNDLE

        # Create session wrapper with client capabilities for ACP tools
        session = AmplifierAgentSession(
            session_id=session_id,
            cwd=cwd,
            bundle=bundle,
            conn=self._conn,
            client_capabilities=self._client_capabilities,
        )

        # Initialize the underlying Amplifier session
        await session.initialize()

        self._sessions[session_id] = session

        logger.info(f"Created ACP session: {session_id} with bundle '{bundle}'")

        return NewSessionResponse(
            sessionId=session_id,
            modes=SessionModeState(
                availableModes=[
                    SessionMode(id="default", name="Default", description="Default agent mode"),
                ],
                currentModeId="default",
            ),
        )

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        session_id: str,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        """Load an existing session.

        Attempts to load a session in the following order:
        1. Check in-memory cache (already active sessions)
        2. Try Amplifier's session manager
        3. Try to find and load from filesystem storage

        Args:
            cwd: Working directory for the session.
            mcp_servers: MCP servers to connect (currently unused).
            session_id: The session ID to load.

        Returns:
            LoadSessionResponse if session found and loaded, None otherwise.
        """
        # Check if we have it cached (already active)
        if session_id in self._sessions:
            logger.info(f"Loaded cached ACP session: {session_id}")
            return LoadSessionResponse(
                modes=SessionModeState(
                    availableModes=[
                        SessionMode(id="default", name="Default", description="Default agent mode"),
                    ],
                    currentModeId="default",
                ),
            )

        # Try to load from Amplifier session manager first
        from ..session import session_manager

        amplifier_session = await session_manager.get(session_id)

        if amplifier_session:
            # Wrap in our session type with client capabilities
            session = AmplifierAgentSession(
                session_id=session_id,
                cwd=cwd,
                bundle=DEFAULT_BUNDLE,
                conn=self._conn,
                client_capabilities=self._client_capabilities,
            )
            session._amplifier_session = amplifier_session
            self._sessions[session_id] = session

            # Register ACP tools for loaded session
            await session._register_acp_tools()

            logger.info(f"Loaded ACP session from manager: {session_id}")

            return LoadSessionResponse(
                modes=SessionModeState(
                    availableModes=[
                        SessionMode(id="default", name="Default", description="Default agent mode"),
                    ],
                    currentModeId="default",
                ),
            )

        # Fallback: Try to find session in filesystem storage
        session_dir = find_session_directory(session_id, cwd)
        if not session_dir:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Load metadata to get session details
        metadata_file = session_dir / "metadata.json"
        bundle = DEFAULT_BUNDLE
        session_cwd = cwd

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                bundle = metadata.get("bundle") or DEFAULT_BUNDLE
                session_cwd = metadata.get("cwd") or cwd
                logger.info(
                    f"Found session metadata: bundle={bundle}, cwd={session_cwd}, "
                    f"turns={metadata.get('turn_count', 0)}"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read session metadata: {e}")

        # Create a new session wrapper that will resume from the stored state
        # The Amplifier session will be created fresh but can restore context
        # from the events.jsonl file if needed
        session = AmplifierAgentSession(
            session_id=session_id,
            cwd=session_cwd,
            bundle=bundle,
            conn=self._conn,
            client_capabilities=self._client_capabilities,
        )

        # Initialize with the session directory for potential state restoration
        await session.initialize(session_dir=session_dir)
        self._sessions[session_id] = session

        logger.info(f"Loaded ACP session from filesystem: {session_id}")

        return LoadSessionResponse(
            modes=SessionModeState(
                availableModes=[
                    SessionMode(id="default", name="Default", description="Default agent mode"),
                ],
                currentModeId="default",
            ),
        )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """List available sessions.

        Discovers sessions from Amplifier's filesystem storage at
        ~/.amplifier/projects/{encoded-path}/sessions/

        Args:
            cursor: Pagination cursor (not yet implemented).
            cwd: If provided, only return sessions for this working directory.
        """
        from acp.schema import ListSessionsResponse, SessionInfo  # type: ignore[import-untyped]

        sessions = []

        # First, add any in-memory sessions (currently active)
        for sid, session in self._sessions.items():
            sessions.append(
                SessionInfo(
                    sessionId=sid,
                    cwd=session.cwd,
                    name=getattr(session, "name", None),
                )
            )

        # Collect IDs of in-memory sessions to avoid duplicates
        in_memory_ids = set(self._sessions.keys())

        # Discover persisted sessions from filesystem
        discovered = await discover_sessions(cwd=cwd, limit=50)

        for session_data in discovered:
            session_id = session_data["session_id"]

            # Skip if already in memory or if it's a child session
            if session_id in in_memory_ids:
                continue
            if session_data.get("is_child"):
                continue  # Don't list child/spawned sessions

            sessions.append(
                SessionInfo(
                    sessionId=session_id,
                    cwd=session_data.get("cwd") or cwd or "",
                    name=session_data.get("name"),
                )
            )

        return ListSessionsResponse(sessions=sessions)

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Process a prompt and stream responses back.

        This is the main entry point for user prompts. It:
        1. Extracts text from the prompt blocks
        2. Executes via Amplifier
        3. Streams updates back via conn.session_update()
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            # Send error message if we have a connection
            if self._conn:
                await self._conn.session_update(
                    session_id,
                    update_agent_message(text_block(f"Error: Session not found: {session_id}")),
                )
            # Use "cancelled" for error cases - "error" is not a valid ACP stopReason
            return PromptResponse(stopReason="cancelled")

        # Pass full content blocks to session for multi-modal handling
        # The session will handle conversion and context pre-population
        stop_reason = await session.execute_prompt(prompt)

        return PromptResponse(stopReason=stop_reason)

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        """Change agent mode."""
        session = self._sessions.get(session_id)
        if session:
            session.current_mode = mode_id
            logger.info(f"Set mode to '{mode_id}' for session {session_id}")
        return SetSessionModeResponse()

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Any:
        """Change the model for a session."""
        logger.info(f"Model change requested: {model_id} for session {session_id}")
        # Model switching not yet implemented in Amplifier session
        from acp.schema import SetSessionModelResponse  # type: ignore[import-untyped]

        return SetSessionModelResponse()

    async def authenticate(
        self,
        method_id: str,
        **kwargs: Any,
    ) -> Any:
        """Handle authentication."""
        logger.info(f"Auth requested: {method_id}")
        from acp.schema import AuthenticateResponse  # type: ignore[import-untyped]

        return AuthenticateResponse()

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fork an existing session."""
        from acp.schema import ForkSessionResponse  # type: ignore[import-untyped]

        # Create a new session based on the existing one
        new_session_id = f"acp_{uuid.uuid4().hex[:12]}"

        original = self._sessions.get(session_id)
        if not original:
            logger.error(f"Cannot fork: session not found: {session_id}")
            return ForkSessionResponse(sessionId=new_session_id)

        # Create new session with same bundle
        session = AmplifierAgentSession(
            session_id=new_session_id,
            cwd=cwd,
            bundle=original.bundle,
            conn=self._conn,
        )
        await session.initialize()
        self._sessions[new_session_id] = session

        logger.info(f"Forked session {session_id} -> {new_session_id}")
        return ForkSessionResponse(sessionId=new_session_id)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resume a session."""
        from acp.schema import ResumeSessionResponse  # type: ignore[import-untyped]

        # Try to load the session
        result = await self.load_session(cwd, mcp_servers or [], session_id)
        if result:
            return ResumeSessionResponse()
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel ongoing execution."""
        session = self._sessions.get(session_id)
        if session:
            await session.cancel()
            logger.info(f"Cancelled session: {session_id}")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods."""
        logger.info(f"Extension method: {method}")
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications."""
        logger.info(f"Extension notification: {method}")

    def _extract_text_content(self, blocks: list[Any]) -> str:
        """Extract text content from content blocks."""
        text_parts = []
        for block in blocks:
            if isinstance(block, TextContentBlock):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts)


class AmplifierAgentSession:
    """Session wrapper that streams Amplifier events as ACP updates.

    This class bridges Amplifier's event system to ACP's session/update notifications.
    Events flow: Amplifier -> Hook -> _on_event() -> conn.session_update()

    ACP client-side tools (ide_terminal, ide_read_file, ide_write_file) are registered
    based on client capabilities during initialization.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str,
        bundle: str,
        conn: Client | None,
        client_capabilities: Any | None = None,
    ) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.bundle = bundle
        self.current_mode = "default"
        self._conn = conn
        self._client_capabilities = client_capabilities
        self._amplifier_session: ManagedSession | None = None
        self._cancel_event = asyncio.Event()
        self._registered_acp_tools: list[str] = []
        # Slash command handler for IDE commands
        self._slash_handler: SlashCommandHandler | None = None
        # ACP approval bridge for IDE permission dialogs
        self._approval_bridge: Any | None = None
        self._tool_tracker: Any | None = None

    async def initialize(self, session_dir: Path | None = None) -> None:
        """Initialize the underlying Amplifier session.

        Args:
            session_dir: Optional path to existing session directory for restoration.
                        If provided, the session will attempt to restore state from
                        the stored events.jsonl file.
        """
        from ..bundle_manager import BundleManager
        from ..session import SessionConfig, session_manager
        from ..transport.base import Event
        from .approval_bridge import ACPApprovalBridge, ToolCallTracker

        # Load and prepare the bundle
        bundle_manager = BundleManager()
        try:
            prepared_bundle = await bundle_manager.load_and_prepare(
                bundle_name=self.bundle,
                working_directory=Path(self.cwd) if self.cwd else None,
            )
        except Exception as e:
            logger.error(f"Failed to load bundle '{self.bundle}': {e}")
            raise RuntimeError(
                f"Failed to load bundle '{self.bundle}'. "
                f"Ensure ANTHROPIC_API_KEY or OPENAI_API_KEY is set. Error: {e}"
            ) from e

        # Create event forwarder that uses the SDK's session_update
        async def on_amplifier_event(event: Event) -> None:
            """Forward Amplifier events to ACP via conn.session_update()."""
            await self._on_event(event)

        # Create ACP approval bridge for IDE permission dialogs
        def get_client() -> Client | None:
            return self._conn

        self._approval_bridge = ACPApprovalBridge(
            session_id=self.session_id,
            get_client=get_client,
        )
        self._tool_tracker = ToolCallTracker

        # Create Amplifier session with ACP approval bridge
        config = SessionConfig(
            bundle=self.bundle,
            working_directory=self.cwd,
            approval_system=self._approval_bridge,
        )
        self._amplifier_session = await session_manager.create(
            config=config,
            auto_initialize=False,
            send_fn=on_amplifier_event,
        )

        # Initialize with prepared bundle
        await self._amplifier_session.initialize(prepared_bundle=prepared_bundle)

        # If restoring from existing session, try to restore context
        if session_dir:
            await self._restore_session_context(session_dir)

        logger.info(f"Amplifier session {self.session_id} initialized")

        # Register ACP client-side tools based on capabilities
        await self._register_acp_tools()

        # Initialize slash command handler
        self._slash_handler = SlashCommandHandler(self._amplifier_session)

        # Send available commands to client
        await self._send_available_commands()

    async def _register_acp_tools(self) -> None:
        """Register ACP client-side tools on this session.

        Tools are registered based on client capabilities:
        - ide_terminal: requires client_capabilities.terminal
        - ide_read_file: requires client_capabilities.fs.readTextFile
        - ide_write_file: requires client_capabilities.fs.writeTextFile
        """
        if not self._amplifier_session:
            logger.warning("Cannot register ACP tools - session not initialized")
            return

        from .tools import register_acp_tools

        # Create closure for lazy client access
        def get_client() -> Client | None:
            return self._conn

        try:
            self._registered_acp_tools = await register_acp_tools(
                session=self._amplifier_session,
                get_client=get_client,
                session_id=self.session_id,
                client_capabilities=self._client_capabilities,
            )
            if self._registered_acp_tools:
                logger.info(
                    f"Registered ACP tools for session {self.session_id}: "
                    f"{self._registered_acp_tools}"
                )
            else:
                logger.debug(
                    f"No ACP tools registered for session {self.session_id} "
                    "(client may not support capabilities)"
                )
        except Exception as e:
            logger.warning(f"Failed to register ACP tools: {e}")

    async def _restore_session_context(self, session_dir: Path) -> None:
        """Restore session context from stored events.

        This method reads the events.jsonl file and extracts conversation
        history summary. Full context restoration requires Amplifier's
        native session resumption mechanism.

        Args:
            session_dir: Path to the session directory containing events.jsonl.
        """
        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            logger.debug(f"No events.jsonl found in {session_dir}, starting fresh")
            return

        try:
            # Count events and extract basic info for logging
            turn_count = 0
            last_prompt = None

            with open(events_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("event", "")
                    data = event.get("data", {})

                    if event_type == "turn:start":
                        turn_count += 1
                        last_prompt = data.get("prompt", "")[:100]  # First 100 chars

            if turn_count > 0:
                logger.info(
                    f"Session {self.session_id} has {turn_count} previous turns. "
                    f"Last prompt: '{last_prompt}...'"
                )
                # Store turn count for potential use
                self._previous_turn_count = turn_count

        except Exception as e:
            logger.warning(f"Failed to read session history: {e}")
            # Continue anyway - session will work but without history awareness

    async def _send_available_commands(self) -> None:
        """Send available slash commands to the ACP client.

        This advertises commands like /help, /mode, /tools to the IDE
        for autocomplete and command palette integration.
        """
        if not self._conn:
            return

        try:
            commands = SlashCommandRegistry.get_commands_for_session(self._amplifier_session)
            update = create_available_commands_update(self._amplifier_session)
            await self._conn.session_update(self.session_id, update)
            logger.debug(f"Sent {len(commands)} available commands to session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to send available commands: {e}")

    # Supported image MIME types for multi-modal content
    _SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

    def _convert_acp_to_amplifier_blocks(
        self,
        blocks: list[Any],
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        """Convert ACP content blocks to Amplifier format.

        Args:
            blocks: List of ACP content blocks (TextContentBlock, ImageContentBlock, etc.)

        Returns:
            amplifier_blocks: List of Amplifier-compatible content blocks
            text_prompt: Extracted text to use as the prompt
            warnings: List of warning messages for unsupported content
        """
        amplifier_blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        warnings: list[str] = []

        for block in blocks:
            # Handle TextContentBlock
            if isinstance(block, TextContentBlock):
                text_parts.append(block.text)
                amplifier_blocks.append({"type": "text", "text": block.text})

            # Handle dict-style text block
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                text_parts.append(text)
                amplifier_blocks.append({"type": "text", "text": text})

            # Handle ImageContentBlock
            elif isinstance(block, ImageContentBlock):
                converted = self._convert_image_block(block)
                if converted:
                    amplifier_blocks.append(converted)
                else:
                    warnings.append(
                        f"Unsupported image type: {getattr(block, 'mimeType', 'unknown')}. "
                        f"Supported types: {', '.join(sorted(self._SUPPORTED_IMAGE_TYPES))}"
                    )

            # Handle AudioContentBlock (not supported)
            elif isinstance(block, AudioContentBlock):
                warnings.append("Audio content is not currently supported.")

            # Handle EmbeddedResourceContentBlock
            elif isinstance(block, EmbeddedResourceContentBlock):
                converted = self._convert_embedded_resource(block)
                if converted:
                    amplifier_blocks.append(converted)
                    # Also extract text for the prompt if it's a text resource
                    if converted.get("type") == "text":
                        text_parts.append(converted.get("text", ""))

            # Handle ResourceContentBlock (external URI - not supported)
            elif isinstance(block, ResourceContentBlock):
                warnings.append(
                    "External resource links cannot be fetched. Please embed content directly."
                )

            # Handle generic object with type attribute
            elif hasattr(block, "type"):
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = getattr(block, "text", "")
                    text_parts.append(text)
                    amplifier_blocks.append({"type": "text", "text": text})
                else:
                    logger.debug(f"Skipping unsupported block type: {block_type}")

        # Build text prompt from all text parts
        text_prompt = "\n".join(text_parts).strip()

        # If no text content after filtering, use fallback
        if not text_prompt and not any(b.get("type") == "image" for b in amplifier_blocks):
            text_prompt = "Please provide content with text or images."

        return amplifier_blocks, text_prompt, warnings

    def _convert_image_block(self, block: ImageContentBlock) -> dict[str, Any] | None:
        """Convert ACP ImageContentBlock to Amplifier image format.

        Args:
            block: ACP ImageContentBlock with data and mimeType

        Returns:
            Amplifier image block format or None if unsupported type:
            {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        """
        # Get MIME type - handle both attribute styles
        mime_type = getattr(block, "mimeType", None) or getattr(block, "mime_type", None)
        if not mime_type:
            return None

        # Check if supported
        if mime_type not in self._SUPPORTED_IMAGE_TYPES:
            return None

        # Get base64 data
        data = getattr(block, "data", None)
        if not data:
            return None

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            },
        }

    def _convert_embedded_resource(
        self, block: EmbeddedResourceContentBlock
    ) -> dict[str, Any] | None:
        """Convert embedded resource to text or image block.

        Args:
            block: ACP EmbeddedResourceContentBlock with resource data

        Returns:
            Amplifier-compatible block (text or image) or None if unsupported
        """
        resource = getattr(block, "resource", None)
        if not resource:
            return None

        # Get URI for context
        uri = getattr(resource, "uri", "") or ""

        # Check if it's a text resource
        text = getattr(resource, "text", None)
        if text is not None:
            # Include URI context if available
            if uri:
                return {"type": "text", "text": f"[Resource: {uri}]\n{text}"}
            return {"type": "text", "text": text}

        # Check if it's a blob resource (potentially an image)
        blob = getattr(resource, "blob", None)
        if blob:
            mime_type = getattr(resource, "mimeType", None) or getattr(resource, "mime_type", None)
            if mime_type and mime_type in self._SUPPORTED_IMAGE_TYPES:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": blob,
                    },
                }

        return None

    async def execute_prompt(self, content: str | list[Any]) -> str:
        """Execute prompt and stream updates back via ACP.

        Handles three types of input:
        1. String content (legacy) - works exactly as before
        2. List of ACP content blocks - converts and handles multi-modal content
        3. Slash commands (/help, /mode, etc.) - handled locally

        For multi-modal content (images), the content is pre-populated into the
        conversation context so the LLM can see it, then execution proceeds with
        the text portion as the prompt.

        Returns stop reason: 'end_turn' or 'cancelled'.
        """
        self._cancel_event.clear()

        # Handle multi-modal content (list of ACP content blocks)
        text_prompt: str
        has_multimodal = False
        warnings: list[str] = []

        if isinstance(content, list):
            # Convert ACP blocks to Amplifier format
            amplifier_blocks, text_prompt, warnings = self._convert_acp_to_amplifier_blocks(content)

            # Check if we have multi-modal content (images)
            has_multimodal = any(b.get("type") == "image" for b in amplifier_blocks)

            # Pre-populate context with multi-modal content if present
            if has_multimodal and self._amplifier_session:
                await self._prepopulate_multimodal_context(amplifier_blocks)
        else:
            # Legacy string content - use as-is
            text_prompt = content

        # Check for slash command (only on text prompts)
        parsed = parse_slash_command(text_prompt)
        if parsed and self._slash_handler:
            return await self._execute_slash_command(parsed)

        if not self._amplifier_session:
            logger.error("Session not initialized")
            return "cancelled"  # "error" is not a valid ACP stopReason

        # Inject warnings about unsupported content as system context
        if warnings and self._conn:
            warning_text = "Note: " + " ".join(warnings)
            await self._conn.session_update(
                self.session_id,
                update_agent_message(text_block(warning_text)),
            )

        try:
            # Execute and let the hook stream events
            async for event in self._amplifier_session.execute(text_prompt):
                if self._cancel_event.is_set():
                    return "cancelled"

                # The yield path events - forward them too
                await self._on_event(event)

            return "end_turn"

        except asyncio.CancelledError:
            return "cancelled"
        except Exception as e:
            logger.exception(f"Error executing prompt: {e}")
            # Send error as agent message
            if self._conn:
                await self._conn.session_update(
                    self.session_id,
                    update_agent_message(text_block(f"Error: {e}")),
                )
            return "cancelled"  # "error" is not a valid ACP stopReason

    async def _prepopulate_multimodal_context(self, amplifier_blocks: list[dict[str, Any]]) -> None:
        """Pre-populate the conversation context with multi-modal content.

        This adds a user message containing all content blocks (text + images)
        to the conversation context before execution. The orchestrator will
        see this message and include it in the LLM request.

        Args:
            amplifier_blocks: List of Amplifier-compatible content blocks
        """
        if not self._amplifier_session:
            return

        try:
            # Get context manager from the internal AmplifierSession
            # ManagedSession wraps AmplifierSession in _amplifier_session attribute
            amplifier_session = getattr(self._amplifier_session, "_amplifier_session", None)
            if not amplifier_session:
                logger.warning("Internal AmplifierSession not available")
                return

            coordinator = getattr(amplifier_session, "coordinator", None)
            if not coordinator:
                logger.warning("Coordinator not available for multi-modal pre-population")
                return

            context = coordinator.get("context")
            if context is None:
                logger.warning("Context manager not available for multi-modal pre-population")
                return

            # Add user message with multi-modal content
            await context.add_message({"role": "user", "content": amplifier_blocks})
            logger.debug(f"Pre-populated context with {len(amplifier_blocks)} content blocks")

        except Exception as e:
            logger.warning(f"Failed to pre-populate multi-modal context: {e}")

    async def _execute_slash_command(self, parsed: Any) -> str:
        """Execute a slash command and send result to client.

        Args:
            parsed: ParsedCommand from parse_slash_command()

        Returns:
            Stop reason: 'end_turn' or 'error'
        """
        if not self._slash_handler:
            logger.error("Slash handler not initialized")
            return "error"

        try:
            result = await self._slash_handler.execute(parsed)

            # If command returns a prompt to execute through Amplifier
            # This is the correct pattern for commands that need tool invocation
            # with full context and orchestration (e.g., recipes)
            if result.execute_as_prompt:
                logger.info(f"Slash command /{parsed.name} translating to Amplifier prompt")
                return await self._execute_amplifier_prompt(result.execute_as_prompt)

            # Send result as agent message (for direct responses)
            if result.send_as_message and self._conn:
                await self._conn.session_update(
                    self.session_id,
                    update_agent_message(text_block(result.message)),
                )

            # If command changed state, send updated commands list
            if result.update_commands:
                await self._send_available_commands()

            return "end_turn" if result.success else "error"

        except Exception as e:
            logger.exception(f"Error executing slash command: {e}")
            if self._conn:
                await self._conn.session_update(
                    self.session_id,
                    update_agent_message(text_block(f"Error: {e}")),
                )
            return "error"

    async def _execute_amplifier_prompt(self, prompt: str) -> str:
        """Execute a prompt through Amplifier's normal flow.

        This is used when slash commands need full orchestration,
        such as recipe execution which requires proper context,
        tool invocation, and event streaming.

        Args:
            prompt: The prompt to execute through Amplifier

        Returns:
            Stop reason from execution
        """
        if not self._amplifier_session:
            logger.error("Session not initialized for prompt execution")
            return "error"

        try:
            # Execute through Amplifier's normal flow
            # This ensures proper orchestration, context, and tool invocation
            async for event in self._amplifier_session.execute(prompt):
                if self._cancel_event.is_set():
                    return "cancelled"
                await self._on_event(event)

            return "end_turn"

        except asyncio.CancelledError:
            return "cancelled"
        except Exception as e:
            logger.exception(f"Error executing Amplifier prompt: {e}")
            if self._conn:
                await self._conn.session_update(
                    self.session_id,
                    update_agent_message(text_block(f"Error: {e}")),
                )
            return "error"

    async def _on_event(self, event: Any) -> None:
        """Map Amplifier event to ACP session update.

        This is called both from the streaming hook (during execution)
        and from the yield path (synthetic events).

        Uses the SDK's session_update() method which properly formats
        and sends the notification.

        ACP Protocol Mapping:
        - tool:pre -> ToolCallStart (sessionUpdate="tool_call")
        - tool:post -> ToolCallUpdate with status="completed"
        - tool:error -> ToolCallUpdate with status="failed"
        - todo:update -> AgentPlanUpdate (sessionUpdate="plan")
        - content_block:* -> update_agent_message
        - thinking:* -> update_agent_thought
        """
        if not self._conn:
            return

        # Get event type and properties
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type", "")
        event_type = event_type or ""

        props = getattr(event, "properties", None)
        if props is None and isinstance(event, dict):
            props = event
        props = props or {}

        try:
            # Map event types to ACP updates
            if event_type == "content_block:delta":
                # Streaming text delta
                delta = props.get("delta", {})
                text = delta.get("text", "")
                if text:
                    await self._conn.session_update(
                        self.session_id,
                        update_agent_message(text_block(text)),
                    )

            elif event_type == "content_block:end":
                # Final content block - may contain the full text
                block = props.get("block", {})
                text = block.get("text", "")
                if text:
                    await self._conn.session_update(
                        self.session_id,
                        update_agent_message(text_block(text)),
                    )

            elif event_type in ("content", "assistant_message", "text"):
                # Direct text content
                text = props.get("text", "")
                if text:
                    await self._conn.session_update(
                        self.session_id,
                        update_agent_message(text_block(text)),
                    )

            elif event_type == "tool:pre":
                # Tool call starting - ACP ToolCallStart
                tool_info = props.get("tool", {})
                tool_name = (
                    tool_info.get("name", "") if isinstance(tool_info, dict) else str(tool_info)
                )
                tool_call_id = props.get("call_id", "")
                arguments = props.get("arguments", {})

                # Track tool call for approval context (so ACPApprovalBridge
                # can include tool info in permission requests)
                if self._tool_tracker:
                    self._tool_tracker.track(tool_call_id, tool_name, arguments)

                # Generate human-readable title from tool name
                title = self._generate_tool_title(tool_name, arguments)

                # Infer tool kind from name
                kind = self._infer_tool_kind(tool_name)

                update = ToolCallStart(
                    session_update="tool_call",
                    tool_call_id=tool_call_id,
                    title=title,
                    kind=kind,
                    status="pending",
                    raw_input=arguments,
                )
                await self._conn.session_update(self.session_id, update)

            elif event_type == "tool:post":
                # Tool call completed - ACP ToolCallUpdate
                # Clear tool tracking context
                if self._tool_tracker:
                    self._tool_tracker.clear()

                update = ToolCallUpdate(
                    tool_call_id=props.get("call_id", ""),
                    status="completed",
                    raw_output=props.get("result"),
                )
                await self._conn.session_update(self.session_id, update)

            elif event_type == "tool:error":
                # Tool call failed - ACP ToolCallUpdate with status="failed"
                # Clear tool tracking context
                if self._tool_tracker:
                    self._tool_tracker.clear()

                error_info = props.get("error", "Unknown error")
                update = ToolCallUpdate(
                    tool_call_id=props.get("call_id", ""),
                    status="failed",
                    raw_output={"error": str(error_info)},
                )
                await self._conn.session_update(self.session_id, update)

            elif event_type == "todo:update":
                # Todo list update - map to ACP AgentPlanUpdate
                await self._handle_todo_update(props)

            elif event_type in ("thinking:delta", "thinking:final", "thinking:start"):
                # Thinking/reasoning content
                text = props.get("text", "") or props.get("content", "")
                if text:
                    await self._conn.session_update(
                        self.session_id,
                        update_agent_thought(text_block(text)),
                    )

            elif event_type == "content_block:start":
                # Content block starting - check if it's thinking
                block = props.get("block", {})
                block_type = block.get("type", "")
                if block_type == "thinking":
                    # Thinking block starting - we'll get content in delta/end
                    pass
                # For text blocks, wait for delta/end to send content

            # Log unmapped events at debug level
            elif event_type and not event_type.startswith(
                ("session:", "execution:", "llm:", "provider:", "prompt:", "orchestrator:")
            ):
                logger.debug(f"Unmapped event type: {event_type}")

        except Exception as e:
            logger.warning(f"Error sending event {event_type}: {e}")

    def _generate_tool_title(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Generate a human-readable title for a tool call.

        Maps common tool names to descriptive titles, optionally incorporating
        key argument values for context.
        """
        # Tool-specific title generation
        title_map = {
            "read_file": lambda args: f"Reading {args.get('file_path', 'file')}",
            "write_file": lambda args: f"Writing {args.get('file_path', 'file')}",
            "edit_file": lambda args: f"Editing {args.get('file_path', 'file')}",
            "glob": lambda args: f"Finding files: {args.get('pattern', '*')}",
            "grep": lambda args: f"Searching for: {args.get('pattern', '...')}",
            "bash": lambda args: "Running command",
            "web_fetch": lambda args: f"Fetching {args.get('url', 'URL')}",
            "web_search": lambda args: f"Searching: {args.get('query', '...')}",
            "task": lambda args: f"Delegating to {args.get('agent', 'agent')}",
            "todo": lambda args: "Updating task list",
            "recipes": lambda args: f"Running recipe: {args.get('operation', 'operation')}",
            "python_check": lambda args: "Checking Python code",
        }

        if tool_name in title_map:
            try:
                return title_map[tool_name](arguments)
            except Exception:
                pass

        # Default: humanize the tool name
        return tool_name.replace("_", " ").title()

    def _infer_tool_kind(self, tool_name: str) -> str:
        """Infer the ACP tool kind from the tool name.

        ACP tool kinds: read, edit, delete, move, search, execute, think, fetch, other
        """
        # Map tool names to ACP kinds
        kind_map = {
            # Read operations
            "read_file": "read",
            "glob": "read",
            "load_skill": "read",
            # Edit operations
            "write_file": "edit",
            "edit_file": "edit",
            # Search operations
            "grep": "search",
            "web_search": "search",
            # Execute operations
            "bash": "execute",
            "python_check": "execute",
            "recipes": "execute",
            # Fetch operations
            "web_fetch": "fetch",
            # Think/plan operations
            "todo": "think",
            "task": "think",
        }

        return kind_map.get(tool_name, "other")

    async def _handle_todo_update(self, props: dict[str, Any]) -> None:
        """Convert Amplifier todo:update event to ACP AgentPlanUpdate.

        Maps Amplifier todo statuses to ACP plan entry statuses:
        - pending -> pending
        - in_progress -> in_progress
        - completed -> completed

        Maps Amplifier priorities (if present) or defaults to medium.
        """
        if not self._conn:
            return

        todos = props.get("todos", [])
        if not todos:
            return

        # Convert todos to ACP PlanEntry format
        entries = []
        for todo in todos:
            # Map status
            status = todo.get("status", "pending")
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"

            # Map priority (default to medium if not specified)
            priority = todo.get("priority", "medium")
            if priority not in ("high", "medium", "low"):
                priority = "medium"

            # Get content - use content field or activeForm as fallback
            content = todo.get("content", "") or todo.get("activeForm", "Task")

            entries.append(
                PlanEntry(
                    content=content,
                    priority=priority,
                    status=status,
                )
            )

        # Send plan update
        update = AgentPlanUpdate(
            session_update="plan",
            entries=entries,
        )
        await self._conn.session_update(self.session_id, update)

    async def cancel(self) -> None:
        """Cancel ongoing execution."""
        self._cancel_event.set()


# Entry point for stdio mode
async def run_stdio_agent() -> None:
    """Run the Amplifier agent over stdio using the official SDK.

    This is the simplest way to expose Amplifier via ACP - it handles
    all transport complexity automatically.

    Usage:
        python -m amplifier_app_runtime.acp.agent
    """
    from acp import run_agent  # type: ignore[import-untyped]

    # Register recipe tools in host registry
    try:
        from ..host_tools import host_tool_registry
        from .recipe_integration import setup_recipe_tools

        await setup_recipe_tools(host_tool_registry)
        logger.debug("Recipe tools registered in host registry")
    except Exception as e:
        logger.warning(f"Failed to register recipe tools: {e}")

    agent = AmplifierAgent()
    logger.info("Starting Amplifier ACP agent (stdio mode)")
    await run_agent(agent)


if __name__ == "__main__":
    # Direct execution is deprecated. Use the package entry point instead:
    #   python -m amplifier_app_runtime.acp
    #
    # The package entry point properly configures logging to stderr BEFORE
    # importing any modules, which is required for stdio transport to work
    # correctly (stdout must be reserved for JSON-RPC messages only).
    import sys

    print(
        "WARNING: Direct execution of agent.py is deprecated.\n"
        "Use: python -m amplifier_app_runtime.acp\n"
        "This ensures proper stdio isolation for ACP protocol.",
        file=sys.stderr,
    )

    # Still run for backward compatibility, but logging may not be properly isolated
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(run_stdio_agent())
