"""Session Manager for Amplifier Server.

Manages Amplifier sessions with full lifecycle support:
- Session creation and configuration via amplifier-foundation
- Prompt execution with streaming via amplifier-core
- Event forwarding to transport layer
- Session persistence and resume
- Agent spawning with context forwarding

Integrates with amplifier-core's AmplifierSession and Coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .session_store import SessionStore
from .transport.base import Event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from amplifier_foundation import PreparedBundle

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Session lifecycle states."""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class SessionMetadata:
    """Metadata for a session."""

    session_id: str
    state: SessionState = SessionState.CREATED
    bundle_name: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    turn_count: int = 0
    cwd: str = field(default_factory=lambda: str(Path.cwd()))
    parent_session_id: str | None = None
    error: str | None = None


@dataclass
class SessionConfig:
    """Configuration for session creation."""

    bundle: str | None = None
    provider: str | None = None
    model: str | None = None
    max_turns: int = 100
    timeout: float = 300.0
    working_directory: str | None = None
    behaviors: list[str] = field(default_factory=list)
    show_thinking: bool = True
    environment: dict[str, str] = field(default_factory=dict)
    # Optional custom approval system (e.g., ACPApprovalBridge for IDE integration)
    approval_system: Any | None = None


class ManagedSession:
    """A managed Amplifier session with transport integration.

    Wraps an AmplifierSession with:
    - Event streaming via ServerStreamingHook
    - Approval handling via ServerApprovalSystem
    - Display notifications via ServerDisplaySystem
    - Agent spawning via ServerSpawnManager
    - State management and persistence
    - Transcript persistence
    """

    def __init__(
        self,
        session_id: str,
        config: SessionConfig,
        send_fn: Callable[[Event], Coroutine[Any, Any, None]] | None = None,
        store: SessionStore | None = None,
    ):
        self.session_id = session_id
        self.config = config
        self.metadata = SessionMetadata(
            session_id=session_id,
            bundle_name=config.bundle,
            cwd=config.working_directory or str(Path.cwd()),
        )

        # Store send function for later use
        self._send = send_fn

        # Protocol handlers (created during initialization)
        self._approval: Any = None
        self._display: Any = None
        self._streaming_hook: Any = None
        self._spawn_manager: Any = None

        # Persistence
        self._store = store

        # Conversation history (for persistence)
        self._messages: list[dict[str, Any]] = []

        # Internal state
        self._amplifier_session: Any = None  # AmplifierSession when loaded
        self._prepared_bundle: PreparedBundle | None = None
        self._lock = asyncio.Lock()
        self._cancel_event = asyncio.Event()

    async def initialize(
        self,
        prepared_bundle: PreparedBundle | None = None,
        initial_transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the Amplifier session.

        Args:
            prepared_bundle: Optional pre-prepared bundle to use
            initial_transcript: Optional transcript to restore (for resume)
        """
        from .protocols import (
            ServerApprovalSystem,
            ServerDisplaySystem,
            ServerSpawnManager,
            ServerStreamingHook,
        )

        async with self._lock:
            if self.metadata.state != SessionState.CREATED:
                raise RuntimeError(f"Cannot initialize session in state {self.metadata.state}")

            try:
                # Create protocol handlers
                # Use custom approval system if provided (e.g., ACPApprovalBridge for IDE)
                if self.config.approval_system is not None:
                    self._approval = self.config.approval_system
                else:
                    self._approval = ServerApprovalSystem(send_fn=self._send)
                self._display = ServerDisplaySystem(send_fn=self._send)
                self._streaming_hook = ServerStreamingHook(
                    send_fn=self._send,
                    show_thinking=self.config.show_thinking,
                )
                self._spawn_manager = ServerSpawnManager()

                # Create real AmplifierSession - NO MOCK MODE
                # Sessions require a real bundle and provider configuration
                if prepared_bundle:
                    self._prepared_bundle = prepared_bundle
                    await self._create_amplifier_session(
                        prepared_bundle,
                        initial_transcript,
                    )
                elif self.config.bundle:
                    # Load bundle if specified
                    try:
                        from .bundle_manager import BundleManager

                        manager = BundleManager()
                        prepared = await manager.load_and_prepare(
                            bundle_name=self.config.bundle,
                            behaviors=self.config.behaviors,
                            working_directory=Path(self.metadata.cwd)
                            if self.metadata.cwd
                            else None,
                        )
                        self._prepared_bundle = prepared
                        await self._create_amplifier_session(prepared, initial_transcript)
                    except Exception as e:
                        # Bundle loading failed - fail clearly, no silent mock fallback
                        logger.error(
                            f"Session {self.session_id} failed to load bundle "
                            f"'{self.config.bundle}': {e}"
                        )
                        raise RuntimeError(
                            f"Failed to load bundle '{self.config.bundle}'. "
                            f"Ensure the bundle exists and providers are configured "
                            f"(ANTHROPIC_API_KEY or OPENAI_API_KEY). Error: {e}"
                        ) from e
                else:
                    # No bundle specified - fail clearly
                    raise RuntimeError(
                        f"Session {self.session_id} requires a bundle to be specified. "
                        "Pass a bundle name in SessionConfig or a prepared_bundle."
                    )

                # Restore transcript if provided and not already restored
                if initial_transcript and not self._messages:
                    self._messages = list(initial_transcript)
                    self.metadata.turn_count = sum(
                        1 for m in initial_transcript if m.get("role") == "user"
                    )
                    logger.info(
                        f"Restored {len(initial_transcript)} messages for session {self.session_id}"
                    )

                self.metadata.state = SessionState.READY
                self.metadata.updated_at = datetime.now(UTC)

                # Save initial metadata
                self._persist_metadata()

                # Notify client
                await self._emit_state_change()

            except Exception as e:
                self.metadata.state = SessionState.ERROR
                self.metadata.error = str(e)
                logger.error(f"Failed to initialize session {self.session_id}: {e}")
                raise

    async def _create_amplifier_session(
        self,
        prepared_bundle: PreparedBundle,
        initial_transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        """Create AmplifierSession from prepared bundle.

        Args:
            prepared_bundle: The prepared bundle to create session from
            initial_transcript: Optional transcript to restore
        """
        from .protocols import register_spawn_capability, register_streaming_hook
        from .resolvers import AppModuleResolver, FallbackResolver

        try:
            # Wrap bundle resolver with app-layer fallback (like CLI does)
            # This allows modules not in the bundle to be resolved from
            # environment variables or installed packages
            fallback_resolver = FallbackResolver()
            prepared_bundle.resolver = AppModuleResolver(  # type: ignore[assignment]
                bundle_resolver=prepared_bundle.resolver,
                fallback_resolver=fallback_resolver,
            )

            # Create session via foundation's factory method
            session = await prepared_bundle.create_session(
                session_id=self.session_id,
                approval_system=self._approval,
                display_system=self._display,
                session_cwd=Path(self.metadata.cwd) if self.metadata.cwd else None,
                is_resumed=initial_transcript is not None,
            )

            # Register streaming hook for all events
            register_streaming_hook(session, self._streaming_hook)

            # Register spawn capability for agent delegation
            register_spawn_capability(session, prepared_bundle, self._spawn_manager)

            # Register host-defined tools
            await self._register_host_tools(session)

            # Restore transcript if provided
            if initial_transcript:
                await self._restore_transcript(session, initial_transcript)

            self._amplifier_session = session
            logger.info(f"Session {self.session_id} initialized with amplifier-core")

        except ImportError as e:
            # No fallback to mock - fail clearly if dependencies missing
            logger.error(
                f"Session {self.session_id} failed: amplifier-core/foundation not available: {e}"
            )
            raise RuntimeError(
                f"Failed to create AmplifierSession: {e}. "
                "Ensure amplifier-core and amplifier-foundation are installed."
            ) from e

    async def _restore_transcript(
        self,
        session: Any,
        transcript: list[dict[str, Any]],
    ) -> None:
        """Restore conversation transcript into a session.

        Args:
            session: The AmplifierSession to restore messages into
            transcript: List of message dicts with role and content
        """
        try:
            # Get the context module from coordinator
            context = session.coordinator.get("context")
            if context and hasattr(context, "set_messages"):
                # Preserve fresh system message if present
                fresh_system_msg = None
                if hasattr(context, "get_messages"):
                    current_msgs = await context.get_messages()
                    system_msgs = [m for m in current_msgs if m.get("role") == "system"]
                    if system_msgs:
                        fresh_system_msg = system_msgs[0]

                # Filter to only user/assistant messages
                filtered = [msg for msg in transcript if msg.get("role") in ("user", "assistant")]
                await context.set_messages(filtered)

                # Re-inject system message if transcript doesn't have one
                if fresh_system_msg:
                    restored_msgs = await context.get_messages()
                    has_system = any(m.get("role") == "system" for m in restored_msgs)
                    if not has_system:
                        await context.set_messages([fresh_system_msg] + restored_msgs)

                self._messages = list(transcript)
                logger.info(f"Restored {len(filtered)} messages via context.set_messages()")
            else:
                logger.warning("Context module not found or doesn't support set_messages()")
        except Exception as e:
            logger.error(f"Failed to restore transcript: {e}")

    async def _register_host_tools(self, session: Any) -> None:
        """Register host-defined tools on this session.

        Host tools are transport-agnostic tools registered by the host application.
        They are available to all sessions regardless of transport (stdio, HTTP, etc.).

        Args:
            session: The AmplifierSession to register tools on
        """
        from .host_tools import host_tool_registry, register_host_tools_on_session

        if host_tool_registry.count == 0:
            logger.debug(f"No host tools registered for session {self.session_id}")
            return

        try:
            registered = await register_host_tools_on_session(
                session=session,
                registry=host_tool_registry,
                session_id=self.session_id,
                cwd=self.metadata.cwd,
            )
            if registered:
                logger.info(
                    f"Registered {len(registered)} host tools on session {self.session_id}: "
                    f"{', '.join(registered)}"
                )
        except Exception as e:
            logger.warning(f"Failed to register host tools on session {self.session_id}: {e}")

    async def execute(self, prompt: str) -> AsyncIterator[Event]:
        """Execute a prompt and stream results.

        Args:
            prompt: The user's prompt

        Yields:
            Events as they occur during execution
        """
        async with self._lock:
            if self.metadata.state not in (SessionState.READY, SessionState.PAUSED):
                raise RuntimeError(f"Cannot execute in state {self.metadata.state}")

            self.metadata.state = SessionState.RUNNING
            self.metadata.turn_count += 1
            self.metadata.updated_at = datetime.now(UTC)
            self._cancel_event.clear()

        # Reset streaming hook sequence
        if self._streaming_hook:
            self._streaming_hook.reset_sequence()

        # Record user message
        self._messages.append(
            {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Emit prompt submit event
        yield Event(
            type="prompt:submit",
            properties={
                "session_id": self.session_id,
                "prompt": prompt,
                "turn": self.metadata.turn_count,
            },
        )

        response_text = ""

        try:
            if not self._amplifier_session:
                raise RuntimeError(
                    f"Session {self.session_id} not properly initialized. "
                    "AmplifierSession is required for execution."
                )

            # Real execution with amplifier-core
            async for event in self._execute_with_amplifier(prompt):
                # Capture response text from content blocks
                if event.type == "content_block:end":
                    block = event.properties.get("block", {})
                    if "text" in block:
                        response_text += block["text"]
                yield event

            # Record assistant message
            if response_text:
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

            # Update state on completion
            async with self._lock:
                self.metadata.state = SessionState.READY
                self.metadata.updated_at = datetime.now(UTC)

            # Persist after each turn
            self._persist_transcript()
            self._persist_metadata()

            # Emit completion event
            yield Event(
                type="prompt:complete",
                properties={
                    "session_id": self.session_id,
                    "turn": self.metadata.turn_count,
                },
            )

        except asyncio.CancelledError:
            async with self._lock:
                self.metadata.state = SessionState.CANCELLED
            self._persist_metadata()
            yield Event(
                type="cancel:completed",
                properties={"session_id": self.session_id},
            )
            raise

        except Exception as e:
            async with self._lock:
                self.metadata.state = SessionState.ERROR
                self.metadata.error = str(e)
            self._persist_metadata()

            yield Event(
                type="error",
                properties={
                    "session_id": self.session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    async def _execute_with_amplifier(self, prompt: str) -> AsyncIterator[Event]:
        """Execute prompt using real Amplifier session.

        Uses the streaming hook's event queue to yield ALL events
        (thinking, tools, content, etc.) as they occur during execution.
        """
        if not self._streaming_hook:
            logger.warning("No streaming hook configured, events will be limited")
            result = await self._amplifier_session.execute(prompt)
            if result:
                yield Event(
                    type="content_block:start",
                    properties={"session_id": self.session_id, "block_type": "text", "index": 0},
                )
                yield Event(
                    type="content_block:end",
                    properties={
                        "session_id": self.session_id,
                        "index": 0,
                        "block": {"text": str(result)},
                    },
                )
            return

        # Start streaming mode - hook will queue events
        self._streaming_hook.start_streaming()

        # Run execution in background task
        async def run_execute():
            try:
                return await self._amplifier_session.execute(prompt)
            finally:
                # Signal end of streaming
                self._streaming_hook.stop_streaming()

        exec_task = asyncio.create_task(run_execute())

        try:
            # Yield events as they arrive from the hook
            async for event in self._streaming_hook.get_events():
                yield event

            # Wait for execution to complete
            result = await exec_task

            # Log result for debugging (content already streamed via events)
            if result:
                logger.debug(f"Execution completed with result: {str(result)[:100]}...")

        except asyncio.CancelledError:
            exec_task.cancel()
            try:
                await exec_task
            except asyncio.CancelledError:
                pass
            raise
        except Exception as e:
            logger.error(f"Execution error in session {self.session_id}: {e}")
            exec_task.cancel()
            try:
                await exec_task
            except asyncio.CancelledError:
                pass
            raise

    async def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancel_event.set()

        # Cancel pending approvals
        if self._approval:
            self._approval.cancel_all()

        # Cancel AmplifierSession if available
        if self._amplifier_session and hasattr(self._amplifier_session, "cancel"):
            await self._amplifier_session.cancel()

        async with self._lock:
            if self.metadata.state == SessionState.RUNNING:
                self.metadata.state = SessionState.CANCELLED

        self._persist_metadata()

        if self._send:
            await self._send(
                Event(
                    type="cancel:requested",
                    properties={"session_id": self.session_id},
                )
            )

    async def handle_approval(self, request_id: str, choice: str) -> bool:
        """Handle an approval response from the client."""
        if not self._approval:
            return False

        return self._approval.handle_response(request_id, choice)

    async def _emit_state_change(self) -> None:
        """Emit session state change event."""
        if self._send:
            await self._send(
                Event(
                    type="session:state",
                    properties={
                        "session_id": self.session_id,
                        "state": self.metadata.state.value,
                        "turn_count": self.metadata.turn_count,
                        "bundle": self.metadata.bundle_name,
                    },
                )
            )

    def _persist_metadata(self) -> None:
        """Persist session metadata to storage."""
        if not self._store:
            return

        self._store.save_metadata(
            session_id=self.session_id,
            bundle_name=self.metadata.bundle_name,
            turn_count=self.metadata.turn_count,
            created_at=self.metadata.created_at,
            updated_at=self.metadata.updated_at,
            cwd=self.metadata.cwd,
            parent_session_id=self.metadata.parent_session_id,
            state=self.metadata.state.value,
            error=self.metadata.error,
        )

    def _persist_transcript(self) -> None:
        """Persist conversation transcript to storage."""
        if not self._store or not self._messages:
            return

        self._store.save_transcript(self.session_id, self._messages)

    def set_send_function(self, send_fn: Callable[[Event], Coroutine[Any, Any, None]]) -> None:
        """Update the send function for all protocol handlers."""
        self._send = send_fn
        if self._approval:
            self._approval.set_send_fn(send_fn)
        if self._display:
            self._display.set_send_fn(send_fn)
        if self._streaming_hook:
            self._streaming_hook.set_send_fn(send_fn)

    def get_transcript(self) -> list[dict[str, Any]]:
        """Get the conversation transcript."""
        return list(self._messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session metadata."""
        return {
            "session_id": self.session_id,
            "state": self.metadata.state.value,
            "bundle": self.metadata.bundle_name,
            "created_at": self.metadata.created_at.isoformat(),
            "updated_at": self.metadata.updated_at.isoformat(),
            "turn_count": self.metadata.turn_count,
            "cwd": self.metadata.cwd,
            "parent_session_id": self.metadata.parent_session_id,
            "error": self.metadata.error,
        }

    async def cleanup(self) -> None:
        """Clean up session resources."""
        if self._amplifier_session:
            try:
                if hasattr(self._amplifier_session, "__aexit__"):
                    await self._amplifier_session.__aexit__(None, None, None)
                elif hasattr(self._amplifier_session, "cleanup"):
                    await self._amplifier_session.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up session {self.session_id}: {e}")


class SessionManager:
    """Manager for all active and saved sessions.

    Provides:
    - Session creation and lookup
    - Session lifecycle management
    - Concurrent session support
    - Session persistence via SessionStore
    - Bundle management via BundleManager
    """

    def __init__(
        self,
        store: SessionStore | None = None,
    ) -> None:
        """Initialize session manager.

        Args:
            store: Optional session store for persistence
        """
        self._store = store or SessionStore()
        self._active: dict[str, ManagedSession] = {}
        self._bundle_manager: Any = None
        self._lock = asyncio.Lock()

    async def _get_bundle_manager(self) -> Any:
        """Get or create the bundle manager."""
        if self._bundle_manager is None:
            from .bundle_manager import BundleManager

            self._bundle_manager = BundleManager()
            await self._bundle_manager.initialize()
        return self._bundle_manager

    async def create(
        self,
        config: SessionConfig | None = None,
        session_id: str | None = None,
        send_fn: Callable[[Event], Coroutine[Any, Any, None]] | None = None,
        auto_initialize: bool = False,
    ) -> ManagedSession:
        """Create a new session.

        Args:
            config: Session configuration
            session_id: Optional session ID (generated if not provided)
            send_fn: Optional function to send events to client
            auto_initialize: If True, automatically initialize the session

        Returns:
            The created ManagedSession (call initialize() to start)
        """
        config = config or SessionConfig()
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"

        session = ManagedSession(
            session_id=session_id,
            config=config,
            send_fn=send_fn,
            store=self._store,
        )

        async with self._lock:
            self._active[session_id] = session

        # Optionally initialize immediately
        if auto_initialize:
            # Load and prepare bundle if specified
            prepared_bundle = None
            if config.bundle:
                try:
                    manager = await self._get_bundle_manager()
                    prepared_bundle = await manager.load_and_prepare(
                        bundle_name=config.bundle,
                        behaviors=config.behaviors,
                        working_directory=Path(config.working_directory)
                        if config.working_directory
                        else None,
                    )
                except Exception as e:
                    logger.error(f"Failed to prepare bundle '{config.bundle}': {e}")
                    # Invalidate cache on failure
                    if self._bundle_manager:
                        await self._bundle_manager.invalidate_cache()
                    raise

            await session.initialize(prepared_bundle=prepared_bundle)

        logger.info(f"Created session {session_id} with bundle {config.bundle}")
        return session

    async def get(self, session_id: str) -> ManagedSession | None:
        """Get an active session by ID.

        Args:
            session_id: The session ID to look up

        Returns:
            The session if found, None otherwise
        """
        return self._active.get(session_id)

    async def resume(
        self,
        session_id: str,
        send_fn: Callable[[Event], Coroutine[Any, Any, None]] | None = None,
        force_bundle: str | None = None,
    ) -> ManagedSession | None:
        """Resume a saved session.

        Args:
            session_id: The session ID to resume
            send_fn: Optional function to send events to client
            force_bundle: Optional bundle to force (overrides saved bundle)

        Returns:
            The resumed session if found, None if session doesn't exist
        """
        # Check if already active
        if session_id in self._active:
            session = self._active[session_id]
            if send_fn:
                session.set_send_function(send_fn)
            return session

        # Try to load from storage
        metadata = self._store.load_metadata(session_id)
        if not metadata:
            return None

        transcript = self._store.load_transcript(session_id)

        # Create config from saved metadata
        bundle_name = force_bundle or metadata.get("bundle_name")
        config = SessionConfig(
            bundle=bundle_name,
            working_directory=metadata.get("cwd"),
        )

        # Create new session
        session = ManagedSession(
            session_id=session_id,
            config=config,
            send_fn=send_fn,
            store=self._store,
        )

        # Restore metadata
        session.metadata.turn_count = metadata.get("turn_count", 0)
        session.metadata.created_at = metadata.get("created_at", datetime.now(UTC))

        # Load and prepare bundle
        prepared_bundle = None
        if bundle_name:
            try:
                manager = await self._get_bundle_manager()
                prepared_bundle = await manager.load_and_prepare(
                    bundle_name=bundle_name,
                    working_directory=Path(config.working_directory)
                    if config.working_directory
                    else None,
                )
            except Exception as e:
                logger.error(f"Failed to prepare bundle '{bundle_name}' for resume: {e}")
                if self._bundle_manager:
                    await self._bundle_manager.invalidate_cache()
                raise

        async with self._lock:
            self._active[session_id] = session

        # Initialize with transcript
        await session.initialize(
            prepared_bundle=prepared_bundle,
            initial_transcript=transcript,
        )

        logger.info(f"Resumed session {session_id}")
        return session

    async def delete(self, session_id: str, delete_saved: bool = True) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete
            delete_saved: If True, also delete from persistent storage

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            session = self._active.pop(session_id, None)

        if session:
            await session.cleanup()

        # Optionally delete from storage
        deleted = False
        if delete_saved:
            deleted = self._store.delete_session(session_id)

        if deleted:
            logger.info(f"Deleted session {session_id}")

        return deleted or session is not None

    async def list_sessions(
        self,
        limit: int = 50,
        include_completed: bool = False,
    ) -> list[dict[str, Any]]:
        """List all sessions.

        Args:
            limit: Maximum number of sessions to return
            include_completed: Whether to include completed sessions

        Returns:
            List of session info dicts
        """
        sessions = []

        # Add active sessions
        for session in self._active.values():
            info = session.to_dict()
            info["is_active"] = True
            sessions.append(info)

        # Add saved sessions from storage
        saved = self._store.list_sessions(limit=limit)
        for saved_info in saved:
            session_id = saved_info.get("session_id")
            if session_id not in self._active:
                saved_info["is_active"] = False
                if include_completed or saved_info.get("state") != "completed":
                    sessions.append(saved_info)

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return sessions[:limit]

    async def list_active(self) -> list[dict[str, Any]]:
        """List all active sessions.

        Returns:
            List of active session info dicts
        """
        return [{**session.to_dict(), "active": True} for session in self._active.values()]

    def list_saved(self, min_turns: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        """List saved sessions from storage.

        Args:
            min_turns: Minimum turn count to include
            limit: Maximum number of sessions to return

        Returns:
            List of saved session info dicts
        """
        return self._store.list_sessions(min_turns=min_turns, limit=limit)

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get info about a session (active or saved).

        Args:
            session_id: The session ID to look up

        Returns:
            Session info dict or None if not found
        """
        # Check active sessions first
        if session_id in self._active:
            info = self._active[session_id].to_dict()
            info["active"] = True
            return info

        # Check storage
        saved = self._store.load_metadata(session_id)
        if saved:
            saved["active"] = False
            return saved

        return None

    def get_active_count(self) -> int:
        """Get count of active sessions."""
        return len(self._active)

    @property
    def active_count(self) -> int:
        """Get count of active (ready/running) sessions."""
        return sum(
            1
            for s in self._active.values()
            if s.metadata.state in (SessionState.READY, SessionState.RUNNING)
        )

    @property
    def total_count(self) -> int:
        """Get total count of sessions in memory."""
        return len(self._active)

    async def cleanup_completed(self, max_age_seconds: int = 86400) -> int:
        """Clean up old completed sessions from memory.

        Args:
            max_age_seconds: Maximum age in seconds for completed sessions

        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
        to_remove = []

        async with self._lock:
            for session_id, session in self._active.items():
                if (
                    session.metadata.state in (SessionState.COMPLETED, SessionState.ERROR)
                    and session.metadata.updated_at < cutoff
                ):
                    to_remove.append(session_id)

            for session_id in to_remove:
                session = self._active.pop(session_id)
                await session.cleanup()

        return len(to_remove)

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old completed sessions from memory.

        Args:
            max_age_hours: Maximum age in hours for completed sessions

        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        to_remove = []

        async with self._lock:
            for session_id, session in self._active.items():
                if (
                    session.metadata.state in (SessionState.COMPLETED, SessionState.ERROR)
                    and session.metadata.updated_at < cutoff
                ):
                    to_remove.append(session_id)

            for session_id in to_remove:
                session = self._active.pop(session_id)
                await session.cleanup()

        return len(to_remove)

    @property
    def store(self) -> SessionStore:
        """Get the session store."""
        return self._store


# Global session manager instance for use by routes and adapters
session_manager = SessionManager()
