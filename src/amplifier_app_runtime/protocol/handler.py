"""Command Handler - Transport-agnostic business logic.

Processes commands and yields correlated events.
All transports (HTTP, WebSocket, stdio) use this same handler.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .commands import Command, CommandType
from .events import Event, EventType

if TYPE_CHECKING:
    from ..session import SessionManager

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles protocol commands and yields correlated events.

    This is the central business logic layer. All transports delegate
    command processing here, ensuring consistent behavior regardless
    of how commands arrive (HTTP, WebSocket, stdio, etc.).

    Usage:
        handler = CommandHandler(session_manager)

        # Process a command
        async for event in handler.handle(command):
            transport.send(event)

    Correlation:
        Every yielded event has `correlation_id` set to the command's `id`,
        enabling clients to match responses with requests.

    Streaming:
        Commands like `prompt.send` yield multiple events with increasing
        `sequence` numbers. The final event has `final=True`.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize handler with session manager.

        Args:
            session_manager: Manager for session CRUD and execution
        """
        self._sessions = session_manager

    async def handle(self, command: Command) -> AsyncIterator[Event]:
        """Process a command and yield correlated events.

        Args:
            command: The command to process

        Yields:
            Events correlated to the command
        """
        logger.debug(f"Handling command: {command.cmd} (id={command.id})")

        try:
            # Dispatch to handler method
            match command.cmd:
                # Session lifecycle
                case CommandType.SESSION_CREATE.value:
                    async for event in self._session_create(command):
                        yield event

                case CommandType.SESSION_GET.value:
                    async for event in self._session_get(command):
                        yield event

                case CommandType.SESSION_INFO.value:
                    async for event in self._session_info(command):
                        yield event

                case CommandType.SESSION_LIST.value:
                    async for event in self._session_list(command):
                        yield event

                case CommandType.SESSION_DELETE.value:
                    async for event in self._session_delete(command):
                        yield event

                # Execution
                case CommandType.PROMPT_SEND.value:
                    async for event in self._prompt_send(command):
                        yield event

                case CommandType.PROMPT_CANCEL.value:
                    async for event in self._prompt_cancel(command):
                        yield event

                # Approval
                case CommandType.APPROVAL_RESPOND.value:
                    async for event in self._approval_respond(command):
                        yield event

                # Server
                case CommandType.PING.value:
                    yield Event.pong(command.id)

                case CommandType.CAPABILITIES.value:
                    async for event in self._capabilities(command):
                        yield event

                # Configuration
                case CommandType.CONFIG_INIT.value:
                    async for event in self._config_init(command):
                        yield event

                case CommandType.CONFIG_GET.value:
                    async for event in self._config_get(command):
                        yield event

                case CommandType.PROVIDER_LIST.value:
                    async for event in self._provider_list(command):
                        yield event

                case CommandType.PROVIDER_DETECT.value:
                    async for event in self._provider_detect(command):
                        yield event

                case CommandType.BUNDLE_LIST.value:
                    async for event in self._bundle_list(command):
                        yield event

                case CommandType.BUNDLE_INSTALL.value:
                    async for event in self._bundle_install(command):
                        yield event

                case CommandType.BUNDLE_ADD.value:
                    async for event in self._bundle_add(command):
                        yield event

                case CommandType.BUNDLE_REMOVE.value:
                    async for event in self._bundle_remove(command):
                        yield event

                case CommandType.BUNDLE_INFO.value:
                    async for event in self._bundle_info(command):
                        yield event

                case CommandType.SESSION_RESET.value:
                    async for event in self._session_reset(command):
                        yield event

                # Agent commands
                case CommandType.AGENTS_LIST.value:
                    async for event in self._agents_list(command):
                        yield event

                case CommandType.AGENTS_INFO.value:
                    async for event in self._agents_info(command):
                        yield event

                # Tool management
                case CommandType.TOOLS_LIST.value:
                    async for event in self._tools_list(command):
                        yield event

                case CommandType.TOOLS_INFO.value:
                    async for event in self._tools_info(command):
                        yield event

                # Slash commands metadata (for TUI/CLI autocomplete)
                case CommandType.SLASH_COMMANDS_LIST.value:
                    async for event in self._slash_commands_list(command):
                        yield event

                case _:
                    yield Event.error(
                        command.id,
                        error=f"Unknown command: {command.cmd}",
                        code="UNKNOWN_COMMAND",
                    )

        except Exception as e:
            logger.exception(f"Error handling command {command.id}: {e}")
            yield Event.error(
                command.id,
                error=str(e),
                code="HANDLER_ERROR",
            )

    # =========================================================================
    # Session Commands
    # =========================================================================

    async def _session_create(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.create command.

        Supports two modes:
        1. Named bundle: {"bundle": "foundation"}
        2. Runtime bundle: {"bundle_definition": {"name": "...", "tools": [...]}}
        """
        from ..session import SessionConfig

        prepared_bundle = None
        bundle_definition = command.get_param("bundle_definition")

        # Handle runtime bundle definition
        if bundle_definition:
            try:
                from amplifier_foundation import Bundle, load_bundle

                from ..bundle_manager import BundleManager

                # Ensure session defaults (orchestrator + context)
                session_config = bundle_definition.get("session", {})
                if "orchestrator" not in session_config:
                    session_config["orchestrator"] = "loop-streaming"
                if "context" not in session_config:
                    session_config["context"] = "context-simple"

                # Create bundle from definition
                bundle = Bundle(
                    name=bundle_definition.get("name", "runtime-bundle"),
                    version=bundle_definition.get("version", "1.0.0"),
                    description=bundle_definition.get("description", ""),
                    providers=bundle_definition.get("providers", []),
                    tools=bundle_definition.get("tools", []),
                    hooks=bundle_definition.get("hooks", []),
                    agents=bundle_definition.get("agents", {}),
                    instruction=bundle_definition.get("instructions")
                    or bundle_definition.get("instruction"),
                    session=session_config,
                )

                # If user explicitly requests a base bundle, compose on top
                if "base" in bundle_definition:
                    base = await load_bundle(bundle_definition["base"])
                    bundle = base.compose(bundle)

                # Auto-detect provider if not specified
                if not bundle_definition.get("providers"):
                    manager = BundleManager()
                    await manager.initialize()
                    provider_bundle = await manager._auto_detect_provider()
                    if provider_bundle:
                        bundle = bundle.compose(provider_bundle)

                # Prepare the bundle
                prepared_bundle = await bundle.prepare()

            except Exception as e:
                yield Event.error(
                    command.id,
                    error=f"Failed to create runtime bundle: {e}",
                    code="BUNDLE_ERROR",
                )
                return

        # Extract config from params
        config = SessionConfig(
            bundle=command.get_param("bundle") if not prepared_bundle else None,
            provider=command.get_param("provider"),
            model=command.get_param("model"),
            working_directory=command.get_param("working_directory"),
            behaviors=command.get_param("behaviors") or [],
        )

        # Create and initialize session
        session = await self._sessions.create(config=config)
        await session.initialize(prepared_bundle=prepared_bundle)

        # Return session info
        yield Event.result(
            command.id,
            data={
                "session_id": session.session_id,
                "state": session.metadata.state.value,
                "bundle": session.metadata.bundle_name,
            },
        )

    async def _session_get(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.get command."""
        session_id = command.require_param("session_id")
        session = await self._sessions.get(session_id)

        if not session:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        yield Event.result(command.id, data=session.to_dict())

    async def _session_info(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.info command - detailed session information."""
        session_id = command.require_param("session_id")

        # Get session info from manager (handles both active and saved)
        info = self._sessions.get_session_info(session_id)

        if not info:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        # Enrich with additional runtime info if session is active
        session = await self._sessions.get(session_id)
        if session:
            # Add tools list if available
            amp_session = getattr(session, "_amplifier_session", None)
            if amp_session:
                config = getattr(amp_session, "config", {})
                if isinstance(config, dict):
                    tools = config.get("tools", [])
                    # Tools can be a list or dict depending on config format
                    if isinstance(tools, dict):
                        info["tools"] = list(tools.keys())
                    elif isinstance(tools, list):
                        info["tools"] = tools

        yield Event.result(command.id, data=info)

    async def _session_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.list command."""
        # Get both active and saved sessions
        active = await self._sessions.list_active()
        saved = self._sessions.list_saved()
        yield Event.result(command.id, data={"active": active, "saved": saved})

    async def _session_delete(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.delete command."""
        session_id = command.require_param("session_id")
        deleted = await self._sessions.delete(session_id)

        if not deleted:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        yield Event.result(command.id, data={"deleted": True, "session_id": session_id})

    # =========================================================================
    # Execution Commands
    # =========================================================================

    async def _prompt_send(self, command: Command) -> AsyncIterator[Event]:
        """Handle prompt.send command.

        This is a streaming command - yields multiple events with
        increasing sequence numbers, all correlated to the command.
        """
        session_id = command.require_param("session_id")
        content = command.require_param("content")
        _stream = command.get_param("stream", True)  # Reserved for future non-streaming mode

        session = await self._sessions.get(session_id)
        if not session:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        # Acknowledge receipt
        yield Event.ack(command.id, message="Processing prompt")

        # Execute and stream events
        sequence = 0
        try:
            async for session_event in session.execute(content):
                # Map session events to protocol events with correlation
                protocol_event = self._map_session_event(
                    session_event,
                    command.id,
                    sequence,
                )
                if protocol_event:
                    yield protocol_event
                    sequence += 1

            # Final completion event
            yield Event.create(
                EventType.RESULT,
                data={
                    "session_id": session_id,
                    "state": session.metadata.state.value,
                    "turn": session.metadata.turn_count,
                },
                correlation_id=command.id,
                sequence=sequence,
                final=True,
            )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="EXECUTION_ERROR",
            )

    async def _prompt_cancel(self, command: Command) -> AsyncIterator[Event]:
        """Handle prompt.cancel command."""
        session_id = command.require_param("session_id")
        session = await self._sessions.get(session_id)

        if not session:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        await session.cancel()

        yield Event.result(
            command.id,
            data={
                "cancelled": True,
                "session_id": session_id,
                "state": session.metadata.state.value,
            },
        )

    # =========================================================================
    # Approval Commands
    # =========================================================================

    async def _approval_respond(self, command: Command) -> AsyncIterator[Event]:
        """Handle approval.respond command."""
        session_id = command.require_param("session_id")
        request_id = command.require_param("request_id")
        choice = command.require_param("choice")

        session = await self._sessions.get(session_id)
        if not session:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        handled = await session.handle_approval(request_id, choice)

        if not handled:
            yield Event.error(
                command.id,
                error=f"Approval request not found: {request_id}",
                code="APPROVAL_NOT_FOUND",
            )
            return

        yield Event.result(
            command.id,
            data={
                "resolved": True,
                "request_id": request_id,
                "choice": choice,
            },
        )

    # =========================================================================
    # Server Commands
    # =========================================================================

    async def _capabilities(self, command: Command) -> AsyncIterator[Event]:
        """Handle capabilities command."""
        yield Event.result(
            command.id,
            data={
                "version": "0.1.0",
                "protocol_version": "1.0",
                "commands": [cmd.value for cmd in CommandType],
                "events": [evt.value for evt in EventType],
                "features": {
                    "streaming": True,
                    "approval": True,
                    "spawning": True,
                },
            },
        )

    # =========================================================================
    # Configuration Commands
    # =========================================================================

    async def _config_init(self, command: Command) -> AsyncIterator[Event]:
        """Handle config.init command.

        Initializes runtime configuration with provider detection and bundle setup.
        Streams progress events during initialization.
        """
        import os

        bundle = command.get_param("bundle", "foundation")
        detect_providers = command.get_param("detect_providers", True)

        # Start event
        yield Event.create(
            EventType.CONFIG_INIT_STARTED,
            data={"bundle": bundle, "detect_providers": detect_providers},
            correlation_id=command.id,
            sequence=0,
        )

        sequence = 1
        detected_providers = []

        # Detect providers from environment
        if detect_providers:
            provider_checks = [
                ("anthropic", "ANTHROPIC_API_KEY"),
                ("openai", "OPENAI_API_KEY"),
                ("azure-openai", "AZURE_OPENAI_API_KEY"),
                ("google", "GOOGLE_API_KEY"),
            ]

            for name, env_var in provider_checks:
                available = os.getenv(env_var) is not None
                if available:
                    detected_providers.append(name)

                yield Event.create(
                    EventType.CONFIG_INIT_PROVIDER_DETECTED,
                    data={
                        "name": name,
                        "env_var": env_var,
                        "available": available,
                    },
                    correlation_id=command.id,
                    sequence=sequence,
                )
                sequence += 1

        # Bundle set event
        yield Event.create(
            EventType.CONFIG_INIT_BUNDLE_SET,
            data={"bundle": bundle},
            correlation_id=command.id,
            sequence=sequence,
        )
        sequence += 1

        # Completed event with full config
        yield Event.result(
            command.id,
            data={
                "initialized": True,
                "config": {
                    "default_bundle": bundle,
                    "providers_detected": detected_providers,
                    "default_provider": detected_providers[0] if detected_providers else None,
                },
            },
        )

    async def _config_get(self, command: Command) -> AsyncIterator[Event]:
        """Handle config.get command."""
        import os
        from pathlib import Path

        # Get current configuration
        config = {
            "data_dir": str(Path.home() / ".amplifier-runtime"),
            "default_bundle": os.getenv("AMPLIFIER_BUNDLE", "foundation"),
            "providers_configured": [],
        }

        # Check configured providers
        provider_vars = [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("azure-openai", "AZURE_OPENAI_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
        ]

        for name, env_var in provider_vars:
            if os.getenv(env_var):
                config["providers_configured"].append(name)

        if config["providers_configured"]:
            config["default_provider"] = config["providers_configured"][0]
        else:
            config["default_provider"] = None

        yield Event.result(command.id, data=config)

    async def _provider_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle provider.list command."""
        import os

        providers = []
        provider_checks = [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("azure-openai", "AZURE_OPENAI_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
        ]

        for name, env_var in provider_checks:
            providers.append(
                {
                    "name": name,
                    "env_var": env_var,
                    "available": os.getenv(env_var) is not None,
                }
            )

        yield Event.result(command.id, data={"providers": providers})

    async def _provider_detect(self, command: Command) -> AsyncIterator[Event]:
        """Handle provider.detect command."""
        import os

        detected = []
        provider_checks = [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("azure-openai", "AZURE_OPENAI_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
        ]

        for name, env_var in provider_checks:
            if os.getenv(env_var):
                detected.append(
                    {
                        "name": name,
                        "env_var": env_var,
                    }
                )

        yield Event.result(
            command.id,
            data={
                "detected": detected,
                "count": len(detected),
                "default": detected[0]["name"] if detected else None,
            },
        )

    async def _bundle_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle bundle.list command."""
        from ..bundle_manager import BundleManager

        manager = BundleManager()
        bundles = await manager.list_bundles()

        yield Event.result(
            command.id,
            data={
                "bundles": [
                    {"name": b.name, "description": b.description, "uri": b.uri} for b in bundles
                ],
            },
        )

    async def _bundle_install(self, command: Command) -> AsyncIterator[Event]:
        """Handle bundle.install command.

        Installs a bundle from a source (git URL, local path, or registry).
        Streams progress events during installation.
        """
        from ..bundle_manager import BundleManager

        source = command.require_param("source")
        name = command.get_param("name")  # Optional, derived from source if not provided

        manager = BundleManager()
        sequence = 0

        # Start event
        yield Event.create(
            EventType.BUNDLE_INSTALL_STARTED,
            data={"source": source, "name": name},
            correlation_id=command.id,
            sequence=sequence,
        )
        sequence += 1

        try:
            # Install with progress callback
            async for progress in manager.install_bundle(source, name):
                yield Event.create(
                    EventType.BUNDLE_INSTALL_PROGRESS,
                    data=progress,
                    correlation_id=command.id,
                    sequence=sequence,
                )
                sequence += 1

            # Get installed bundle info
            installed = await manager.get_bundle_info(name or manager.name_from_source(source))

            yield Event.result(
                command.id,
                data={
                    "installed": True,
                    "name": installed.name,
                    "path": str(installed.path),
                    "source": source,
                },
            )

        except Exception as e:
            yield Event.create(
                EventType.BUNDLE_INSTALL_ERROR,
                data={"error": str(e), "source": source},
                correlation_id=command.id,
                sequence=sequence,
                final=True,
            )

    async def _bundle_add(self, command: Command) -> AsyncIterator[Event]:
        """Handle bundle.add command.

        Registers a local bundle path with a name.
        """
        from ..bundle_manager import BundleManager

        path = command.require_param("path")
        name = command.require_param("name")

        manager = BundleManager()

        try:
            bundle_info = await manager.add_local_bundle(path, name)
            yield Event.result(
                command.id,
                data={
                    "added": True,
                    "name": bundle_info.name,
                    "path": str(bundle_info.path),
                },
            )
        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="BUNDLE_ADD_FAILED",
            )

    async def _bundle_remove(self, command: Command) -> AsyncIterator[Event]:
        """Handle bundle.remove command."""
        from ..bundle_manager import BundleManager

        name = command.require_param("name")

        manager = BundleManager()

        try:
            removed = await manager.remove_bundle(name)
            yield Event.result(
                command.id,
                data={"removed": removed, "name": name},
            )
        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="BUNDLE_REMOVE_FAILED",
            )

    async def _bundle_info(self, command: Command) -> AsyncIterator[Event]:
        """Handle bundle.info command."""
        from ..bundle_manager import BundleManager

        name = command.require_param("name")

        manager = BundleManager()

        try:
            info = await manager.get_bundle_info(name)
            yield Event.result(
                command.id,
                data={
                    "name": info.name,
                    "description": info.description,
                    "uri": info.uri,
                    "path": str(info.path) if info.path else None,
                    "source": info.source,
                },
            )
        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="BUNDLE_NOT_FOUND",
            )

    # =========================================================================
    # Session Reset
    # =========================================================================

    async def _session_reset(self, command: Command) -> AsyncIterator[Event]:
        """Handle session.reset command.

        Resets a session with optional new bundle configuration.
        Creates a new session and optionally preserves history.
        """
        from ..session import SessionConfig

        session_id = command.require_param("session_id")
        bundle = command.get_param("bundle")
        preserve_history = command.get_param("preserve_history", False)

        # Get old session
        old_session = await self._sessions.get(session_id)
        if not old_session:
            yield Event.error(
                command.id,
                error=f"Session not found: {session_id}",
                code="SESSION_NOT_FOUND",
            )
            return

        # Get bundle name from old session metadata
        old_bundle = old_session.metadata.bundle_name or "foundation"
        new_bundle = bundle or old_bundle

        # Start event
        yield Event.create(
            EventType.SESSION_RESET_STARTED,
            data={
                "session_id": session_id,
                "bundle": new_bundle,
                "preserve_history": preserve_history,
            },
            correlation_id=command.id,
            sequence=0,
        )

        try:
            # Create new session config
            # Use working directory from old session metadata (cwd field)
            config = SessionConfig(
                bundle=new_bundle,
                working_directory=old_session.metadata.cwd,
            )

            # Create new session
            new_session = await self._sessions.create(config=config)
            await new_session.initialize()

            # TODO: If preserve_history, copy conversation history
            # This would require serializing/deserializing messages

            # Delete old session
            await self._sessions.delete(session_id)

            # Completed event
            yield Event.create(
                EventType.SESSION_RESET_COMPLETED,
                data={
                    "old_session_id": session_id,
                    "new_session_id": new_session.session_id,
                    "bundle": new_session.metadata.bundle_name,
                    "state": new_session.metadata.state.value,
                },
                correlation_id=command.id,
                sequence=1,
                final=True,
            )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="SESSION_RESET_FAILED",
            )

    # =========================================================================
    # Agent Commands
    # =========================================================================

    async def _agents_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle agents.list command.

        Lists all available agents from the current bundle configuration.
        Can optionally filter by session to show agents available in that session's bundle.
        """
        session_id = command.get_param("session_id")

        try:
            agents = []

            if session_id:
                # Get agents from specific session's bundle
                session = await self._sessions.get(session_id)
                if not session:
                    yield Event.error(
                        command.id,
                        error=f"Session not found: {session_id}",
                        code="SESSION_NOT_FOUND",
                    )
                    return

                # Extract agents from session's coordinator config
                if hasattr(session, "_amplifier_session") and session._amplifier_session:
                    config = getattr(session._amplifier_session, "config", {})
                    if isinstance(config, dict):
                        agent_configs = config.get("agents", {})
                    else:
                        agent_configs = getattr(config, "agents", {}) or {}

                    for name, agent_config in agent_configs.items():
                        if isinstance(agent_config, dict):
                            agents.append(
                                {
                                    "name": name,
                                    "description": agent_config.get("description", ""),
                                    "bundle": agent_config.get("bundle", ""),
                                }
                            )
                        else:
                            # Handle object-style config
                            agents.append(
                                {
                                    "name": name,
                                    "description": getattr(agent_config, "description", ""),
                                    "bundle": getattr(agent_config, "bundle", ""),
                                }
                            )
            else:
                # List agents from all active sessions
                all_sessions = await self._sessions.list_active()
                seen_agents = set()

                for sess_info in all_sessions:
                    session = await self._sessions.get(sess_info.session_id)
                    if (
                        session
                        and hasattr(session, "_amplifier_session")
                        and session._amplifier_session
                    ):
                        config = getattr(session._amplifier_session, "config", {})
                        if isinstance(config, dict):
                            agent_configs = config.get("agents", {})
                        else:
                            agent_configs = getattr(config, "agents", {}) or {}

                        for name, agent_config in agent_configs.items():
                            if name not in seen_agents:
                                seen_agents.add(name)
                                if isinstance(agent_config, dict):
                                    agents.append(
                                        {
                                            "name": name,
                                            "description": agent_config.get("description", ""),
                                            "bundle": agent_config.get("bundle", ""),
                                        }
                                    )
                                else:
                                    agents.append(
                                        {
                                            "name": name,
                                            "description": getattr(agent_config, "description", ""),
                                            "bundle": getattr(agent_config, "bundle", ""),
                                        }
                                    )

            yield Event.result(
                command.id,
                data={"agents": agents, "count": len(agents)},
            )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="AGENTS_LIST_FAILED",
            )

    async def _agents_info(self, command: Command) -> AsyncIterator[Event]:
        """Handle agents.info command.

        Gets detailed information about a specific agent.
        """
        agent_name = command.require_param("name")
        session_id = command.get_param("session_id")

        try:
            agent_info = None

            # Find the agent in session config
            if session_id:
                session = await self._sessions.get(session_id)
                if (
                    session
                    and hasattr(session, "_amplifier_session")
                    and session._amplifier_session
                ):
                    config = getattr(session._amplifier_session, "config", {})
                    if isinstance(config, dict):
                        agent_configs = config.get("agents", {})
                    else:
                        agent_configs = getattr(config, "agents", {}) or {}

                    if agent_name in agent_configs:
                        agent_config = agent_configs[agent_name]
                        if isinstance(agent_config, dict):
                            agent_info = {
                                "name": agent_name,
                                "description": agent_config.get("description", ""),
                                "bundle": agent_config.get("bundle", ""),
                                "instructions": agent_config.get("instructions", ""),
                            }
                        else:
                            agent_info = {
                                "name": agent_name,
                                "description": getattr(agent_config, "description", ""),
                                "bundle": getattr(agent_config, "bundle", ""),
                                "instructions": getattr(agent_config, "instructions", ""),
                            }

            if not agent_info:
                # Search all sessions
                all_sessions = await self._sessions.list_active()
                for sess_info in all_sessions:
                    session = await self._sessions.get(sess_info.session_id)
                    if (
                        session
                        and hasattr(session, "_amplifier_session")
                        and session._amplifier_session
                    ):
                        config = getattr(session._amplifier_session, "config", {})
                        if isinstance(config, dict):
                            agent_configs = config.get("agents", {})
                        else:
                            agent_configs = getattr(config, "agents", {}) or {}

                        if agent_name in agent_configs:
                            agent_config = agent_configs[agent_name]
                            if isinstance(agent_config, dict):
                                agent_info = {
                                    "name": agent_name,
                                    "description": agent_config.get("description", ""),
                                    "bundle": agent_config.get("bundle", ""),
                                    "instructions": agent_config.get("instructions", ""),
                                }
                            else:
                                agent_info = {
                                    "name": agent_name,
                                    "description": getattr(agent_config, "description", ""),
                                    "bundle": getattr(agent_config, "bundle", ""),
                                    "instructions": getattr(agent_config, "instructions", ""),
                                }
                            break

            if agent_info:
                yield Event.result(command.id, data=agent_info)
            else:
                yield Event.error(
                    command.id,
                    error=f"Agent not found: {agent_name}",
                    code="AGENT_NOT_FOUND",
                )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="AGENTS_INFO_FAILED",
            )

    async def _tools_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle tools.list command.

        Lists all available tools from the current session's bundle configuration.
        """
        session_id = command.get_param("session_id")

        try:
            tools: list[dict[str, Any]] = []

            def extract_tools_from_session(session: Any) -> list[dict[str, Any]]:
                """Extract tool list from a ManagedSession."""
                result = []
                amp_session = getattr(session, "_amplifier_session", None)
                if not amp_session:
                    return result

                config = getattr(amp_session, "config", {})
                if not isinstance(config, dict):
                    return result

                tool_configs = config.get("tools", {})

                # Handle both dict and list formats
                if isinstance(tool_configs, dict):
                    for name, tool_config in tool_configs.items():
                        if isinstance(tool_config, dict):
                            result.append(
                                {
                                    "name": name,
                                    "description": tool_config.get("description", ""),
                                    "module": tool_config.get("module", ""),
                                }
                            )
                        else:
                            result.append(
                                {
                                    "name": name,
                                    "description": getattr(tool_config, "description", ""),
                                    "module": getattr(tool_config, "module", ""),
                                }
                            )
                elif isinstance(tool_configs, list):
                    for tool in tool_configs:
                        if isinstance(tool, str):
                            result.append({"name": tool, "description": "", "module": tool})
                        elif isinstance(tool, dict):
                            name = tool.get("name", tool.get("module", "unknown"))
                            result.append(
                                {
                                    "name": name,
                                    "description": tool.get("description", ""),
                                    "module": tool.get("module", name),
                                }
                            )
                return result

            if session_id:
                session = await self._sessions.get(session_id)
                if not session:
                    yield Event.error(
                        command.id,
                        error=f"Session not found: {session_id}",
                        code="SESSION_NOT_FOUND",
                    )
                    return
                tools = extract_tools_from_session(session)
            else:
                # List tools from all active sessions
                all_sessions = await self._sessions.list_active()
                seen_tools: set[str] = set()

                for sess_info in all_sessions:
                    session = await self._sessions.get(sess_info.get("session_id", ""))
                    if session:
                        for tool in extract_tools_from_session(session):
                            if tool["name"] not in seen_tools:
                                seen_tools.add(tool["name"])
                                tools.append(tool)

            yield Event.result(
                command.id,
                data={"tools": tools, "count": len(tools)},
            )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="TOOLS_LIST_FAILED",
            )

    async def _tools_info(self, command: Command) -> AsyncIterator[Event]:
        """Handle tools.info command.

        Gets detailed information about a specific tool.
        """
        tool_name = command.require_param("name")
        session_id = command.get_param("session_id")

        try:
            tool_info: dict[str, Any] | None = None

            if session_id:
                session = await self._sessions.get(session_id)
                if not session:
                    yield Event.error(
                        command.id,
                        error=f"Session not found: {session_id}",
                        code="SESSION_NOT_FOUND",
                    )
                    return

                amp_session = getattr(session, "_amplifier_session", None)
                if amp_session:
                    config = getattr(amp_session, "config", {})
                    if isinstance(config, dict):
                        tool_configs = config.get("tools", {})
                        if isinstance(tool_configs, dict) and tool_name in tool_configs:
                            tc = tool_configs[tool_name]
                            if isinstance(tc, dict):
                                tool_info = {
                                    "name": tool_name,
                                    "description": tc.get("description", ""),
                                    "module": tc.get("module", ""),
                                    "config": tc.get("config", {}),
                                }
                            else:
                                tool_info = {
                                    "name": tool_name,
                                    "description": getattr(tc, "description", ""),
                                    "module": getattr(tc, "module", ""),
                                }

            if not tool_info:
                # Search all sessions
                all_sessions = await self._sessions.list_active()
                for sess_info in all_sessions:
                    session = await self._sessions.get(sess_info.get("session_id", ""))
                    if session:
                        amp_session = getattr(session, "_amplifier_session", None)
                        if amp_session:
                            config = getattr(amp_session, "config", {})
                            if isinstance(config, dict):
                                tool_configs = config.get("tools", {})
                                if isinstance(tool_configs, dict) and tool_name in tool_configs:
                                    tc = tool_configs[tool_name]
                                    if isinstance(tc, dict):
                                        tool_info = {
                                            "name": tool_name,
                                            "description": tc.get("description", ""),
                                            "module": tc.get("module", ""),
                                            "config": tc.get("config", {}),
                                        }
                                    else:
                                        tool_info = {
                                            "name": tool_name,
                                            "description": getattr(tc, "description", ""),
                                            "module": getattr(tc, "module", ""),
                                        }
                                    break

            if tool_info:
                yield Event.result(command.id, data=tool_info)
            else:
                yield Event.error(
                    command.id,
                    error=f"Tool not found: {tool_name}",
                    code="TOOL_NOT_FOUND",
                )

        except Exception as e:
            yield Event.error(
                command.id,
                error=str(e),
                code="TOOLS_INFO_FAILED",
            )

    async def _slash_commands_list(self, command: Command) -> AsyncIterator[Event]:
        """Handle slash_commands.list command.

        Returns the list of available slash commands for TUI/CLI autocomplete.
        This enables clients to stay in sync with available commands.
        """
        # Define the canonical list of slash commands
        # This mirrors what app-cli supports
        commands = [
            {
                "name": "help",
                "aliases": ["h", "?"],
                "description": "Show available commands",
                "subcommands": [],
            },
            {
                "name": "bundle",
                "aliases": ["b"],
                "description": "Bundle management",
                "subcommands": [
                    {"name": "list", "aliases": ["ls"], "description": "List available bundles"},
                    {"name": "install", "aliases": ["i"], "description": "Install a bundle"},
                    {"name": "add", "aliases": [], "description": "Add a bundle from URL"},
                    {"name": "remove", "aliases": ["rm"], "description": "Remove a bundle"},
                    {"name": "use", "aliases": [], "description": "Set active bundle"},
                    {"name": "info", "aliases": [], "description": "Show bundle details"},
                ],
            },
            {
                "name": "agents",
                "aliases": ["agent", "a"],
                "description": "List available agents",
                "subcommands": [
                    {"name": "list", "aliases": ["ls"], "description": "List agents"},
                    {"name": "info", "aliases": [], "description": "Show agent details"},
                ],
            },
            {
                "name": "tools",
                "aliases": ["t"],
                "description": "List available tools",
                "subcommands": [],
            },
            {
                "name": "mode",
                "aliases": [],
                "description": "Set or toggle a mode",
                "subcommands": [],
            },
            {
                "name": "modes",
                "aliases": [],
                "description": "List available modes",
                "subcommands": [],
            },
            {
                "name": "session",
                "aliases": ["s"],
                "description": "Session management",
                "subcommands": [
                    {"name": "list", "aliases": ["ls"], "description": "List sessions"},
                    {"name": "info", "aliases": [], "description": "Show session info"},
                    {"name": "switch", "aliases": [], "description": "Switch to session"},
                ],
            },
            {
                "name": "reset",
                "aliases": [],
                "description": "Reset current session",
                "subcommands": [],
                "flags": ["--bundle", "--preserve"],
            },
            {
                "name": "init",
                "aliases": [],
                "description": "Initialize configuration",
                "subcommands": [],
            },
            {
                "name": "config",
                "aliases": [],
                "description": "Show current configuration",
                "subcommands": [],
            },
            {
                "name": "status",
                "aliases": [],
                "description": "Show session status",
                "subcommands": [],
            },
            {
                "name": "save",
                "aliases": [],
                "description": "Save conversation transcript",
                "subcommands": [],
            },
            {
                "name": "clear",
                "aliases": [],
                "description": "Clear conversation context",
                "subcommands": [],
            },
            {
                "name": "rename",
                "aliases": [],
                "description": "Rename current session",
                "subcommands": [],
            },
            {
                "name": "fork",
                "aliases": [],
                "description": "Fork session at turn N",
                "subcommands": [],
            },
            {
                "name": "allowed-dirs",
                "aliases": [],
                "description": "Manage allowed write directories",
                "subcommands": [],
            },
            {
                "name": "denied-dirs",
                "aliases": [],
                "description": "Manage denied write directories",
                "subcommands": [],
            },
            {
                "name": "quit",
                "aliases": ["exit", "q"],
                "description": "Exit the application",
                "subcommands": [],
            },
        ]

        # Mode shortcuts
        mode_shortcuts = [
            {
                "name": "careful",
                "aliases": [],
                "description": "Full capability with confirmation for destructive actions",
                "is_mode_shortcut": True,
            },
            {
                "name": "explore",
                "aliases": [],
                "description": "Zero-footprint exploration - understand before acting",
                "is_mode_shortcut": True,
            },
            {
                "name": "plan",
                "aliases": [],
                "description": "Analyze, strategize, and organize - but don't implement",
                "is_mode_shortcut": True,
            },
        ]

        yield Event.result(
            command.id,
            data={
                "commands": commands,
                "mode_shortcuts": mode_shortcuts,
                "count": len(commands) + len(mode_shortcuts),
            },
        )

    # =========================================================================
    # Event Mapping
    # =========================================================================

    def _map_session_event(
        self,
        session_event: Any,
        correlation_id: str,
        sequence: int,
    ) -> Event | None:
        """Map a session event to a protocol event.

        Args:
            session_event: Event from ManagedSession.execute()
            correlation_id: Command ID for correlation
            sequence: Current sequence number

        Returns:
            Protocol Event or None if event should be skipped
        """
        # Session events have .type and .properties
        event_type = session_event.type
        props = session_event.properties

        match event_type:
            case "content_block:start":
                return Event.create(
                    EventType.CONTENT_START,
                    data={
                        "block_type": props.get("block_type", "text"),
                        "block_index": props.get("index", 0),
                    },
                    correlation_id=correlation_id,
                    sequence=sequence,
                )

            case "content_block:delta":
                delta = props.get("delta", {})
                delta_text = delta.get("text", "") if isinstance(delta, dict) else str(delta)
                return Event.content_delta(
                    correlation_id=correlation_id,
                    delta=delta_text,
                    sequence=sequence,
                    block_index=props.get("index", 0),
                )

            case "content_block:end":
                block = props.get("block", {})
                content = block.get("text", "") if isinstance(block, dict) else str(block)
                return Event.create(
                    EventType.CONTENT_END,
                    data={
                        "content": content,
                        "block_index": props.get("index", 0),
                    },
                    correlation_id=correlation_id,
                    sequence=sequence,
                )

            case "tool:pre":
                return Event.tool_call(
                    correlation_id=correlation_id,
                    tool_name=props.get("tool_name", "unknown"),
                    tool_call_id=props.get("tool_call_id", ""),
                    arguments=props.get("tool_input", {}),
                    sequence=sequence,
                )

            case "tool:post":
                result = props.get("result", {})
                output = result.get("output", "") if isinstance(result, dict) else str(result)
                return Event.tool_result(
                    correlation_id=correlation_id,
                    tool_call_id=props.get("tool_call_id", ""),
                    output=output,
                    sequence=sequence,
                )

            case "approval:required":
                return Event.approval_required(
                    correlation_id=correlation_id,
                    request_id=props.get("request_id", ""),
                    prompt=props.get("prompt", ""),
                    options=props.get("options", ["yes", "no"]),
                    timeout=props.get("timeout", 30.0),
                    sequence=sequence,
                )

            case "prompt:submit" | "prompt:complete":
                # Skip these - handled at higher level
                return None

            case "error":
                return Event.error(
                    correlation_id,
                    error=props.get("error", "Unknown error"),
                    code=props.get("error_type", "UNKNOWN"),
                )

            case _:
                # Pass through other events with raw data
                return Event.create(
                    event_type.replace(":", "."),
                    data=props,
                    correlation_id=correlation_id,
                    sequence=sequence,
                )
