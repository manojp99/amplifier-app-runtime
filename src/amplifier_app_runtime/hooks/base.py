"""Base classes for server hooks.

Hooks allow bundles to extend the server's input/output capabilities
without modifying core server code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..session import SessionManager

logger = logging.getLogger(__name__)


class ServerHook(ABC):
    """Base class for all server hooks.

    Hooks extend server capabilities by:
    - Injecting external inputs into sessions
    - Processing session outputs to external destinations
    - Observing and reacting to server events
    """

    name: str = "base_hook"

    @abstractmethod
    async def start(self, session_manager: SessionManager) -> None:
        """Start the hook.

        Called when server starts. Hook receives reference to
        SessionManager for accessing sessions.

        Args:
            session_manager: The server's session manager
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop hook and cleanup resources."""


class InputHook(ServerHook):
    """Hook for injecting external inputs into sessions.

    Examples:
    - Notification collector (Windows, mobile)
    - Calendar event watcher
    - Email digest generator
    - Scheduled triggers
    - Webhook receiver
    """

    @abstractmethod
    async def poll(self) -> list[dict[str, Any]]:
        """Poll for new inputs.

        Returns:
            List of input items to inject. Each item should have:
            - content: str - The content to inject
            - session_id: str | None - Target session (None = default)
            - role: str - Message role (default: "user")
        """


class OutputHook(ServerHook):
    """Hook for processing session outputs.

    Examples:
    - Push notification sender
    - Webhook delivery
    - Email sender
    - Slack/Teams poster
    """

    @abstractmethod
    async def send(self, event: str, data: dict[str, Any]) -> bool:
        """Send an output.

        Args:
            event: Event type (e.g., "notification", "webhook")
            data: Event data

        Returns:
            True if sent successfully
        """

    def should_handle(self, event: str, data: dict[str, Any]) -> bool:
        """Check if this hook should handle the event.

        Override to filter which events this hook processes.

        Args:
            event: Event type
            data: Event data

        Returns:
            True if this hook should handle the event
        """
        return True


class HookRegistry:
    """Registry for server hooks.

    Manages hook lifecycle and provides methods for triggering hooks.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, ServerHook] = {}
        self._input_hooks: list[InputHook] = []
        self._output_hooks: list[OutputHook] = []
        self._session_manager: SessionManager | None = None

    def register(self, hook: ServerHook) -> None:
        """Register a hook.

        Args:
            hook: Hook instance to register

        Raises:
            ValueError: If hook already registered
        """
        if hook.name in self._hooks:
            raise ValueError(f"Hook already registered: {hook.name}")

        self._hooks[hook.name] = hook

        if isinstance(hook, InputHook):
            self._input_hooks.append(hook)
        if isinstance(hook, OutputHook):
            self._output_hooks.append(hook)

        logger.info(f"Registered hook: {hook.name}")

    def unregister(self, name: str) -> None:
        """Unregister a hook by name.

        Args:
            name: Hook name to unregister
        """
        hook = self._hooks.pop(name, None)
        if hook:
            if isinstance(hook, InputHook) and hook in self._input_hooks:
                self._input_hooks.remove(hook)
            if isinstance(hook, OutputHook) and hook in self._output_hooks:
                self._output_hooks.remove(hook)
            logger.info(f"Unregistered hook: {name}")

    async def start_all(self, session_manager: SessionManager) -> None:
        """Start all registered hooks.

        Args:
            session_manager: The session manager to pass to hooks
        """
        self._session_manager = session_manager

        for hook in self._hooks.values():
            try:
                await hook.start(session_manager)
                logger.info(f"Started hook: {hook.name}")
            except Exception as e:
                logger.error(f"Failed to start hook {hook.name}: {e}")

    async def stop_all(self) -> None:
        """Stop all registered hooks."""
        for hook in self._hooks.values():
            try:
                await hook.stop()
                logger.info(f"Stopped hook: {hook.name}")
            except Exception as e:
                logger.error(f"Failed to stop hook {hook.name}: {e}")

    async def poll_inputs(self) -> list[dict[str, Any]]:
        """Poll all input hooks for new inputs.

        Returns:
            List of input items from all hooks
        """
        inputs = []

        for hook in self._input_hooks:
            try:
                items = await hook.poll()
                inputs.extend(items)
            except Exception as e:
                logger.error(f"Input hook {hook.name} poll error: {e}")

        return inputs

    async def dispatch_output(
        self,
        event: str,
        data: dict[str, Any],
    ) -> dict[str, bool]:
        """Dispatch output event to interested output hooks.

        Args:
            event: Event type
            data: Event data

        Returns:
            Dict of hook_name -> success
        """
        results = {}

        for hook in self._output_hooks:
            if hook.should_handle(event, data):
                try:
                    results[hook.name] = await hook.send(event, data)
                except Exception as e:
                    logger.error(f"Output hook {hook.name} send error: {e}")
                    results[hook.name] = False

        return results

    def list_hooks(self) -> list[dict[str, Any]]:
        """List all registered hooks.

        Returns:
            List of hook info dicts
        """
        return [
            {
                "name": hook.name,
                "type": type(hook).__name__,
                "is_input": isinstance(hook, InputHook),
                "is_output": isinstance(hook, OutputHook),
            }
            for hook in self._hooks.values()
        ]
