"""Amplifier to ACP event mapping.

This module handles the conversion of Amplifier events to ACP session updates.
Extracted from agent.py to improve maintainability and testability.

Event Mapping:
- tool:pre -> ToolCallStart (sessionUpdate="tool_call")
- tool:post -> ToolCallUpdate with status="completed"
- tool:error -> ToolCallUpdate with status="failed"
- todo:update -> AgentPlanUpdate (sessionUpdate="plan")
- content_block:* -> update_agent_message
- thinking:* -> update_agent_thought
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from acp import (  # type: ignore[import-untyped]
    text_block,
    update_agent_message,
    update_agent_thought,
)
from acp.schema import (  # type: ignore[import-untyped]
    AgentPlanUpdate,
    PlanEntry,
    ToolCallStart,
    ToolCallUpdate,
)

from .tool_metadata import get_tool_kind, get_tool_title

if TYPE_CHECKING:
    from acp.schema import SessionUpdate  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class EventMapResult:
    """Result of mapping an Amplifier event to ACP.

    Attributes:
        update: The ACP SessionUpdate to send, or None if no update needed
        track_tool: If set, indicates a tool call should be tracked (call_id, name, args)
        clear_tool_tracking: If True, tool tracking should be cleared
    """

    update: SessionUpdate | None = None
    track_tool: tuple[str, str, dict[str, Any]] | None = None
    clear_tool_tracking: bool = False


@dataclass
class AmplifierToAcpEventMapper:
    """Maps Amplifier events to ACP session updates.

    This class encapsulates the event mapping logic, making it easier to test
    and maintain separately from the session management code.

    Usage:
        mapper = AmplifierToAcpEventMapper()
        result = mapper.map_event(event)
        if result.update:
            await conn.session_update(session_id, result.update)
    """

    # Event types that are expected but don't need mapping
    _ignored_prefixes: tuple[str, ...] = field(
        default=(
            "session:",
            "execution:",
            "llm:",
            "provider:",
            "prompt:",
            "orchestrator:",
        ),
        repr=False,
    )

    # Recipe plan state (for tracking step progress)
    _current_plan: list[PlanEntry] = field(default_factory=list, init=False)
    _recipe_session_id: str | None = field(default=None, init=False)

    def map_event(self, event: Any) -> EventMapResult:
        """Map an Amplifier event to an ACP session update.

        Args:
            event: Amplifier event (object with type/properties or dict)

        Returns:
            EventMapResult with the ACP update (if any) and side-effect flags
        """
        # Extract event type and properties
        event_type = self._get_event_type(event)
        props = self._get_event_props(event)

        if not event_type:
            return EventMapResult()

        # Dispatch to specific handler
        handler = self._get_handler(event_type)
        if handler:
            return handler(props)

        # Log unmapped events at debug level (but not expected ones)
        if not event_type.startswith(self._ignored_prefixes):
            logger.debug(f"Unmapped event type: {event_type}")

        return EventMapResult()

    def _get_event_type(self, event: Any) -> str:
        """Extract event type from event object or dict."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type", "")
        return event_type or ""

    def _get_event_props(self, event: Any) -> dict[str, Any]:
        """Extract properties from event object or dict."""
        props = getattr(event, "properties", None)
        if props is None and isinstance(event, dict):
            props = event
        return props or {}

    def _get_handler(self, event_type: str) -> Any:
        """Get the handler method for an event type."""
        handlers = {
            "content_block:delta": self._handle_content_delta,
            "content_block:end": self._handle_content_end,
            "content_block:start": self._handle_content_start,
            "content": self._handle_text_content,
            "assistant_message": self._handle_text_content,
            "text": self._handle_text_content,
            "tool:pre": self._handle_tool_pre,
            "tool:post": self._handle_tool_post,
            "tool:error": self._handle_tool_error,
            "todo:update": self._handle_todo_update,
            "thinking:delta": self._handle_thinking,
            "thinking:final": self._handle_thinking,
            "thinking:start": self._handle_thinking,
            # Recipe event handlers (Phase 1: ready for future events)
            "recipe:session:start": self._handle_recipe_session_start,
            "recipe:step:start": self._handle_recipe_step_start,
            "recipe:step:complete": self._handle_recipe_step_complete,
            "recipe:approval:pending": self._handle_recipe_approval_pending,
            "recipe:session:complete": self._handle_recipe_session_complete,
        }
        return handlers.get(event_type)

    # =========================================================================
    # Content handlers
    # =========================================================================

    def _handle_content_delta(self, props: dict[str, Any]) -> EventMapResult:
        """Handle streaming text delta."""
        delta = props.get("delta", {})
        text = delta.get("text", "")
        if text:
            return EventMapResult(update=update_agent_message(text_block(text)))
        return EventMapResult()

    def _handle_content_end(self, props: dict[str, Any]) -> EventMapResult:
        """Handle final content block."""
        block = props.get("block", {})
        text = block.get("text", "")
        if text:
            return EventMapResult(update=update_agent_message(text_block(text)))
        return EventMapResult()

    def _handle_content_start(self, props: dict[str, Any]) -> EventMapResult:
        """Handle content block starting (mostly a no-op, waits for delta/end)."""
        # Content block starting - check if it's thinking
        block = props.get("block", {})
        block_type = block.get("type", "")
        if block_type == "thinking":
            # Thinking block starting - we'll get content in delta/end
            pass
        # For text blocks, wait for delta/end to send content
        return EventMapResult()

    def _handle_text_content(self, props: dict[str, Any]) -> EventMapResult:
        """Handle direct text content events."""
        text = props.get("text", "")
        if text:
            return EventMapResult(update=update_agent_message(text_block(text)))
        return EventMapResult()

    # =========================================================================
    # Tool call handlers
    # =========================================================================

    def _handle_tool_pre(self, props: dict[str, Any]) -> EventMapResult:
        """Handle tool call starting - ACP ToolCallStart."""
        tool_info = props.get("tool", {})
        tool_name = tool_info.get("name", "") if isinstance(tool_info, dict) else str(tool_info)
        tool_call_id = props.get("call_id", "")
        arguments = props.get("arguments", {})

        # Generate human-readable title and kind
        title = get_tool_title(tool_name, arguments)
        kind = get_tool_kind(tool_name)

        update = ToolCallStart(
            session_update="tool_call",
            tool_call_id=tool_call_id,
            title=title,
            kind=kind,
            status="pending",
            raw_input=arguments,
        )

        return EventMapResult(
            update=update,
            track_tool=(tool_call_id, tool_name, arguments),
        )

    def _handle_tool_post(self, props: dict[str, Any]) -> EventMapResult:
        """Handle tool call completed - ACP ToolCallUpdate."""
        update = ToolCallUpdate(
            tool_call_id=props.get("call_id", ""),
            status="completed",
            raw_output=props.get("result"),
        )
        return EventMapResult(update=update, clear_tool_tracking=True)

    def _handle_tool_error(self, props: dict[str, Any]) -> EventMapResult:
        """Handle tool call failed - ACP ToolCallUpdate with status='failed'."""
        error_info = props.get("error", "Unknown error")
        update = ToolCallUpdate(
            tool_call_id=props.get("call_id", ""),
            status="failed",
            raw_output={"error": str(error_info)},
        )
        return EventMapResult(update=update, clear_tool_tracking=True)

    # =========================================================================
    # Planning/thinking handlers
    # =========================================================================

    def _handle_todo_update(self, props: dict[str, Any]) -> EventMapResult:
        """Handle todo list update - map to ACP AgentPlanUpdate."""
        todos = props.get("todos", [])
        if not todos:
            return EventMapResult()

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
                    status=status,
                    priority=priority,
                )
            )

        update = AgentPlanUpdate(
            session_update="plan",
            entries=entries,
        )
        return EventMapResult(update=update)

    def _handle_thinking(self, props: dict[str, Any]) -> EventMapResult:
        """Handle thinking/reasoning content."""
        text = props.get("text", "") or props.get("content", "")
        if text:
            return EventMapResult(update=update_agent_thought(text_block(text)))
        return EventMapResult()

    # =========================================================================
    # Recipe event handlers (Phase 1: Ready for future recipe events)
    # =========================================================================

    def _handle_recipe_session_start(self, props: dict[str, Any]) -> EventMapResult:
        """Handle recipe:session:start - create initial plan.

        Maps recipe steps to ACP plan entries with pending status.

        Expected props:
            session_id: str - Recipe session ID
            recipe_name: str - Recipe name
            steps: list[dict] - Step definitions with name, agent, status

        Returns:
            EventMapResult with AgentPlanUpdate
        """
        steps = props.get("steps", [])
        if not steps:
            return EventMapResult()

        # Store recipe session ID for tracking
        self._recipe_session_id = props.get("session_id")

        # Create plan entries for all steps
        entries = []
        for i, step in enumerate(steps):
            step_name = step.get("name", f"step_{i + 1}")
            agent = step.get("agent", "unknown")
            status = step.get("status", "pending")

            # Normalize status
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"

            entries.append(
                PlanEntry(
                    content=f"{i + 1}. {step_name} ({agent})",
                    status=status,
                    priority="medium",
                )
            )

        # Cache plan state for updates
        self._current_plan = entries

        update = AgentPlanUpdate(
            session_update="plan",
            entries=entries,
        )

        logger.debug(f"Recipe session started: {props.get('recipe_name')} with {len(steps)} steps")

        return EventMapResult(update=update)

    def _handle_recipe_step_start(self, props: dict[str, Any]) -> EventMapResult:
        """Handle recipe:step:start - update specific step to in_progress.

        Expected props:
            session_id: str - Recipe session ID
            step_index: int - Index of step starting
            step_name: str - Name of step

        Returns:
            EventMapResult with updated plan
        """
        step_index = props.get("step_index", 0)

        # Update plan item status
        if 0 <= step_index < len(self._current_plan):
            self._current_plan[step_index].status = "in_progress"

        update = AgentPlanUpdate(
            session_update="plan",
            entries=list(self._current_plan),
        )

        logger.debug(f"Recipe step {step_index} started: {props.get('step_name')}")

        return EventMapResult(update=update)

    def _handle_recipe_step_complete(self, props: dict[str, Any]) -> EventMapResult:
        """Handle recipe:step:complete - mark step as completed.

        Expected props:
            session_id: str - Recipe session ID
            step_index: int - Index of completed step
            step_name: str - Name of step
            result: str - Step result (optional)

        Returns:
            EventMapResult with updated plan
        """
        step_index = props.get("step_index", 0)

        # Update plan item status
        if 0 <= step_index < len(self._current_plan):
            self._current_plan[step_index].status = "completed"

        update = AgentPlanUpdate(
            session_update="plan",
            entries=list(self._current_plan),
        )

        logger.debug(f"Recipe step {step_index} completed: {props.get('step_name')}")

        return EventMapResult(update=update)

    def _handle_recipe_approval_pending(self, props: dict[str, Any]) -> EventMapResult:
        """Handle recipe:approval:pending - signal approval gate reached.

        Uses AgentPlanUpdate to show current progress and sends an agent
        message notifying about the approval requirement.

        Expected props:
            session_id: str - Recipe session ID
            stage_name: str - Stage waiting for approval
            prompt: str - Approval prompt text
            timeout_seconds: int - Timeout (optional)

        Returns:
            EventMapResult with plan update + approval notification
        """
        stage_name = props.get("stage_name", "stage")
        prompt = props.get("prompt", "Approve to continue?")

        # Send plan update with current state
        plan_update = AgentPlanUpdate(
            session_update="plan",
            entries=list(self._current_plan),
        )

        # Log approval pending
        logger.info(f"Recipe approval pending: {stage_name} - {prompt}")

        return EventMapResult(update=plan_update)

    def _handle_recipe_session_complete(self, props: dict[str, Any]) -> EventMapResult:
        """Handle recipe:session:complete - mark all remaining steps completed.

        Expected props:
            session_id: str - Recipe session ID
            status: str - Completion status (success/failure/cancelled)
            total_steps: int - Total steps executed
            duration_seconds: float - Execution duration

        Returns:
            EventMapResult with final plan state
        """
        status = props.get("status", "success")

        # Mark all remaining steps as completed (in_progress or pending)
        for entry in self._current_plan:
            if entry.status != "completed":
                entry.status = "completed"

        update = AgentPlanUpdate(
            session_update="plan",
            entries=list(self._current_plan),
        )

        logger.info(
            f"Recipe session completed: {status} "
            f"({props.get('total_steps', 0)} steps, "
            f"{props.get('duration_seconds', 0):.1f}s)"
        )

        # Clear plan state
        self._current_plan = []
        self._recipe_session_id = None

        return EventMapResult(update=update)
