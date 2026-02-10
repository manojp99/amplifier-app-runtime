"""Tests for hook system."""

from __future__ import annotations

from typing import Any

import pytest

from amplifier_app_runtime.hooks import HookRegistry, InputHook, OutputHook
from amplifier_app_runtime.session import SessionManager


class TestInputHook(InputHook):
    """Test input hook implementation."""

    name = "test_input"

    def __init__(self):
        self.started = False
        self.session_manager = None
        self.items: list[dict[str, Any]] = []

    async def start(self, session_manager):
        self.started = True
        self.session_manager = session_manager

    async def stop(self):
        self.started = False

    async def poll(self):
        return self.items


class TestOutputHook(OutputHook):
    """Test output hook implementation."""

    name = "test_output"

    def __init__(self):
        self.started = False
        self.session_manager = None
        self.sent_events: list[tuple[str, dict[str, Any]]] = []

    async def start(self, session_manager):
        self.started = True
        self.session_manager = session_manager

    async def stop(self):
        self.started = False

    async def send(self, event, data):
        self.sent_events.append((event, data))
        return True


class FilteredOutputHook(OutputHook):
    """Output hook that filters events."""

    name = "filtered_output"

    def __init__(self):
        self.sent_events: list[tuple[str, dict[str, Any]]] = []

    async def start(self, session_manager):
        pass

    async def stop(self):
        pass

    async def send(self, event, data):
        self.sent_events.append((event, data))
        return True

    def should_handle(self, event, data):
        # Only handle "notification" events
        return event == "notification"


class TestHookRegistration:
    """Test hook registration and listing."""

    def test_register_input_hook(self):
        """Test registering an input hook."""
        registry = HookRegistry()
        hook = TestInputHook()

        registry.register(hook)

        assert "test_input" in registry._hooks
        assert hook in registry._input_hooks

    def test_register_output_hook(self):
        """Test registering an output hook."""
        registry = HookRegistry()
        hook = TestOutputHook()

        registry.register(hook)

        assert "test_output" in registry._hooks
        assert hook in registry._output_hooks

    def test_register_duplicate_raises(self):
        """Test registering duplicate hook raises ValueError."""
        registry = HookRegistry()
        hook1 = TestInputHook()
        hook2 = TestInputHook()

        registry.register(hook1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(hook2)

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        registry = HookRegistry()
        hook = TestInputHook()

        registry.register(hook)
        assert "test_input" in registry._hooks

        registry.unregister("test_input")
        assert "test_input" not in registry._hooks
        assert hook not in registry._input_hooks

    def test_unregister_nonexistent_is_noop(self):
        """Test unregistering nonexistent hook doesn't raise."""
        registry = HookRegistry()

        # Should not raise
        registry.unregister("nonexistent")

    def test_list_hooks(self):
        """Test listing registered hooks."""
        registry = HookRegistry()

        input_hook = TestInputHook()
        output_hook = TestOutputHook()

        registry.register(input_hook)
        registry.register(output_hook)

        hooks = registry.list_hooks()

        assert len(hooks) == 2
        assert any(h["name"] == "test_input" and h["is_input"] for h in hooks)
        assert any(h["name"] == "test_output" and h["is_output"] for h in hooks)


class TestHookLifecycle:
    """Test hook lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_all_hooks(self):
        """Test starting all registered hooks."""
        registry = HookRegistry()
        session_manager = SessionManager()

        hook1 = TestInputHook()
        hook2 = TestOutputHook()

        registry.register(hook1)
        registry.register(hook2)

        assert not hook1.started
        assert not hook2.started

        await registry.start_all(session_manager)

        assert hook1.started
        assert hook2.started
        assert hook1.session_manager is session_manager
        assert hook2.session_manager is session_manager

    @pytest.mark.asyncio
    async def test_stop_all_hooks(self):
        """Test stopping all registered hooks."""
        registry = HookRegistry()
        session_manager = SessionManager()

        hook = TestInputHook()
        registry.register(hook)

        await registry.start_all(session_manager)
        assert hook.started

        await registry.stop_all()
        assert not hook.started

    @pytest.mark.asyncio
    async def test_start_continues_on_error(self):
        """Test start_all continues if a hook fails to start."""
        registry = HookRegistry()
        session_manager = SessionManager()

        # Hook that raises on start
        class FailingHook(InputHook):
            name = "failing_hook"

            async def start(self, session_manager):
                raise RuntimeError("Start failed")

            async def stop(self):
                pass

            async def poll(self):
                return []

        failing_hook = FailingHook()
        good_hook = TestInputHook()

        registry.register(failing_hook)
        registry.register(good_hook)

        # Should not raise
        await registry.start_all(session_manager)

        # Good hook should still start
        assert good_hook.started


class TestInputHookPolling:
    """Test input hook polling."""

    @pytest.mark.asyncio
    async def test_poll_inputs_empty(self):
        """Test polling with no hooks returns empty list."""
        registry = HookRegistry()

        inputs = await registry.poll_inputs()
        assert inputs == []

    @pytest.mark.asyncio
    async def test_poll_inputs_single_hook(self):
        """Test polling single input hook."""
        registry = HookRegistry()
        hook = TestInputHook()

        hook.items = [
            {"content": "Message 1", "session_id": "sess1", "role": "user"},
            {"content": "Message 2", "session_id": "sess2", "role": "user"},
        ]

        registry.register(hook)

        inputs = await registry.poll_inputs()

        assert len(inputs) == 2
        assert inputs[0]["content"] == "Message 1"
        assert inputs[1]["content"] == "Message 2"

    @pytest.mark.asyncio
    async def test_poll_inputs_multiple_hooks(self):
        """Test polling multiple input hooks."""
        registry = HookRegistry()

        hook1 = TestInputHook()
        hook1.name = "hook1"
        hook1.items = [{"content": "From hook1"}]

        hook2 = TestInputHook()
        hook2.name = "hook2"
        hook2.items = [{"content": "From hook2"}]

        registry.register(hook1)
        registry.register(hook2)

        inputs = await registry.poll_inputs()

        assert len(inputs) == 2
        contents = [item["content"] for item in inputs]
        assert "From hook1" in contents
        assert "From hook2" in contents

    @pytest.mark.asyncio
    async def test_poll_continues_on_error(self):
        """Test poll_inputs continues if a hook fails."""

        class FailingHook(InputHook):
            name = "failing_hook"

            async def start(self, session_manager):
                pass

            async def stop(self):
                pass

            async def poll(self):
                raise RuntimeError("Poll failed")

        registry = HookRegistry()

        failing_hook = FailingHook()
        good_hook = TestInputHook()
        good_hook.items = [{"content": "Success"}]

        registry.register(failing_hook)
        registry.register(good_hook)

        # Should not raise
        inputs = await registry.poll_inputs()

        # Should still get good hook's items
        assert len(inputs) == 1
        assert inputs[0]["content"] == "Success"


class TestOutputHookDispatching:
    """Test output hook event dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_output_no_hooks(self):
        """Test dispatching with no hooks returns empty dict."""
        registry = HookRegistry()

        results = await registry.dispatch_output("notification", {"message": "test"})

        assert results == {}

    @pytest.mark.asyncio
    async def test_dispatch_output_single_hook(self):
        """Test dispatching to single output hook."""
        registry = HookRegistry()
        hook = TestOutputHook()

        registry.register(hook)

        results = await registry.dispatch_output("notification", {"message": "Test notification"})

        assert results == {"test_output": True}
        assert len(hook.sent_events) == 1
        assert hook.sent_events[0][0] == "notification"
        assert hook.sent_events[0][1]["message"] == "Test notification"

    @pytest.mark.asyncio
    async def test_dispatch_output_multiple_hooks(self):
        """Test dispatching to multiple output hooks."""
        registry = HookRegistry()

        hook1 = TestOutputHook()
        hook1.name = "hook1"

        hook2 = TestOutputHook()
        hook2.name = "hook2"

        registry.register(hook1)
        registry.register(hook2)

        results = await registry.dispatch_output("event", {"data": "test"})

        assert results == {"hook1": True, "hook2": True}
        assert len(hook1.sent_events) == 1
        assert len(hook2.sent_events) == 1

    @pytest.mark.asyncio
    async def test_dispatch_output_filtered(self):
        """Test should_handle filters events correctly."""
        registry = HookRegistry()

        # Hook that only handles "notification" events
        filtered_hook = FilteredOutputHook()
        registry.register(filtered_hook)

        # Dispatch notification - should handle
        results1 = await registry.dispatch_output("notification", {"msg": "test"})
        assert results1 == {"filtered_output": True}
        assert len(filtered_hook.sent_events) == 1

        # Dispatch different event - should not handle
        results2 = await registry.dispatch_output("webhook", {"url": "test"})
        assert results2 == {}
        assert len(filtered_hook.sent_events) == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_dispatch_continues_on_error(self):
        """Test dispatch_output continues if a hook fails."""

        class FailingHook(OutputHook):
            name = "failing_hook"

            async def start(self, session_manager):
                pass

            async def stop(self):
                pass

            async def send(self, event, data):
                raise RuntimeError("Send failed")

        registry = HookRegistry()

        failing_hook = FailingHook()
        good_hook = TestOutputHook()

        registry.register(failing_hook)
        registry.register(good_hook)

        # Should not raise
        results = await registry.dispatch_output("event", {"data": "test"})

        # Good hook succeeds, failing hook marked as failed
        assert results["failing_hook"] is False
        assert results["test_output"] is True


class TestSessionManagerHooks:
    """Test SessionManager hook integration."""

    @pytest.mark.asyncio
    async def test_session_manager_has_hooks_property(self):
        """Test SessionManager exposes hooks property."""
        manager = SessionManager()

        assert hasattr(manager, "hooks")
        assert isinstance(manager.hooks, HookRegistry)

    @pytest.mark.asyncio
    async def test_session_manager_start_hooks(self):
        """Test SessionManager can start hooks."""
        registry = HookRegistry()
        hook = TestInputHook()
        registry.register(hook)

        manager = SessionManager(hook_registry=registry)

        assert not hook.started

        await manager.start_hooks()

        assert hook.started
        assert hook.session_manager is manager

    @pytest.mark.asyncio
    async def test_session_manager_stop_hooks(self):
        """Test SessionManager can stop hooks."""
        registry = HookRegistry()
        hook = TestInputHook()
        registry.register(hook)

        manager = SessionManager(hook_registry=registry)

        await manager.start_hooks()
        assert hook.started

        await manager.stop_hooks()
        assert not hook.started

    @pytest.mark.asyncio
    async def test_custom_hook_registry(self):
        """Test SessionManager accepts custom hook registry."""
        custom_registry = HookRegistry()
        hook = TestInputHook()
        custom_registry.register(hook)

        manager = SessionManager(hook_registry=custom_registry)

        assert manager.hooks is custom_registry
        assert "test_input" in manager.hooks._hooks
