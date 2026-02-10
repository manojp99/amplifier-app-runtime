"""Tests for context injection functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_app_runtime.session import ManagedSession, SessionConfig, SessionManager


class TestContextInjection:
    """Test suite for context injection without execution."""

    @pytest.mark.asyncio
    async def test_inject_context_adds_message(self):
        """Test inject_context adds message to session context."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock AmplifierSession and context
        mock_context = MagicMock()
        mock_context.add_message = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Inject context
        await session.inject_context("Test notification", role="user")

        # Verify add_message was called
        mock_context.add_message.assert_called_once()
        call_args = mock_context.add_message.call_args[0][0]
        assert call_args["role"] == "user"
        assert call_args["content"] == "Test notification"

    @pytest.mark.asyncio
    async def test_inject_context_tracks_locally(self):
        """Test inject_context adds to local message history."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock AmplifierSession and context
        mock_context = MagicMock()
        mock_context.add_message = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Inject context
        await session.inject_context("Test message", role="user")

        # Verify local tracking
        assert len(session._messages) == 1
        assert session._messages[0]["role"] == "user"
        assert session._messages[0]["content"] == "Test message"
        assert "timestamp" in session._messages[0]

    @pytest.mark.asyncio
    async def test_inject_context_different_roles(self):
        """Test inject_context works with different message roles."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context
        mock_context = MagicMock()
        mock_context.add_message = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Inject as different roles
        await session.inject_context("User message", role="user")
        await session.inject_context("System directive", role="system")
        await session.inject_context("Assistant note", role="assistant")

        # Verify all were added
        assert mock_context.add_message.call_count == 3
        assert len(session._messages) == 3

    @pytest.mark.asyncio
    async def test_inject_context_not_initialized(self):
        """Test inject_context raises if session not initialized."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Don't initialize - _amplifier_session is None

        with pytest.raises(RuntimeError, match="not initialized"):
            await session.inject_context("Test")

    @pytest.mark.asyncio
    async def test_inject_context_no_context_module(self):
        """Test inject_context raises if context module unavailable."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock AmplifierSession but no context module
        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = None

        session._amplifier_session = mock_amplifier_session

        with pytest.raises(RuntimeError, match="Context module not available"):
            await session.inject_context("Test")


class TestClearContext:
    """Test suite for clearing session context."""

    @pytest.mark.asyncio
    async def test_clear_context_preserves_system(self):
        """Test clear_context preserves system prompt by default."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context with system message and user messages
        system_msg = {"role": "system", "content": "System prompt"}
        user_msg = {"role": "user", "content": "User message"}

        mock_context = MagicMock()
        mock_context.get_messages = AsyncMock(return_value=[system_msg, user_msg])
        mock_context.set_messages = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session
        session._messages = [system_msg, user_msg]

        # Clear context (preserve system)
        await session.clear_context(preserve_system=True)

        # Verify set_messages called with only system message
        mock_context.set_messages.assert_called_once()
        call_args = mock_context.set_messages.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == system_msg

        # Verify local messages cleared
        assert len(session._messages) == 0

    @pytest.mark.asyncio
    async def test_clear_context_removes_all(self):
        """Test clear_context removes everything when preserve_system=False."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context
        mock_context = MagicMock()
        mock_context.set_messages = AsyncMock()

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session
        session._messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]

        # Clear all
        await session.clear_context(preserve_system=False)

        # Verify set_messages called with empty list
        mock_context.set_messages.assert_called_once_with([])

        # Verify local messages cleared
        assert len(session._messages) == 0

    @pytest.mark.asyncio
    async def test_clear_context_fallback_to_clear_method(self):
        """Test clear_context falls back to clear() method if set_messages unavailable."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context with clear() but no set_messages
        mock_context = MagicMock()
        mock_context.clear = AsyncMock()
        del mock_context.set_messages  # Remove set_messages

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Clear context
        await session.clear_context(preserve_system=False)

        # Verify clear() was called
        mock_context.clear.assert_called_once()


class TestSessionManagerContextMethods:
    """Test SessionManager convenience methods for context operations."""

    @pytest.mark.asyncio
    async def test_session_manager_inject_context(self):
        """Test SessionManager.inject_context delegates to session."""
        manager = SessionManager()

        # Create mock session
        mock_session = MagicMock()
        mock_session.inject_context = AsyncMock()

        manager._active["test_sess"] = mock_session

        # Inject via manager
        await manager.inject_context("test_sess", "Test content", role="system")

        # Verify delegation
        mock_session.inject_context.assert_called_once_with("Test content", "system")

    @pytest.mark.asyncio
    async def test_session_manager_inject_context_not_found(self):
        """Test inject_context raises if session not found."""
        manager = SessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            await manager.inject_context("nonexistent", "Test")

    @pytest.mark.asyncio
    async def test_session_manager_clear_context(self):
        """Test SessionManager.clear_context delegates to session."""
        manager = SessionManager()

        # Create mock session
        mock_session = MagicMock()
        mock_session.clear_context = AsyncMock()

        manager._active["test_sess"] = mock_session

        # Clear via manager
        await manager.clear_context("test_sess", preserve_system=False)

        # Verify delegation
        mock_session.clear_context.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_session_manager_clear_context_not_found(self):
        """Test clear_context raises if session not found."""
        manager = SessionManager()

        with pytest.raises(ValueError, match="Session not found"):
            await manager.clear_context("nonexistent")


class TestContextInjectionIntegration:
    """Integration tests for context injection workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_inject_multiple_then_clear(self):
        """Test injecting multiple messages then clearing."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context
        messages_store = []

        mock_context = MagicMock()
        mock_context.add_message = AsyncMock(side_effect=lambda m: messages_store.append(m))
        mock_context.get_messages = AsyncMock(return_value=lambda: list(messages_store))
        mock_context.set_messages = AsyncMock(side_effect=lambda m: messages_store.clear())

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Inject multiple messages
        await session.inject_context("Message 1", role="user")
        await session.inject_context("Message 2", role="user")
        await session.inject_context("Message 3", role="user")

        assert len(session._messages) == 3

        # Clear context
        await session.clear_context(preserve_system=False)

        # Verify cleared
        assert len(session._messages) == 0
        mock_context.set_messages.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_inject_preserves_across_executions(self):
        """Test injected context persists for execution."""
        session = ManagedSession(
            session_id="test_session",
            config=SessionConfig(bundle="foundation"),
        )

        # Mock context that tracks messages
        messages_store = []

        mock_context = MagicMock()
        mock_context.add_message = AsyncMock(side_effect=lambda m: messages_store.append(m))
        mock_context.get_messages = AsyncMock(return_value=messages_store.copy())

        mock_amplifier_session = MagicMock()
        mock_amplifier_session.coordinator.get.return_value = mock_context

        session._amplifier_session = mock_amplifier_session

        # Inject context before execution
        await session.inject_context("Important context: User prefers brevity", role="system")

        # Verify context was added and persists
        assert len(messages_store) == 1
        assert messages_store[0]["role"] == "system"
        assert "brevity" in messages_store[0]["content"]
