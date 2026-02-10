"""Tests for minimal session creation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_app_runtime.session import SessionManager


class TestMinimalSessionCreation:
    """Test suite for optimized minimal session creation."""

    @pytest.mark.asyncio
    async def test_create_minimal_session_basic(self):
        """Test basic minimal session creation."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        session = await manager.create_minimal()

        assert session is not None
        assert session.session_id.startswith("sess_minimal_")
        assert session.config.bundle == "foundation"
        assert session.config.show_thinking is False

    @pytest.mark.asyncio
    async def test_create_minimal_with_custom_id(self):
        """Test minimal session with custom ID."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        custom_id = "scorer_session"
        session = await manager.create_minimal(session_id=custom_id)

        assert session is not None
        assert session.session_id == custom_id

    @pytest.mark.asyncio
    async def test_create_minimal_with_custom_prompt(self):
        """Test minimal session with custom system prompt."""
        manager = SessionManager()

        custom_prompt = "You are a scoring agent. Respond with JSON only."

        # Mock bundle manager and context
        mock_context = MagicMock()
        mock_context.get_messages = AsyncMock(return_value=[])
        mock_context.set_messages = AsyncMock()

        mock_session = MagicMock()
        mock_session.coordinator.get.return_value = mock_context

        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(return_value=mock_session)

        mock_bundle_manager = MagicMock()
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        _session = await manager.create_minimal(system_prompt=custom_prompt)

        # Verify custom system prompt was set
        assert mock_context.set_messages.called

    @pytest.mark.asyncio
    async def test_minimal_session_uses_haiku(self):
        """Test minimal session configures Haiku provider."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        await manager.create_minimal()

        # Verify load_and_prepare called with Haiku config
        call_args = mock_bundle_manager.load_and_prepare.call_args
        provider_config = call_args[1]["provider_config"]

        assert provider_config["module"] == "provider-anthropic"
        assert provider_config["config"]["model"] == "claude-haiku-3-5-20241022"
        assert provider_config["config"]["max_tokens"] == 300

    @pytest.mark.asyncio
    async def test_minimal_session_no_behaviors(self):
        """Test minimal session loads foundation with no behaviors."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        await manager.create_minimal()

        # Verify load_and_prepare called with empty behaviors
        call_args = mock_bundle_manager.load_and_prepare.call_args
        behaviors = call_args[1]["behaviors"]

        assert behaviors == []

    @pytest.mark.asyncio
    async def test_minimal_session_no_persistence(self):
        """Test minimal session doesn't use SessionStore."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        session = await manager.create_minimal()

        # Minimal sessions don't persist
        assert session._store is None

    @pytest.mark.asyncio
    async def test_minimal_session_added_to_active(self):
        """Test minimal session is tracked in active sessions."""
        manager = SessionManager()

        # Mock bundle manager
        mock_bundle_manager = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.create_session = AsyncMock(
            return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
        )
        mock_bundle_manager.load_and_prepare = AsyncMock(return_value=mock_prepared)
        manager._bundle_manager = mock_bundle_manager

        session = await manager.create_minimal(session_id="minimal_test")

        # Verify in active sessions
        assert "minimal_test" in manager._active
        assert manager._active["minimal_test"] is session


class TestMinimalSessionCaching:
    """Test minimal sessions benefit from bundle caching."""

    @pytest.mark.asyncio
    async def test_multiple_minimal_sessions_share_cache(self):
        """Test multiple minimal sessions reuse cached bundles."""
        manager = SessionManager()

        # Mock bundle manager with real cache tracking
        mock_bundle_manager = MagicMock()
        mock_bundle_manager._prepared_cache = {}  # Real cache

        # Mock prepare to track caching
        prepare_count = 0

        async def mock_prepare(*args, **kwargs):
            nonlocal prepare_count
            prepare_count += 1

            # Return mock prepared bundle
            mock_prepared = MagicMock()
            mock_prepared.create_session = AsyncMock(
                return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
            )
            return mock_prepared

        # Mock load_and_prepare to populate cache on first call
        first_call = True

        async def mock_load_and_prepare(*args, **kwargs):
            nonlocal first_call
            cache_key = "foundation:no-behaviors:provider-hash"

            if first_call:
                # First call: prepare and cache
                prepared = await mock_prepare(*args, **kwargs)
                mock_bundle_manager._prepared_cache[cache_key] = prepared
                first_call = False
                return prepared
            else:
                # Subsequent calls: return from cache
                return mock_bundle_manager._prepared_cache[cache_key]

        mock_bundle_manager.load_and_prepare = mock_load_and_prepare
        manager._bundle_manager = mock_bundle_manager

        # Create first minimal session
        _session1 = await manager.create_minimal(session_id="min1")

        # Create second minimal session
        _session2 = await manager.create_minimal(session_id="min2")

        # Verify prepare only called once (second used cache)
        assert prepare_count == 1
        assert len(mock_bundle_manager._prepared_cache) == 1

    @pytest.mark.asyncio
    async def test_minimal_and_full_sessions_different_cache(self):
        """Test minimal and full sessions use different cache entries."""
        manager = SessionManager()

        # Mock bundle manager with cache tracking
        mock_bundle_manager = MagicMock()
        call_count = 0

        async def mock_load(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_prepared = MagicMock()
            mock_prepared.create_session = AsyncMock(
                return_value=MagicMock(coordinator=MagicMock(get=MagicMock(return_value=None)))
            )
            return mock_prepared

        mock_bundle_manager.load_and_prepare = mock_load
        manager._bundle_manager = mock_bundle_manager

        # Create minimal session (Haiku, no behaviors)
        await manager.create_minimal(session_id="minimal")

        # Create full session (different config)
        from amplifier_app_runtime.session import SessionConfig

        await manager.create(
            config=SessionConfig(bundle="foundation", behaviors=["agents"]),
            session_id="full",
            auto_initialize=True,
        )

        # Different configs = different cache entries = 2 calls
        assert call_count == 2
