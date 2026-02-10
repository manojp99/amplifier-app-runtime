"""Unit tests for bundle_manager module.

Tests the BundleManager for bundle loading and provider detection.
Focus on testable components - avoid mocking internal foundation imports.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from amplifier_app_runtime.bundle_manager import BundleInfo, BundleManager

# =============================================================================
# BundleInfo Tests
# =============================================================================


class TestBundleInfo:
    """Tests for BundleInfo dataclass."""

    def test_create_with_name_only(self) -> None:
        """BundleInfo can be created with just name."""
        info = BundleInfo(name="test-bundle")
        assert info.name == "test-bundle"
        assert info.description == ""
        assert info.uri is None

    def test_create_with_all_fields(self) -> None:
        """BundleInfo accepts all fields."""
        info = BundleInfo(
            name="my-bundle",
            description="A test bundle",
            uri="git+https://github.com/org/repo",
        )
        assert info.name == "my-bundle"
        assert info.description == "A test bundle"
        assert info.uri == "git+https://github.com/org/repo"

    def test_equality(self) -> None:
        """BundleInfo supports equality comparison."""
        info1 = BundleInfo(name="test", description="desc")
        info2 = BundleInfo(name="test", description="desc")
        assert info1 == info2

    def test_inequality(self) -> None:
        """BundleInfo detects differences."""
        info1 = BundleInfo(name="test1")
        info2 = BundleInfo(name="test2")
        assert info1 != info2


# =============================================================================
# BundleManager Initialization Tests
# =============================================================================


class TestBundleManagerInit:
    """Tests for BundleManager initialization."""

    def test_init_not_initialized(self) -> None:
        """BundleManager starts uninitialized."""
        manager = BundleManager()
        assert manager._initialized is False
        assert manager._registry is None

    def test_registry_raises_before_init(self) -> None:
        """registry property raises before initialize()."""
        manager = BundleManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.registry

    def test_registry_returns_value_after_manual_init(self) -> None:
        """registry property returns value when set."""
        manager = BundleManager()
        mock_registry = MagicMock()
        manager._registry = mock_registry
        manager._initialized = True

        assert manager.registry is mock_registry

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """initialize() is idempotent - second call is no-op."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        original_registry = manager._registry

        # Should not raise or change anything
        await manager.initialize()

        assert manager._initialized is True
        assert manager._registry is original_registry


# =============================================================================
# BundleManager List Bundles Tests
# =============================================================================


class TestBundleManagerListBundles:
    """Tests for list_bundles method."""

    @pytest.mark.asyncio
    async def test_list_bundles_returns_list(self) -> None:
        """list_bundles returns list of BundleInfo."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        result = await manager.list_bundles()

        assert isinstance(result, list)
        assert all(isinstance(b, BundleInfo) for b in result)

    @pytest.mark.asyncio
    async def test_list_bundles_includes_foundation(self) -> None:
        """list_bundles includes foundation bundle."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        result = await manager.list_bundles()

        names = [b.name for b in result]
        assert "foundation" in names

    @pytest.mark.asyncio
    async def test_list_bundles_includes_amplifier_dev(self) -> None:
        """list_bundles includes amplifier-dev bundle."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        result = await manager.list_bundles()

        names = [b.name for b in result]
        assert "amplifier-dev" in names

    @pytest.mark.asyncio
    async def test_list_bundles_has_descriptions(self) -> None:
        """list_bundles provides descriptions."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        result = await manager.list_bundles()

        for bundle_info in result:
            assert bundle_info.description != ""

    @pytest.mark.asyncio
    async def test_list_bundles_count(self) -> None:
        """list_bundles returns expected number of bundles."""
        manager = BundleManager()
        manager._initialized = True
        manager._registry = MagicMock()

        result = await manager.list_bundles()

        # Currently returns 2 hardcoded bundles
        assert len(result) >= 2


# =============================================================================
# BundleManager Cache Invalidation Tests
# =============================================================================


class TestBundleManagerCache:
    """Tests for cache invalidation."""

    def test_invalidate_cache_clears_all(self) -> None:
        """invalidate_cache clears both bundle and prepared caches."""
        manager = BundleManager()

        # Populate both caches
        manager._bundle_cache["foundation"] = "mock"
        manager._prepared_cache["foundation:a:b"] = "mock"

        manager.invalidate_cache()

        assert len(manager._bundle_cache) == 0
        assert len(manager._prepared_cache) == 0

    def test_invalidate_cache_specific_bundle(self) -> None:
        """invalidate_cache can target specific bundle."""
        manager = BundleManager()

        # Populate caches
        manager._bundle_cache["foundation"] = "mock1"
        manager._bundle_cache["recipes"] = "mock2"
        manager._prepared_cache["foundation:a:b"] = "mock3"
        manager._prepared_cache["recipes:c:d"] = "mock4"

        # Invalidate only foundation
        manager.invalidate_cache("foundation")

        # Foundation removed, recipes preserved
        assert "foundation" not in manager._bundle_cache
        assert "recipes" in manager._bundle_cache
        assert "foundation:a:b" not in manager._prepared_cache
        assert "recipes:c:d" in manager._prepared_cache

    def test_invalidate_cache_with_registry(self) -> None:
        """invalidate_cache also clears registry cache if available."""
        manager = BundleManager()
        mock_registry = MagicMock()
        mock_registry.clear_cache = MagicMock()
        manager._registry = mock_registry

        manager.invalidate_cache()

        # Registry cache also cleared
        mock_registry.clear_cache.assert_called_once()

    def test_invalidate_cache_no_registry_graceful(self) -> None:
        """invalidate_cache handles missing registry gracefully."""
        manager = BundleManager()
        manager._registry = None

        # Should not raise
        manager.invalidate_cache()
