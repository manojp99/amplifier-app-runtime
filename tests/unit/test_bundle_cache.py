"""Tests for bundle caching functionality."""

from __future__ import annotations

import pytest

from amplifier_app_runtime.bundle_manager import BundleManager


class TestBundleCaching:
    """Test suite for two-tier bundle caching."""

    @pytest.mark.asyncio
    async def test_bundle_cache_hit(self):
        """Test bundle is loaded once and cached."""
        manager = BundleManager()
        await manager.initialize()

        # First load
        bundle1 = await manager._load_bundle_cached("foundation")

        # Second load should hit cache
        bundle2 = await manager._load_bundle_cached("foundation")

        assert bundle1 is bundle2  # Same object reference

    @pytest.mark.asyncio
    async def test_prepared_cache_same_config(self):
        """Test same configuration reuses prepared bundle."""
        manager = BundleManager()
        await manager.initialize()

        # Load twice with same config
        prepared1 = await manager.load_and_prepare(bundle_name="foundation")

        prepared2 = await manager.load_and_prepare(bundle_name="foundation")

        # Should be same cached object
        assert prepared1 is prepared2

    @pytest.mark.asyncio
    async def test_prepared_cache_different_behaviors(self):
        """Test different behaviors create different cache entries."""
        manager = BundleManager()
        await manager.initialize()

        # Load with different behaviors
        prepared1 = await manager.load_and_prepare(
            bundle_name="foundation",
            behaviors=["agents"],
        )

        prepared2 = await manager.load_and_prepare(
            bundle_name="foundation",
            behaviors=["streaming"],
        )

        # Different configs = different cache entries
        assert prepared1 is not prepared2

    @pytest.mark.asyncio
    async def test_prepared_cache_different_providers(self):
        """Test different provider configs create different cache entries."""
        manager = BundleManager()
        await manager.initialize()

        # Load with different provider configs
        prepared1 = await manager.load_and_prepare(
            bundle_name="foundation",
            provider_config={"module": "provider-anthropic"},
        )

        prepared2 = await manager.load_and_prepare(
            bundle_name="foundation",
            provider_config={"module": "provider-openai"},
        )

        # Different configs = different prepared bundles
        assert prepared1 is not prepared2

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation produces consistent keys."""
        manager = BundleManager()

        # Same inputs = same key
        key1 = manager._make_cache_key("foundation", None, None)
        key2 = manager._make_cache_key("foundation", None, None)
        assert key1 == key2

        # Different behaviors = different key
        key3 = manager._make_cache_key("foundation", ["agents"], None)
        assert key3 != key1

        # Different provider = different key
        key4 = manager._make_cache_key("foundation", None, {"module": "provider-anthropic"})
        assert key4 != key1

    def test_cache_invalidation_all(self):
        """Test cache invalidation clears all caches."""
        manager = BundleManager()

        # Populate caches manually
        manager._bundle_cache["foundation"] = "mock_bundle"
        manager._prepared_cache["foundation:hash:hash"] = "mock_prepared"

        assert len(manager._bundle_cache) > 0
        assert len(manager._prepared_cache) > 0

        # Invalidate all
        manager.invalidate_cache()

        assert len(manager._bundle_cache) == 0
        assert len(manager._prepared_cache) == 0

    def test_cache_invalidation_specific(self):
        """Test cache invalidation for specific bundle."""
        manager = BundleManager()

        # Populate caches
        manager._bundle_cache["foundation"] = "mock_bundle"
        manager._bundle_cache["recipes"] = "mock_bundle2"
        manager._prepared_cache["foundation:a:b"] = "mock_prepared1"
        manager._prepared_cache["foundation:c:d"] = "mock_prepared2"
        manager._prepared_cache["recipes:e:f"] = "mock_prepared3"

        # Invalidate only foundation
        manager.invalidate_cache("foundation")

        # foundation removed, recipes preserved
        assert "foundation" not in manager._bundle_cache
        assert "recipes" in manager._bundle_cache
        assert "foundation:a:b" not in manager._prepared_cache
        assert "foundation:c:d" not in manager._prepared_cache
        assert "recipes:e:f" in manager._prepared_cache

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        manager = BundleManager()

        # Populate caches
        manager._bundle_cache["foundation"] = "mock"
        manager._prepared_cache["foundation:a:b"] = "mock"

        stats = manager.get_cache_stats()

        assert stats["bundle_cache_size"] == 1
        assert stats["prepared_cache_size"] == 1
        assert "foundation" in stats["bundle_cache_keys"]
        assert "foundation:a:b" in stats["prepared_cache_keys"]

    @pytest.mark.asyncio
    async def test_cache_survives_multiple_sessions(self):
        """Test cached bundles are reused across session creations."""
        manager = BundleManager()
        await manager.initialize()

        # Create multiple sessions (simulated)
        prepared1 = await manager.load_and_prepare("foundation")
        prepared2 = await manager.load_and_prepare("foundation")
        prepared3 = await manager.load_and_prepare("foundation")

        # All should be same cached instance
        assert prepared1 is prepared2 is prepared3

        # Verify cache hit
        stats = manager.get_cache_stats()
        assert stats["bundle_cache_size"] >= 1
        assert stats["prepared_cache_size"] >= 1
