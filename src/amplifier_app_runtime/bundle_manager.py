"""Bundle manager for Amplifier Server App.

Thin wrapper around amplifier-foundation's bundle system.
Provides bundle loading, preparation, and provider detection.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amplifier_foundation import Bundle
    from amplifier_foundation.registry import BundleRegistry

logger = logging.getLogger(__name__)


@dataclass
class BundleInfo:
    """Minimal info about a bundle for API responses."""

    name: str
    description: str = ""
    uri: str | None = None
    path: Path | None = None
    source: str | None = None  # "builtin", "git", "local"


class BundleManager:
    """Thin wrapper around amplifier-foundation's bundle system.

    Responsibilities (app-layer policy):
    - Provide registry instance
    - Compose provider credentials at runtime
    - Auto-detect providers from environment

    NOT responsible for (foundation handles):
    - Bundle discovery, loading, parsing
    - Module activation and resolution
    - Session creation internals
    """

    def __init__(self) -> None:
        """Initialize bundle manager."""
        self._registry: BundleRegistry | None = None
        self._initialized = False

        # Two-tier cache for bundle reuse across sessions
        self._bundle_cache: dict[str, Any] = {}  # uri -> Bundle
        self._prepared_cache: dict[str, Any] = {}  # cache_key -> PreparedBundle

    async def initialize(self) -> None:
        """Initialize by importing foundation and creating registry."""
        if self._initialized:
            return

        try:
            from amplifier_foundation.registry import BundleRegistry

            self._registry = BundleRegistry()
            self._initialized = True
            logger.info("Bundle manager initialized with amplifier-foundation")

        except ImportError as e:
            logger.error(f"Failed to import amplifier-foundation: {e}")
            raise RuntimeError(
                "amplifier-foundation not available. Install with: pip install amplifier-foundation"
            ) from e

    @property
    def registry(self) -> BundleRegistry:
        """Get the bundle registry. Must call initialize() first."""
        if not self._registry:
            raise RuntimeError("BundleManager not initialized. Call initialize() first.")
        return self._registry

    async def load_and_prepare(
        self,
        bundle_name: str,
        behaviors: list[str] | None = None,
        provider_config: dict[str, Any] | None = None,
        working_directory: Path | None = None,
    ) -> Any:  # PreparedBundle
        """Load a bundle, compose behaviors, inject provider config, and prepare.

        NOW WITH TWO-TIER CACHING:
        - Level 1: Cache loaded bundles by URI
        - Level 2: Cache prepared bundles by (name, behaviors, provider_hash)

        Args:
            bundle_name: Bundle to load (e.g., "foundation", "amplifier-dev")
            behaviors: Optional behavior bundles to compose
            provider_config: Optional provider config to inject
            working_directory: Working directory for session (passed to create_session)

        Returns:
            PreparedBundle ready for create_session()
        """
        await self.initialize()

        from amplifier_foundation import Bundle

        # Generate cache key for prepared bundle (L2 cache check)
        cache_key = self._make_cache_key(bundle_name, behaviors, provider_config)

        # Check prepared cache first (fast path)
        if cache_key in self._prepared_cache:
            logger.debug(f"Using cached prepared bundle: {cache_key}")
            return self._prepared_cache[cache_key]

        # Load the base bundle (with L1 cache)
        bundle = await self._load_bundle_cached(bundle_name)
        logger.info(f"Loaded bundle: {bundle_name}")

        # Compose with behaviors if specified
        if behaviors:
            for behavior_name in behaviors:
                # Behaviors are typically namespaced like "foundation:behaviors/streaming-ui"
                behavior_ref = behavior_name
                if ":" not in behavior_name and "/" not in behavior_name:
                    # Short name - assume foundation behavior
                    behavior_ref = f"foundation:behaviors/{behavior_name}"

                try:
                    behavior_bundle = await self._load_bundle_cached(behavior_ref)
                    bundle = bundle.compose(behavior_bundle)
                    logger.info(f"Composed behavior: {behavior_name}")
                except Exception as e:
                    logger.warning(f"Failed to load behavior '{behavior_name}': {e}")

        # Enable debug for event visibility
        debug_bundle = Bundle(
            name="server-debug-config",
            version="1.0.0",
            session={"debug": True, "raw_debug": True},
        )
        bundle = bundle.compose(debug_bundle)

        # Compose with provider config if specified
        if provider_config:
            provider_bundle = Bundle(
                name="app-provider-config",
                version="1.0.0",
                providers=[provider_config],
            )
            bundle = bundle.compose(provider_bundle)
            logger.info(f"Injected provider config: {provider_config.get('module')}")
        else:
            # Auto-detect provider from environment
            provider_bundle = await self._auto_detect_provider()
            if provider_bundle:
                bundle = bundle.compose(provider_bundle)

        # Prepare the bundle (expensive: downloads dependencies)
        prepared = await bundle.prepare()

        # Cache the prepared bundle (L2 cache)
        self._prepared_cache[cache_key] = prepared
        logger.info(f"Bundle prepared and cached: {cache_key}")

        return prepared

    async def _auto_detect_provider(self) -> Bundle | None:
        """Auto-detect ALL providers from environment variables.

        Returns:
            Provider Bundle with all detected providers, None if no API keys found.
        """
        from amplifier_foundation import Bundle

        # Define all supported providers with their env vars and git sources
        provider_configs = [
            {
                "name": "anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "module": "provider-anthropic",
                "source": "git+https://github.com/microsoft/amplifier-module-provider-anthropic@main",
            },
            {
                "name": "openai",
                "env_var": "OPENAI_API_KEY",
                "module": "provider-openai",
                "source": "git+https://github.com/microsoft/amplifier-module-provider-openai@main",
            },
            {
                "name": "azure-openai",
                "env_var": "AZURE_OPENAI_API_KEY",
                "module": "provider-azure-openai",
                "source": "git+https://github.com/microsoft/amplifier-module-provider-azure-openai@main",
            },
            {
                "name": "gemini",
                "env_var": "GOOGLE_API_KEY",
                "module": "provider-gemini",
                "source": "git+https://github.com/microsoft/amplifier-module-provider-gemini@main",
            },
        ]

        # Collect all providers with API keys set
        detected_providers: list[dict[str, Any]] = []

        for config in provider_configs:
            if os.getenv(config["env_var"]):
                detected_providers.append(
                    {
                        "module": config["module"],
                        "source": config["source"],
                        "config": {
                            "debug": True,
                            "raw_debug": True,
                        },
                    }
                )
                logger.info(f"Auto-detected {config['name']} provider ({config['env_var']} is set)")

        if not detected_providers:
            logger.warning(
                "No provider API keys found in environment. "
                "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "AZURE_OPENAI_API_KEY, GOOGLE_API_KEY"
            )
            return None

        # Create bundle with ALL detected providers
        try:
            provider_bundle = Bundle(
                name="auto-providers",
                version="1.0.0",
                providers=detected_providers,
            )
            logger.info(f"Created provider bundle with {len(detected_providers)} provider(s)")
            return provider_bundle
        except Exception as e:
            logger.warning(f"Failed to create provider bundle: {e}")
            return None

    async def list_bundles(self) -> list[BundleInfo]:
        """List available bundles.

        Returns:
            List of BundleInfo with name and description.
        """
        await self.initialize()

        bundles = [
            BundleInfo(
                name="foundation", description="Core foundation bundle with tools and agents"
            ),
            BundleInfo(
                name="amplifier-dev", description="Bundle for Amplifier ecosystem development"
            ),
        ]

        return bundles

    # =========================================================================
    # Bundle Installation & Management
    # =========================================================================

    def _get_bundles_dir(self) -> Path:
        """Get the bundles directory, creating if needed."""
        bundles_dir = Path.home() / ".amplifier-runtime" / "bundles"
        bundles_dir.mkdir(parents=True, exist_ok=True)
        return bundles_dir

    def _get_registry_file(self) -> Path:
        """Get the bundle registry file path."""
        return Path.home() / ".amplifier-runtime" / "bundle-registry.yaml"

    def _load_registry_data(self) -> dict[str, Any]:
        """Load the bundle registry data."""
        import yaml

        registry_file = self._get_registry_file()
        if registry_file.exists():
            with open(registry_file) as f:
                return yaml.safe_load(f) or {"bundles": {}}
        return {"bundles": {}}

    def _save_registry_data(self, data: dict[str, Any]) -> None:
        """Save the bundle registry data."""
        import yaml

        registry_file = self._get_registry_file()
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def name_from_source(self, source: str) -> str:
        """Extract bundle name from source URL.

        Examples:
            git+https://github.com/microsoft/amplifier-bundle-recipes -> recipes
            /home/user/my-bundle -> my-bundle
        """
        import re

        # Remove git+ prefix and .git suffix
        clean = source.replace("git+", "").replace(".git", "")

        # Extract repo name from URL
        name = clean.rstrip("/").split("/")[-1] if "/" in clean else clean

        # Remove common prefixes
        name = re.sub(r"^amplifier-bundle-", "", name)
        name = re.sub(r"^amplifier-", "", name)

        return name or "unknown"

    async def install_bundle(
        self, source: str, name: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Install a bundle from source.

        Yields progress events during installation.

        Args:
            source: Git URL or local path
            name: Optional name (derived from source if not provided)

        Yields:
            Progress dicts with 'stage' and 'message' keys
        """
        import subprocess

        derived_name = name or self.name_from_source(source)
        bundles_dir = self._get_bundles_dir()
        target_dir = bundles_dir / derived_name

        # Yield: Starting
        yield {"stage": "starting", "message": f"Installing bundle: {derived_name}"}

        try:
            if source.startswith("git+") or source.startswith("https://"):
                # Git clone
                git_url = source.replace("git+", "")

                yield {"stage": "cloning", "message": f"Cloning from {git_url}"}

                # Remove existing if present
                if target_dir.exists():
                    import shutil

                    shutil.rmtree(target_dir)

                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", git_url, str(target_dir)],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Git clone failed: {result.stderr}")

                yield {"stage": "cloned", "message": "Repository cloned successfully"}

            elif Path(source).exists():
                # Local path - create symlink
                source_path = Path(source).resolve()

                yield {"stage": "linking", "message": f"Linking local bundle from {source_path}"}

                if target_dir.exists():
                    target_dir.unlink() if target_dir.is_symlink() else None

                target_dir.symlink_to(source_path)

                yield {"stage": "linked", "message": "Local bundle linked"}

            else:
                raise ValueError(f"Invalid source: {source}")

            # Validate bundle
            yield {"stage": "validating", "message": "Validating bundle structure"}

            bundle_file = target_dir / "bundle.md"
            if not bundle_file.exists():
                raise ValueError(f"No bundle.md found in {target_dir}")

            yield {"stage": "validated", "message": "Bundle structure valid"}

            # Register in bundle registry
            yield {"stage": "registering", "message": "Registering bundle"}

            registry_data = self._load_registry_data()
            registry_data["bundles"][derived_name] = {
                "source": source,
                "path": str(target_dir),
                "installed_at": self._now_iso(),
            }
            self._save_registry_data(registry_data)

            yield {
                "stage": "completed",
                "message": f"Bundle '{derived_name}' installed successfully",
            }

        except Exception as e:
            yield {"stage": "error", "message": str(e)}
            raise

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        from datetime import datetime

        return datetime.now(UTC).isoformat()

    async def add_local_bundle(self, path: str, name: str) -> BundleInfo:
        """Register a local bundle path.

        Args:
            path: Path to local bundle directory
            name: Name to register the bundle as

        Returns:
            BundleInfo for the added bundle
        """
        source_path = Path(path).resolve()

        if not source_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        bundle_file = source_path / "bundle.md"
        if not bundle_file.exists():
            raise ValueError(f"No bundle.md found in {path}")

        # Register in bundle registry
        registry_data = self._load_registry_data()
        registry_data["bundles"][name] = {
            "source": "local",
            "path": str(source_path),
            "added_at": self._now_iso(),
        }
        self._save_registry_data(registry_data)

        return BundleInfo(
            name=name,
            description=f"Local bundle from {path}",
            path=source_path,
            source="local",
        )

    async def remove_bundle(self, name: str) -> bool:
        """Remove a bundle registration.

        Args:
            name: Bundle name to remove

        Returns:
            True if removed, False if not found
        """
        registry_data = self._load_registry_data()

        if name not in registry_data["bundles"]:
            return False

        bundle_info = registry_data["bundles"][name]

        # Remove from registry
        del registry_data["bundles"][name]
        self._save_registry_data(registry_data)

        # Optionally remove files (only for git-installed bundles)
        if bundle_info.get("source", "").startswith("git"):
            bundle_path = Path(bundle_info.get("path", ""))
            if bundle_path.exists() and bundle_path.is_relative_to(self._get_bundles_dir()):
                import shutil

                shutil.rmtree(bundle_path)

        return True

    async def get_bundle_info(self, name: str) -> BundleInfo:
        """Get information about a bundle.

        Args:
            name: Bundle name

        Returns:
            BundleInfo with details

        Raises:
            ValueError if bundle not found
        """
        # Check builtin bundles first
        if name in ("foundation", "amplifier-dev"):
            return BundleInfo(
                name=name,
                description=f"Built-in {name} bundle",
                source="builtin",
            )

        # Check registry
        registry_data = self._load_registry_data()

        if name not in registry_data["bundles"]:
            raise ValueError(f"Bundle not found: {name}")

        bundle_data = registry_data["bundles"][name]

        return BundleInfo(
            name=name,
            description=f"Installed bundle: {name}",
            path=Path(bundle_data["path"]) if bundle_data.get("path") else None,
            source=bundle_data.get("source"),
            uri=bundle_data.get("source"),
        )

    async def _load_bundle_cached(self, bundle_uri: str) -> Any:  # Bundle
        """Load bundle with Level 1 caching.

        Args:
            bundle_uri: Bundle URI to load

        Returns:
            Loaded Bundle object
        """
        if bundle_uri not in self._bundle_cache:
            from amplifier_foundation.registry import load_bundle

            bundle = await load_bundle(bundle_uri, registry=self._registry)
            self._bundle_cache[bundle_uri] = bundle
            logger.debug(f"Loaded and cached bundle: {bundle_uri}")
        else:
            logger.debug(f"Using cached bundle: {bundle_uri}")
        return self._bundle_cache[bundle_uri]

    def _make_cache_key(
        self,
        bundle_name: str,
        behaviors: list[str] | None,
        provider_config: dict[str, Any] | None,
    ) -> str:
        """Generate cache key for prepared bundle.

        Key format: {bundle}:{behaviors_hash}:{provider_hash}
        Different configs = different cache entries

        Args:
            bundle_name: Base bundle name
            behaviors: Optional behavior list
            provider_config: Optional provider config

        Returns:
            Cache key string
        """
        import hashlib
        import json

        parts = [bundle_name]

        # Hash behaviors list
        if behaviors:
            behaviors_str = ",".join(sorted(behaviors))
            behaviors_hash = hashlib.md5(behaviors_str.encode()).hexdigest()[:8]
            parts.append(behaviors_hash)
        else:
            parts.append("no-behaviors")

        # Hash provider config
        if provider_config:
            config_str = json.dumps(provider_config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            parts.append(config_hash)
        else:
            parts.append("no-provider")

        return ":".join(parts)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dict with cache sizes and keys
        """
        return {
            "bundle_cache_size": len(self._bundle_cache),
            "prepared_cache_size": len(self._prepared_cache),
            "bundle_cache_keys": list(self._bundle_cache.keys()),
            "prepared_cache_keys": list(self._prepared_cache.keys()),
        }

    def invalidate_cache(self, bundle_uri: str | None = None) -> None:
        """Invalidate bundle cache.

        Args:
            bundle_uri: Specific bundle to invalidate, or None for all
        """
        if bundle_uri:
            # Invalidate specific bundle
            self._bundle_cache.pop(bundle_uri, None)
            # Invalidate all prepared bundles using this bundle
            to_remove = [k for k in self._prepared_cache if k.startswith(bundle_uri + ":")]
            for key in to_remove:
                self._prepared_cache.pop(key)
            logger.info(f"Invalidated cache for bundle: {bundle_uri}")
        else:
            # Invalidate all
            self._bundle_cache.clear()
            self._prepared_cache.clear()
            logger.info("Invalidated all bundle caches")

        # Also clear the registry's cache if available
        try:
            if self._registry and hasattr(self._registry, "clear_cache"):
                self._registry.clear_cache()  # type: ignore[attr-defined]
                logger.debug("Cleared bundle registry cache")
        except Exception as e:
            logger.debug(f"Registry cache clear not available: {e}")
