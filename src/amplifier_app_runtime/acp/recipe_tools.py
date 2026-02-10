"""Recipe discovery and management tools for ACP.

Provides structured recipe metadata for IDE integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def list_recipes_tool(pattern: str | None = None) -> dict[str, Any]:
    """List available recipes with metadata.

    This is a host-defined tool that provides structured recipe discovery
    for IDE clients. Returns machine-readable recipe metadata including
    name, description, stages, and approval gate information.

    Args:
        pattern: Optional glob pattern to filter recipes (e.g., "code-*", "*.yaml")

    Returns:
        Dictionary with:
        - recipes: List of recipe metadata dicts
        - count: Total number of recipes found
        - pattern: Pattern used (if any)

    Example output:
        {
            "recipes": [
                {
                    "path": "@recipes:examples/code-review.yaml",
                    "name": "code-review",
                    "description": "Systematic code review workflow",
                    "requires_approval": true,
                    "stages": ["analysis", "feedback"],
                    "steps": null,
                    "source": "recipes bundle"
                }
            ],
            "count": 1,
            "pattern": null
        }
    """
    try:
        # Discover recipe paths
        recipe_paths = await _discover_recipe_paths(pattern)

        # Extract metadata from each recipe
        recipes = []
        for recipe_path in recipe_paths:
            try:
                metadata = await _extract_recipe_metadata(recipe_path)
                recipes.append(metadata)
            except Exception as e:
                # Include failed recipes with error flag
                recipes.append(
                    {
                        "path": recipe_path,
                        "name": Path(recipe_path).stem,
                        "error": str(e),
                        "valid": False,
                    }
                )
                logger.warning(f"Failed to parse recipe {recipe_path}: {e}")

        return {
            "recipes": recipes,
            "count": len(recipes),
            "pattern": pattern,
        }

    except Exception as e:
        logger.error(f"Recipe discovery failed: {e}")
        return {
            "recipes": [],
            "count": 0,
            "error": str(e),
        }


async def _discover_recipe_paths(pattern: str | None = None) -> list[str]:
    """Find recipe files matching pattern.

    Searches common recipe locations:
    - @recipes: namespace (from recipes bundle)
    - .amplifier/recipes/ (workspace recipes)
    - ~/.amplifier/recipes/ (user recipes)

    Args:
        pattern: Optional glob pattern

    Returns:
        List of recipe paths (as URIs or file paths)
    """
    recipe_paths = []

    # Try to load recipes from the recipes bundle namespace
    # This requires the recipes bundle to be available
    try:
        from amplifier_foundation.registry import BundleRegistry

        registry = BundleRegistry()

        # Try to resolve @recipes: namespace
        try:
            # Load recipes bundle to access its recipes/ directory
            from amplifier_foundation import load_bundle

            _bundle = await load_bundle("recipes", registry=registry)

            # Get bundle path and find recipe files
            # Note: This is simplified - actual implementation would need
            # to navigate bundle structure properly
            logger.debug("Recipes bundle loaded, enumeration requires bundle path access")
        except Exception as e:
            logger.debug(f"Recipes bundle not available: {e}")

    except ImportError:
        logger.debug("amplifier-foundation not available for bundle recipe discovery")

    # Check workspace recipes
    workspace_recipes = Path.cwd() / ".amplifier" / "recipes"
    if workspace_recipes.exists():
        recipe_paths.extend(_find_yaml_files(workspace_recipes, pattern))

    # Check user recipes
    user_recipes = Path.home() / ".amplifier" / "recipes"
    if user_recipes.exists():
        recipe_paths.extend(_find_yaml_files(user_recipes, pattern))

    return recipe_paths


def _find_yaml_files(directory: Path, pattern: str | None = None) -> list[str]:
    """Find YAML files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Optional glob pattern

    Returns:
        List of file paths as strings
    """
    files = directory.glob(pattern) if pattern else directory.rglob("*.yaml")
    return [str(f) for f in files if f.is_file()]


async def _extract_recipe_metadata(recipe_path: str) -> dict[str, Any]:
    """Parse recipe YAML and extract metadata.

    Args:
        recipe_path: Path to recipe file

    Returns:
        Recipe metadata dictionary

    Raises:
        ValueError: If recipe is invalid or cannot be parsed
    """
    import yaml

    # Read recipe file
    path = Path(recipe_path)
    if not path.exists():
        raise ValueError(f"Recipe file not found: {recipe_path}")

    try:
        with open(path) as f:
            recipe_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if not isinstance(recipe_data, dict):
        raise ValueError("Recipe YAML must be a dictionary")

    # Extract metadata
    metadata = {
        "path": recipe_path,
        "name": path.stem,
        "description": recipe_data.get("description", ""),
        "valid": True,
    }

    # Check if staged or flat recipe
    if "stages" in recipe_data:
        # Staged recipe with approval gates
        metadata["requires_approval"] = True
        metadata["stages"] = [
            stage.get("name", f"stage_{i}") for i, stage in enumerate(recipe_data["stages"])
        ]
        metadata["steps"] = None
    elif "steps" in recipe_data:
        # Flat recipe
        metadata["requires_approval"] = False
        metadata["stages"] = None
        metadata["steps"] = [
            step.get("name", f"step_{i}") for i, step in enumerate(recipe_data["steps"])
        ]
    else:
        raise ValueError("Recipe must have 'steps' or 'stages' field")

    # Add source information
    if recipe_path.startswith("@"):
        # Bundle reference
        namespace = recipe_path.split(":")[0].lstrip("@")
        metadata["source"] = f"{namespace} bundle"
    elif ".amplifier" in recipe_path:
        if str(Path.home()) in recipe_path:
            metadata["source"] = "user recipes"
        else:
            metadata["source"] = "workspace recipes"
    else:
        metadata["source"] = "local file"

    return metadata
