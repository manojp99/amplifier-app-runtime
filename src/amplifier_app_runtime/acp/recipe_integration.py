"""Recipe integration setup for ACP.

Registers recipe discovery tool in the host tool registry.
"""

from __future__ import annotations

import logging
from typing import Any

from ..host_tools import HostToolDefinition, ToolContext, ToolResult

logger = logging.getLogger(__name__)


async def setup_recipe_tools(registry: Any) -> None:
    """Register recipe tools in the host tool registry.

    Args:
        registry: HostToolRegistry to register tools in
    """
    from .recipe_tools import list_recipes_tool

    async def list_recipes_handler(tool_input: dict, context: ToolContext) -> ToolResult:
        """Handler for list_recipes tool.

        Args:
            tool_input: Tool input with optional 'pattern' field
            context: Tool execution context

        Returns:
            ToolResult with recipe list
        """
        try:
            pattern = tool_input.get("pattern")
            result = await list_recipes_tool(pattern=pattern)

            return ToolResult(
                success=True,
                output=result,
            )
        except Exception as e:
            logger.error(f"Recipe discovery failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                output={"recipes": [], "count": 0},
            )

    # Register list_recipes tool
    await registry.register(
        HostToolDefinition(
            name="list_recipes",
            description=(
                "List available recipe workflows with metadata. Recipes are "
                "multi-step AI agent orchestration workflows. Returns structured "
                "data including recipe name, description, stages (for approval "
                "gates), and steps."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Optional glob pattern to filter recipes (e.g., 'code-*', '*.yaml')"
                        ),
                    }
                },
            },
            handler=list_recipes_handler,
        )
    )

    logger.info("Registered recipe tools in host registry")
