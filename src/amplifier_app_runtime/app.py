"""Amplifier Server Application.

Creates the Starlette ASGI application with all routes.

Route organization when ACP is DISABLED (default):
- /health - Health check endpoints
- /event - SSE event streaming
- /ws - WebSocket transport
- /v1/* - Protocol-based routes
- /session/* - Session management (legacy)

Route organization when ACP is ENABLED (--acp-enabled):
- /health - Health check endpoints (shared)
- /acp/* - Agent Client Protocol routes (JSON-RPC over HTTP/WS)
- /amplifier/event - SSE event streaming (namespaced)
- /amplifier/ws - WebSocket transport (namespaced)
- /amplifier/v1/* - Protocol-based routes (namespaced)
- /amplifier/session/* - Session management (namespaced, legacy)

CRITICAL: When ACP is enabled:
- STDIO is exclusively owned by ACP for JSON-RPC protocol
- Amplifier's internal STDIO transport MUST NOT be used
- All Amplifier events flow through ACP's session_update() notifications
- HTTP/WS/SSE routes are namespaced under /amplifier/ to avoid conflicts
"""

import os

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount, Route

from .routes import (
    event_routes,
    health_routes,
    modules_routes,
    protocol_routes,
    session_routes,
    websocket_routes,
)


def create_app(*, use_protocol_routes: bool = True) -> Starlette:
    """Create the Amplifier backend application.

    Args:
        use_protocol_routes: If True, use protocol-based routes at /v1/
                            If False, use legacy session routes

    Returns:
        Configured Starlette application
    """
    # Check if ACP is enabled via environment variable
    acp_enabled = os.environ.get("AMPLIFIER_ACP_ENABLED", "").lower() in ("1", "true", "yes")

    # Combine all routes
    routes: list[Route | Mount] = []

    # Health routes are always at root (shared by both ACP and Amplifier)
    routes.extend(health_routes)

    # Module discovery routes (also at root for easy access)
    routes.extend(modules_routes)

    if acp_enabled:
        # =================================================================
        # ACP MODE: ACP owns the root, Amplifier routes are namespaced
        # =================================================================
        from .acp import acp_routes

        # ACP routes at root level
        routes.extend(acp_routes)  # /acp/rpc, /acp/events, /acp/ws

        # Amplifier routes namespaced under /amplifier/
        # This prevents any conflict with ACP protocol
        amplifier_routes: list[Route | Mount] = []
        amplifier_routes.extend(event_routes)  # /amplifier/event
        amplifier_routes.extend(websocket_routes)  # /amplifier/ws

        if use_protocol_routes:
            amplifier_routes.append(Mount("/v1", routes=protocol_routes))
            amplifier_routes.extend(protocol_routes)
        else:
            amplifier_routes.extend(session_routes)

        routes.append(Mount("/amplifier", routes=amplifier_routes))
    else:
        # =================================================================
        # STANDARD MODE: Amplifier routes at root (no ACP)
        # =================================================================
        routes.extend(event_routes)
        routes.extend(websocket_routes)

        if use_protocol_routes:
            routes.append(Mount("/v1", routes=protocol_routes))
            routes.extend(protocol_routes)
        else:
            routes.extend(session_routes)

    # CORS middleware for local development
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for local dev
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]

    app = Starlette(routes=routes, middleware=middleware)
    return app


# Lazy singleton for embedded mode
_app: Starlette | None = None


def get_app() -> Starlette:
    """Get or create the application singleton.

    Used by embedded mode to ensure single app instance.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app
