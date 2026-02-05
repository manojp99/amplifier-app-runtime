"""Amplifier Runtime CLI.

Default mode is stdio (for IDE/subprocess integration).
Use --http to run as HTTP server.

Usage:
    amplifier-runtime                     # Stdio mode (default)
    amplifier-runtime --http              # HTTP server mode
    amplifier-runtime --http --port 8080  # HTTP with custom port
    amplifier-runtime --http --acp        # HTTP with ACP endpoints
    amplifier-runtime --health            # Check HTTP server health

    amplifier-runtime session list        # List saved sessions
    amplifier-runtime session info <id>   # Show session details
    amplifier-runtime session resume <id> # Resume a session
    amplifier-runtime session delete <id> # Delete a session

    amplifier-runtime bundle list         # List available bundles
    amplifier-runtime bundle info <name>  # Show bundle details

    amplifier-runtime provider list       # List providers
    amplifier-runtime provider check <n>  # Check provider status

    amplifier-runtime config              # Show configuration
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
import httpx

if TYPE_CHECKING:
    pass

# Output format options
FORMAT_TABLE = "table"
FORMAT_JSON = "json"


def format_datetime(dt: datetime | str | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        # Parse ISO format
        try:
            dt = datetime.fromisoformat(dt.rstrip("Z"))
        except ValueError:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M")


def truncate(text: str | None, max_len: int = 50) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _save_env_var(env_file: Any, var_name: str, value: str) -> None:
    """Save an environment variable to a .env file.

    Appends or updates the variable in the file.

    Args:
        env_file: Path to the .env file
        var_name: Name of the environment variable
        value: Value to save
    """
    from pathlib import Path

    env_path = Path(env_file)
    lines: list[str] = []

    # Read existing content
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    # Update or append the variable
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{var_name}="):
            lines[i] = f"{var_name}={value}"
            found = True
            break

    if not found:
        lines.append(f"{var_name}={value}")

    # Write back
    env_path.write_text("\n".join(lines) + "\n")


@click.group(invoke_without_command=True)
@click.option("--http", "http_mode", is_flag=True, help="Run as HTTP server instead of stdio")
@click.option("--host", default="127.0.0.1", help="Host to bind to (HTTP mode)")
@click.option("--port", default=4096, help="Port to bind to (HTTP mode)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development (HTTP mode)")
@click.option("--acp", "acp_enabled", is_flag=True, help="Enable ACP endpoints (HTTP mode)")
@click.option("--health", "health_check", is_flag=True, help="Check HTTP server health and exit")
@click.option("--health-url", default="http://localhost:4096", help="Server URL for health check")
@click.option(
    "--host-tools",
    "host_tools_path",
    type=click.Path(exists=True),
    help="Path to YAML file defining host tools",
)
@click.option(
    "--host-tools-module",
    "host_tools_module",
    help="Python module containing host tool definitions (e.g., myapp.tools)",
)
@click.pass_context
def main(
    ctx: click.Context,
    http_mode: bool,
    host: str,
    port: int,
    reload: bool,
    acp_enabled: bool,
    health_check: bool,
    health_url: str,
    host_tools_path: str | None,
    host_tools_module: str | None,
) -> None:
    """Amplifier Runtime - AI agent server for IDE integrations.

    By default, runs in stdio mode for subprocess/IPC communication.
    Use --http to run as an HTTP server.
    """
    # If a subcommand is invoked, let it handle everything
    if ctx.invoked_subcommand is not None:
        return

    # Validate flag combinations
    if acp_enabled and not http_mode:
        raise click.UsageError(
            "--acp requires --http mode. "
            "ACP endpoints are only available when running as an HTTP server.\n\n"
            "Usage:\n"
            "  amplifier-runtime --http --acp     # HTTP server with ACP endpoints\n"
            "  amplifier-runtime                  # Stdio mode (no ACP endpoints)"
        )

    if reload and not http_mode:
        raise click.UsageError(
            "--reload requires --http mode. "
            "Auto-reload is only available when running as an HTTP server."
        )

    if (host != "127.0.0.1" or port != 4096) and not http_mode and not health_check:
        raise click.UsageError(
            "--host and --port require --http mode. "
            "These options are only available when running as an HTTP server."
        )

    # Handle --health flag
    if health_check:
        _do_health_check(health_url)
        return

    # Load host tools if specified
    if host_tools_path or host_tools_module:
        _load_host_tools(host_tools_path, host_tools_module)

    # Run in appropriate mode
    if http_mode:
        _run_http_server(host, port, reload, acp_enabled)
    else:
        _run_stdio_server()


def _load_host_tools(yaml_path: str | None, module_name: str | None) -> None:
    """Load host tools from YAML file or Python module.

    Args:
        yaml_path: Path to YAML file defining tools
        module_name: Python module containing tool definitions
    """
    from .host_tools import HostToolDefinition, host_tool_registry

    tools_loaded = 0

    # Load from YAML file
    if yaml_path:
        import importlib

        import yaml

        click.echo(f"Loading host tools from {yaml_path}", err=True)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        tools_config = config.get("tools", [])
        for tool_config in tools_config:
            name = tool_config.get("name")
            description = tool_config.get("description", "")
            parameters = tool_config.get("parameters", {"type": "object"})
            module_path = tool_config.get("module")
            function_name = tool_config.get("function")

            if not all([name, module_path, function_name]):
                click.echo(
                    f"Skipping incomplete tool definition: {tool_config}",
                    err=True,
                )
                continue

            try:
                # Import the handler function
                mod = importlib.import_module(module_path)
                handler = getattr(mod, function_name)

                # Register the tool
                definition = HostToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    requires_approval=tool_config.get("requires_approval", False),
                    timeout=tool_config.get("timeout"),
                    category=tool_config.get("category"),
                )
                asyncio.run(host_tool_registry.register(definition))
                tools_loaded += 1
                click.echo(f"  Registered: {name}", err=True)

            except Exception as e:
                click.echo(f"  Failed to load {name}: {e}", err=True)

    # Load from Python module
    if module_name:
        import importlib

        click.echo(f"Loading host tools from module {module_name}", err=True)

        try:
            # Import the module - this will trigger any @host_tool decorators
            mod = importlib.import_module(module_name)

            # Look for a setup function
            if hasattr(mod, "setup_host_tools"):
                setup_fn = mod.setup_host_tools
                if asyncio.iscoroutinefunction(setup_fn):
                    asyncio.run(setup_fn(host_tool_registry))
                else:
                    setup_fn(host_tool_registry)

            # Count tools registered
            tools_loaded = host_tool_registry.count
            click.echo(f"  Module loaded, {tools_loaded} tools registered", err=True)

        except ImportError as e:
            click.echo(f"Failed to import module {module_name}: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error loading tools from {module_name}: {e}", err=True)
            sys.exit(1)

    if tools_loaded > 0:
        click.echo(f"Total host tools loaded: {tools_loaded}", err=True)


def _do_health_check(url: str) -> None:
    """Check server health."""

    async def check() -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    data = response.json()
                    click.echo(f"Server is healthy: {data}")
                else:
                    click.echo(f"Server returned {response.status_code}", err=True)
                    sys.exit(1)
        except httpx.ConnectError:
            click.echo(f"Cannot connect to server at {url}", err=True)
            sys.exit(1)

    asyncio.run(check())


def _run_http_server(host: str, port: int, reload: bool, acp_enabled: bool) -> None:
    """Run HTTP server mode."""
    import os

    import uvicorn

    # Pass ACP flag via environment variable for the app factory
    if acp_enabled:
        os.environ["AMPLIFIER_ACP_ENABLED"] = "1"
        click.echo(f"Starting Amplifier runtime on http://{host}:{port} (ACP enabled)", err=True)
        click.echo("  ACP endpoints: /acp/rpc, /acp/events, /acp/ws", err=True)
    else:
        os.environ.pop("AMPLIFIER_ACP_ENABLED", None)
        click.echo(f"Starting Amplifier runtime on http://{host}:{port}", err=True)

    click.echo("Press Ctrl+C to stop", err=True)

    uvicorn.run(
        "amplifier_app_runtime.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def _run_stdio_server() -> None:
    """Run stdio server mode (default).

    Uses native Amplifier protocol over stdin/stdout (JSON lines).
    For ACP protocol, use --http --acp instead.
    """
    from .stdio import run_native_stdio

    try:
        asyncio.run(run_native_stdio())
    except KeyboardInterrupt:
        click.echo("\nShutting down", err=True)


# =============================================================================
# Session Commands
# =============================================================================


@main.group()
def session() -> None:
    """Manage saved sessions."""


@session.command("list")
@click.option("--all", "show_all", is_flag=True, help="Include sub-sessions (spawned agents)")
@click.option("--limit", "-n", default=20, help="Maximum sessions to show")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice([FORMAT_TABLE, FORMAT_JSON]),
    default=FORMAT_TABLE,
    help="Output format",
)
def session_list(show_all: bool, limit: int, output_format: str) -> None:
    """List saved sessions.

    Examples:

        # List recent sessions
        amplifier-runtime session list

        # Include agent sub-sessions
        amplifier-runtime session list --all

        # JSON output for scripting
        amplifier-runtime session list --format json
    """
    from .session_store import SessionStore

    store = SessionStore()
    sessions = store.list_sessions(
        top_level_only=not show_all,
        min_turns=0 if show_all else 1,
        limit=limit,
    )

    if not sessions:
        click.echo("No sessions found.")
        return

    if output_format == FORMAT_JSON:
        click.echo(json.dumps(sessions, indent=2, ensure_ascii=False, default=str))
        return

    # Table format
    click.echo(f"{'ID':<20} {'Bundle':<15} {'Turns':>6} {'Updated':<17} {'State':<10}")
    click.echo("-" * 75)

    for s in sessions:
        session_id = s.get("session_id", "?")[:20]
        bundle = truncate(s.get("bundle_name") or "default", 15)
        turns = s.get("turn_count", 0)
        updated = format_datetime(s.get("updated_at"))
        state = s.get("state", "?")[:10]

        click.echo(f"{session_id:<20} {bundle:<15} {turns:>6} {updated:<17} {state:<10}")

    click.echo(f"\nTotal: {len(sessions)} session(s)")


@session.command("info")
@click.argument("session_id")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice([FORMAT_TABLE, FORMAT_JSON]),
    default=FORMAT_TABLE,
    help="Output format",
)
@click.option("--transcript", "-t", is_flag=True, help="Include conversation transcript")
def session_info(session_id: str, output_format: str, transcript: bool) -> None:
    """Show detailed information about a session.

    Examples:

        # Show session info
        amplifier-runtime session info sess_abc123

        # Include transcript
        amplifier-runtime session info sess_abc123 --transcript

        # JSON output
        amplifier-runtime session info sess_abc123 --format json --transcript
    """
    from .session_store import SessionStore

    store = SessionStore()
    info = store.get_session_summary(session_id)

    if info is None:
        click.echo(f"Session not found: {session_id}", err=True)
        sys.exit(1)

    if transcript:
        info["transcript"] = store.load_transcript(session_id)

    if output_format == FORMAT_JSON:
        click.echo(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        return

    # Table format
    click.echo(f"Session: {info.get('session_id')}")
    click.echo(f"Bundle:  {info.get('bundle_name') or 'default'}")
    click.echo(f"State:   {info.get('state', 'unknown')}")
    click.echo(f"Turns:   {info.get('turn_count', 0)}")
    click.echo(f"Created: {format_datetime(info.get('created_at'))}")
    click.echo(f"Updated: {format_datetime(info.get('updated_at'))}")

    if info.get("cwd"):
        click.echo(f"CWD:     {info.get('cwd')}")

    if info.get("parent_session_id"):
        click.echo(f"Parent:  {info.get('parent_session_id')}")

    if info.get("error"):
        click.echo(f"Error:   {info.get('error')}")

    # Preview
    if info.get("first_user_message"):
        click.echo(f"\nFirst prompt: {info.get('first_user_message')}")

    if info.get("last_assistant_message"):
        click.echo(f"Last response: {info.get('last_assistant_message')}")

    # Transcript
    if transcript and info.get("transcript"):
        click.echo("\n--- Transcript ---")
        for msg in info["transcript"]:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if isinstance(content, str):
                content = truncate(content, 200)
            click.echo(f"\n[{role}]")
            click.echo(content)


@session.command("resume")
@click.argument("session_id")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def session_resume(session_id: str, output_json: bool) -> None:
    """Resume a session and show its state.

    Examples:

        # Show session state
        amplifier-runtime session resume sess_abc123

        # JSON output
        amplifier-runtime session resume sess_abc123 --json
    """
    from .session import SessionManager

    async def execute() -> None:
        manager = SessionManager()
        managed_session = await manager.resume(session_id)

        if not managed_session:
            click.echo(f"Session not found: {session_id}", err=True)
            sys.exit(1)

        click.echo(f"Resumed session {session_id}", err=True)
        click.echo(
            f"Turn count: {managed_session.metadata.turn_count}, "
            f"Messages: {len(managed_session.get_transcript())}",
            err=True,
        )

        if output_json:
            click.echo(json.dumps(managed_session.to_dict(), indent=2, default=str))

    asyncio.run(execute())


@session.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def session_delete(session_id: str, yes: bool) -> None:
    """Delete a saved session.

    Examples:

        # Delete with confirmation
        amplifier-runtime session delete sess_abc123

        # Skip confirmation
        amplifier-runtime session delete sess_abc123 --yes
    """
    from .session_store import SessionStore

    store = SessionStore()

    if not store.session_exists(session_id):
        click.echo(f"Session not found: {session_id}", err=True)
        sys.exit(1)

    if not yes and not click.confirm(f"Delete session {session_id}?"):
        click.echo("Cancelled.")
        return

    if store.delete_session(session_id):
        click.echo(f"Deleted session {session_id}")
    else:
        click.echo(f"Failed to delete session {session_id}", err=True)
        sys.exit(1)


@session.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation", required=True)
def session_clear(yes: bool) -> None:
    """Delete all saved sessions.

    Examples:

        amplifier-runtime session clear --yes
    """
    from .session_store import SessionStore

    store = SessionStore()

    if not yes:
        click.echo("Must pass --yes to confirm deletion of all sessions.", err=True)
        sys.exit(1)

    count = store.delete_all_sessions(confirm=True)
    click.echo(f"Deleted {count} session(s)")


# =============================================================================
# Bundle Commands
# =============================================================================


@main.group()
def bundle() -> None:
    """Manage bundles."""


@bundle.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice([FORMAT_TABLE, FORMAT_JSON]),
    default=FORMAT_TABLE,
    help="Output format",
)
def bundle_list(output_format: str) -> None:
    """List available bundles.

    Examples:

        amplifier-runtime bundle list
        amplifier-runtime bundle list --format json
    """

    async def run() -> None:
        from .bundle_manager import BundleManager

        manager = BundleManager()
        bundles = await manager.list_bundles()

        if output_format == FORMAT_JSON:
            click.echo(
                json.dumps(
                    [{"name": b.name, "description": b.description, "uri": b.uri} for b in bundles],
                    indent=2,
                )
            )
            return

        click.echo(f"{'Name':<20} {'Description':<50}")
        click.echo("-" * 72)
        for b in bundles:
            click.echo(f"{b.name:<20} {truncate(b.description, 50):<50}")

    asyncio.run(run())


@bundle.command("info")
@click.argument("bundle_name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice([FORMAT_TABLE, FORMAT_JSON]),
    default=FORMAT_TABLE,
    help="Output format",
)
def bundle_info(bundle_name: str, output_format: str) -> None:
    """Show information about a bundle.

    Examples:

        amplifier-runtime bundle info foundation
        amplifier-runtime bundle info amplifier-dev --format json
    """

    async def run() -> None:
        from .bundle_manager import BundleManager

        manager = BundleManager()
        try:
            prepared = await manager.load_and_prepare(bundle_name)

            info = {
                "name": bundle_name,
                "tools": [],
                "agents": [],
                "providers": [],
            }

            # Extract available tools
            if hasattr(prepared, "bundle") and hasattr(prepared.bundle, "tools"):
                info["tools"] = [
                    t.get("name", t.get("module", "unknown")) for t in (prepared.bundle.tools or [])
                ]

            # Extract available agents
            if hasattr(prepared, "bundle") and hasattr(prepared.bundle, "agents"):
                info["agents"] = list((prepared.bundle.agents or {}).keys())

            if output_format == FORMAT_JSON:
                click.echo(json.dumps(info, indent=2))
                return

            click.echo(f"Bundle: {bundle_name}")
            click.echo(f"Tools:  {', '.join(info['tools']) or 'none'}")
            click.echo(f"Agents: {', '.join(info['agents']) or 'none'}")

        except Exception as e:
            click.echo(f"Error loading bundle '{bundle_name}': {e}", err=True)
            sys.exit(1)

    asyncio.run(run())


# =============================================================================
# Provider Commands
# =============================================================================


@main.group()
def provider() -> None:
    """Manage providers."""


@provider.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice([FORMAT_TABLE, FORMAT_JSON]),
    default=FORMAT_TABLE,
    help="Output format",
)
def provider_list(output_format: str) -> None:
    """List available providers.

    Shows all known providers with their installed and configured status.

    Examples:

        amplifier-runtime provider list
    """
    import os

    from .provider_sources import get_installed_providers

    providers = get_installed_providers()

    # Add configured status based on env vars
    for p in providers:
        env_var = p.get("env_var")
        p["configured"] = bool(os.getenv(env_var)) if env_var else False

    if output_format == FORMAT_JSON:
        click.echo(json.dumps(providers, indent=2))
        return

    click.echo(f"{'Provider':<15} {'Installed':<12} {'Configured':<12} {'Env Var':<25}")
    click.echo("-" * 65)
    for p in providers:
        installed = (
            click.style("yes", fg="green") if p["installed"] else click.style("no", fg="yellow")
        )
        configured = (
            click.style("yes", fg="green") if p["configured"] else click.style("no", fg="red")
        )
        env_var = p.get("env_var") or "-"
        click.echo(f"{p['display_name']:<15} {installed:<21} {configured:<21} {env_var:<25}")

    # Summary
    installed_count = sum(1 for p in providers if p["installed"])
    configured_count = sum(1 for p in providers if p["configured"])
    click.echo(f"\n{installed_count}/{len(providers)} installed, {configured_count} configured")


@provider.command("install")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def provider_install(quiet: bool) -> None:
    """Install all known provider modules.

    Downloads and installs all provider modules so they are available
    for use. This does NOT configure them - you still need to set
    the appropriate API key environment variables.

    Examples:

        amplifier-runtime provider install
        amplifier-runtime provider install -q
    """
    from .provider_sources import install_known_providers

    if not quiet:
        click.echo("Installing provider modules...")
        click.echo("(This downloads provider code - API keys are still needed for configuration)\n")

    installed = install_known_providers(verbose=not quiet, quiet=quiet)

    if not quiet:
        click.echo(f"\n✓ Installed {len(installed)} provider(s)")
        click.echo("\nTo configure a provider, set its API key environment variable:")
        click.echo("  export ANTHROPIC_API_KEY=your-key")
        click.echo("  export OPENAI_API_KEY=your-key")
        click.echo("  export AZURE_OPENAI_API_KEY=your-key")
        click.echo("  export GOOGLE_API_KEY=your-key")


@provider.command("check")
@click.argument("provider_name")
def provider_check(provider_name: str) -> None:
    """Check if a provider is configured and working.

    Examples:

        amplifier-runtime provider check anthropic
        amplifier-runtime provider check openai
    """
    import os

    env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure-openai": "AZURE_OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = env_vars.get(provider_name.lower())
    if not env_var:
        click.echo(f"Unknown provider: {provider_name}", err=True)
        click.echo(f"Available providers: {', '.join(env_vars.keys())}")
        sys.exit(1)

    if os.getenv(env_var):
        click.echo(f"{provider_name}: " + click.style("configured", fg="green"))
        click.echo(f"Environment variable {env_var} is set")
    else:
        click.echo(f"{provider_name}: " + click.style("not configured", fg="red"))
        click.echo(f"Set environment variable {env_var} to enable this provider")
        sys.exit(1)


# =============================================================================
# Init Command
# =============================================================================


# Known providers bundled with the runtime
BUNDLED_PROVIDERS = [
    {
        "name": "anthropic",
        "display_name": "Anthropic",
        "module": "provider-anthropic",
        "env_var": "ANTHROPIC_API_KEY",
    },
    {
        "name": "openai",
        "display_name": "OpenAI",
        "module": "provider-openai",
        "env_var": "OPENAI_API_KEY",
    },
    {
        "name": "azure-openai",
        "display_name": "Azure OpenAI",
        "module": "provider-azure-openai",
        "env_var": "AZURE_OPENAI_API_KEY",
    },
    {
        "name": "gemini",
        "display_name": "Google Gemini",
        "module": "provider-gemini",
        "env_var": "GOOGLE_API_KEY",
    },
    {
        "name": "ollama",
        "display_name": "Ollama",
        "module": "provider-ollama",
        "env_var": "OLLAMA_HOST",
    },
    {
        "name": "vllm",
        "display_name": "vLLM",
        "module": "provider-vllm",
        "env_var": "VLLM_API_BASE",
    },
]


@main.command("init")
@click.option("--bundle", "-b", default=None, help="Default bundle to use")
@click.option("--provider", "-p", default=None, help="Default provider (anthropic, openai, etc.)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing configuration")
@click.option("--yes", "-y", is_flag=True, help="Non-interactive mode: use env vars and defaults")
def init_config(bundle: str | None, provider: str | None, force: bool, yes: bool) -> None:
    """Interactive first-time setup wizard.

    Providers are bundled with the runtime - no download needed.
    This wizard helps you configure your API keys and select a default model.

    Examples:

        # Interactive setup (recommended)
        amplifier-runtime init

        # Non-interactive with auto-detection from env vars
        amplifier-runtime init --yes

        # Specify provider and bundle directly
        amplifier-runtime init --provider anthropic --bundle foundation
    """
    import os
    from pathlib import Path

    import yaml

    from .key_manager import KeyManager
    from .provider_config_utils import configure_provider

    config_dir = Path.home() / ".amplifier-runtime"
    settings_file = config_dir / "settings.yaml"

    # Check for TTY in interactive mode
    if not yes and not sys.stdin.isatty():
        click.echo("Error: Interactive mode requires a TTY.", err=True)
        click.echo("Use --yes flag for non-interactive setup.")
        click.echo("\nExample:")
        click.echo("  amplifier-runtime init --yes")
        sys.exit(1)

    # Check if already initialized
    if settings_file.exists() and not force:
        click.echo(f"Configuration already exists at {settings_file}")
        click.echo("Use --force to overwrite existing configuration.")
        if not yes and not click.confirm("Continue anyway?"):
            return

    # Create config directory
    config_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Welcome
    # =========================================================================
    click.echo()
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo(click.style("  Welcome to Amplifier Runtime Setup!", fg="cyan", bold=True))
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo()

    # Initialize key manager (loads existing keys from ~/.amplifier-runtime/keys.env)
    key_manager = KeyManager()

    # =========================================================================
    # Step 1: Show available providers and their status
    # =========================================================================
    click.echo(click.style("Step 1: Available Providers", bold=True))
    click.echo(click.style("(Providers are bundled - no download needed)", dim=True))
    click.echo()

    # Mark which providers are configured (have API keys)
    providers: list[dict[str, Any]] = []
    for p in BUNDLED_PROVIDERS:
        p_copy: dict[str, Any] = dict(p)
        env_var = p_copy.get("env_var")
        p_copy["configured"] = bool(env_var and os.getenv(env_var))
        providers.append(p_copy)

        if p_copy["configured"]:
            click.echo(
                f"  ✓ {p_copy['display_name']:<15} "
                + click.style("configured", fg="green")
                + f" ({env_var})"
            )
        elif not yes:
            click.echo(
                f"  - {p_copy['display_name']:<15} "
                + click.style("not configured", fg="yellow")
                + f" (set {env_var})"
            )

    click.echo()

    # =========================================================================
    # Step 2: Select provider
    # =========================================================================
    selected_provider = provider
    provider_config: dict[str, Any] = {}

    if not selected_provider and providers:
        if yes:
            # Non-interactive: use first configured provider
            configured = [p for p in providers if p["configured"]]
            if configured:
                selected_provider = configured[0]["name"]
            else:
                click.echo(click.style("Error: No provider API keys found!", fg="red"))
                click.echo("\nSet one of these environment variables:")
                for p in providers:
                    click.echo(f"  export {p['env_var']}=your-api-key")
                sys.exit(1)
        else:
            # Interactive: let user choose from ALL providers
            click.echo(click.style("Step 2: Select Provider", bold=True))
            click.echo()

            # Find default (first configured, or first overall)
            default_idx = 1
            for idx, p in enumerate(providers, 1):
                if p["configured"]:
                    default_idx = idx
                    break

            for idx, p in enumerate(providers, 1):
                status = (
                    click.style("✓", fg="green")
                    if p["configured"]
                    else click.style("-", fg="yellow")
                )
                click.echo(f"  [{idx}] {status} {p['display_name']}")

            click.echo()
            click.echo(click.style("  ✓ = API key configured", fg="green", dim=True))
            click.echo()

            choices = [str(i) for i in range(1, len(providers) + 1)]
            choice = click.prompt(
                "Which provider?",
                default=str(default_idx),
                type=click.Choice(choices),
            )
            selected_p = providers[int(choice) - 1]
            selected_provider = selected_p["name"]

            # =========================================================================
            # Step 3: Configure the selected provider (API key + model selection)
            # =========================================================================
            click.echo()
            click.echo(click.style(f"Step 3: Configure {selected_p['display_name']}", bold=True))

            # Use the full configuration flow from provider_config_utils
            try:
                provider_config = (
                    configure_provider(
                        selected_p["module"],
                        key_manager,
                        existing_config=None,
                        non_interactive=False,
                    )
                    or {}
                )
            except Exception as e:
                click.echo(click.style(f"Configuration error: {e}", fg="red"))
                provider_config = {}

            click.echo()

    # =========================================================================
    # Step 4: Select default bundle
    # =========================================================================
    selected_bundle = bundle
    if not selected_bundle:
        if yes:
            selected_bundle = "foundation"
        else:
            click.echo(click.style("Step 4: Select Default Bundle", bold=True))
            click.echo()
            click.echo("  [1] foundation (recommended)")
            click.echo("  [2] amplifier-dev")
            click.echo()
            bundle_choice = click.prompt(
                "Which bundle?",
                default="1",
                type=click.Choice(["1", "2"]),
            )
            selected_bundle = "foundation" if bundle_choice == "1" else "amplifier-dev"
            click.echo()

    # =========================================================================
    # Save configuration
    # =========================================================================
    settings: dict[str, Any] = {
        "version": "1.0",
        "default_bundle": selected_bundle,
    }

    if selected_provider:
        # Build provider config
        provider_module = f"provider-{selected_provider}"
        settings["default_provider"] = selected_provider
        settings["providers"] = [
            {
                "module": provider_module,
                "config": {
                    "priority": 1,
                    **provider_config,  # Include model and other config
                },
            }
        ]

    # Write settings file
    with open(settings_file, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

    # =========================================================================
    # Done!
    # =========================================================================
    click.echo(click.style("=" * 60, fg="green"))
    click.echo(click.style("  ✓ Setup Complete!", fg="green", bold=True))
    click.echo(click.style("=" * 60, fg="green"))
    click.echo()
    click.echo(f"  Configuration: {settings_file}")
    click.echo(f"  Bundle:        {selected_bundle}")
    if selected_provider:
        click.echo(f"  Provider:      {selected_provider}")
        if provider_config.get("default_model"):
            click.echo(f"  Model:         {provider_config['default_model']}")
    click.echo()
    click.echo("To start the runtime:")
    click.echo("  amplifier-runtime           # Stdio mode (for TUI/IDE)")
    click.echo("  amplifier-runtime --http    # HTTP server mode")
    click.echo()


# =============================================================================
# Config Command
# =============================================================================


@main.command("config")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def show_config(output_json: bool) -> None:
    """Show current configuration.

    Examples:

        amplifier-runtime config
        amplifier-runtime config --json
    """
    import os
    from pathlib import Path

    config = {
        "data_dir": str(Path.home() / ".amplifier-runtime"),
        "default_bundle": os.getenv("AMPLIFIER_BUNDLE", "foundation"),
        "default_provider": None,
        "providers_configured": [],
    }

    # Check configured providers
    provider_vars = [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("azure-openai", "AZURE_OPENAI_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ]

    for name, env_var in provider_vars:
        if os.getenv(env_var):
            config["providers_configured"].append(name)
            if config["default_provider"] is None:
                config["default_provider"] = name

    if output_json:
        click.echo(json.dumps(config, indent=2))
        return

    click.echo("Amplifier Runtime Configuration")
    click.echo("-" * 40)
    click.echo(f"Data directory:     {config['data_dir']}")
    click.echo(f"Default bundle:     {config['default_bundle']}")
    click.echo(f"Default provider:   {config['default_provider'] or 'none'}")
    click.echo(f"Providers ready:    {', '.join(config['providers_configured']) or 'none'}")


if __name__ == "__main__":
    main()
