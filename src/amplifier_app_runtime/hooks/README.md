# Hook Extensibility System

The hook system allows bundles and applications to extend the Amplifier Runtime's capabilities without modifying core code.

## Overview

Hooks provide two primary extension points:

1. **InputHooks** - Inject external inputs into sessions (notifications, webhooks, scheduled events)
2. **OutputHooks** - Process session outputs to external destinations (push notifications, webhooks, email)

## Hook Types

### ServerHook (Base Class)

All hooks inherit from `ServerHook`:

```python
from amplifier_app_runtime.hooks import ServerHook

class MyHook(ServerHook):
    name = "my_hook"
    
    async def start(self, session_manager):
        """Called when server starts."""
        pass
    
    async def stop(self):
        """Called when server stops."""
        pass
```

### InputHook

Injects external inputs into sessions:

```python
from amplifier_app_runtime.hooks import InputHook

class NotificationHook(InputHook):
    name = "notifications"
    
    async def start(self, session_manager):
        self.session_manager = session_manager
        # Set up notification listener
    
    async def poll(self):
        """Return list of items to inject."""
        return [
            {
                "content": "New email from boss",
                "session_id": "main",  # Or None for default
                "role": "user",
            }
        ]
    
    async def stop(self):
        # Cleanup resources
        pass
```

### OutputHook

Processes session outputs to external destinations:

```python
from amplifier_app_runtime.hooks import OutputHook

class PushNotificationHook(OutputHook):
    name = "push_notifications"
    
    async def start(self, session_manager):
        # Set up push notification service
        pass
    
    def should_handle(self, event, data):
        """Filter which events to handle."""
        return event == "notification"
    
    async def send(self, event, data):
        """Send to external service."""
        # Send push notification
        return True  # Success
    
    async def stop(self):
        pass
```

## Using Hooks

### Registration

Register hooks with the `HookRegistry`:

```python
from amplifier_app_runtime.hooks import HookRegistry
from amplifier_app_runtime.session import SessionManager

# Create registry
hook_registry = HookRegistry()

# Register hooks
hook_registry.register(NotificationHook())
hook_registry.register(PushNotificationHook())

# Create session manager with hooks
manager = SessionManager(hook_registry=hook_registry)

# Start hooks
await manager.start_hooks()
```

### Polling Input Hooks

Input hooks are polled for new items:

```python
# Poll all input hooks
inputs = await manager.hooks.poll_inputs()

# Inject into sessions
for item in inputs:
    await manager.inject_context(
        session_id=item["session_id"] or default_session_id,
        content=item["content"],
        role=item.get("role", "user"),
    )
```

### Dispatching to Output Hooks

Output hooks receive events from sessions:

```python
# Dispatch event to all interested hooks
results = await manager.hooks.dispatch_output(
    event="notification",
    data={"message": "Task completed", "priority": "high"},
)

# results = {"push_notifications": True, "email_sender": True}
```

## Example: Calendar Integration

```python
from amplifier_app_runtime.hooks import InputHook
import asyncio

class CalendarHook(InputHook):
    name = "calendar_events"
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.session_manager = None
        self._task = None
    
    async def start(self, session_manager):
        self.session_manager = session_manager
        # Start background polling task
        self._task = asyncio.create_task(self._background_poll())
    
    async def poll(self):
        """Check for upcoming events."""
        # Query calendar API
        events = await self._fetch_upcoming_events()
        
        return [
            {
                "content": f"[CALENDAR] {event['title']} in {event['minutes']} minutes",
                "session_id": None,  # Default session
                "role": "user",
            }
            for event in events
        ]
    
    async def _background_poll(self):
        """Background task that polls periodically."""
        while True:
            try:
                items = await self.poll()
                for item in items:
                    await self.session_manager.inject_context(
                        session_id=item["session_id"] or "main",
                        content=item["content"],
                        role=item.get("role", "user"),
                    )
            except Exception as e:
                logger.error(f"Calendar poll error: {e}")
            
            await asyncio.sleep(300)  # Poll every 5 minutes
    
    async def stop(self):
        if self._task:
            self._task.cancel()
    
    async def _fetch_upcoming_events(self):
        # Implementation
        return []
```

## Example: Slack Integration

```python
from amplifier_app_runtime.hooks import OutputHook

class SlackHook(OutputHook):
    name = "slack_output"
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    async def start(self, session_manager):
        self.session_manager = session_manager
    
    def should_handle(self, event, data):
        """Only handle notifications marked for Slack."""
        return event == "notification" and data.get("channel") == "slack"
    
    async def send(self, event, data):
        """Post to Slack."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                self.webhook_url,
                json={"text": data.get("message", "")},
            )
            return response.status == 200
    
    async def stop(self):
        pass
```

## Hook Lifecycle

```
Server Start
    ↓
HookRegistry.start_all(session_manager)
    ↓
Each hook's start() called
    ↓
Hooks are active (polling/dispatching)
    ↓
Server Shutdown
    ↓
HookRegistry.stop_all()
    ↓
Each hook's stop() called
```

## Error Handling

Hooks are fault-tolerant:

- If a hook fails to start, other hooks continue
- If a hook fails during poll(), it's skipped for that cycle
- If a hook fails during send(), it's marked as failed but other hooks continue

## Best Practices

### 1. Keep poll() Fast

Input hooks are polled frequently. Keep operations lightweight:

```python
async def poll(self):
    # ❌ Bad: Expensive API call every poll
    events = await expensive_api_call()
    
    # ✅ Good: Return cached results, update in background
    return self._cached_events
```

### 2. Use Background Tasks for Continuous Monitoring

```python
async def start(self, session_manager):
    self._task = asyncio.create_task(self._monitor())

async def _monitor(self):
    while True:
        # Monitor and cache results
        await asyncio.sleep(interval)

async def poll(self):
    return self._cached_items  # Fast
```

### 3. Filter Events Efficiently

```python
def should_handle(self, event, data):
    # Quick filtering before expensive send()
    if event != "notification":
        return False
    if data.get("priority") != "high":
        return False
    return True
```

### 4. Handle Errors Gracefully

```python
async def send(self, event, data):
    try:
        # Attempt send
        return True
    except Exception as e:
        logger.error(f"Failed to send: {e}")
        return False  # Don't raise - let other hooks continue
```

## Integration with Context Injection

Hooks work seamlessly with the context injection API:

```python
# Hook polls for inputs
inputs = await registry.poll_inputs()

# Inject into sessions
for item in inputs:
    await session_manager.inject_context(
        session_id=item["session_id"],
        content=item["content"],
        role=item.get("role", "user"),
    )
```

## Complete Example: Always-On Assistant

```python
from amplifier_app_runtime.hooks import HookRegistry, InputHook, OutputHook
from amplifier_app_runtime.session import SessionManager, SessionConfig

# Define hooks
class WindowsNotificationHook(InputHook):
    name = "windows_notifications"
    # ... implementation ...

class MobileNotificationHook(InputHook):
    name = "mobile_notifications"
    # ... implementation ...

class PushHook(OutputHook):
    name = "push_sender"
    # ... implementation ...

# Create registry and register hooks
registry = HookRegistry()
registry.register(WindowsNotificationHook())
registry.register(MobileNotificationHook())
registry.register(PushHook())

# Create session manager with hooks
manager = SessionManager(hook_registry=registry)
await manager.start_hooks()

# Create a persistent session
session = await manager.create(
    config=SessionConfig(bundle="foundation"),
    session_id="personal-hub",
    auto_initialize=True,
)

# Polling loop (in server)
while True:
    # Poll input hooks
    inputs = await registry.poll_inputs()
    
    # Inject into session
    for item in inputs:
        await manager.inject_context(
            session_id="personal-hub",
            content=item["content"],
            role="user",
        )
    
    await asyncio.sleep(5)  # Poll every 5 seconds
```

## See Also

- `base.py` - Hook base classes and registry implementation
- `../session.py` - Context injection API
- `../../examples/` - Complete hook examples
