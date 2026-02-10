"""Hook extensibility system for Amplifier Runtime."""

from .base import HookRegistry, InputHook, OutputHook, ServerHook

__all__ = [
    "ServerHook",
    "InputHook",
    "OutputHook",
    "HookRegistry",
]
