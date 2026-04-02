"""Registry for deterministic tool plugin registration and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tools.base import ToolPlugin


class ToolRegistrationError(ValueError):
    """Raised when plugin registration fails."""


class ToolResolutionError(KeyError):
    """Raised when plugin lookup fails."""


@dataclass
class ToolRegistry:
    """Deterministic plugin registry with uniqueness guarantees."""

    _plugins: dict[str, ToolPlugin] = field(default_factory=dict)

    def register(self, plugin: ToolPlugin) -> None:
        if not plugin.name.strip():
            raise ToolRegistrationError("Plugin name cannot be empty")
        if plugin.name in self._plugins:
            raise ToolRegistrationError(f"Duplicate plugin name: {plugin.name}")
        self._plugins[plugin.name] = plugin

    def register_many(self, plugins: list[ToolPlugin]) -> None:
        for plugin in plugins:
            self.register(plugin)

    def resolve(self, name: str) -> ToolPlugin:
        plugin = self._plugins.get(name)
        if plugin is None:
            raise ToolResolutionError(name)
        return plugin

    def allowlist(self) -> set[str]:
        return set(self._plugins.keys())

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())
