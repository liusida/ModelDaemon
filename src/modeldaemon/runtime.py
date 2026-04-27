"""API for user scripts executed by the daemon."""

from __future__ import annotations

from typing import Any, Mapping

_registry: Mapping[str, Any] | None = None


def set_registry(models: Mapping[str, Any] | None) -> None:
    global _registry
    _registry = models


def get_model(name: str = "default") -> Any:
    """Return a model object loaded by the daemon.

    Only works while your script is running inside ``modeldaemon run``.
    """
    if _registry is None:
        msg = (
            "ModelDaemon registry is not set. "
            "Run your script with: modeldaemon run your_script.py"
        )
        raise RuntimeError(msg)
    if name not in _registry:
        known = ", ".join(sorted(_registry)) or "(empty)"
        raise KeyError(f"Unknown model {name!r}. Known: {known}")
    return _registry[name]


def list_models() -> list[str]:
    if _registry is None:
        return []
    return sorted(_registry)
