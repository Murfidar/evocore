"""Shared optimizer configuration export helpers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from evocore.core.serialization import canonical_json_hash, json_safe, stable_json_dumps

if TYPE_CHECKING:
    from evocore.callbacks import Callback

HookType = Literal["termination", "artifact", "environment"]
HookReproducibility = Literal["configured", "partial"]
ReproducibilityStatus = Literal["full", "partial"]


@runtime_checkable
class ConfigurableComponent(Protocol):
    """Protocol for custom algorithm components with stable config identity."""

    def config_signature(self) -> dict[str, Any]:
        """Return a JSON-safe canonical component signature."""

    def validate_compatibility(self, gene_space) -> None:
        """Raise ConfigurationError when incompatible with a gene space."""


@dataclass(frozen=True)
class OptimizerConfig:
    """Canonical optimizer configuration used for comparison and hashing."""

    optimizer_type: str
    parameters: Mapping[str, Any]
    components: Mapping[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Export this config as a deterministic JSON-safe signature."""
        return json_safe(
            {
                "schema_version": self.schema_version,
                "optimizer_type": self.optimizer_type,
                "parameters": self.parameters,
                "components": self.components,
            }
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Export this config as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)

    def hash(self) -> str:
        """Return the stable hash for this config signature."""
        return config_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeHookSignature:
    """Stable metadata for a non-core runtime hook."""

    hook_type: HookType
    identity: str
    config: Mapping[str, Any] = field(default_factory=dict)
    reproducibility: HookReproducibility = "configured"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Export this hook signature as JSON-safe metadata."""
        return json_safe(
            {
                "hook_type": self.hook_type,
                "identity": self.identity,
                "config": self.config,
                "reproducibility": self.reproducibility,
                "notes": self.notes,
            }
        )


def config_hash(signature: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a config signature."""
    return canonical_json_hash(signature)


def stable_object_identity(obj: object) -> str:
    """Return a stable module-qualified identity for a function, class, or object."""
    target = obj if inspect.isfunction(obj) or inspect.isclass(obj) else obj.__class__
    module = getattr(target, "__module__", target.__class__.__module__)
    qualname = getattr(target, "__qualname__", target.__class__.__qualname__)
    return f"{module}.{qualname}"


def callback_hook_signature(callback: Callback) -> RuntimeHookSignature:
    """Return the runtime hook signature for a callback instance."""
    from evocore.callbacks import CheckpointCallback, EarlyStopping, MetricsLogger, ProgressBar

    identity = stable_object_identity(callback)
    if isinstance(callback, EarlyStopping):
        return RuntimeHookSignature(
            hook_type="termination",
            identity=identity,
            config={"patience": callback.patience, "min_delta": callback.min_delta},
        )
    if isinstance(callback, CheckpointCallback):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={"path": callback.path, "every": callback.every},
        )
    if isinstance(callback, MetricsLogger):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={"path": callback.path},
        )
    if isinstance(callback, ProgressBar):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={},
        )
    return RuntimeHookSignature(
        hook_type="termination",
        identity=identity,
        config={},
        reproducibility="partial",
        notes=(
            f"{callback.__class__.__name__} may affect termination or side effects without a stable hook signature.",
        ),
    )


def callback_hook_signatures(callbacks: list[Callback]) -> tuple[RuntimeHookSignature, ...]:
    """Return hook signatures for callback instances."""
    return tuple(callback_hook_signature(callback) for callback in callbacks)


def reproducibility_from_hooks(
    hooks: tuple[RuntimeHookSignature, ...],
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes derived from hook signatures."""
    notes: list[str] = []
    for hook in hooks:
        if hook.reproducibility == "partial":
            notes.extend(hook.notes)
    if notes:
        return "partial", tuple(notes)
    return "full", ()


__all__ = [
    "ConfigurableComponent",
    "HookReproducibility",
    "HookType",
    "OptimizerConfig",
    "ReproducibilityStatus",
    "RuntimeHookSignature",
    "callback_hook_signature",
    "callback_hook_signatures",
    "config_hash",
    "reproducibility_from_hooks",
    "stable_object_identity",
]
