"""Constraint and repair records for projection-aware search spaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ConstraintViolation:
    """Describe one deterministic projection or domain validation violation."""

    code: str
    message: str
    names: tuple[str, ...] = ()
    hook_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairRecord:
    """Describe one deterministic projection repair."""

    name: str
    previous: object
    repaired: object
    reason: str
    hook_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ParameterRepair(Protocol):
    """Protocol for deterministic domain-parameter repair hooks."""

    checkpointable: bool

    def repair(
        self,
        parameters: Mapping[str, object],
    ) -> tuple[Mapping[str, object], Sequence[RepairRecord]]:
        """Return repaired parameters and ordered repair records."""
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        """Return a JSON-safe repair behavior signature."""
        raise NotImplementedError


@runtime_checkable
class ParameterValidator(Protocol):
    """Protocol for deterministic domain-parameter validation hooks."""

    checkpointable: bool

    def validate(self, parameters: Mapping[str, object]) -> Sequence[ConstraintViolation]:
        """Return ordered constraint violations for a parameter mapping."""
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        """Return a JSON-safe validation behavior signature."""
        raise NotImplementedError


__all__ = [
    "ConstraintViolation",
    "ParameterRepair",
    "ParameterValidator",
    "RepairRecord",
]
