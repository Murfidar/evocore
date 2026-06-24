"""Constraint and repair records for projection-aware search spaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


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


__all__ = [
    "ConstraintViolation",
    "RepairRecord",
]
