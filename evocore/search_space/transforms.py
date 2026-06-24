"""Portable parameter transforms for projection-aware workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.search_space.solutions import GeneValue


@runtime_checkable
class ParameterTransform(Protocol):
    """Map between optimizer-native values and domain parameter values."""

    checkpointable: bool

    def decode(self, value: GeneValue) -> object:
        """Decode an optimizer value into a domain value."""
        raise NotImplementedError

    def encode(self, value: object) -> GeneValue:
        """Encode a domain value into an optimizer value."""
        raise NotImplementedError

    def signature(self) -> dict[str, object]:
        """Return a stable JSON-safe transform signature."""
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTransform:
    """Pass values through without changing representation."""

    checkpointable: bool = True

    def decode(self, value: GeneValue) -> object:
        """Return the optimizer value unchanged."""
        return value

    def encode(self, value: object) -> GeneValue:
        """Return the domain value unchanged when it is a valid gene value."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return float(value)
        raise ConfigurationError("IdentityTransform requires a finite bool, int, or float.")

    def signature(self) -> dict[str, object]:
        """Return the stable identity transform signature."""
        return {"type": "identity", "version": 1}


@dataclass(frozen=True)
class BinaryThresholdTransform:
    """Decode numeric optimizer values into booleans by threshold."""

    threshold: float = 0.5
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.threshold)):
            raise ConfigurationError("BinaryThresholdTransform threshold must be finite.")

    def decode(self, value: GeneValue) -> bool:
        """Return whether the numeric value meets or exceeds the threshold."""
        return float(value) >= float(self.threshold)

    def encode(self, value: object) -> float:
        """Encode truthy values as 1.0 and falsey values as 0.0."""
        return 1.0 if bool(value) else 0.0

    def signature(self) -> dict[str, object]:
        """Return the stable binary-threshold transform signature."""
        return {"type": "binary_threshold", "version": 1, "threshold": float(self.threshold)}


@dataclass(frozen=True)
class ExponentialIntegerTransform:
    """Decode logarithmic optimizer coordinates into positive integers."""

    base: float
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.base)) or float(self.base) <= 1.0:
            raise ConfigurationError("ExponentialIntegerTransform base must be finite and > 1.")

    def decode(self, value: GeneValue) -> int:
        """Decode a logarithmic coordinate into a rounded positive integer."""
        return max(0, int(round(float(self.base) ** float(value))))

    def encode(self, value: object) -> float:
        """Encode a positive integer into logarithmic optimizer coordinates."""
        numeric = int(value)
        if numeric <= 0:
            raise ConfigurationError("ExponentialIntegerTransform encode requires value > 0.")
        return math.log(float(numeric), float(self.base))

    def signature(self) -> dict[str, object]:
        """Return the stable exponential-integer transform signature."""
        return {"type": "exponential_integer", "version": 1, "base": float(self.base)}


@dataclass(frozen=True)
class OutputNameTransform:
    """Carry a stable output-name annotation while leaving values unchanged."""

    output_name: str
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not self.output_name:
            raise ConfigurationError("OutputNameTransform output_name must be non-empty.")

    def decode(self, value: GeneValue) -> object:
        """Return the value unchanged while preserving output-name metadata."""
        return value

    def encode(self, value: object) -> GeneValue:
        """Encode the value using identity semantics."""
        return IdentityTransform().encode(value)

    def signature(self) -> dict[str, object]:
        """Return the stable output-name transform signature."""
        return {"type": "output_name", "version": 1, "output_name": self.output_name}


__all__ = [
    "BinaryThresholdTransform",
    "ExponentialIntegerTransform",
    "IdentityTransform",
    "OutputNameTransform",
    "ParameterTransform",
]
