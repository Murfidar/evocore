"""Gene space definitions for evocore optimizers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from evocore.exceptions import ConfigurationError
from evocore.exporting import canonical_json_hash, stable_json_dumps

GeneKind = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class GeneDef:
    """Describe one named optimization gene.

    Args:
        name: Unique gene name used for parameter dictionaries.
        kind: Gene kind: `"float"`, `"int"`, or `"bool"`.
        low: Inclusive lower bound for float and integer genes.
        high: Inclusive upper bound for float and integer genes.
        sigma: Optional mutation sigma fraction in `(0, 1]`.

    Raises:
        ConfigurationError: If the name, kind, bounds, or sigma are invalid.
    """

    name: str
    kind: GeneKind
    low: float | int | None = None
    high: float | int | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("GeneDef name must be a non-empty string.")
        if self.kind not in ("float", "int", "bool"):
            raise ConfigurationError("GeneDef kind must be 'float', 'int', or 'bool'.")

        if self.kind == "bool":
            if self.low is not None or self.high is not None:
                raise ConfigurationError(
                    "bool genes do not use bounds; pass GeneDef(name, 'bool')."
                )
        else:
            if self.low is None or self.high is None:
                raise ConfigurationError(f"bounds required for {self.kind} gene '{self.name}'.")
            if not math.isfinite(float(self.low)) or not math.isfinite(float(self.high)):
                raise ConfigurationError(f"GeneDef('{self.name}') requires finite numeric bounds.")
            if self.low > self.high:
                raise ConfigurationError(f"GeneDef('{self.name}') requires low <= high.")
            if self.kind == "int" and (
                not isinstance(self.low, int) or not isinstance(self.high, int)
            ):
                raise ConfigurationError(
                    f"GeneDef('{self.name}') with kind='int' requires integer bounds."
                )

        if self.sigma is not None and not (0.0 < self.sigma <= 1.0):
            raise ConfigurationError("GeneDef sigma must be in (0, 1].")

    @property
    def is_fixed(self) -> bool:
        """Return whether this gene is a fixed numeric value."""
        return self.kind in ("float", "int") and self.low == self.high


class GeneSpace:
    """Collect gene definitions used by optimization engines."""

    def __init__(self, genes: Sequence[GeneDef], *, has_names: bool = True) -> None:
        if not genes:
            raise ConfigurationError("GeneSpace requires at least one GeneDef.")

        self._genes = tuple(genes)
        self._has_names = bool(has_names)

        if self._has_names:
            seen: set[str] = set()
            for gene in self._genes:
                if gene.name in seen:
                    raise ConfigurationError(f"Duplicate gene name: {gene.name!r}.")
                seen.add(gene.name)

    @classmethod
    def uniform(cls, low: float, high: float, length: int) -> GeneSpace:
        """Create an unnamed float gene space with shared bounds.

        Args:
            low: Lower bound for each gene.
            high: Upper bound for each gene.
            length: Number of float genes.

        Returns:
            A `GeneSpace` with `length` float genes.

        Raises:
            ConfigurationError: If `length <= 0` or `low >= high`.
        """
        if length <= 0:
            raise ConfigurationError("GeneSpace.uniform length must be positive.")
        if not math.isfinite(float(low)) or not math.isfinite(float(high)):
            raise ConfigurationError("GeneSpace.uniform requires finite numeric bounds.")
        if low >= high:
            raise ConfigurationError("GeneSpace.uniform requires low < high.")
        return cls(
            [
                GeneDef(f"gene_{index}", "float", float(low), float(high))
                for index in range(length)
            ],
            has_names=False,
        )

    @property
    def genes(self) -> tuple[GeneDef, ...]:
        """Return the ordered gene definitions."""
        return self._genes

    @property
    def length(self) -> int:
        """Return the number of genes in the space."""
        return len(self._genes)

    @property
    def names(self) -> list[str]:
        """Return the gene names in definition order."""
        return [gene.name for gene in self._genes]

    @property
    def kinds(self) -> list[str]:
        """Return the gene kinds in definition order."""
        return [gene.kind for gene in self._genes]

    @property
    def fixed_indices(self) -> list[int]:
        """Return indices of fixed numeric genes."""
        return [index for index, gene in enumerate(self._genes) if gene.is_fixed]

    @property
    def variable_indices(self) -> list[int]:
        """Return indices of genes that participate in variation."""
        return [index for index, gene in enumerate(self._genes) if not gene.is_fixed]

    @property
    def fixed_count(self) -> int:
        """Return the number of fixed numeric genes."""
        return len(self.fixed_indices)

    @property
    def variable_count(self) -> int:
        """Return the number of variable genes."""
        return len(self.variable_indices)

    @property
    def bounds(self) -> list[tuple[float | int, float | int] | None]:
        """Return Python-facing bounds for each gene."""
        return [None if gene.kind == "bool" else (gene.low, gene.high) for gene in self._genes]

    @property
    def rust_bounds(self) -> list[tuple[float, float]]:
        """Return bounds encoded for the Rust extension boundary."""
        bounds: list[tuple[float, float]] = []
        for gene in self._genes:
            if gene.kind == "bool":
                bounds.append((0.0, 1.0))
            else:
                bounds.append((float(gene.low), float(gene.high)))
        return bounds

    @property
    def has_names(self) -> bool:
        """Return whether the gene space exposes named parameters."""
        return self._has_names

    def signature(self) -> dict[str, Any]:
        """Return the stable canonical signature for this gene space."""
        return {
            "schema_version": 1,
            "genes": [
                {
                    "name": gene.name,
                    "kind": gene.kind,
                    "low": gene.low,
                    "high": gene.high,
                    "sigma": gene.sigma,
                    "is_fixed": gene.is_fixed,
                }
                for gene in self._genes
            ],
            "has_names": self._has_names,
            "length": self.length,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export this gene space as its stable canonical signature."""
        return self.signature()

    def hash(self) -> str:
        """Return a stable SHA-256 hash for this gene-space signature."""
        return canonical_json_hash(self.signature())

    def to_json(self, *, indent: int | None = None) -> str:
        """Export this gene space as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)

    def validate_genes(self, values: Sequence[float | int | bool]) -> None:
        """Validate decoded Python gene values against this gene space."""
        if len(values) != self.length:
            raise ConfigurationError(f"GeneSpace expected {self.length} genes, got {len(values)}.")

        for index, (value, gene) in enumerate(zip(values, self._genes, strict=False)):
            label = f"Gene {gene.name!r} at index {index}"

            if gene.kind == "bool":
                if type(value) is not bool:
                    raise ConfigurationError(
                        f"{label} expects bool, got {type(value).__name__}."
                    )
                continue

            if gene.kind == "int":
                if type(value) is not int:
                    raise ConfigurationError(f"{label} expects int, got {type(value).__name__}.")
                if value < gene.low or value > gene.high:
                    raise ConfigurationError(
                        f"{label} must be within [{gene.low}, {gene.high}], got {value}."
                    )
                continue

            if type(value) not in (int, float):
                raise ConfigurationError(f"{label} expects float, got {type(value).__name__}.")

            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ConfigurationError(f"{label} must be finite, got {value}.")
            if numeric_value < float(gene.low) or numeric_value > float(gene.high):
                raise ConfigurationError(
                    f"{label} must be within [{gene.low}, {gene.high}], got {value}."
                )

    def params_for(
        self, genes: Sequence[float | int | bool]
    ) -> dict[str, float | int | bool] | None:
        """Map genes to parameter names when the space has names.

        Args:
            genes: Decoded Python gene values.

        Returns:
            A name-to-value dictionary, or `None` for unnamed spaces.

        Raises:
            ConfigurationError: If the number of values does not match the gene space length.
        """
        if len(genes) != self.length:
            raise ConfigurationError(
                f"Expected {self.length} genes for params mapping, got {len(genes)}."
            )
        if not self._has_names:
            return None
        return dict(zip(self.names, genes))
