"""Gene space definitions for evocore optimizers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from evocore.exceptions import ConfigurationError

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
            if self.low >= self.high:
                raise ConfigurationError(f"GeneDef('{self.name}') requires low < high.")
            if self.kind == "int" and (
                not isinstance(self.low, int) or not isinstance(self.high, int)
            ):
                raise ConfigurationError(
                    f"GeneDef('{self.name}') with kind='int' requires integer bounds."
                )

        if self.sigma is not None and not (0.0 < self.sigma <= 1.0):
            raise ConfigurationError("GeneDef sigma must be in (0, 1].")


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
        if low >= high:
            raise ConfigurationError("GeneSpace.uniform requires low < high.")
        return cls(
            [GeneDef(f"gene_{index}", "float", float(low), float(high)) for index in range(length)],
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

    def params_for(self, genes: Sequence[float | int | bool]) -> dict[str, float | int | bool] | None:
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
