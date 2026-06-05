"""Operator encoding and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from evocore.core.errors import ConfigurationError
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    normalize_crossover_operator,
    normalize_mutation_operator,
    resolve_operator_domain,
    validate_operator_set,
)
from evocore.search_space.genes import Gene, GeneSpace
from evocore.search_space.solutions import GeneValue, Solution


def _validate_gene_count(gene_space: GeneSpace, values: Sequence[object], *, label: str) -> None:
    if len(values) != gene_space.length:
        raise ConfigurationError(f"{label} expected {gene_space.length} genes, got {len(values)}.")


def repair_gene_value(value: object, gene: Gene) -> GeneValue:
    """Repair one decoded or encoded value according to a gene definition."""
    if gene.kind == "bool":
        if type(value) is bool:
            return value
        if isinstance(value, int | float) and type(value) is not bool:
            return float(value) >= 0.5
        raise ConfigurationError(
            f"Gene {gene.name!r} expects bool-compatible value, got {type(value).__name__}."
        )

    if not isinstance(value, int | float) or type(value) is bool:
        raise ConfigurationError(
            f"Gene {gene.name!r} expects numeric-compatible value, got {type(value).__name__}."
        )

    low = float(gene.low)
    high = float(gene.high)
    if gene.kind == "int":
        rounded = float(round(float(value)))
        return int(min(max(rounded, low), high))
    return float(min(max(float(value), low), high))


def repair_gene_values(gene_space: GeneSpace, values: Sequence[object]) -> list[GeneValue]:
    """Repair a full gene vector and validate it against the gene space."""
    _validate_gene_count(gene_space, values, label="Gene repair")
    repaired = [
        repair_gene_value(value, gene)
        for value, gene in zip(values, gene_space.genes, strict=False)
    ]
    gene_space.validate_genes(repaired)
    return repaired


def encode_gene_values(gene_space: GeneSpace, values: Sequence[GeneValue]) -> list[float]:
    """Encode validated Python gene values into Rust/operator floats."""
    gene_space.validate_genes(values)
    encoded: list[float] = []
    for value, gene in zip(values, gene_space.genes, strict=False):
        if gene.kind == "bool":
            encoded.append(1.0 if bool(value) else 0.0)
        elif gene.kind == "int":
            encoded.append(float(int(value)))
        else:
            encoded.append(float(value))
    return encoded


def decode_gene_values(gene_space: GeneSpace, encoded: Sequence[float]) -> list[GeneValue]:
    """Decode and repair Rust/operator floats into Python gene values."""
    return repair_gene_values(gene_space, encoded)


class OperatorCodec:
    """Validate operators and translate values across the PyO3 boundary."""

    def __init__(
        self,
        gene_space: GeneSpace,
        crossover: str | CrossoverOperator,
        mutation: str | MutationOperator,
    ) -> None:
        self.gene_space = gene_space
        self.crossover_operator = resolve_operator_domain(
            normalize_crossover_operator(
                crossover,
                probability=0.9,
                eta=2.0,
                alpha=0.5,
            ),
            gene_space,
        )
        self.mutation_operator = resolve_operator_domain(
            normalize_mutation_operator(
                mutation,
                probability=0.1,
                individual_probability=1.0,
                sigma=0.2,
            ),
            gene_space,
        )
        self.crossover = self.crossover_operator.name
        self.mutation = self.mutation_operator.name
        self._validate()

    def _validate(self) -> None:

        validate_operator_set(
            gene_space=self.gene_space,
            crossover=self.crossover_operator,
            mutation=self.mutation_operator,
            selection=SelectionOperator.tournament(),
            bounds_policy=BoundsPolicy.clamp(),
        )

    @property
    def gene_kinds(self) -> list[str]:
        """Return the Rust-facing gene kind strings."""
        return self.gene_space.kinds

    @property
    def gene_bounds(self) -> list[tuple[float, float]]:
        """Return the Rust-facing floating-point bounds."""
        return self.gene_space.rust_bounds

    def encode_values(self, values: Sequence[GeneValue]) -> list[float]:
        """Encode Python values into the float vector used by Rust."""
        return encode_gene_values(self.gene_space, values)

    def decode_values(self, genes_f64: Sequence[float]) -> list[GeneValue]:
        """Decode Rust float vectors back into Python gene values."""
        return decode_gene_values(self.gene_space, genes_f64)

    def encode_population(self, solutions: Sequence[Solution]) -> list[list[float]]:
        """Encode a SolutionSet of solutions for Rust calls."""
        return [self.encode_values(solution.values) for solution in solutions]

    def decode_solution(
        self,
        genes_f64: Sequence[float],
        *,
        score: float | None = None,
        score_valid: bool = False,
        metadata: dict | None = None,
    ) -> Solution:
        """Decode one Rust-side genome into a `Solution`."""
        values = self.decode_values(genes_f64)
        solution_metadata = dict(metadata or {})
        params = self.gene_space.params_for(values)
        if params is not None:
            solution_metadata["params"] = params
        return Solution(
            list(values),
            score=score,
            score_valid=score_valid,
            metadata=solution_metadata,
        )

    def decode_population(self, population_f64: Sequence[Sequence[float]]) -> list[Solution]:
        """Decode a Rust-side SolutionSet into Python solutions."""
        return [self.decode_solution(genes_f64) for genes_f64 in population_f64]

    def sigma_abs_list(self, global_sigma_fraction: float) -> list[float]:
        """Return per-gene absolute mutation sigmas for Rust operators."""
        if not (0.0 <= global_sigma_fraction <= 1.0):
            raise ConfigurationError("mutation_sigma must be in [0, 1].")

        sigmas: list[float] = []
        for gene in self.gene_space.genes:
            if gene.kind == "bool":
                sigmas.append(0.0)
                continue

            low = float(gene.low)
            high = float(gene.high)
            fraction = gene.sigma if gene.sigma is not None else global_sigma_fraction
            sigmas.append(float(fraction) * (high - low))
        return sigmas
