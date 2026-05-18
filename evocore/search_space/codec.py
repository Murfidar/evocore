"""Operator encoding and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from evocore.core.errors import ConfigurationError
from evocore.search_space.genes import GeneSpace
from evocore.search_space.solutions import Solution

NUMERIC_CROSSOVERS = {"sbx", "blx", "uniform"}
BINARY_CROSSOVERS = {"one_point", "two_point", "uniform"}
NUMERIC_MUTATIONS = {"gaussian", "uniform"}
BINARY_MUTATIONS = {"bit_flip"}


class OperatorCodec:
    """Validate operators and translate values across the PyO3 boundary."""

    def __init__(self, gene_space: GeneSpace, crossover: str, mutation: str) -> None:
        self.gene_space = gene_space
        self.crossover = crossover
        self.mutation = mutation
        self._validate()

    def _validate(self) -> None:
        kinds = set(self.gene_space.kinds)
        if "bool" in kinds and len(kinds) > 1:
            raise ConfigurationError(
                "GeneSpace contains bool genes alongside float/int genes. "
                "Use a binary-only space or encode booleans as int genes with low=0, high=1."
            )

        if kinds == {"bool"}:
            if self.crossover not in BINARY_CROSSOVERS:
                raise ConfigurationError(
                    "binary GeneSpace requires crossover='one_point', 'two_point', or 'uniform'."
                )
            if self.mutation not in BINARY_MUTATIONS:
                raise ConfigurationError("binary GeneSpace requires mutation='bit_flip'.")
        else:
            if self.crossover not in NUMERIC_CROSSOVERS:
                raise ConfigurationError(
                    "float/int GeneSpace requires crossover='sbx', 'blx', or 'uniform'."
                )
            if self.mutation not in NUMERIC_MUTATIONS:
                raise ConfigurationError(
                    "float/int GeneSpace requires mutation='gaussian' or 'uniform'."
                )

    @property
    def gene_kinds(self) -> list[str]:
        """Return the Rust-facing gene kind strings."""
        return self.gene_space.kinds

    @property
    def gene_bounds(self) -> list[tuple[float, float]]:
        """Return the Rust-facing floating-point bounds."""
        return self.gene_space.rust_bounds

    def encode_values(self, values: Sequence[float | int | bool]) -> list[float]:
        """Encode Python values into the float vector used by Rust."""
        self.gene_space.validate_genes(values)

        encoded: list[float] = []
        for value, gene in zip(values, self.gene_space.genes):
            if gene.kind == "bool":
                encoded.append(1.0 if bool(value) else 0.0)
            elif gene.kind == "int":
                encoded.append(float(int(value)))
            else:
                encoded.append(float(value))
        return encoded

    def decode_values(self, genes_f64: Sequence[float]) -> list[float | int | bool]:
        """Decode Rust float vectors back into Python gene values."""
        if len(genes_f64) != self.gene_space.length:
            raise ConfigurationError(
                f"Expected {self.gene_space.length} encoded genes, got {len(genes_f64)}."
            )

        decoded: list[float | int | bool] = []
        for value, gene in zip(genes_f64, self.gene_space.genes):
            if gene.kind == "bool":
                decoded.append(bool(value >= 0.5))
            elif gene.kind == "int":
                decoded.append(int(round(value)))
            else:
                decoded.append(float(value))
        return decoded

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
