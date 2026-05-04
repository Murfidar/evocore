from __future__ import annotations

from collections.abc import Sequence

from evocore import _core
from evocore.callbacks import Callback
from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneSpace
from evocore.individual import Individual
from evocore.operators import OperatorSet


class CMAESEngine:
    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        initial_mean: list[float] | None = None,
        initial_sigma: float = 0.3,
        generations: int = 300,
        parallel: str = "none",
        n_workers: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        seed: int = 0,
        track_diversity: bool = False,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for CMAESEngine. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if "bool" in gene_space.kinds:
            raise ConfigurationError(
                "CMAESEngine does not support bool genes; use float/int genes only."
            )
        if parallel == "process":
            raise ConfigurationError(
                "CMAESEngine does not support parallel='process'.\n"
                "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable.\n"
                "  Fix: use parallel='thread' if your fitness function releases the GIL, or parallel='none'.\n"
                "  Note: parallel='process' is supported by GAEngine, not CMAESEngine."
            )
        if parallel not in ("none", "thread"):
            raise ConfigurationError("CMAESEngine parallel must be 'none' or 'thread'.")
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if generations < 0:
            raise ConfigurationError("generations must be >= 0.")
        if not (initial_sigma > 0.0):
            raise ConfigurationError("initial_sigma must be > 0.")
        if initial_mean is not None and len(initial_mean) != gene_space.length:
            raise ConfigurationError("initial_mean length must match gene_space.length.")

        self.gene_space = gene_space
        self.population_size = population_size
        self.initial_mean = initial_mean
        self.initial_sigma = initial_sigma
        self.generations = generations
        self.parallel = parallel
        self.n_workers = n_workers
        self.callbacks = list(callbacks or [])
        self.seed = int(seed)
        self.track_diversity = track_diversity
        self.operators = OperatorSet(gene_space, "sbx", "gaussian")
        self._fitness_warning_emitted = False

    @property
    def _bounds_list(self) -> list[tuple[float, float]]:
        return self.operators.gene_bounds

    def _initial_mean_encoded(self) -> list[float]:
        if self.initial_mean is not None:
            return [float(value) for value in self.initial_mean]
        return _core.init_population(self._bounds_list, self.operators.gene_kinds, 1, self.seed)[0]

    def _sigma_abs(self) -> float:
        spans = [high - low for low, high in self._bounds_list]
        return self.initial_sigma * (sum(spans) / len(spans))

    def _apply_bounds_and_round(self, genes_f64: Sequence[float]) -> list[float]:
        rounded: list[float] = []
        for value, gene, (low, high) in zip(genes_f64, self.gene_space.genes, self._bounds_list):
            clamped = max(low, min(high, float(value)))
            if gene.kind == "int":
                clamped = float(round(clamped))
                clamped = max(low, min(high, clamped))
            rounded.append(clamped)
        return rounded

    def _decode_individual(
        self,
        genes_f64: Sequence[float],
        fitness: float | None = None,
    ) -> Individual:
        return self.operators.decode_individual(
            genes_f64,
            fitness=fitness,
            fitness_valid=fitness is not None,
        )
