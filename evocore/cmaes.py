"""CMA-ES engine backed by Rust covariance state."""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.evaluation import Candidate, EvaluationRecord, OptimizationTelemetry
from evocore.exceptions import ConfigurationError, FitnessError, FitnessWarning
from evocore.ga import EngineStateSummary, RunResult
from evocore.gene_space import GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ThreadParallel
from evocore.stats import Logbook, LogEntry

logger = logging.getLogger(__name__)


class CMAESEngine:
    """Run covariance matrix adaptation evolution strategy optimization.

    Args:
        gene_space: Float or integer gene definitions.
        population_size: Number of sampled candidates per generation.
        initial_mean: Optional encoded initial mean.
        initial_sigma: Initial sigma fraction relative to gene bounds.
        generations: Maximum number of generations to run.
        parallel: Evaluation mode: `"none"` or `"thread"`.
        n_workers: Worker count for thread mode.
        callbacks: Optional callbacks invoked during the run.
        seed: Master seed for deterministic sampling.
        track_diversity: Whether to record per-gene diversity.

    Raises:
        ConfigurationError: If configuration is invalid or process parallelism is requested.
    """

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
        if gene_space.fixed_count:
            raise ConfigurationError(
                "CMAESEngine does not support fixed numeric genes yet. "
                "Use GAEngine for full-genome fixed genes, or remove fixed genes from the CMA-ES GeneSpace."
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
        self._state: _core.PyCMAESState | None = None
        self._event_index = 0
        self._pending_samples_by_id: dict[str, list[float]] = {}
        self._candidates_by_id: dict[str, Candidate] = {}
        self.vnext_telemetry = OptimizationTelemetry()

    @property
    def generation(self) -> int:
        """Return the current CMA generation."""
        return 0 if self._state is None else int(self._state.generation)

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

    def _normalise_fitness_result(
        self,
        result,
        ind: Individual,
        gen: int,
        idx: int,
    ) -> tuple[float, int]:
        metrics = {}
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise FitnessError("fitness_fn tuple return must be (float, dict).")
            result, metrics = result

        try:
            fitness = float(result)
        except (TypeError, ValueError) as exc:
            raise FitnessError(
                f"fitness_fn must return a float, got {type(result)!r} at generation {gen}, index {idx}."
            ) from exc

        ind.metadata["metrics"] = dict(metrics)
        if not math.isfinite(fitness):
            ind.metadata["raw_fitness"] = fitness
            ind.fitness = float("-inf")
            ind.fitness_valid = True
            return float("-inf"), 1

        ind.fitness = fitness
        ind.fitness_valid = True
        return fitness, 0

    def _evaluate_all(
        self,
        individuals: Sequence[Individual],
        fitness_fn: Callable[[Individual], float | tuple[float, dict]],
        gen: int,
    ) -> tuple[list[float], int]:
        if self.parallel == "thread":
            logger.debug(
                "CMA-ES thread evaluation generation=%s n_workers=%s population=%s",
                gen,
                self.n_workers,
                len(individuals),
            )
            try:
                raw_results = ThreadParallel(self.n_workers).evaluate(individuals, fitness_fn)
            except Exception as exc:
                raise FitnessError(
                    f"fitness_fn raised {type(exc).__name__} during thread evaluation at generation {gen}. "
                    f"Original error: {exc}"
                ) from exc
        else:
            raw_results = []
            for idx, ind in enumerate(individuals):
                try:
                    raw_results.append(fitness_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"fitness_fn raised {type(exc).__name__} for individual at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc

        fitnesses: list[float] = []
        nan_count = 0
        for idx, (ind, raw) in enumerate(zip(individuals, raw_results, strict=False)):
            fitness, bad_count = self._normalise_fitness_result(raw, ind, gen, idx)
            fitnesses.append(fitness)
            nan_count += bad_count

        if nan_count and not self._fitness_warning_emitted:
            logger.warning(
                "CMA-ES generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
                gen,
                nan_count,
            )
            warnings.warn(
                f"{nan_count} individuals in generation {gen} returned NaN or Inf fitness. "
                "They have been assigned fitness=-inf for selection.",
                FitnessWarning,
                stacklevel=2,
            )
            self._fitness_warning_emitted = True

        return fitnesses, nan_count

    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(seed=self.seed, generations=self.generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _ensure_state(self) -> _core.PyCMAESState:
        if self._state is None:
            self._state = _core.PyCMAESState(
                self._initial_mean_encoded(),
                self._sigma_abs(),
                self.population_size,
                self._bounds_list,
            )
        return self._state

    def ask(self, n: int | None = None) -> list[Candidate]:
        """Return a CMA candidate batch."""
        if n is not None and int(n) != self.population_size:
            raise ConfigurationError(
                "CMAESEngine.ask currently requires n to equal population_size."
            )
        state = self._ensure_state()
        event_index = self._event_index
        samples_continuous = state.ask(self.seed, event_index)
        samples_discrete = [
            self._apply_bounds_and_round(sample) for sample in samples_continuous
        ]
        candidates: list[Candidate] = []
        for index, sample in enumerate(samples_discrete):
            individual = self._decode_individual(sample)
            candidate_id = _core.candidate_id(self.seed, event_index, index)
            candidate = Candidate(
                candidate_id=candidate_id,
                genes=list(individual.genes),
                params=individual.params,
                origin="cma_sample",
                event_index=event_index,
            )
            self._pending_samples_by_id[candidate_id] = list(samples_continuous[index])
            self._candidates_by_id[candidate_id] = candidate
            candidates.append(candidate)
        self._event_index += 1
        self.vnext_telemetry.record_proposed(len(candidates))
        return candidates

    def tell(self, records: Sequence[EvaluationRecord]) -> EngineStateSummary:
        """Update CMA state from trusted evaluation records."""
        trusted_records: list[EvaluationRecord] = []
        partial = surrogate = rejected = 0
        for record in records:
            candidate = self._candidates_by_id.get(record.candidate_id)
            if candidate is None:
                raise FitnessError(
                    f"tell() received unknown candidate_id: {record.candidate_id!r}"
                )
            candidate.apply_record(record)
            if record.confidence == "trusted_full":
                trusted_records.append(record)
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence in ("partial", "cached"):
                partial += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, rung=record.rung)

        if len(trusted_records) == self.population_size:
            samples = [
                self._pending_samples_by_id[record.candidate_id]
                for record in trusted_records
            ]
            fitnesses = [
                float(record.score)
                for record in trusted_records
                if record.score is not None
            ]
            self._ensure_state().tell(samples, fitnesses)

        return EngineStateSummary(
            accepted_count=len(records),
            trusted_count=len(trusted_records),
            partial_count=partial,
            surrogate_count=surrogate,
            rejected_count=rejected,
        )

    def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
        """Run one CMA-ES optimization.

        Args:
            fitness_fn: Callable receiving an `Individual` and returning either a fitness
                float or `(fitness, metrics_dict)`.

        Returns:
            Run result containing the best individual, final population, logbook, and timing.

        Raises:
            FitnessError: If the fitness function raises or returns an invalid value.

        Warns:
            FitnessWarning: When NaN or Inf fitness values are assigned `-inf`.
        """
        self._fitness_warning_emitted = False
        self._bind_callbacks()

        start = time.perf_counter()
        state = _core.PyCMAESState(
            self._initial_mean_encoded(),
            self._sigma_abs(),
            self.population_size,
            self._bounds_list,
        )
        logbook = Logbook()
        elite_history: list[Individual] = []
        diversity_history: list[list[float]] = []
        final_population = Population([])
        n_evaluations = 0
        stopped_early = False

        for gen in range(self.generations):
            for callback in self.callbacks:
                callback.on_generation_start(gen, final_population)
            if self._callbacks_should_stop():
                stopped_early = True
                break

            gen_start = time.perf_counter()
            samples_continuous = state.ask(self.seed, gen)
            samples_discrete = [
                self._apply_bounds_and_round(sample) for sample in samples_continuous
            ]
            individuals = [self._decode_individual(sample) for sample in samples_discrete]
            fitnesses, nan_count = self._evaluate_all(individuals, fitness_fn, gen)
            n_evaluations += len(individuals)
            state.tell(samples_continuous, fitnesses)

            final_population = Population(individuals)
            info = GenerationInfo(gen, nan_count, 0)
            best = final_population.best(1)[0]
            diversity = final_population.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            elite_history.append(best.clone())
            logbook.append(
                LogEntry(
                    gen=gen,
                    best_fitness=float(best.fitness),
                    mean_fitness=final_population.mean_fitness(),
                    std_fitness=final_population.std_fitness(),
                    wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
                    n_evaluations=len(individuals),
                    nan_fitness_count=nan_count,
                    cached_count=0,
                    diversity=diversity,
                    custom=dict(best.metadata.get("metrics", {})),
                )
            )
            logger.info(
                "CMA-ES generation=%s best_fitness=%s mean_fitness=%s nan_fitness_count=%s",
                gen,
                float(best.fitness),
                final_population.mean_fitness(),
                nan_count,
            )
            for callback in self.callbacks:
                callback.on_generation_end(gen, final_population, info)
            if self._callbacks_should_stop():
                stopped_early = True
                break

        if len(final_population):
            best = final_population.best(1)[0]
            best_individual = best.clone()
            best_fitness = float(best.fitness)
        else:
            best_individual = Individual([], fitness=float("-inf"), fitness_valid=True)
            best_fitness = float("-inf")

        result = RunResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            final_population=final_population,
            logbook=logbook,
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_history=elite_history,
            diversity_history=diversity_history,
            seed=self.seed,
            stopped_early=stopped_early,
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result
