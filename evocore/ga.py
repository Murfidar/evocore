"""Genetic algorithm engine and run result containers."""

from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import time
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Literal

from evocore import _core
from evocore.batches import CandidateBatch, batch_id_from_seed
from evocore.callbacks import Callback, GenerationInfo
from evocore.evaluation import (
    Candidate,
    Direction,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    OptimizationTelemetry,
    TellResult,
    is_state_update_confidence,
    score_for_direction,
)
from evocore.exceptions import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    FitnessError,
    FitnessWarning,
)
from evocore.exporting import json_safe, package_version, stable_json_dumps
from evocore.gene_space import GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.policies import MultiFidelityPolicy
from evocore.protocols import Evaluator
from evocore.scheduler import EvaluationScheduler
from evocore.stats import (
    EventHistory,
    EventRecord,
    Logbook,
    LogEntry,
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)

logger = logging.getLogger(__name__)

StopReason = Literal["generations", "max_evaluations", "callback"]


@dataclass
class RunResult:
    """Store the outcome of one optimization run."""

    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
    n_evaluations: int
    elite_history: list[Individual]
    diversity_history: list[list[float]]
    seed: int
    stopped_early: bool
    max_evaluations: int | None = None
    stop_reason: StopReason = "generations"
    budget_reached: bool = False
    telemetry: OptimizationTelemetry = field(default_factory=OptimizationTelemetry)
    direction: Direction = "maximize"
    engine_type: str = ""
    best_candidate_id: str | None = None
    best_score: float | None = None
    history: EventHistory = field(default_factory=EventHistory)
    reproducibility: ReproducibilityMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export this run result as a stable JSON-safe dictionary."""
        best_score = self.best_score if self.best_score is not None else self.best_fitness
        payload: dict[str, Any] = {
            "schema_version": 1,
            "engine_type": self.engine_type,
            "direction": self.direction,
            "seed": self.seed,
            "best": {
                "fitness": self.best_fitness,
                "score": best_score,
                "candidate_id": self.best_candidate_id,
                "genes": list(self.best_individual.genes),
                "params": self.best_individual.metadata.get("params"),
            },
            "stop": {
                "stopped_early": self.stopped_early,
                "reason": self.stop_reason,
            },
            "budget": {
                "max_evaluations": self.max_evaluations,
                "budget_reached": self.budget_reached,
            },
            "n_evaluations": self.n_evaluations,
            "reproducibility": (
                self.reproducibility.to_dict() if self.reproducibility is not None else None
            ),
            "telemetry": self.telemetry.to_dict(),
            "history": self.history.to_dict(),
            "logbook": self.logbook.to_dict(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export this run result as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return event history as a DataFrame, falling back to generation logbook rows."""
        if len(self.history):
            return self.history.to_dataframe()
        return self.logbook.to_dataframe()


@dataclass
class MultiRunResult:
    """Store the aggregated outcome of multiple GA runs."""

    best: RunResult
    all_runs: list[RunResult]
    n_runs: int
    wall_time_seconds: float
    direction: Direction = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_n(self, n: int) -> list[RunResult]:
        """Return the top `n` runs sorted by best fitness."""
        return self.all_runs[:n]

    def fitness_summary(self) -> dict[str, float]:
        """Return summary statistics across best fitness values."""
        values = [run.best_fitness for run in self.all_runs]
        return {
            "mean": mean(values) if values else float("nan"),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export aggregate run results as a stable JSON-safe dictionary."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "direction": self.direction,
            "n_runs": self.n_runs,
            "best": self.best.to_dict(include_runtime=include_runtime),
            "runs": [run.to_dict(include_runtime=include_runtime) for run in self.all_runs],
            "fitness_summary": self.fitness_summary(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export aggregate run results as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return one pandas DataFrame row per child run."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "MultiRunResult.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        rows = [
            {
                "run_index": index,
                "seed": run.seed,
                "best_fitness": run.best_fitness,
                "best_score": run.best_score if run.best_score is not None else run.best_fitness,
                "n_evaluations": run.n_evaluations,
            }
            for index, run in enumerate(self.all_runs)
        ]
        return pd.DataFrame(rows)


def _run_child_engine(engine: GAEngine, seed: int, evaluator: Evaluator) -> RunResult:
    return engine._copy_with_seed(seed).run(evaluator)


class GAEngine:
    """Run deterministic genetic algorithm optimization over a gene space.

    Args:
        gene_space: Gene definitions for individuals.
        population_size: Number of individuals per generation.
        generations: Maximum number of generations to run.
        crossover: Crossover operator name. Numeric spaces support `"sbx"`, `"blx"`,
            and `"uniform"`; binary spaces support `"one_point"`, `"two_point"`, and
            `"uniform"`.
        crossover_prob: Probability of applying crossover.
        crossover_eta: Eta parameter for simulated binary crossover.
        crossover_alpha: Alpha parameter for blend crossover.
        mutation: Mutation operator name.
        mutation_prob: Per-gene mutation probability once an offspring is selected for mutation.
        mutation_individual_prob: Per-offspring probability of applying mutation.
        mutation_sigma: Global mutation sigma fraction.
        mutation_sigma_schedule: Sigma schedule name.
        mutation_sigma_end: Final sigma fraction for decay schedules.
        selection: Selection operator name.
        tournament_size: Number of candidates per tournament.
        elitism: Number of best individuals copied into each generation.
        parallel: Evaluation mode: `"none"`, `"thread"`, or `"process"`.
        n_workers: Worker count for parallel modes.
        process_initializer: Optional initializer for process workers.
        process_initargs: Arguments passed to the process initializer.
        seed: Master seed for deterministic reproducibility.
        track_diversity: Whether to record per-gene diversity.
        callbacks: Optional callbacks invoked during the run.
        max_evaluations: Optional hard cap on fitness calls.

    Raises:
        ConfigurationError: If engine configuration is invalid.
    """

    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 100,
        generations: int = 100,
        crossover: str = "sbx",
        crossover_prob: float = 0.9,
        crossover_eta: float = 2.0,
        crossover_alpha: float = 0.5,
        mutation: str = "gaussian",
        mutation_prob: float = 0.1,
        mutation_individual_prob: float = 1.0,
        mutation_sigma: float = 0.2,
        mutation_sigma_schedule: str = "constant",
        mutation_sigma_end: float = 0.02,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 1,
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: Callable[..., object] | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for GAEngine. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if generations < 0:
            raise ConfigurationError("generations must be >= 0.")
        if max_evaluations is not None and max_evaluations <= 0:
            raise ConfigurationError("max_evaluations must be positive when provided.")
        if elitism < 0 or elitism >= population_size:
            raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
        if parallel not in ("none", "thread", "process"):
            raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
        if selection not in ("tournament", "roulette", "rank"):
            raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
        if not (0.0 <= mutation_individual_prob <= 1.0):
            raise ConfigurationError("mutation_individual_prob must be in [0, 1].")
        if mutation_sigma_schedule not in ("constant", "linear_decay", "cosine_decay"):
            raise ConfigurationError(
                "mutation_sigma_schedule must be 'constant', 'linear_decay', or 'cosine_decay'."
            )

        self.gene_space = gene_space
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.crossover_alpha = crossover_alpha
        self.mutation = mutation
        self.mutation_prob = mutation_prob
        self.mutation_individual_prob = mutation_individual_prob
        self.mutation_sigma = mutation_sigma
        self.mutation_sigma_schedule = mutation_sigma_schedule
        self.mutation_sigma_end = mutation_sigma_end
        self.selection = selection
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        if direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.track_diversity = track_diversity
        self.callbacks = list(callbacks or [])
        self.operators = OperatorSet(gene_space, crossover, mutation)
        self._fitness_warning_emitted = False
        self._reset_vnext_state()

        self._warn_if_large_int_gene_without_sigma()

    def _reset_vnext_state(self) -> None:
        """Reset state used by the vNext ask/tell and run APIs."""
        self._event_index = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._trusted_population_vnext: list[Candidate] = []
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
        self.history = EventHistory()

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if len(batch.records_by_key) < len(batch.candidate_ids)
        )

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return (
            self.best_candidate.candidate_id,
            self.best_candidate.best_state_score(self.direction),
        )

    def _record_state_candidate(self, candidate: Candidate) -> None:
        if not any(
            existing.candidate_id == candidate.candidate_id
            for existing in self._trusted_population_vnext
        ):
            self._trusted_population_vnext.append(candidate)
        if self.best_candidate is None or candidate.state_comparison_score(
            self.direction
        ) > self.best_candidate.state_comparison_score(self.direction):
            self.best_candidate = candidate

    def state_summary(self) -> EngineStateSummary:
        """Return a stable read-only vNext state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return EngineStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=len(self._trusted_population_vnext),
            telemetry=self.vnext_telemetry,
        )

    def _warn_if_large_int_gene_without_sigma(self) -> None:
        for gene in self.gene_space.genes:
            if gene.kind == "int" and gene.sigma is None and (gene.high - gene.low) > 100:
                sigma_abs = self.mutation_sigma * (gene.high - gene.low)
                warnings.warn(
                    f'GeneDef("{gene.name}", "int", {gene.low}, {gene.high}) has range '
                    f"{gene.high - gene.low} and no per-gene sigma. With mutation_sigma={self.mutation_sigma}, "
                    f"sigma_abs={sigma_abs:g} may prevent fine-tuning. Consider sigma=0.03.",
                    category=ConfigurationWarning,
                    stacklevel=2,
                )

    def _normalise_fitness_result(
        self, result, ind: Individual, gen: int, idx: int
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

    def _remaining_evaluations(self, n_evaluations: int) -> int | None:
        if self.max_evaluations is None:
            return None
        return max(self.max_evaluations - n_evaluations, 0)

    @staticmethod
    def _fitnesses_for_selection(individuals: Sequence[Individual]) -> list[float]:
        return [
            float(ind.fitness) if ind.fitness is not None and ind.fitness_valid else float("-inf")
            for ind in individuals
        ]

    def _evaluate_with_budget(
        self,
        individuals: Sequence[Individual],
        fitness_fn: Callable,
        gen: int,
        n_evaluations: int,
    ) -> tuple[list[Individual], list[float], int, int]:
        working = list(individuals)
        pending = [ind for ind in working if not ind.fitness_valid]
        remaining = self._remaining_evaluations(n_evaluations)
        if remaining == 0:
            evaluated = [ind for ind in working if ind.fitness_valid]
            return evaluated, self._fitnesses_for_selection(evaluated), 0, 0

        to_evaluate = pending if remaining is None else pending[:remaining]
        nan_count = 0
        if to_evaluate:
            _, nan_count = self._evaluate_all(to_evaluate, fitness_fn, gen=gen)

        evaluated_now = len(to_evaluate)
        evaluated = [ind for ind in working if ind.fitness_valid]
        return evaluated, self._fitnesses_for_selection(evaluated), evaluated_now, nan_count

    def _evaluate_all(
        self, individuals: Sequence[Individual], fitness_fn: Callable, gen: int
    ) -> tuple[list[float], int]:
        pending = [ind for ind in individuals if not ind.fitness_valid]
        if self.parallel == "process":
            logger.debug(
                "GA process evaluation generation=%s n_workers=%s pending=%s",
                gen,
                self.n_workers,
                len(pending),
            )
            ensure_picklable(fitness_fn, context="parallel='process'")
            try:
                with ProcessParallel(
                    self.n_workers,
                    initializer=self.process_initializer,
                    initargs=self.process_initargs,
                ) as parallel:
                    raw_results = parallel.evaluate(pending, fitness_fn)
            except Exception as exc:
                raise FitnessError(
                    f"fitness_fn raised {type(exc).__name__} during process evaluation at generation {gen}. "
                    f"Original error: {exc}"
                ) from exc
        elif self.parallel == "thread":
            logger.debug(
                "GA thread evaluation generation=%s n_workers=%s pending=%s",
                gen,
                self.n_workers,
                len(pending),
            )
            try:
                raw_results = ThreadParallel(self.n_workers).evaluate(pending, fitness_fn)
            except Exception as exc:
                raise FitnessError(
                    f"fitness_fn raised {type(exc).__name__} during thread evaluation at generation {gen}. "
                    f"Original error: {exc}"
                ) from exc
        else:
            raw_results = []
            for idx, ind in enumerate(pending):
                try:
                    raw_results.append(fitness_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"fitness_fn raised {type(exc).__name__} for individual at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc

        nan_count = 0
        for raw_idx, (ind, raw) in enumerate(zip(pending, raw_results, strict=False)):
            _, n_bad = self._normalise_fitness_result(raw, ind, gen, raw_idx)
            nan_count += n_bad

        if nan_count and not self._fitness_warning_emitted:
            logger.warning(
                "GA generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
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

        return [
            float(ind.fitness) if ind.fitness is not None else float("-inf") for ind in individuals
        ], nan_count

    def _compute_sigma_fraction(self, gen: int) -> float:
        if self.generations <= 1 or self.mutation_sigma_schedule == "constant":
            return self.mutation_sigma

        t = gen / max(1, self.generations - 1)
        if self.mutation_sigma_schedule == "linear_decay":
            return self.mutation_sigma + t * (self.mutation_sigma_end - self.mutation_sigma)
        if self.mutation_sigma_schedule == "cosine_decay":
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return self.mutation_sigma_end + cosine * (
                self.mutation_sigma - self.mutation_sigma_end
            )
        raise ConfigurationError("unknown mutation_sigma_schedule")

    def _initial_population(self) -> list[Individual]:
        population_size = self.population_size
        if self.max_evaluations is not None:
            population_size = min(population_size, self.max_evaluations)
        encoded = _core.init_population(
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            population_size,
            self.seed,
        )
        return self.operators.decode_population(encoded)

    def _clone_elites(self, population: Sequence[Individual]) -> list[Individual]:
        if self.elitism == 0:
            return []
        return [ind.clone() for ind in Population(population).best(self.elitism)]

    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(seed=self.seed, generations=self.generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _log_entry(
        self,
        gen: int,
        pop: Population,
        gen_start: float,
        n_evaluations: int,
        info: GenerationInfo,
        diversity: list[float],
    ) -> LogEntry:
        best = pop.best(1)[0]
        return LogEntry(
            gen=gen,
            best_fitness=float(best.fitness),
            mean_fitness=pop.mean_fitness(),
            std_fitness=pop.std_fitness(),
            wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
            n_evaluations=n_evaluations,
            nan_fitness_count=info.nan_fitness_count,
            cached_count=info.cached_count,
            diversity=diversity,
            custom=dict(best.metadata.get("metrics", {})),
        )

    def _make_offspring(
        self,
        working_population: Sequence[Individual],
        fitnesses: Sequence[float],
        gen: int,
        offspring_count: int,
    ) -> list[Individual]:
        if offspring_count <= 0:
            return []

        sigma_list = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
        offspring_encoded = _core.reproduce_population(
            self.operators.encode_population(working_population),
            fitnesses,
            self.crossover,
            self.crossover_prob,
            self.crossover_eta,
            self.crossover_alpha,
            self.mutation,
            self.mutation_prob,
            sigma_list,
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.selection,
            self.tournament_size,
            offspring_count,
            self.seed,
            gen,
            self.mutation_individual_prob,
        )
        return self.operators.decode_population(offspring_encoded)

    def _record_generation(
        self,
        *,
        gen: int,
        gen_start: float,
        n_evaluations: int,
        eval_before: int,
        elites: Sequence[Individual],
        nan_count: int,
        population: Population,
        elite_history: list[Individual],
        diversity_history: list[list[float]],
        logbook: Logbook,
    ) -> GenerationInfo:
        info = GenerationInfo(gen, nan_count, len(elites))
        diversity = population.diversity() if self.track_diversity else []
        if self.track_diversity:
            diversity_history.append(diversity)
        elite_history.append(population.best(1)[0].clone())
        logbook.append(
            self._log_entry(
                gen, population, gen_start, n_evaluations - eval_before, info, diversity
            )
        )
        logger.info(
            "GA generation=%s best_fitness=%s mean_fitness=%s nan_fitness_count=%s cached_count=%s",
            gen,
            float(population.best(1)[0].fitness),
            population.mean_fitness(),
            nan_count,
            len(elites),
        )
        return info

    def _run_generation(
        self,
        *,
        working_population: Sequence[Individual],
        fitnesses: Sequence[float],
        fitness_fn: Callable[[Individual], float | tuple[float, dict]],
        gen: int,
        n_evaluations: int,
        elite_history: list[Individual],
        diversity_history: list[list[float]],
        logbook: Logbook,
    ) -> tuple[list[Individual], list[float], int, bool, StopReason]:
        gen_start = time.perf_counter()
        current_pop = Population(working_population)
        for callback in self.callbacks:
            callback.on_generation_start(gen, current_pop)
        if self._callbacks_should_stop():
            return list(working_population), list(fitnesses), n_evaluations, True, "callback"

        elites = self._clone_elites(working_population)
        for elite in elites:
            elite.fitness_valid = True

        offspring = self._make_offspring(
            working_population,
            fitnesses,
            gen,
            self.population_size - len(elites),
        )
        next_population = elites + offspring

        eval_before = n_evaluations
        next_population, fitnesses, evaluated_now, nan_count = self._evaluate_with_budget(
            next_population,
            fitness_fn,
            gen=gen,
            n_evaluations=n_evaluations,
        )
        n_evaluations += evaluated_now
        pop_obj = Population(next_population)
        info = self._record_generation(
            gen=gen,
            gen_start=gen_start,
            n_evaluations=n_evaluations,
            eval_before=eval_before,
            elites=elites,
            nan_count=nan_count,
            population=pop_obj,
            elite_history=elite_history,
            diversity_history=diversity_history,
            logbook=logbook,
        )

        for callback in self.callbacks:
            callback.on_generation_end(gen, pop_obj, info)
        if self._callbacks_should_stop():
            return next_population, fitnesses, n_evaluations, True, "callback"
        if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
            return next_population, fitnesses, n_evaluations, True, "max_evaluations"
        return next_population, fitnesses, n_evaluations, False, "generations"

    def _copy_with_seed(self, seed: int) -> GAEngine:
        return GAEngine(
            gene_space=self.gene_space,
            population_size=self.population_size,
            generations=self.generations,
            crossover=self.crossover,
            crossover_prob=self.crossover_prob,
            crossover_eta=self.crossover_eta,
            crossover_alpha=self.crossover_alpha,
            mutation=self.mutation,
            mutation_prob=self.mutation_prob,
            mutation_individual_prob=self.mutation_individual_prob,
            mutation_sigma=self.mutation_sigma,
            mutation_sigma_schedule=self.mutation_sigma_schedule,
            mutation_sigma_end=self.mutation_sigma_end,
            selection=self.selection,
            tournament_size=self.tournament_size,
            elitism=self.elitism,
            parallel=self.parallel,
            n_workers=self.n_workers,
            process_initializer=self.process_initializer,
            process_initargs=self.process_initargs,
            seed=int(seed),
            direction=self.direction,
            max_evaluations=self.max_evaluations,
            track_diversity=self.track_diversity,
            callbacks=copy.deepcopy(self.callbacks),
        )

    def _run_from_population(
        self,
        population: Sequence[Individual],
        fitness_fn: Callable[[Individual], float | tuple[float, dict]],
        *,
        start_generation: int,
    ) -> RunResult:
        if self.parallel == "process":
            ensure_picklable(fitness_fn, context="parallel='process'")

        self._fitness_warning_emitted = False
        self._bind_callbacks()

        start = time.perf_counter()
        logbook = Logbook()
        working_population = [ind.clone() for ind in population]
        working_population, fitnesses, evaluated_now, _ = self._evaluate_with_budget(
            working_population,
            fitness_fn,
            gen=start_generation - 1,
            n_evaluations=0,
        )
        n_evaluations = evaluated_now
        elite_history: list[Individual] = []
        diversity_history: list[list[float]] = []
        stopped_early = False
        stop_reason: StopReason = "generations"
        if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
            stopped_early = True
            stop_reason = "max_evaluations"

        for gen in range(start_generation, self.generations):
            if stop_reason == "max_evaluations":
                break
            (
                working_population,
                fitnesses,
                n_evaluations,
                generation_stopped,
                generation_stop_reason,
            ) = self._run_generation(
                working_population=working_population,
                fitnesses=fitnesses,
                fitness_fn=fitness_fn,
                gen=gen,
                n_evaluations=n_evaluations,
                elite_history=elite_history,
                diversity_history=diversity_history,
                logbook=logbook,
            )
            if generation_stopped:
                stopped_early = True
                stop_reason = generation_stop_reason
                break

        if not working_population:
            raise FitnessError("GA run produced no evaluated individuals.")
        final_population = Population(working_population)
        best = final_population.best(1)[0]
        result = RunResult(
            best_individual=best.clone(),
            best_fitness=float(best.fitness),
            final_population=final_population,
            logbook=logbook,
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_history=elite_history,
            diversity_history=diversity_history,
            seed=self.seed,
            stopped_early=stopped_early,
            max_evaluations=self.max_evaluations,
            stop_reason=stop_reason,
            budget_reached=(
                self.max_evaluations is not None and n_evaluations >= self.max_evaluations
            ),
            direction=self.direction,
            engine_type="GAEngine",
            best_score=float(best.fitness),
            history=self._generation_history(logbook),
            reproducibility=self._reproducibility_metadata(),
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result

    def _candidate_from_genes(
        self,
        genes: list[float | int | bool],
        *,
        batch_id: str,
        origin: str,
        event_index: int,
        candidate_index: int,
        parents: Sequence[str] = (),
    ) -> Candidate:
        candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
        params = self.gene_space.params_for(genes)
        return Candidate(
            candidate_id=candidate_id,
            genes=list(genes),
            batch_id=batch_id,
            params=params,
            origin=origin,
            parents=parents,
            event_index=event_index,
        )

    def ask(self, n: int | None = None) -> list[Candidate]:
        """Return vNext candidates for external evaluation."""
        count = int(n or self.population_size)
        if count <= 0:
            raise ConfigurationError("ask(n) requires n > 0.")

        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        if not self._trusted_population_vnext:
            encoded = _core.init_population(
                self.operators.gene_bounds,
                self.operators.gene_kinds,
                count,
                int(_core.py_derive_seed(self.seed, event_index, 0, _core.OP_INIT)),
            )
            individuals = self.operators.decode_population(encoded)
            candidates = [
                self._candidate_from_genes(
                    individual.genes,
                    batch_id=batch_id,
                    origin="random",
                    event_index=event_index,
                    candidate_index=index,
                )
                for index, individual in enumerate(individuals)
            ]
        else:
            trusted_individuals = [
                Individual(
                    list(candidate.genes),
                    fitness=candidate.state_comparison_score(self.direction),
                    fitness_valid=True,
                    metadata={"params": candidate.params} if candidate.params else {},
                )
                for candidate in self._trusted_population_vnext
            ]
            fitnesses = [individual.fitness or float("-inf") for individual in trusted_individuals]
            offspring = self._make_offspring(
                trusted_individuals,
                fitnesses,
                gen=event_index,
                offspring_count=count,
            )
            candidates = [
                self._candidate_from_genes(
                    individual.genes,
                    batch_id=batch_id,
                    origin="mutation",
                    event_index=event_index,
                    candidate_index=index,
                )
                for index, individual in enumerate(offspring)
            ]

        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        )
        self._event_index += 1
        self.vnext_telemetry.record_proposed_candidates(candidates)
        self._append_ask_events(candidates)
        return candidates

    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        """Update GA state from vNext evaluation records."""
        trusted = partial = surrogate = cached = rejected = 0
        touched_batch_ids: set[str] = set()
        for record in records:
            candidate = self._candidates_by_id.get(record.candidate_id)
            if candidate is None:
                raise FitnessError(
                    f"tell() received unknown candidate_id: {record.candidate_id!r}"
                )
            if record.batch_id is not None and record.batch_id not in self._batches_by_id:
                raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
            batch = self._batches_by_id.get(candidate.batch_id)
            if batch is None:
                raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
            batch.accept_record(record)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            self._append_tell_event(candidate, record)
            if is_state_update_confidence(record.confidence):
                self._record_state_candidate(candidate)
            if record.confidence == "trusted_full":
                trusted += 1
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "cached":
                cached += 1
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "partial":
                partial += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, rung=record.rung)

        self._trusted_population_vnext.sort(
            key=lambda candidate: candidate.state_comparison_score(self.direction), reverse=True
        )
        self._trusted_population_vnext = self._trusted_population_vnext[: self.population_size]
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        consumed_batch_ids = tuple(
            batch_id
            for batch_id in touched_batch_ids
            if len(self._batches_by_id[batch_id].records_by_key)
            >= len(self._batches_by_id[batch_id].candidate_ids)
        )
        return TellResult(
            accepted_count=len(records),
            trusted_count=trusted,
            partial_count=partial,
            surrogate_count=surrogate,
            cached_count=cached,
            rejected_count=rejected,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=consumed_batch_ids,
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
        )

    def _evaluation_context(
        self,
        assigned: Sequence[Candidate],
        rung,
    ) -> EvaluationContext:
        batch_ids = {candidate.batch_id for candidate in assigned}
        if len(batch_ids) != 1:
            raise FitnessError(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            )
        batch_id = next(iter(batch_ids))
        event_index = assigned[0].event_index if assigned else self._event_index
        return EvaluationContext(
            rung=rung,
            batch_id=batch_id,
            event_index=event_index,
            direction=self.direction,
            budget=rung.budget,
        )

    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        """Reject incomplete or mismatched synchronous evaluator results."""
        expected_ids = [candidate.candidate_id for candidate in assigned]
        returned_ids = [record.candidate_id for record in records]
        expected_counts = Counter(expected_ids)
        returned_counts = Counter(returned_ids)

        missing_ids = [
            candidate_id for candidate_id in expected_ids if returned_counts[candidate_id] == 0
        ]
        unexpected_ids = [
            candidate_id for candidate_id in returned_counts if candidate_id not in expected_counts
        ]
        duplicate_ids = [
            candidate_id
            for candidate_id, count in returned_counts.items()
            if count > expected_counts[candidate_id]
        ]

        if missing_ids:
            raise FitnessError(
                "Evaluator returned missing evaluation records for candidate_ids: "
                f"{sorted(set(missing_ids))!r}."
            )
        if unexpected_ids:
            raise FitnessError(
                "Evaluator returned unknown evaluation records for candidate_ids: "
                f"{sorted(unexpected_ids)!r}."
            )
        if duplicate_ids:
            raise FitnessError(
                "Evaluator returned duplicate evaluation records for candidate_ids: "
                f"{sorted(duplicate_ids)!r}."
            )

        batch_ids = {candidate.batch_id for candidate in assigned}
        if len(batch_ids) != 1:
            raise FitnessError(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            )
        expected_batch_id = next(iter(batch_ids))
        for record in records:
            if record.batch_id is not None and record.batch_id != expected_batch_id:
                raise FitnessError(
                    f"Evaluator returned record batch_id {record.batch_id!r} for batch "
                    f"{expected_batch_id!r}."
                )

    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed candidates."""
        for candidate in candidates:
            self.history.append(
                EventRecord(
                    event_index=len(self.history),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(),
                    generation=candidate.generation,
                    origin=candidate.origin,
                    parents=tuple(candidate.parents),
                    genes=tuple(candidate.genes),
                    params=dict(candidate.params) if candidate.params is not None else None,
                    metadata=dict(candidate.metadata),
                )
            )

    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        raw_score = float(record.score) if record.score is not None else None
        comparison_score = (
            score_for_direction(raw_score, self.direction)
            if raw_score is not None and math.isfinite(raw_score)
            else None
        )
        self.history.append(
            EventRecord(
                event_index=len(self.history),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(),
                generation=candidate.generation,
                rung=record.rung,
                confidence=record.confidence,
                raw_score=raw_score,
                comparison_score=comparison_score,
                cost=record.cost,
                status=candidate.status,
                origin=candidate.origin,
                parents=tuple(candidate.parents),
                genes=tuple(candidate.genes),
                params=dict(candidate.params) if candidate.params is not None else None,
                metrics=dict(record.metrics),
                metadata=dict(record.metadata),
            )
        )

    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable GA constructor configuration."""
        return json_safe(
            {
                "population_size": self.population_size,
                "generations": self.generations,
                "crossover": self.crossover,
                "crossover_prob": self.crossover_prob,
                "crossover_eta": self.crossover_eta,
                "crossover_alpha": self.crossover_alpha,
                "mutation": self.mutation,
                "mutation_prob": self.mutation_prob,
                "mutation_individual_prob": self.mutation_individual_prob,
                "mutation_sigma": self.mutation_sigma,
                "mutation_sigma_schedule": self.mutation_sigma_schedule,
                "mutation_sigma_end": self.mutation_sigma_end,
                "selection": self.selection,
                "tournament_size": self.tournament_size,
                "elitism": self.elitism,
                "parallel": self.parallel,
                "n_workers": self.n_workers,
                "max_evaluations": self.max_evaluations,
                "track_diversity": self.track_diversity,
            }
        )

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = gene_space_signature(self.gene_space)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            engine_type="GAEngine",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=gene_space_hash(signature),
            optimizer_config=self._optimizer_config(),
        )

    def _generation_history(self, logbook: Logbook) -> EventHistory:
        """Convert generation logbook entries into generation events."""
        history = EventHistory()
        for entry in logbook:
            history.append(
                EventRecord(
                    event_index=len(history),
                    event_type="generation",
                    generation=entry.gen,
                    raw_score=entry.best_fitness,
                    comparison_score=score_for_direction(entry.best_fitness, self.direction),
                    metrics=entry.to_dict(),
                )
            )
        return history

    def run(self, evaluator: Evaluator, policy: MultiFidelityPolicy | None = None) -> RunResult:
        """Run vNext policy-driven GA optimization."""
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "GAEngine.run requires an evaluator with evaluate(candidates, context)."
            )
        self._reset_vnext_state()

        resolved_policy = policy or MultiFidelityPolicy.single_full(
            budget=max(1, self.population_size * max(1, self.generations)),
            batch_size=self.population_size,
        )
        scheduler = EvaluationScheduler(resolved_policy)
        start = time.perf_counter()
        n_evaluations = 0
        final_candidates: list[Candidate] = []

        while (
            self.vnext_telemetry.candidates_full_evaluated < resolved_policy.full_evaluation_budget
        ):
            remaining = (
                resolved_policy.full_evaluation_budget
                - self.vnext_telemetry.candidates_full_evaluated
            )
            batch_size = min(resolved_policy.batch_size or self.population_size, remaining)
            active_candidates = self.ask(batch_size)

            for rung in resolved_policy.rungs:
                assigned = scheduler.assign_rung(active_candidates, rung_name=rung.name)
                context = self._evaluation_context(assigned, rung)
                records = list(evaluator.evaluate(assigned, context))
                self._validate_evaluator_records(assigned, records)
                self.tell(records)
                if rung.confidence == "trusted_full":
                    n_evaluations += len(records)
                    final_candidates.extend(assigned)
                    break
                active_candidates = scheduler.promote(assigned, completed_rung=rung.name)

        if self.best_candidate is None:
            # Defensive: only possible if policy has no trusted_full rung,
            # which MultiFidelityPolicy.__post_init__ already rejects.
            best = Individual([0.0], fitness=float("-inf"), fitness_valid=False)
            return RunResult(
                best_individual=best,
                best_fitness=float("-inf"),
                final_population=Population([best]),
                logbook=Logbook(),
                wall_time_seconds=time.perf_counter() - start,
                n_evaluations=n_evaluations,
                elite_history=[],
                diversity_history=[],
                seed=self.seed,
                stopped_early=True,
                max_evaluations=resolved_policy.full_evaluation_budget,
                stop_reason="max_evaluations",
                budget_reached=True,
                telemetry=self.vnext_telemetry,
                direction=self.direction,
                engine_type="GAEngine",
                best_score=float("-inf"),
                history=self.history,
                reproducibility=self._reproducibility_metadata(),
            )

        best = Individual(
            list(self.best_candidate.genes),
            fitness=self.best_candidate.best_state_score(self.direction),
            fitness_valid=True,
            metadata={
                "params": self.best_candidate.params,
                "candidate_id": self.best_candidate.candidate_id,
            },
        )
        final_population = Population(
            [
                Individual(
                    list(candidate.genes),
                    fitness=candidate.best_state_score(self.direction),
                    fitness_valid=candidate.confidence is not None
                    and is_state_update_confidence(candidate.confidence),
                    metadata={"params": candidate.params, "candidate_id": candidate.candidate_id},
                )
                for candidate in final_candidates
            ]
        )
        return RunResult(
            best_individual=best,
            best_fitness=float(best.fitness),
            final_population=final_population,
            logbook=Logbook(),
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_history=[],
            diversity_history=[],
            seed=self.seed,
            stopped_early=True,
            max_evaluations=resolved_policy.full_evaluation_budget,
            stop_reason="max_evaluations",
            budget_reached=True,
            telemetry=self.vnext_telemetry,
            direction=self.direction,
            engine_type="GAEngine",
            best_candidate_id=self.best_candidate.candidate_id,
            best_score=float(best.fitness),
            history=self.history,
            reproducibility=self._reproducibility_metadata(),
        )

    def run_multiple(
        self,
        evaluator: Evaluator,
        n_runs: int = 10,
        aggregate: str = "best",
        run_parallel: bool = False,
    ) -> MultiRunResult:
        """Run multiple deterministic child runs from derived seeds.

        Args:
            evaluator: Evaluator passed to each child run.
            n_runs: Number of child runs.
            aggregate: Aggregation mode. `"best"` and `"all"` are accepted.
            run_parallel: Whether to execute child runs in spawned processes.

        Returns:
            Multi-run result sorted by descending best fitness.

        Raises:
            ConfigurationError: If `n_runs`, `aggregate`, or pickle constraints are invalid.
        """
        if n_runs <= 0:
            raise ConfigurationError("n_runs must be positive.")
        if aggregate not in ("best", "all"):
            raise ConfigurationError("aggregate must be 'best' or 'all'.")

        child_seeds = [
            int(_core.py_derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
            for run_idx in range(n_runs)
        ]
        logger.debug("GA run_multiple n_runs=%s child_seeds=%s", n_runs, child_seeds)

        started = time.perf_counter()
        if run_parallel:
            ensure_picklable(evaluator, context="run_multiple(run_parallel=True)")
            ensure_picklable(self, context="run_multiple(run_parallel=True) engine")

            import concurrent.futures
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_runs, self.n_workers or os.cpu_count() or 1),
                mp_context=ctx,
            )
            try:
                futures = [
                    pool.submit(_run_child_engine, self, seed, evaluator) for seed in child_seeds
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            finally:
                pool.shutdown(cancel_futures=True, wait=False)
        else:
            results = [self._copy_with_seed(seed).run(evaluator) for seed in child_seeds]

        results.sort(
            key=lambda run: score_for_direction(run.best_fitness, self.direction),
            reverse=True,
        )
        return MultiRunResult(
            best=results[0],
            all_runs=results,
            n_runs=n_runs,
            wall_time_seconds=time.perf_counter() - started,
            direction=self.direction,
        )

    def resume(self, fitness_fn: Callable, checkpoint: str) -> RunResult:
        """Resume a GA run from a checkpoint file.

        Args:
            fitness_fn: Fitness callable used for remaining generations.
            checkpoint: Path to a checkpoint written by `CheckpointCallback`.

        Returns:
            Run result for the resumed optimization.

        Raises:
            CheckpointError: If the file is missing, corrupt, incompatible, or has a
                seed that differs from the engine seed.
        """
        if not os.path.exists(checkpoint):
            directory = os.path.dirname(checkpoint) or "."
            available = []
            if os.path.isdir(directory):
                available = sorted(
                    name for name in os.listdir(directory) if name.startswith("checkpoint_gen_")
                )
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} not found. Available checkpoints: "
                f"{', '.join(available) or 'none'}"
            )

        try:
            with open(checkpoint, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} is corrupt or incompatible: {exc}"
            ) from exc

        population = payload.get("population")
        if not isinstance(population, list) or not all(
            isinstance(individual, Individual) for individual in population
        ):
            raise CheckpointError(
                "checkpoint payload must contain a list[Individual] under key 'population'."
            )

        saved_generation = int(payload.get("generation", -1))
        saved_seed = payload.get("seed")
        if saved_seed is not None and int(saved_seed) != self.seed:
            raise CheckpointError(
                f"checkpoint seed {saved_seed} does not match engine seed {self.seed}."
            )

        return self._run_from_population(
            population,
            fitness_fn,
            start_generation=saved_generation + 1,
        )
