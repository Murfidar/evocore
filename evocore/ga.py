from __future__ import annotations

import copy
import logging
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Sequence

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.exceptions import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    FitnessError,
    FitnessWarning,
)
from evocore.gene_space import GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.stats import LogEntry, Logbook

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
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


@dataclass
class MultiRunResult:
    best: RunResult
    all_runs: list[RunResult]
    n_runs: int
    wall_time_seconds: float

    def best_n(self, n: int) -> list[RunResult]:
        return self.all_runs[:n]

    def fitness_summary(self) -> dict[str, float]:
        values = [run.best_fitness for run in self.all_runs]
        return {
            "mean": mean(values) if values else float("nan"),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }


def _run_child_engine(engine: "GAEngine", seed: int, fitness_fn: Callable) -> RunResult:
    return engine._copy_with_seed(seed).run(fitness_fn)


class GAEngine:
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
        mutation_sigma: float = 0.2,
        mutation_sigma_schedule: str = "constant",
        mutation_sigma_end: float = 0.02,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 1,
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer=None,
        process_initargs=(),
        seed: int = 0,
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
        if elitism < 0 or elitism >= population_size:
            raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
        if parallel not in ("none", "thread", "process"):
            raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
        if selection not in ("tournament", "roulette", "rank"):
            raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
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
        self.track_diversity = track_diversity
        self.callbacks = list(callbacks or [])
        self.operators = OperatorSet(gene_space, crossover, mutation)
        self._fitness_warning_emitted = False

        for gene in gene_space.genes:
            if gene.kind == "int" and gene.sigma is None and (gene.high - gene.low) > 100:
                sigma_abs = mutation_sigma * (gene.high - gene.low)
                warnings.warn(
                    f'GeneDef("{gene.name}", "int", {gene.low}, {gene.high}) has range '
                    f"{gene.high - gene.low} and no per-gene sigma. With mutation_sigma={mutation_sigma}, "
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
                raw_results = ProcessParallel(
                    self.n_workers,
                    initializer=self.process_initializer,
                    initargs=self.process_initargs,
                ).evaluate(pending, fitness_fn)
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
        encoded = _core.init_population(
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.population_size,
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

    def _copy_with_seed(self, seed: int) -> "GAEngine":
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
        initial_pending = sum(1 for ind in working_population if not ind.fitness_valid)
        fitnesses, _ = self._evaluate_all(
            working_population, fitness_fn, gen=start_generation - 1
        )
        n_evaluations = initial_pending
        elite_history: list[Individual] = []
        diversity_history: list[list[float]] = []
        stopped_early = False

        for gen in range(start_generation, self.generations):
            gen_start = time.perf_counter()
            current_pop = Population(working_population)
            for callback in self.callbacks:
                callback.on_generation_start(gen, current_pop)
            if self._callbacks_should_stop():
                stopped_early = True
                break

            elites = self._clone_elites(working_population)
            for elite in elites:
                elite.fitness_valid = True

            offspring_count = self.population_size - len(elites)
            offspring_encoded: list[list[float]] = []
            if offspring_count:
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
                )

            offspring = self.operators.decode_population(offspring_encoded)
            working_population = elites + offspring

            eval_before = n_evaluations
            evaluated_now = sum(1 for ind in working_population if not ind.fitness_valid)
            fitnesses, nan_count = self._evaluate_all(working_population, fitness_fn, gen=gen)
            n_evaluations += evaluated_now

            pop_obj = Population(working_population)
            info = GenerationInfo(gen, nan_count, len(elites))
            diversity = pop_obj.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            elite_history.append(pop_obj.best(1)[0].clone())
            logbook.append(
                self._log_entry(gen, pop_obj, gen_start, n_evaluations - eval_before, info, diversity)
            )
            logger.info(
                "GA generation=%s best_fitness=%s mean_fitness=%s nan_fitness_count=%s cached_count=%s",
                gen,
                float(pop_obj.best(1)[0].fitness),
                pop_obj.mean_fitness(),
                nan_count,
                len(elites),
            )

            for callback in self.callbacks:
                callback.on_generation_end(gen, pop_obj, info)
            if self._callbacks_should_stop():
                stopped_early = True
                break

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
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result

    def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
        return self._run_from_population(
            self._initial_population(),
            fitness_fn,
            start_generation=0,
        )

    def run_multiple(
        self,
        fitness_fn: Callable,
        n_runs: int = 10,
        aggregate: str = "best",
        run_parallel: bool = False,
    ) -> MultiRunResult:
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
            ensure_picklable(fitness_fn, context="run_multiple(run_parallel=True)")
            ensure_picklable(self, context="run_multiple(run_parallel=True) engine")

            import concurrent.futures
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_runs, self.n_workers or os.cpu_count() or 1),
                mp_context=ctx,
            )
            try:
                futures = [pool.submit(_run_child_engine, self, seed, fitness_fn) for seed in child_seeds]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            finally:
                pool.shutdown(cancel_futures=True, wait=False)
        else:
            results = [self._copy_with_seed(seed).run(fitness_fn) for seed in child_seeds]

        results.sort(key=lambda run: run.best_fitness, reverse=True)
        return MultiRunResult(
            best=results[0],
            all_runs=results,
            n_runs=n_runs,
            wall_time_seconds=time.perf_counter() - started,
        )

    def resume(self, fitness_fn: Callable, checkpoint: str) -> RunResult:
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
