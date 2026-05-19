from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence

from evocore.callbacks import GenerationInfo
from evocore.core.errors import FitnessError
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.results import (
    GenerationHistory,
    GenerationRecord,
    OptimizationResult,
    StopReason,
    append_run_stop_event,
)
from evocore.search_space import Solution, SolutionSet

logger = logging.getLogger(__name__)


class GeneticAlgorithmGenerationLoopMixin:
    """Classic generation-loop execution helpers for GA."""

    def _normalise_fitness_result(
        self, result, ind: Solution, gen: int, idx: int
    ) -> tuple[float, int]:
        metrics = {}
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise FitnessError("objective_fn tuple return must be (float, dict).")
            result, metrics = result

        try:
            fitness = float(result)
        except (TypeError, ValueError) as exc:
            raise FitnessError(
                f"objective_fn must return a float, got {type(result)!r} at generation {gen}, index {idx}."
            ) from exc

        ind.metadata["metrics"] = dict(metrics)
        if not math.isfinite(fitness):
            raise FitnessError(
                f"objective_fn must return a finite float at generation {gen}, index {idx}; "
                f"got {fitness!r}."
            )

        ind.score = fitness
        ind.score_valid = True
        return fitness, 0

    def _remaining_evaluations(self, n_evaluations: int) -> int | None:
        if self.max_evaluations is None:
            return None
        return max(self.max_evaluations - n_evaluations, 0)

    @staticmethod
    def _fitnesses_for_selection(individuals: Sequence[Solution]) -> list[float]:
        return [
            float(ind.score) if ind.score is not None and ind.score_valid else float("-inf")
            for ind in individuals
        ]

    def _evaluate_with_budget(
        self,
        individuals: Sequence[Solution],
        objective_fn: Callable,
        gen: int,
        n_evaluations: int,
    ) -> tuple[list[Solution], list[float], int, int]:
        working = list(individuals)
        pending = [ind for ind in working if not ind.score_valid]
        remaining = self._remaining_evaluations(n_evaluations)
        if remaining == 0:
            evaluated = [ind for ind in working if ind.score_valid]
            return evaluated, self._fitnesses_for_selection(evaluated), 0, 0

        to_evaluate = pending if remaining is None else pending[:remaining]
        nan_count = 0
        if to_evaluate:
            _, nan_count = self._evaluate_all(to_evaluate, objective_fn, gen=gen)

        evaluated_now = len(to_evaluate)
        evaluated = [ind for ind in working if ind.score_valid]
        return evaluated, self._fitnesses_for_selection(evaluated), evaluated_now, nan_count

    def _evaluate_all(
        self, individuals: Sequence[Solution], objective_fn: Callable, gen: int
    ) -> tuple[list[float], int]:
        pending = [ind for ind in individuals if not ind.score_valid]
        if self.parallel == "process":
            logger.debug(
                "GA process evaluation generation=%s n_workers=%s pending=%s",
                gen,
                self.n_workers,
                len(pending),
            )
            ensure_picklable(objective_fn, context="parallel='process'")
            try:
                with ProcessParallel(
                    self.n_workers,
                    initializer=self.process_initializer,
                    initargs=self.process_initargs,
                ) as parallel:
                    raw_results = parallel.evaluate(pending, objective_fn)
            except Exception as exc:
                raise FitnessError(
                    f"objective_fn raised {type(exc).__name__} during process evaluation at generation {gen}. "
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
                raw_results = ThreadParallel(self.n_workers).evaluate(pending, objective_fn)
            except Exception as exc:
                raise FitnessError(
                    f"objective_fn raised {type(exc).__name__} during thread evaluation at generation {gen}. "
                    f"Original error: {exc}"
                ) from exc
        else:
            raw_results = []
            for idx, ind in enumerate(pending):
                try:
                    raw_results.append(objective_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"objective_fn raised {type(exc).__name__} for Solution at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc

        nan_count = 0
        for raw_idx, (ind, raw) in enumerate(zip(pending, raw_results, strict=False)):
            _, n_bad = self._normalise_fitness_result(raw, ind, gen, raw_idx)
            nan_count += n_bad

        return [
            float(ind.score) if ind.score is not None else float("-inf") for ind in individuals
        ], nan_count

    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(
                seed=self.seed,
                max_generations=self.max_generations,
                checkpoint_factory=self.checkpoint,
            )

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _log_entry(
        self,
        gen: int,
        pop: SolutionSet,
        gen_start: float,
        n_evaluations: int,
        info: GenerationInfo,
        diversity: list[float],
    ) -> GenerationRecord:
        best = pop.best(1)[0]
        return GenerationRecord(
            gen=gen,
            best_score=float(best.score),
            mean_score=pop.mean_score(),
            std_score=pop.std_score(),
            wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
            n_evaluations=n_evaluations,
            nan_score_count=info.nan_score_count,
            cached_count=info.cached_count,
            diversity=diversity,
            custom=dict(best.metadata.get("metrics", {})),
        )

    def _record_generation(
        self,
        *,
        gen: int,
        gen_start: float,
        n_evaluations: int,
        eval_before: int,
        elites: Sequence[Solution],
        nan_count: int,
        solutions: SolutionSet,
        elite_history: list[Solution],
        diversity_history: list[list[float]],
        generation_history: GenerationHistory,
    ) -> GenerationInfo:
        info = GenerationInfo(gen, nan_count, len(elites))
        diversity = solutions.diversity() if self.track_diversity else []
        if self.track_diversity:
            diversity_history.append(diversity)
        elite_history.append(solutions.best(1)[0].clone())
        generation_history.append(
            self._log_entry(
                gen, solutions, gen_start, n_evaluations - eval_before, info, diversity
            )
        )
        logger.info(
            "GA generation=%s best_score=%s mean_score=%s nan_score_count=%s cached_count=%s",
            gen,
            float(solutions.best(1)[0].score),
            solutions.mean_score(),
            nan_count,
            len(elites),
        )
        return info

    def _run_generation(
        self,
        *,
        working_population: Sequence[Solution],
        fitnesses: Sequence[float],
        objective_fn: Callable[[Solution], float | tuple[float, dict]],
        gen: int,
        n_evaluations: int,
        elite_history: list[Solution],
        diversity_history: list[list[float]],
        generation_history: GenerationHistory,
    ) -> tuple[list[Solution], list[float], int, bool, StopReason]:
        gen_start = time.perf_counter()
        current_pop = SolutionSet(working_population)
        for callback in self.callbacks:
            callback.on_generation_start(gen, current_pop)
        if self._callbacks_should_stop():
            return list(working_population), list(fitnesses), n_evaluations, True, "callback"

        elites = self._clone_elites(working_population)
        for elite in elites:
            elite.score_valid = True

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
            objective_fn,
            gen=gen,
            n_evaluations=n_evaluations,
        )
        n_evaluations += evaluated_now
        pop_obj = SolutionSet(next_population)
        info = self._record_generation(
            gen=gen,
            gen_start=gen_start,
            n_evaluations=n_evaluations,
            eval_before=eval_before,
            elites=elites,
            nan_count=nan_count,
            solutions=pop_obj,
            elite_history=elite_history,
            diversity_history=diversity_history,
            generation_history=generation_history,
        )

        for callback in self.callbacks:
            callback.on_generation_end(gen, pop_obj, info)
        if self._callbacks_should_stop():
            return next_population, fitnesses, n_evaluations, True, "callback"
        if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
            return next_population, fitnesses, n_evaluations, True, "max_evaluations"
        return next_population, fitnesses, n_evaluations, False, "max_generations"

    def _run_from_population(
        self,
        solutions: Sequence[Solution],
        objective_fn: Callable[[Solution], float | tuple[float, dict]],
        *,
        start_generation: int,
    ) -> OptimizationResult:
        if self.parallel == "process":
            ensure_picklable(objective_fn, context="parallel='process'")

        self._fitness_warning_emitted = False
        self._bind_callbacks()

        start = time.perf_counter()
        generation_history = GenerationHistory()
        working_population = [ind.clone() for ind in solutions]
        working_population, fitnesses, evaluated_now, _ = self._evaluate_with_budget(
            working_population,
            objective_fn,
            gen=start_generation - 1,
            n_evaluations=0,
        )
        n_evaluations = evaluated_now
        elite_history: list[Solution] = []
        diversity_history: list[list[float]] = []
        stop_reason: StopReason = "max_generations"
        for gen in range(start_generation, self.max_generations):
            # No early break here — let _run_generation check callbacks before budget
            (
                working_population,
                fitnesses,
                n_evaluations,
                generation_stopped,
                generation_stop_reason,
            ) = self._run_generation(
                working_population=working_population,
                fitnesses=fitnesses,
                objective_fn=objective_fn,
                gen=gen,
                n_evaluations=n_evaluations,
                elite_history=elite_history,
                diversity_history=diversity_history,
                generation_history=generation_history,
            )
            if generation_stopped:
                stop_reason = generation_stop_reason
                break

        if not working_population:
            raise FitnessError("GA run produced no evaluated individuals.")
        final_solutions = SolutionSet(working_population)
        best = final_solutions.best(1)[0]
        result = OptimizationResult(
            best_solution=best.clone(),
            best_score=float(best.score),
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_history,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=self.max_evaluations,
            direction=self.direction,
            optimizer_type="GeneticAlgorithmOptimizer",
            events=self._generation_history(generation_history),
            reproducibility=self._reproducibility_metadata(),
        )
        append_run_stop_event(
            result.events,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result


__all__ = ["GeneticAlgorithmGenerationLoopMixin"]
