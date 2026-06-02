"""Differential Evolution optimizer engine."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from evocore.callbacks import Callback, GenerationInfo
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.core.serialization import package_version
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    Direction,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    Evaluator,
    OptimizationTelemetry,
    OptimizerStateSummary,
    candidate_to_solution,
)
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.de.ask_tell import DifferentialEvolutionAskTellMixin
from evocore.optimizers.de.checkpointing import DifferentialEvolutionCheckpointingMixin
from evocore.optimizers.de.config import (
    build_de_config,
    de_reproducibility_status,
    de_runtime_hooks,
    validate_de_compatibility,
)
from evocore.results import (
    EventHistory,
    GenerationHistory,
    GenerationRecord,
    OptimizationResult,
    ReproducibilityMetadata,
    StopReason,
    append_run_stop_event,
)
from evocore.search_space import GeneSpace, SolutionSet


def _evaluate_one_candidate(
    args: tuple[Evaluator, Candidate, EvaluationContext],
) -> EvaluationRecord:
    evaluator, candidate, context = args
    return evaluator.evaluate([candidate], context)[0]


class DifferentialEvolutionOptimizer(
    DifferentialEvolutionCheckpointingMixin,
    DifferentialEvolutionAskTellMixin,
):
    """Run Differential Evolution over a flat EvoCore GeneSpace."""

    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        max_generations: int = 300,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.9,
        strategy: str = "rand1bin",
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: object | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            from evocore.core.errors import ConfigurationError

            raise ConfigurationError(
                f"DifferentialEvolutionOptimizer got unexpected argument(s): {unknown}."
            )
        self.gene_space = gene_space
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.strategy = str(strategy)
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.track_diversity = bool(track_diversity)
        self.callbacks = list(callbacks or [])
        validate_de_compatibility(self)
        self._reset_vnext_state()

    def _reset_vnext_state(self) -> None:
        """Reset state used by DE ask/tell and run APIs."""
        self._event_index = 0
        self.generation = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._target_candidate_ids: list[str] = []
        self._trial_target_slots: dict[str, int] = {}
        self._trial_target_candidate_ids: dict[str, str] = {}
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
        self.events = EventHistory()

    def _trusted_count(self) -> int:
        return len(self._target_candidate_ids)

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return self.best_candidate.candidate_id, self.best_candidate.best_state_score(
            self.direction
        )

    def state_summary(self) -> OptimizerStateSummary:
        """Return a stable read-only DE state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return OptimizerStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            telemetry=self.vnext_telemetry,
        )

    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_de_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer and gene-space compatibility."""
        validate_de_compatibility(self)

    def _optimizer_config(self) -> dict[str, Any]:
        return self.config_signature()

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        status, notes = de_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="DifferentialEvolutionOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=de_runtime_hooks(self),
        )

    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(seed=self.seed, max_generations=self.max_generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _evaluate_candidates(
        self,
        candidates: Sequence[Candidate],
        evaluator: Evaluator,
        context: EvaluationContext,
    ) -> list[EvaluationRecord]:
        if self.parallel == "process":
            ensure_picklable(
                evaluator, context="DifferentialEvolutionOptimizer.run parallel='process'"
            )
            with ProcessParallel(
                self.n_workers,
                initializer=self.process_initializer,
                initargs=self.process_initargs,
            ) as parallel:
                return parallel.evaluate(
                    [(evaluator, candidate, context) for candidate in candidates],
                    _evaluate_one_candidate,
                )
        if self.parallel == "thread":
            return ThreadParallel(self.n_workers).evaluate(
                candidates,
                lambda candidate: evaluator.evaluate([candidate], context)[0],
            )
        return list(evaluator.evaluate(candidates, context))

    def _evaluation_context(self, candidates, stage: EvaluationStage) -> EvaluationContext:
        batch_ids = {candidate.batch_id for candidate in candidates}
        if len(batch_ids) != 1:
            raise FitnessError("DE run candidates must belong to exactly one batch.")
        return EvaluationContext(
            stage=stage,
            batch_id=next(iter(batch_ids)),
            event_index=candidates[0].event_index if candidates else self._event_index,
            direction=self.direction,
            budget=stage.budget,
        )

    def _target_solutions(self) -> SolutionSet:
        return SolutionSet(
            [
                candidate_to_solution(
                    self._candidates_by_id[candidate_id],
                    direction=self.direction,
                    gene_space=self.gene_space,
                )
                for candidate_id in self._target_candidate_ids
            ]
        )

    def _append_generation_record(
        self,
        history: GenerationHistory,
        *,
        gen: int,
        gen_start: float,
        n_evaluations: int,
    ) -> None:
        solutions = self._target_solutions()
        if not len(solutions):
            return
        best = solutions.best(1)[0]
        history.append(
            GenerationRecord(
                gen=gen,
                best_score=float(best.score),
                mean_score=solutions.mean_score(),
                std_score=solutions.std_score(),
                wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
                n_evaluations=n_evaluations,
                nan_score_count=0,
                cached_count=0,
                diversity=solutions.diversity() if self.track_diversity else [],
                custom=dict(best.metadata.get("metrics", {})),
            )
        )

    def run(self, evaluator: Evaluator, policy=None) -> OptimizationResult:  # noqa: PLR0912, PLR0915
        """Run one synchronous evaluator-driven DE optimization."""
        if policy is not None:
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run does not support policy yet."
            )
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run requires an evaluator with evaluate(candidates, context)."
            )
        self._reset_vnext_state()
        self._bind_callbacks()
        stage = EvaluationStage(
            name="full",
            budget=1.0,
            promote_fraction=1.0,
            confidence="trusted_full",
        )
        started = time.perf_counter()
        generation_history = GenerationHistory()
        diversity_history: list[list[float]] = []
        elite_history = []
        n_evaluations = 0
        stop_reason: StopReason = "max_generations"

        initial = self.ask(self.population_size)
        initial_context = self._evaluation_context(initial, stage)
        initial_records = self._evaluate_candidates(initial, evaluator, initial_context)
        self.tell(initial_records)
        n_evaluations += len(initial_records)

        for gen in range(self.max_generations):
            gen_start = time.perf_counter()
            current_solutions = self._target_solutions()
            for callback in self.callbacks:
                callback.on_generation_start(gen, current_solutions)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break
            if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
                stop_reason = "max_evaluations"
                break
            remaining = (
                self.population_size
                if self.max_evaluations is None
                else max(self.max_evaluations - n_evaluations, 0)
            )
            trial_count = min(self.population_size, remaining)
            if trial_count <= 0:
                stop_reason = "max_evaluations"
                break
            trials = self.ask(trial_count)
            context = self._evaluation_context(trials, stage)
            records = self._evaluate_candidates(trials, evaluator, context)
            self.tell(records)
            n_evaluations += len(records)
            self._append_generation_record(
                generation_history,
                gen=gen,
                gen_start=gen_start,
                n_evaluations=len(records),
            )
            solutions = self._target_solutions()
            diversity = solutions.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            if len(solutions):
                elite_history.append(solutions.best(1)[0].clone())
            info = GenerationInfo(gen, 0, 0)
            for callback in self.callbacks:
                callback.on_generation_end(gen, solutions, info)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break

        final_solutions = self._target_solutions()
        if not len(final_solutions):
            raise FitnessError("DE run produced no evaluated target candidates.")
        best_solution = final_solutions.best(1)[0].clone()
        result = OptimizationResult(
            best_solution=best_solution,
            best_score=float(best_solution.score),
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - started,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_history,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=self.max_evaluations,
            telemetry=self.vnext_telemetry,
            direction=self.direction,
            optimizer_type="DifferentialEvolutionOptimizer",
            best_candidate_id=self.best_candidate.candidate_id if self.best_candidate else None,
            events=self.events,
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
