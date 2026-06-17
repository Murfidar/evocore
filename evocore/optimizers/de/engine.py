"""Differential Evolution optimizer engine."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from evocore.callbacks import Callback, GenerationInfo
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.core.serialization import package_version
from evocore.lifecycle import (
    BudgetPolicy,
    BudgetScheduler,
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
    is_state_update_confidence,
)
from evocore.lifecycle.ask_tell_helpers import (
    evaluation_context_for_candidates,
    validate_evaluator_records,
)
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.de.adaptive import initial_strategy_state
from evocore.optimizers.de.ask_tell import DifferentialEvolutionAskTellMixin
from evocore.optimizers.de.checkpointing import DifferentialEvolutionCheckpointingMixin
from evocore.optimizers.de.config import (
    build_de_config,
    de_reproducibility_status,
    de_runtime_hooks,
    validate_de_compatibility,
)
from evocore.optimizers.de.external import DifferentialEvolutionExternalStateMixin
from evocore.optimizers.de.multi_run import DifferentialEvolutionMultiRunMixin
from evocore.results import (
    EventHistory,
    GenerationHistory,
    GenerationRecord,
    OptimizationResult,
    ReproducibilityMetadata,
    StopReason,
    append_run_stop_event,
)
from evocore.search_space import GeneSpace, Solution, SolutionSet


def _evaluate_one_candidate(
    args: tuple[Evaluator, Candidate, EvaluationContext],
) -> EvaluationRecord:
    evaluator, candidate, context = args
    return evaluator.evaluate([candidate], context)[0]


class DifferentialEvolutionOptimizer(
    DifferentialEvolutionExternalStateMixin,
    DifferentialEvolutionCheckpointingMixin,
    DifferentialEvolutionAskTellMixin,
    DifferentialEvolutionMultiRunMixin,
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
        self._de_strategy_state = initial_strategy_state(
            strategy=self.strategy,
            population_size=self.population_size,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
        )

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

    def _resolve_policy(self, policy: BudgetPolicy | None) -> BudgetPolicy:
        """Resolve explicit or constructor shorthand budget settings."""
        if policy is not None and not isinstance(policy, BudgetPolicy):
            raise ConfigurationError("policy must be a BudgetPolicy when provided.")
        if policy is not None:
            return policy
        max_evaluations = self.max_evaluations
        if max_evaluations is None:
            max_evaluations = max(1, self.population_size * (self.max_generations + 1))
        return BudgetPolicy.single_full(
            max_evaluations=max_evaluations,
            batch_size=self.population_size,
        )

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
        return evaluation_context_for_candidates(
            candidates,
            stage,
            direction=self.direction,
            fallback_event_index=self._event_index,
            batch_error_message="DE run candidates must belong to exactly one batch.",
        )

    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        """Reject incomplete or mismatched synchronous evaluator results."""
        validate_evaluator_records(
            assigned,
            records,
            batch_error_message="DE run candidates must belong to exactly one batch.",
        )

    def _candidate_has_terminal_record(self, candidate: Candidate) -> bool:
        batch = self._batches_by_id[candidate.batch_id]
        return any(
            record.candidate_id == candidate.candidate_id
            and (is_state_update_confidence(record.confidence) or record.confidence == "rejected")
            for record in batch.records_by_key.values()
        )

    def _screened_out_records(
        self,
        candidates: Sequence[Candidate],
        *,
        completed_stage: str,
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        synthetic_stage = f"{completed_stage}__de_screened_out"
        for candidate in candidates:
            if self._candidate_has_terminal_record(candidate):
                continue
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=None,
                    confidence="rejected",
                    stage=synthetic_stage,
                    cost=0.0,
                    metadata={
                        "reason": "not_promoted",
                        "completed_stage": completed_stage,
                        "target_candidate_id": candidate.metadata.get("target_candidate_id"),
                        "target_slot": candidate.metadata.get("target_slot"),
                    },
                )
            )
        return records

    def _reject_screened_out(
        self,
        candidates: Sequence[Candidate],
        *,
        completed_stage: str,
    ) -> None:
        records = self._screened_out_records(candidates, completed_stage=completed_stage)
        if records:
            self.tell(records)

    @staticmethod
    def _project_final_stage_count(candidate_count: int, policy: BudgetPolicy) -> int:
        count = int(candidate_count)
        if count <= 0:
            return 0
        for stage in policy.stages:
            if stage.confidence == "trusted_full":
                return count
            promote_count = max(1, int(math.ceil(count * stage.promote_fraction)))
            exploration_count = int(math.floor(count * policy.exploration_fraction))
            audit_count = int(math.floor(count * policy.audit_fraction))
            count = min(count, promote_count + exploration_count + audit_count)
        return 0

    def _candidate_count_for_remaining_budget(
        self,
        *,
        candidate_limit: int,
        remaining: int,
        policy: BudgetPolicy,
    ) -> int:
        if candidate_limit <= 0 or remaining <= 0:
            return 0
        for count in range(int(candidate_limit), 0, -1):
            projected = self._project_final_stage_count(count, policy)
            if 0 < projected <= remaining:
                return count
        return 0

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

    def _best_target_candidate(self) -> Candidate | None:
        if not self._target_candidate_ids:
            return None
        return max(
            (self._candidates_by_id[candidate_id] for candidate_id in self._target_candidate_ids),
            key=lambda candidate: candidate.state_comparison_score(self.direction),
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
        best_candidate = self._best_target_candidate()
        if best_candidate is None:
            return
        best = candidate_to_solution(
            best_candidate,
            direction=self.direction,
            gene_space=self.gene_space,
        )
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

    def _evaluate_policy_stages(
        self,
        candidates: Sequence[Candidate],
        evaluator: Evaluator,
        scheduler: BudgetScheduler,
        policy: BudgetPolicy,
    ) -> tuple[int, list[Candidate], list[EvaluationRecord]]:
        active_candidates = list(candidates)
        for stage in policy.stages:
            if not active_candidates:
                return 0, [], []
            assigned = scheduler.assign_stage(active_candidates, stage_name=stage.name)
            context = self._evaluation_context(assigned, stage)
            records = self._evaluate_candidates(assigned, evaluator, context)
            self._validate_evaluator_records(assigned, records)

            if stage.confidence == "trusted_full":
                self.tell(records)
                state_eligible_count = sum(
                    1 for record in records if is_state_update_confidence(record.confidence)
                )
                fresh_full_count = sum(
                    1 for record in records if record.confidence == "trusted_full"
                )
                if state_eligible_count == 0:
                    raise FitnessError(
                        "Evaluator returned no state-eligible records for the final stage; "
                        "trusted_full or cached records are required."
                    )
                return fresh_full_count, list(assigned), list(records)

            if any(is_state_update_confidence(record.confidence) for record in records):
                raise FitnessError(
                    "Evaluator returned state-eligible records before final stage; "
                    "only the final policy stage may update DE target slots."
                )
            self.tell(records)
            promoted = scheduler.promote(assigned, completed_stage=stage.name)
            promoted_ids = {candidate.candidate_id for candidate in promoted}
            screened_out = [
                candidate for candidate in assigned if candidate.candidate_id not in promoted_ids
            ]
            self._reject_screened_out(screened_out, completed_stage=stage.name)
            active_candidates = promoted

        return 0, [], []

    def _build_run_result(
        self,
        *,
        started: float,
        generation_history: GenerationHistory,
        diversity_history: list[list[float]],
        elite_history: list[Solution],
        n_evaluations: int,
        stop_reason: StopReason,
        max_evaluations: int,
    ) -> OptimizationResult:
        final_solutions = self._target_solutions()
        best_candidate = self._best_target_candidate()
        if best_candidate is not None:
            best_solution = candidate_to_solution(
                best_candidate,
                direction=self.direction,
                gene_space=self.gene_space,
            )
            best_score = float(best_solution.score)
        else:
            best_solution = Solution([], score=float("-inf"), score_valid=False)
            final_solutions = SolutionSet([best_solution])
            best_score = float("-inf")

        result = OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - started,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_history,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=max_evaluations,
            telemetry=self.vnext_telemetry,
            direction=self.direction,
            optimizer_type="DifferentialEvolutionOptimizer",
            best_candidate_id=best_candidate.candidate_id if best_candidate else None,
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

    def run(  # noqa: PLR0912, PLR0915
        self,
        evaluator: Evaluator,
        policy: BudgetPolicy | None = None,
    ) -> OptimizationResult:
        """Run one synchronous policy-driven DE optimization."""
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run requires an evaluator with evaluate(candidates, context)."
            )
        resolved_policy = self._resolve_policy(policy)
        scheduler = BudgetScheduler(resolved_policy)
        self._reset_vnext_state()
        self._bind_callbacks()

        started = time.perf_counter()
        generation_history = GenerationHistory()
        diversity_history: list[list[float]] = []
        elite_history: list[Solution] = []
        n_evaluations = 0
        stop_reason: StopReason = "max_generations"

        while (
            len(self._target_candidate_ids) < self.population_size
            and self.vnext_telemetry.candidates_full_evaluated < resolved_policy.max_evaluations
        ):
            remaining = (
                resolved_policy.max_evaluations - self.vnext_telemetry.candidates_full_evaluated
            )
            candidate_limit = min(
                resolved_policy.batch_size or self.population_size,
                self.population_size - len(self._target_candidate_ids),
            )
            batch_size = self._candidate_count_for_remaining_budget(
                candidate_limit=candidate_limit,
                remaining=remaining,
                policy=resolved_policy,
            )
            if batch_size <= 0:
                stop_reason = "max_evaluations"
                break
            candidates = self.ask(batch_size)
            fresh_count, _, _ = self._evaluate_policy_stages(
                candidates,
                evaluator,
                scheduler,
                resolved_policy,
            )
            n_evaluations += fresh_count

        if self.vnext_telemetry.candidates_full_evaluated >= resolved_policy.max_evaluations:
            stop_reason = "max_evaluations"

        for gen in range(self.max_generations):
            if stop_reason == "max_evaluations":
                break
            gen_start = time.perf_counter()
            current_solutions = self._target_solutions()
            for callback in self.callbacks:
                callback.on_generation_start(gen, current_solutions)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break

            remaining = (
                resolved_policy.max_evaluations - self.vnext_telemetry.candidates_full_evaluated
            )
            candidate_limit = min(
                self.population_size,
                resolved_policy.batch_size or self.population_size,
            )
            trial_count = self._candidate_count_for_remaining_budget(
                candidate_limit=candidate_limit,
                remaining=remaining,
                policy=resolved_policy,
            )
            if trial_count <= 0:
                stop_reason = "max_evaluations"
                break

            trials = self.ask(trial_count)
            fresh_count, final_candidates, _ = self._evaluate_policy_stages(
                trials,
                evaluator,
                scheduler,
                resolved_policy,
            )
            n_evaluations += fresh_count
            self._append_generation_record(
                generation_history,
                gen=gen,
                gen_start=gen_start,
                n_evaluations=fresh_count,
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
            if (
                fresh_count > 0
                and self.vnext_telemetry.candidates_full_evaluated
                >= resolved_policy.max_evaluations
            ):
                stop_reason = "max_evaluations"
                break
            if not final_candidates:
                stop_reason = "max_evaluations"
                break

        return self._build_run_result(
            started=started,
            generation_history=generation_history,
            diversity_history=diversity_history,
            elite_history=elite_history,
            n_evaluations=n_evaluations,
            stop_reason=stop_reason,
            max_evaluations=resolved_policy.max_evaluations,
        )
