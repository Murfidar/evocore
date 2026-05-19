from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.lifecycle import (
    BudgetPolicy,
    BudgetScheduler,
    Candidate,
    CandidateBatch,
    EvaluationContext,
    EvaluationRecord,
    Evaluator,
    UpdateResult,
    batch_id_from_seed,
    candidate_to_solution,
    is_state_update_confidence,
    score_for_direction,
    solution_to_candidate,
)
from evocore.results import (
    EventRecord,
    GenerationHistory,
    OptimizationResult,
    append_run_stop_event,
)
from evocore.search_space import Solution, SolutionSet


class GeneticAlgorithmAskTellMixin:
    """Ask/tell lifecycle and policy-driven GA execution."""

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
        return solution_to_candidate(
            Solution(genes),
            gene_space=self.gene_space,
            candidate_id=candidate_id,
            batch_id=batch_id,
            origin=origin,
            event_index=event_index,
            parents=parents,
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
                    solution.values,
                    batch_id=batch_id,
                    origin="random",
                    event_index=event_index,
                    candidate_index=index,
                )
                for index, solution in enumerate(individuals)
            ]
        else:
            trusted_individuals = [
                Solution(
                    list(candidate.genes),
                    score=candidate.state_comparison_score(self.direction),
                    score_valid=True,
                    metadata={"params": candidate.params} if candidate.params else {},
                )
                for candidate in self._trusted_population_vnext
            ]
            fitnesses = [solution.score or float("-inf") for solution in trusted_individuals]
            offspring = self._make_offspring(
                trusted_individuals,
                fitnesses,
                gen=event_index,
                offspring_count=count,
            )
            candidates = [
                self._candidate_from_genes(
                    solution.values,
                    batch_id=batch_id,
                    origin="mutation",
                    event_index=event_index,
                    candidate_index=index,
                )
                for index, solution in enumerate(offspring)
            ]

        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        )
        self._event_index += 1
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)
        return candidates

    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
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
                self.vnext_telemetry.record_full(1, stage=record.stage, cost=record.cost)
            elif record.confidence == "cached":
                cached += 1
                self.vnext_telemetry.record_cached(1, stage=record.stage, cost=record.cost)
            elif record.confidence == "partial":
                partial += 1
                self.vnext_telemetry.record_partial(1, stage=record.stage, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, stage=record.stage)

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
        return UpdateResult(
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
        stage,
    ) -> EvaluationContext:
        batch_ids = {candidate.batch_id for candidate in assigned}
        if len(batch_ids) != 1:
            raise FitnessError(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            )
        batch_id = next(iter(batch_ids))
        event_index = assigned[0].event_index if assigned else self._event_index
        return EvaluationContext(
            stage=stage,
            batch_id=batch_id,
            event_index=event_index,
            direction=self.direction,
            budget=stage.budget,
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
            self.events.append(
                EventRecord(
                    event_index=len(self.events),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(self.gene_space),
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
        self.events.append(
            EventRecord(
                event_index=len(self.events),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(self.gene_space),
                generation=candidate.generation,
                stage=record.stage,
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

    def run(self, evaluator: Evaluator, policy: BudgetPolicy | None = None) -> OptimizationResult:
        """Run vNext policy-driven GA optimization."""
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "GeneticAlgorithmOptimizer.run requires an evaluator with evaluate(candidates, context)."
            )
        self._reset_vnext_state()

        resolved_policy = policy or BudgetPolicy.single_full(
            max_evaluations=max(1, self.population_size * max(1, self.max_generations)),
            batch_size=self.population_size,
        )
        scheduler = BudgetScheduler(resolved_policy)
        start = time.perf_counter()
        n_evaluations = 0
        final_candidates: list[Candidate] = []

        while self.vnext_telemetry.candidates_full_evaluated < resolved_policy.max_evaluations:
            remaining = (
                resolved_policy.max_evaluations - self.vnext_telemetry.candidates_full_evaluated
            )
            batch_size = min(resolved_policy.batch_size or self.population_size, remaining)
            active_candidates = self.ask(batch_size)

            for stage in resolved_policy.stages:
                assigned = scheduler.assign_stage(active_candidates, stage_name=stage.name)
                context = self._evaluation_context(assigned, stage)
                records = list(evaluator.evaluate(assigned, context))
                self._validate_evaluator_records(assigned, records)
                self.tell(records)
                if stage.confidence == "trusted_full":
                    fresh_count = sum(
                        1 for record in records if record.confidence == "trusted_full"
                    )
                    n_evaluations += fresh_count
                    final_candidates.extend(assigned)
                    if fresh_count == 0:
                        raise FitnessError(
                            "Evaluator returned no fresh trusted_full records for the final stage; "
                            "cached records do not consume full-evaluation budget."
                        )
                    break
                active_candidates = scheduler.promote(assigned, completed_stage=stage.name)

        if self.best_candidate is None:
            # Defensive: only possible if policy has no trusted_full stage,
            # which BudgetPolicy.__post_init__ already rejects.
            best = Solution([0.0], score=float("-inf"), score_valid=False)
            result = OptimizationResult(
                best_solution=best,
                best_score=float("-inf"),
                final_solutions=SolutionSet([best]),
                generations=GenerationHistory(),
                wall_time_seconds=time.perf_counter() - start,
                n_evaluations=n_evaluations,
                elite_solutions=[],
                diversity_by_generation=[],
                seed=self.seed,
                stop_reason="max_evaluations",
                max_generations=self.max_generations,
                max_evaluations=resolved_policy.max_evaluations,
                telemetry=self.vnext_telemetry,
                direction=self.direction,
                optimizer_type="GeneticAlgorithmOptimizer",
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
            return result

        best = candidate_to_solution(
            self.best_candidate,
            direction=self.direction,
            gene_space=self.gene_space,
        )
        final_solutions = SolutionSet(
            [
                candidate_to_solution(
                    candidate,
                    direction=self.direction,
                    gene_space=self.gene_space,
                )
                for candidate in final_candidates
            ]
        )
        result = OptimizationResult(
            best_solution=best,
            best_score=float(best.score),
            final_solutions=final_solutions,
            generations=GenerationHistory(),
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_solutions=[],
            diversity_by_generation=[],
            seed=self.seed,
            stop_reason="max_evaluations",
            max_generations=self.max_generations,
            max_evaluations=resolved_policy.max_evaluations,
            telemetry=self.vnext_telemetry,
            direction=self.direction,
            optimizer_type="GeneticAlgorithmOptimizer",
            best_candidate_id=self.best_candidate.candidate_id,
            reproducibility=self._reproducibility_metadata(),
            events=self.events,
        )
        append_run_stop_event(
            result.events,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
        return result


__all__ = ["GeneticAlgorithmAskTellMixin"]
