from __future__ import annotations

from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.lifecycle import (
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
    is_state_update_confidence,
    is_trusted_confidence,
    score_for_direction,
)
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    record_evaluation_telemetry,
)


class CMAESAskTellMixin:
    """Ask/tell lifecycle helpers for CMA-ES."""

    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        return candidate_and_batch_for_record(
            record,
            self._candidates_by_id,
            self._batches_by_id,
        )

    def _apply_record_confidence(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        trusted_records: list[EvaluationRecord],
    ) -> str:
        if is_trusted_confidence(record.confidence) and (
            self.best_candidate is None
            or candidate.state_comparison_score(self.direction)
            > self.best_candidate.state_comparison_score(self.direction)
        ):
            self.best_candidate = candidate
        if record.confidence == "trusted_full":
            trusted_records.append(record)
        return record_evaluation_telemetry(self.vnext_telemetry, record)

    def _consume_complete_batch(self, batch: CandidateBatch) -> bool:
        ordered_records = batch.ordered_state_update_records()
        if ordered_records is None or batch.consumed:
            return False
        samples = []
        fitnesses = []
        for record in ordered_records:
            sample = batch.continuous_samples_by_id.get(record.candidate_id)
            if sample is None:
                raise FitnessError(
                    f"missing continuous sample for candidate_id {record.candidate_id!r}."
                )
            samples.append(sample)
            if record.score is None:
                raise FitnessError(
                    f"trusted_full record for candidate_id {record.candidate_id!r} is missing score."
                )
            fitnesses.append(score_for_direction(float(record.score), self.direction))
        self._ensure_state().tell(samples, fitnesses)
        batch.consumed = True
        return True

    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed CMA candidates."""
        append_candidate_ask_events(self.events, candidates, self.gene_space)

    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        append_candidate_tell_event(
            self.events,
            candidate,
            record,
            self.gene_space,
            self.direction,
        )

    def ask(self, n: int | None = None) -> list[Candidate]:
        """Return a CMA candidate batch."""
        if n is not None and int(n) != self.population_size:
            raise ConfigurationError(
                "CMAESOptimizer.ask currently requires n to equal population_size."
            )
        state = self._ensure_state()
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        samples_continuous = state.ask(self.seed, event_index)
        samples_discrete = [
            self._apply_integer_strategy(
                sample,
                event_index=event_index,
                candidate_index=index,
            )
            for index, sample in enumerate(samples_continuous)
        ]
        candidates: list[Candidate] = []
        continuous_samples_by_id: dict[str, list[float]] = {}
        for index, sample in enumerate(samples_discrete):
            solution = self._decode_solution(sample)
            candidate_id = _core.candidate_id(self.seed, event_index, index)
            candidate = Candidate(
                candidate_id=candidate_id,
                genes=list(solution.values),
                batch_id=batch_id,
                params=solution.params,
                origin="cma_sample",
                event_index=event_index,
            )
            continuous_samples_by_id[candidate_id] = list(samples_continuous[index])
            self._candidates_by_id[candidate_id] = candidate
            candidates.append(candidate)
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            continuous_samples_by_id=continuous_samples_by_id,
        )
        self._event_index += 1
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)
        return candidates

    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
        """Update CMA state from trusted evaluation records."""
        trusted_records: list[EvaluationRecord] = []
        acceptance_decisions: list[AcceptanceDecision] = []
        counts = {
            "partial": 0,
            "surrogate": 0,
            "cached": 0,
            "constraint_penalty": 0,
            "rejected": 0,
        }
        touched_batch_ids: set[str] = set()
        consumed_batch_ids: set[str] = set()
        for record in records:
            candidate, batch = self._candidate_and_batch_for_record(record)
            batch.accept_record(record, reject_consumed_state_record=True)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            self._append_tell_event(candidate, record)
            confidence = self._apply_record_confidence(candidate, record, trusted_records)
            if is_state_update_confidence(record.confidence):
                acceptance_decisions.append(
                    AcceptanceDecision(
                        candidate_id=record.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="state_record_accepted",
                    )
                )
            if confidence in counts:
                counts[confidence] += 1

        for batch_id in touched_batch_ids:
            batch = self._batches_by_id[batch_id]
            if self._consume_complete_batch(batch):
                consumed_batch_ids.add(batch.batch_id)

        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=len(records),
            trusted_count=len(trusted_records),
            partial_count=counts["partial"],
            surrogate_count=counts["surrogate"],
            cached_count=counts["cached"],
            rejected_count=counts["rejected"],
            penalty_count=counts["constraint_penalty"],
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=tuple(sorted(consumed_batch_ids)),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
            acceptance_decisions=tuple(acceptance_decisions),
            state_accepted_count=len(acceptance_decisions),
        )


__all__ = ["CMAESAskTellMixin"]
