from __future__ import annotations

import math
from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
    is_state_update_confidence,
    score_for_direction,
)
from evocore.results import EventRecord


class CMAESAskTellMixin:
    """Ask/tell lifecycle helpers for CMA-ES."""

    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        candidate = self._candidates_by_id.get(record.candidate_id)
        if candidate is None:
            raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
        if record.batch_id is not None and record.batch_id not in self._batches_by_id:
            raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
        batch = self._batches_by_id.get(candidate.batch_id)
        if batch is None:
            raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
        return candidate, batch

    def _apply_record_confidence(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        trusted_records: list[EvaluationRecord],
    ) -> str:
        if is_state_update_confidence(record.confidence) and (
            self.best_candidate is None
            or candidate.state_comparison_score(self.direction)
            > self.best_candidate.state_comparison_score(self.direction)
        ):
            self.best_candidate = candidate
        if record.confidence == "trusted_full":
            trusted_records.append(record)
            self.vnext_telemetry.record_full(1, stage=record.stage, cost=record.cost)
            return "trusted"
        if record.confidence == "cached":
            self.vnext_telemetry.record_cached(1, stage=record.stage, cost=record.cost)
            return "cached"
        if record.confidence == "partial":
            self.vnext_telemetry.record_partial(1, stage=record.stage, cost=record.cost)
            return "partial"
        if record.confidence == "surrogate":
            self.vnext_telemetry.record_screened(1)
            return "surrogate"
        self.vnext_telemetry.record_eliminated(1, stage=record.stage)
        return "rejected"

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
        samples_discrete = [self._apply_bounds_and_round(sample) for sample in samples_continuous]
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
        counts = {"partial": 0, "surrogate": 0, "cached": 0, "rejected": 0}
        touched_batch_ids: set[str] = set()
        consumed_batch_ids: set[str] = set()
        for record in records:
            candidate, batch = self._candidate_and_batch_for_record(record)
            batch.accept_record(record, reject_consumed_state_record=True)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            self._append_tell_event(candidate, record)
            confidence = self._apply_record_confidence(candidate, record, trusted_records)
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
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=tuple(sorted(consumed_batch_ids)),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
        )


__all__ = ["CMAESAskTellMixin"]
