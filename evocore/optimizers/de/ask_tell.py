from __future__ import annotations

from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import (
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
    is_state_update_confidence,
    is_trusted_confidence,
    solution_to_candidate,
)
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    record_evaluation_telemetry,
)
from evocore.optimizers.de.adaptive import JDEAdaptiveState, JDETrialParameters
from evocore.optimizers.de.kernel import DERustKernelAdapter
from evocore.optimizers.de.strategies import TrialProposal
from evocore.search_space import Solution, decode_gene_values


class DifferentialEvolutionAskTellMixin:
    """Ask/tell lifecycle helpers for Differential Evolution."""

    def _candidate_from_genes(
        self,
        genes: Sequence[float | int | bool],
        *,
        batch_id: str,
        origin: str,
        event_index: int,
        candidate_index: int,
        metadata: dict | None = None,
    ) -> Candidate:
        candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
        candidate = solution_to_candidate(
            Solution(list(genes)),
            gene_space=self.gene_space,
            candidate_id=candidate_id,
            batch_id=batch_id,
            origin=origin,
            event_index=event_index,
        )
        candidate.generation = self.generation
        candidate.metadata.update(dict(metadata or {}))
        return candidate

    def _initial_candidates(self, count: int, event_index: int, batch_id: str) -> list[Candidate]:
        encoded_population = _core.init_population(
            self.gene_space.rust_bounds,
            self.gene_space.kinds,
            count,
            int(_core.py_derive_seed(self.seed, event_index, 0, _core.OP_INIT)),
        )
        return [
            self._candidate_from_genes(
                decode_gene_values(self.gene_space, encoded),
                batch_id=batch_id,
                origin="random",
                event_index=event_index,
                candidate_index=index,
            )
            for index, encoded in enumerate(encoded_population)
        ]

    def _target_candidate(self, slot: int) -> Candidate:
        return self._candidates_by_id[self._target_candidate_ids[slot]]

    def _target_population(self) -> list[Candidate]:
        return [
            self._candidates_by_id[candidate_id] for candidate_id in self._target_candidate_ids
        ]

    def _rust_trial_proposals(self, count: int) -> list[TrialProposal]:
        target_population = self._target_population()
        trial_count = min(int(count), len(target_population))
        target_slots = list(range(trial_count))
        scores = [candidate.best_state_score(self.direction) for candidate in target_population]
        jde_state = None
        to_rust_committed_state = getattr(
            self._de_strategy_state,
            "to_rust_committed_state",
            None,
        )
        if callable(to_rust_committed_state):
            jde_state = to_rust_committed_state()

        return DERustKernelAdapter().generate_trials(
            target_population=target_population,
            scores=scores,
            gene_space=self.gene_space,
            strategy=self.strategy,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
            seed=self.seed,
            generation=self.generation,
            target_slots=target_slots,
            direction=self.direction,
            jde_state=jde_state,
        )

    def _record_pending_strategy_trial(self, candidate: Candidate) -> None:
        if not isinstance(self._de_strategy_state, JDEAdaptiveState):
            return
        self._de_strategy_state.register_pending(
            candidate.candidate_id,
            JDETrialParameters(
                target_slot=int(candidate.metadata["adaptive_slot"]),
                mutation_factor=float(candidate.metadata["mutation_factor"]),
                crossover_rate=float(candidate.metadata["crossover_rate"]),
            ),
        )

    def _complete_pending_strategy_trial(self, candidate_id: str, *, accepted: bool) -> None:
        if isinstance(self._de_strategy_state, JDEAdaptiveState):
            self._de_strategy_state.complete_pending(candidate_id, accepted=accepted)

    def _discard_pending_strategy_trial(self, candidate_id: str) -> None:
        if isinstance(self._de_strategy_state, JDEAdaptiveState):
            self._de_strategy_state.discard_pending(candidate_id)

    def _trial_candidates(self, count: int, event_index: int, batch_id: str) -> list[Candidate]:
        target_count = len(self._target_candidate_ids)
        trial_count = min(count, target_count)
        candidates: list[Candidate] = []
        proposals = self._rust_trial_proposals(trial_count)
        for target_slot, proposal in enumerate(proposals):
            target = self._target_candidate(target_slot)
            genes = proposal.genes
            metadata = dict(proposal.metadata)
            metadata["target_candidate_id"] = target.candidate_id
            candidate = self._candidate_from_genes(
                genes,
                batch_id=batch_id,
                origin="mutation",
                event_index=event_index,
                candidate_index=target_slot,
                metadata=metadata,
            )
            self._trial_target_slots[candidate.candidate_id] = target_slot
            self._trial_target_candidate_ids[candidate.candidate_id] = target.candidate_id
            self._record_pending_strategy_trial(candidate)
            candidates.append(candidate)
        return candidates

    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        append_candidate_ask_events(self.events, candidates, self.gene_space)

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id for batch_id, batch in self._batches_by_id.items() if not batch.consumed
        )

    def ask(self, n: int | None = None) -> list[Candidate]:
        """Return initialization or trial candidates for external evaluation."""
        count = self.population_size if n is None else int(n)
        if count <= 0:
            raise ConfigurationError("ask(n) requires n > 0.")
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        if len(self._target_candidate_ids) < self.population_size:
            needed = self.population_size - len(self._target_candidate_ids)
            candidates = self._initial_candidates(min(count, needed), event_index, batch_id)
        else:
            candidates = self._trial_candidates(count, event_index, batch_id)
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

    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        return candidate_and_batch_for_record(
            record,
            self._candidates_by_id,
            self._batches_by_id,
        )

    def _record_best_candidate(self, candidate: Candidate) -> None:
        if self.best_candidate is None or candidate.state_comparison_score(
            self.direction
        ) > self.best_candidate.state_comparison_score(self.direction):
            self.best_candidate = candidate

    def _apply_telemetry_for_record(self, record: EvaluationRecord) -> str:
        return record_evaluation_telemetry(self.vnext_telemetry, record)

    def _append_tell_event(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        *,
        metadata: dict | None = None,
    ) -> None:
        append_candidate_tell_event(
            self.events,
            candidate,
            record,
            self.gene_space,
            self.direction,
            metadata=metadata,
        )

    def _batch_complete_for_de(self, batch: CandidateBatch) -> bool:
        terminal_candidate_ids = {
            record.candidate_id
            for record in batch.records_by_key.values()
            if is_state_update_confidence(record.confidence) or record.confidence == "rejected"
        }
        return all(candidate_id in terminal_candidate_ids for candidate_id in batch.candidate_ids)

    def _apply_trial_replacement(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        batch: CandidateBatch,
    ) -> AcceptanceDecision:
        target_slot = self._trial_target_slots[candidate.candidate_id]
        target_candidate_id = self._trial_target_candidate_ids[candidate.candidate_id]
        target = self._candidates_by_id[target_candidate_id]
        accepted = candidate.state_comparison_score(
            self.direction
        ) >= target.state_comparison_score(self.direction)
        if accepted:
            self._target_candidate_ids[target_slot] = candidate.candidate_id
            self._record_best_candidate(candidate)
            reason = "trial_replaced_target"
        else:
            self._record_best_candidate(target)
            reason = "trial_kept_target"
        decision = AcceptanceDecision(
            candidate_id=candidate.candidate_id,
            batch_id=batch.batch_id,
            accepted_for_state=accepted,
            reason=reason,
            target_candidate_id=target_candidate_id,
            target_slot=target_slot,
        )
        self._complete_pending_strategy_trial(
            candidate.candidate_id,
            accepted=accepted,
        )
        self._append_tell_event(
            candidate,
            record,
            metadata={
                "accepted_for_state": accepted,
                "acceptance_reason": reason,
                "target_candidate_id": target_candidate_id,
                "target_slot": target_slot,
            },
        )
        return decision

    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
        """Apply evaluation records and return a DE update summary."""
        counts = {
            "trusted": 0,
            "partial": 0,
            "surrogate": 0,
            "cached": 0,
            "constraint_penalty": 0,
            "rejected": 0,
        }
        consumed_batch_ids: set[str] = set()
        acceptance_decisions: list[AcceptanceDecision] = []
        for record in records:
            candidate, batch = self._candidate_and_batch_for_record(record)
            batch.accept_record(record, reject_consumed_state_record=True)
            candidate.apply_record(record)
            confidence = self._apply_telemetry_for_record(record)
            counts[confidence] += 1
            if is_state_update_confidence(record.confidence):
                if not is_trusted_confidence(record.confidence):
                    if candidate.candidate_id in self._trial_target_slots:
                        self._complete_pending_strategy_trial(
                            candidate.candidate_id,
                            accepted=False,
                        )
                    decision = AcceptanceDecision(
                        candidate_id=candidate.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="constraint_penalty_state_record",
                        target_slot=self._trial_target_slots.get(candidate.candidate_id),
                    )
                    acceptance_decisions.append(decision)
                    self._append_tell_event(
                        candidate,
                        record,
                        metadata={
                            "accepted_for_state": True,
                            "acceptance_reason": decision.reason,
                            "target_slot": decision.target_slot,
                        },
                    )
                elif candidate.candidate_id not in self._trial_target_slots:
                    self._target_candidate_ids.append(candidate.candidate_id)
                    self._record_best_candidate(candidate)
                    decision = AcceptanceDecision(
                        candidate_id=candidate.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="initial_target_accepted",
                        target_slot=len(self._target_candidate_ids) - 1,
                    )
                    acceptance_decisions.append(decision)
                    self._append_tell_event(
                        candidate,
                        record,
                        metadata={
                            "accepted_for_state": True,
                            "acceptance_reason": decision.reason,
                            "target_slot": decision.target_slot,
                        },
                    )
                else:
                    decision = self._apply_trial_replacement(candidate, record, batch)
                    acceptance_decisions.append(decision)
            else:
                if record.confidence == "rejected":
                    self._discard_pending_strategy_trial(candidate.candidate_id)
                self._append_tell_event(
                    candidate,
                    record,
                    metadata={
                        "accepted_for_state": False,
                        "acceptance_reason": "record_not_state_eligible",
                    },
                )
            if self._batch_complete_for_de(batch):
                batch.consumed = True
                consumed_batch_ids.add(batch.batch_id)
                if all(
                    candidate_id in self._trial_target_slots
                    for candidate_id in batch.candidate_ids
                ):
                    self.generation += 1
                for candidate_id in batch.candidate_ids:
                    self._trial_target_slots.pop(candidate_id, None)
                    self._trial_target_candidate_ids.pop(candidate_id, None)
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=len(records),
            trusted_count=counts["trusted"],
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
            state_accepted_count=sum(
                1 for decision in acceptance_decisions if decision.accepted_for_state
            ),
        )
