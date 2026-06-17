from __future__ import annotations

from collections.abc import Mapping, Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import (
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    CandidateOrigin,
    EvaluationConfidence,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
    solution_to_candidate,
)
from evocore.lifecycle.ask_tell_helpers import record_evaluation_telemetry
from evocore.lifecycle.external import (
    CandidateSnapshot,
    CmaMeanStrategy,
    ExternalStateCapabilities,
    InjectionMode,
    InjectionResult,
    PopulationSnapshot,
    SnapshotScope,
    WarmStartMode,
    WarmStartRecord,
    build_candidate_snapshot,
    build_population_snapshot,
    resolve_warm_start_values,
    top_candidate_snapshots,
)
from evocore.search_space import Solution


class GeneticAlgorithmExternalStateMixin:
    """External-state integration API for GeneticAlgorithmOptimizer."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        """Return GA external-state support flags."""
        return ExternalStateCapabilities(
            warm_start_before_ask=True,
            warm_start_after_ask=True,
            proposed_candidate_injection=True,
            state_candidate_injection=False,
            tracked_only_injection=True,
            population_snapshots=True,
            top_candidate_snapshots=True,
            cached_record_helpers=True,
        )

    def _external_known_candidates(self) -> list[Candidate]:
        return list(self._candidates_by_id.values())

    def _external_trusted_candidates(self) -> list[Candidate]:
        return list(self._trusted_population_vnext)

    def _external_pending_candidates(self) -> list[Candidate]:
        pending_batch_ids = set(self._pending_batch_ids())
        return [
            candidate
            for candidate in self._candidates_by_id.values()
            if candidate.batch_id in pending_batch_ids
        ]

    def _external_scored_candidates(self) -> list[Candidate]:
        return [candidate for candidate in self._candidates_by_id.values() if candidate.scores]

    def _external_candidates_for_scope(self, scope: SnapshotScope) -> list[Candidate]:
        if scope == "trusted":
            return self._external_trusted_candidates()
        if scope == "known":
            return self._external_known_candidates()
        if scope == "pending":
            return self._external_pending_candidates()
        if scope == "scored":
            return self._external_scored_candidates()
        raise ConfigurationError(
            "candidate snapshot scope must be 'trusted', 'known', 'pending', or 'scored'."
        )

    def _external_existing_hashes(self) -> dict[str, Candidate]:
        return {
            self.gene_space.value_hash(candidate.genes): candidate
            for candidate in self._candidates_by_id.values()
        }

    def _external_candidate_from_record(
        self,
        record: WarmStartRecord,
        *,
        batch_id: str,
        event_index: int,
        candidate_index: int,
        origin: CandidateOrigin,
        metadata: Mapping[str, object] | None = None,
    ) -> Candidate:
        values = resolve_warm_start_values(record, self.gene_space)
        candidate_metadata = dict(metadata or {})
        candidate_metadata.update(dict(record.metadata))
        return solution_to_candidate(
            Solution(list(values)),
            gene_space=self.gene_space,
            candidate_id=_core.candidate_id(self.seed, event_index, candidate_index),
            batch_id=batch_id,
            origin=origin,
            event_index=event_index,
            metadata=candidate_metadata,
        )

    def candidate_snapshot(
        self,
        *,
        scope: SnapshotScope = "trusted",
    ) -> PopulationSnapshot:
        """Return a detached snapshot of GA candidate state."""
        candidates = self._external_candidates_for_scope(scope)
        return build_population_snapshot(
            optimizer_type="GeneticAlgorithmOptimizer",
            direction=self.direction,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=len(self._trusted_population_vnext),
            candidates=candidates,
            gene_space=self.gene_space,
            telemetry=self.vnext_telemetry,
            metadata={"scope": scope},
        )

    def top_candidates(
        self,
        k: int = 10,
        *,
        scope: SnapshotScope = "trusted",
        confidence: tuple[EvaluationConfidence, ...] = ("trusted_full", "cached"),
    ) -> tuple[CandidateSnapshot, ...]:
        """Return top-k detached GA candidate snapshots."""
        return top_candidate_snapshots(
            self._external_candidates_for_scope(scope),
            k=k,
            gene_space=self.gene_space,
            direction=self.direction,
            confidence=confidence,
        )

    def _external_update_result(
        self,
        *,
        accepted_count: int,
        trusted_count: int,
        cached_count: int,
        partial_count: int = 0,
        surrogate_count: int = 0,
        rejected_count: int = 0,
        acceptance_decisions: Sequence[AcceptanceDecision] = (),
    ) -> UpdateResult:
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=accepted_count,
            trusted_count=trusted_count,
            partial_count=partial_count,
            surrogate_count=surrogate_count,
            cached_count=cached_count,
            rejected_count=rejected_count,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=(),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
            acceptance_decisions=tuple(acceptance_decisions),
            state_accepted_count=len(acceptance_decisions),
        )

    def warm_start(  # noqa: PLR0912
        self,
        records: Sequence[WarmStartRecord],
        *,
        deduplicate: bool = True,
        mode: WarmStartMode = "state",
        cma_mean_strategy: CmaMeanStrategy = "best",
        top_k: int | None = None,
    ) -> UpdateResult:
        """Initialize or track GA state from scored external candidates."""
        if mode not in ("state", "tracked"):
            raise ConfigurationError("warm_start mode must be 'state' or 'tracked'.")
        if cma_mean_strategy not in ("best", "top_k_centroid"):
            raise ConfigurationError("cma_mean_strategy must be 'best' or 'top_k_centroid'.")
        if top_k is not None and int(top_k) <= 0:
            raise ConfigurationError("top_k must be positive when provided.")

        known_hashes = self._external_existing_hashes()
        accepted_hashes: dict[str, Candidate] = {}
        accepted: list[tuple[Candidate, WarmStartRecord]] = []
        skipped: list[Candidate] = []
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)

        for record in records:
            values = resolve_warm_start_values(record, self.gene_space)
            candidate_hash = self.gene_space.value_hash(values)
            duplicate = known_hashes.get(candidate_hash) or accepted_hashes.get(candidate_hash)
            if deduplicate and duplicate is not None:
                skipped.append(duplicate)
                continue
            candidate = self._external_candidate_from_record(
                record,
                batch_id=batch_id,
                event_index=event_index,
                candidate_index=len(accepted),
                origin="memory_seed",
                metadata={"external_state_mode": mode},
            )
            accepted.append((candidate, record))
            accepted_hashes[candidate_hash] = candidate

        if not accepted:
            return self._external_update_result(
                accepted_count=0,
                trusted_count=0,
                cached_count=0,
                acceptance_decisions=(),
            )

        candidates = [candidate for candidate, _ in accepted]
        batch = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        )
        self._batches_by_id[batch_id] = batch
        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)

        trusted = cached = 0
        acceptance_decisions: list[AcceptanceDecision] = []
        for candidate, warm_record in accepted:
            record = EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=batch_id,
                score=warm_record.score,
                confidence=warm_record.confidence,
                stage=warm_record.stage,
                cost=warm_record.cost,
                metrics=dict(warm_record.metrics),
                metadata=dict(warm_record.metadata),
            )
            batch.accept_record(record)
            candidate.apply_record(record)
            self._append_tell_event(candidate, record)
            label = record_evaluation_telemetry(self.vnext_telemetry, record)
            if label == "trusted":
                trusted += 1
            elif label == "cached":
                cached += 1
            if mode == "state":
                self._record_state_candidate(candidate)
                acceptance_decisions.append(
                    AcceptanceDecision(
                        candidate_id=candidate.candidate_id,
                        batch_id=batch_id,
                        accepted_for_state=True,
                        reason="warm_start_state_accepted",
                    )
                )

        if mode == "state":
            self._trusted_population_vnext.sort(
                key=lambda candidate: candidate.state_comparison_score(self.direction),
                reverse=True,
            )
            self._trusted_population_vnext = self._trusted_population_vnext[: self.population_size]
            if self._trusted_population_vnext:
                self.best_candidate = self._trusted_population_vnext[0]
        self._event_index += 1

        return self._external_update_result(
            accepted_count=len(accepted),
            trusted_count=trusted,
            cached_count=cached,
            acceptance_decisions=acceptance_decisions,
        )

    def inject_candidates(
        self,
        records: Sequence[WarmStartRecord],
        *,
        origin: CandidateOrigin = "memory_seed",
        mode: InjectionMode = "proposed",
        deduplicate: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> InjectionResult:
        """Inject proposed or tracked candidates into the GA lifecycle."""
        if mode not in ("proposed", "tracked"):
            raise ConfigurationError("inject_candidates mode must be 'proposed' or 'tracked'.")

        known_hashes = self._external_existing_hashes()
        accepted_hashes: dict[str, Candidate] = {}
        accepted: list[Candidate] = []
        skipped: list[Candidate] = []
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)

        for record in records:
            values = resolve_warm_start_values(record, self.gene_space)
            candidate_hash = self.gene_space.value_hash(values)
            duplicate = known_hashes.get(candidate_hash) or accepted_hashes.get(candidate_hash)
            if deduplicate and duplicate is not None:
                skipped.append(duplicate)
                continue
            candidate = self._external_candidate_from_record(
                record,
                batch_id=batch_id,
                event_index=event_index,
                candidate_index=len(accepted),
                origin=origin,
                metadata={
                    "external_state_mode": mode,
                    **dict(metadata or {}),
                },
            )
            accepted.append(candidate)
            accepted_hashes[candidate_hash] = candidate

        if not accepted:
            return InjectionResult(
                accepted=(),
                skipped_duplicates=tuple(
                    build_candidate_snapshot(
                        candidate,
                        gene_space=self.gene_space,
                        direction=self.direction,
                    )
                    for candidate in skipped
                ),
                rejected=(),
            )

        for candidate in accepted:
            self._candidates_by_id[candidate.candidate_id] = candidate
        if mode == "proposed":
            self._batches_by_id[batch_id] = CandidateBatch(
                batch_id=batch_id,
                candidate_ids=tuple(candidate.candidate_id for candidate in accepted),
            )
        self.vnext_telemetry.record_proposed_candidates(accepted, gene_space=self.gene_space)
        self._append_ask_events(accepted)
        self._event_index += 1

        return InjectionResult(
            accepted=tuple(
                build_candidate_snapshot(
                    candidate,
                    gene_space=self.gene_space,
                    direction=self.direction,
                )
                for candidate in accepted
            ),
            skipped_duplicates=tuple(
                build_candidate_snapshot(
                    candidate,
                    gene_space=self.gene_space,
                    direction=self.direction,
                )
                for candidate in skipped
            ),
            rejected=(),
        )


__all__ = ["GeneticAlgorithmExternalStateMixin"]
