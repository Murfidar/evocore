from __future__ import annotations

from collections.abc import Mapping, Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import (
    TRUSTED_CONFIDENCES,
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    CandidateOrigin,
    EvaluationConfidence,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
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
from evocore.search_space import encode_gene_values, repair_gene_values


class CMAESExternalStateMixin:
    """External-state integration API for CMAESOptimizer."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        """Return CMA-ES external-state support flags."""
        return ExternalStateCapabilities(
            warm_start_before_ask=True,
            warm_start_after_ask=False,
            proposed_candidate_injection=False,
            state_candidate_injection=False,
            tracked_only_injection=True,
            population_snapshots=True,
            top_candidate_snapshots=True,
            cached_record_helpers=True,
        )

    def _external_known_candidates(self) -> list[Candidate]:
        return list(self._candidates_by_id.values())

    def _external_pending_candidates(self) -> list[Candidate]:
        pending_batch_ids = set(self._pending_batch_ids())
        return [
            candidate
            for candidate in self._candidates_by_id.values()
            if candidate.batch_id in pending_batch_ids
        ]

    def _external_scored_candidates(self) -> list[Candidate]:
        return [candidate for candidate in self._candidates_by_id.values() if candidate.scores]

    def _external_is_trusted_candidate(self, candidate: Candidate) -> bool:
        if candidate.metadata.get("external_state_mode") == "tracked":
            return False
        return any(score.confidence in TRUSTED_CONFIDENCES for score in candidate.scores.values())

    def _external_trusted_candidates(self) -> list[Candidate]:
        return [
            candidate
            for candidate in self._candidates_by_id.values()
            if self._external_is_trusted_candidate(candidate)
        ]

    def _trusted_count(self) -> int:
        return len(self._external_trusted_candidates())

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

    def candidate_snapshot(
        self,
        *,
        scope: SnapshotScope = "trusted",
    ) -> PopulationSnapshot:
        """Return a detached snapshot of CMA-ES candidate state."""
        return build_population_snapshot(
            optimizer_type="CMAESOptimizer",
            direction=self.direction,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            candidates=self._external_candidates_for_scope(scope),
            gene_space=self.gene_space,
            telemetry=self.vnext_telemetry,
            metadata={"scope": scope},
        )

    def top_candidates(
        self,
        k: int = 10,
        *,
        scope: SnapshotScope = "trusted",
        confidence: tuple[EvaluationConfidence, ...] = TRUSTED_CONFIDENCES,
    ) -> tuple[CandidateSnapshot, ...]:
        """Return top-k detached CMA-ES candidate snapshots."""
        return top_candidate_snapshots(
            self._external_candidates_for_scope(scope),
            k=k,
            gene_space=self.gene_space,
            direction=self.direction,
            confidence=confidence,
        )

    def _external_records_to_candidates(
        self,
        records: Sequence[WarmStartRecord],
        *,
        event_index: int,
        batch_id: str,
        origin: CandidateOrigin,
        mode: str,
        metadata: Mapping[str, object] | None = None,
        deduplicate: bool,
    ) -> tuple[list[tuple[Candidate, WarmStartRecord]], list[Candidate]]:
        known_hashes = self._external_existing_hashes()
        accepted_hashes: dict[str, Candidate] = {}
        accepted: list[tuple[Candidate, WarmStartRecord]] = []
        skipped: list[Candidate] = []
        for record in records:
            values = resolve_warm_start_values(record, self.gene_space)
            candidate_hash = self.gene_space.value_hash(values)
            duplicate = known_hashes.get(candidate_hash) or accepted_hashes.get(candidate_hash)
            if deduplicate and duplicate is not None:
                skipped.append(duplicate)
                continue
            candidate_id = _core.candidate_id(self.seed, event_index, len(accepted))
            candidate_metadata = dict(record.metadata)
            candidate_metadata.update({**dict(metadata or {}), "external_state_mode": mode})
            candidate = Candidate(
                candidate_id=candidate_id,
                genes=list(values),
                batch_id=batch_id,
                params=self.gene_space.params_for(values),
                origin=origin,
                event_index=event_index,
                generation=self.generation,
                metadata=candidate_metadata,
            )
            accepted.append((candidate, record))
            accepted_hashes[candidate_hash] = candidate
        return accepted, skipped

    def _apply_external_records(
        self,
        accepted: Sequence[tuple[Candidate, WarmStartRecord]],
        *,
        batch: CandidateBatch,
        update_best: bool,
        decision_reason: str,
    ) -> tuple[int, int, list[AcceptanceDecision]]:
        trusted = cached = 0
        decisions: list[AcceptanceDecision] = []
        for candidate, warm_record in accepted:
            record = EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=batch.batch_id,
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
            if update_best:
                if self.best_candidate is None or candidate.state_comparison_score(
                    self.direction
                ) > self.best_candidate.state_comparison_score(self.direction):
                    self.best_candidate = candidate
                decisions.append(
                    AcceptanceDecision(
                        candidate_id=candidate.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason=decision_reason,
                    )
                )
        return trusted, cached, decisions

    def _set_initial_mean_from_warm_start(
        self,
        accepted: Sequence[tuple[Candidate, WarmStartRecord]],
        *,
        cma_mean_strategy: CmaMeanStrategy,
        top_k: int | None,
    ) -> None:
        ranked = sorted(
            (candidate for candidate, _ in accepted),
            key=lambda candidate: candidate.state_comparison_score(self.direction),
            reverse=True,
        )
        if not ranked:
            return
        if cma_mean_strategy == "best":
            self.initial_mean = encode_gene_values(self.gene_space, ranked[0].genes)
            return
        count = len(ranked) if top_k is None else int(top_k)
        if count <= 0:
            raise ConfigurationError("top_k must be positive when provided.")
        selected = ranked[:count]
        centroid = [
            sum(float(candidate.genes[index]) for candidate in selected) / len(selected)
            for index in range(self.gene_space.length)
        ]
        repaired = repair_gene_values(self.gene_space, centroid)
        self.initial_mean = encode_gene_values(self.gene_space, repaired)

    def warm_start(
        self,
        records: Sequence[WarmStartRecord],
        *,
        deduplicate: bool = True,
        mode: WarmStartMode = "state",
        cma_mean_strategy: CmaMeanStrategy = "best",
        top_k: int | None = None,
    ) -> UpdateResult:
        """Initialize or track CMA-ES state from scored external candidates."""
        if mode not in ("state", "tracked"):
            raise ConfigurationError("warm_start mode must be 'state' or 'tracked'.")
        if cma_mean_strategy not in ("best", "top_k_centroid"):
            raise ConfigurationError("cma_mean_strategy must be 'best' or 'top_k_centroid'.")
        if top_k is not None and int(top_k) <= 0:
            raise ConfigurationError("top_k must be positive when provided.")
        if mode == "state" and self._state is not None:
            raise ConfigurationError(
                "warm_start(mode='state') must run before the first CMA-ES ask."
            )

        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        accepted, _ = self._external_records_to_candidates(
            records,
            event_index=event_index,
            batch_id=batch_id,
            origin="memory_seed",
            mode=mode,
            deduplicate=deduplicate,
        )
        if not accepted:
            best_candidate_id, best_score = self._best_candidate_id_and_score()
            return UpdateResult(
                accepted_count=0,
                trusted_count=0,
                partial_count=0,
                surrogate_count=0,
                cached_count=0,
                rejected_count=0,
                best_candidate_id=best_candidate_id,
                best_score=best_score,
                pending_batch_ids=self._pending_batch_ids(),
                telemetry=self.vnext_telemetry,
            )

        candidates = [candidate for candidate, _ in accepted]
        batch = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            consumed=True,
        )
        self._batches_by_id[batch_id] = batch
        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)
        trusted, cached, decisions = self._apply_external_records(
            accepted,
            batch=batch,
            update_best=mode == "state",
            decision_reason="warm_start_initial_mean",
        )
        if mode == "state":
            self._set_initial_mean_from_warm_start(
                accepted,
                cma_mean_strategy=cma_mean_strategy,
                top_k=top_k,
            )
        self._event_index += 1
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=len(accepted),
            trusted_count=trusted,
            partial_count=0,
            surrogate_count=0,
            cached_count=cached,
            rejected_count=0,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=(batch_id,),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
            acceptance_decisions=tuple(decisions),
            state_accepted_count=len(decisions),
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
        """Track externally supplied CMA-ES candidates without mutating covariance state."""
        if mode != "tracked":
            raise ConfigurationError(
                "CMAESOptimizer supports inject_candidates(mode='tracked') in Phase 1."
            )

        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        accepted, skipped = self._external_records_to_candidates(
            records,
            event_index=event_index,
            batch_id=batch_id,
            origin=origin,
            mode=mode,
            metadata=metadata,
            deduplicate=deduplicate,
        )
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

        candidates = [candidate for candidate, _ in accepted]
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            consumed=True,
        )
        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)
        self._event_index += 1
        return InjectionResult(
            accepted=tuple(
                build_candidate_snapshot(
                    candidate,
                    gene_space=self.gene_space,
                    direction=self.direction,
                )
                for candidate in candidates
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


__all__ = ["CMAESExternalStateMixin"]
