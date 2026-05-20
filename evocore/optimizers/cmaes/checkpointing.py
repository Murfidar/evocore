from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from evocore import _core
from evocore.core.errors import CheckpointError
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
from evocore.results import CheckpointSnapshot, validate_checkpoint_identity
from evocore.results import load_checkpoint as load_checkpoint_payload
from evocore.results import save_checkpoint as save_checkpoint_payload

CMAES_ASK_TELL_STATE_KIND = "cmaes_ask_tell"
CMAES_CHECKPOINT_STATE_SCHEMA_VERSION = 1


class CMAESCheckpointingMixin:
    """Stable checkpoint helpers for CMA-ES ask/tell workflows."""

    @staticmethod
    def load_checkpoint(checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
        """Load a stable checkpoint file."""
        return load_checkpoint_payload(checkpoint)

    @staticmethod
    def save_checkpoint(
        checkpoint: str | os.PathLike[str],
        snapshot: CheckpointSnapshot | Mapping[str, Any],
    ) -> None:
        """Save a stable checkpoint file."""
        save_checkpoint_payload(checkpoint, snapshot)

    def _validate_stable_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        validate_checkpoint_identity(
            payload,
            optimizer_type="CMAESOptimizer",
            gene_space_hash=self.gene_space.hash(),
            optimizer_config_hash=self.config_hash(),
            seed=self.seed,
            direction=self.direction,
        )

    def ask_tell_checkpoint(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        """Return a stable CMA-ES ask/tell runtime checkpoint snapshot."""
        best_candidate_id = (
            None if self.best_candidate is None else self.best_candidate.candidate_id
        )
        state_payload = {
            "state_kind": CMAES_ASK_TELL_STATE_KIND,
            "event_index": self._event_index,
            "cmaes_state": self._ensure_state().to_dict(),
            "candidates_by_id": {
                candidate_id: candidate_to_checkpoint(candidate)
                for candidate_id, candidate in sorted(self._candidates_by_id.items())
            },
            "batches_by_id": {
                batch_id: batch_to_checkpoint(batch)
                for batch_id, batch in sorted(self._batches_by_id.items())
            },
            "best_candidate_id": best_candidate_id,
            "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            "events": event_history_to_checkpoint(self.events),
        }
        return CheckpointSnapshot(
            optimizer_type="CMAESOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "mode": "ask_tell",
                "event_index": self._event_index,
                "pending_batch_ids": list(self._pending_batch_ids()),
                "best_candidate_id": best_candidate_id,
            },
            state={
                "optimizer_type": "CMAESOptimizer",
                "schema_version": CMAES_CHECKPOINT_STATE_SCHEMA_VERSION,
                "payload": state_payload,
            },
            audit={
                "events": event_history_to_checkpoint(self.events),
                "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            },
            metadata=dict(metadata or {}),
        )

    def _ask_tell_payload_from_checkpoint(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state = payload["state"]
        if state.get("schema_version") != CMAES_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise CheckpointError("checkpoint state.schema_version must be 1.")
        state_payload = state["payload"]
        if state_payload.get("state_kind") != CMAES_ASK_TELL_STATE_KIND:
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by "
                "CMA-ES ask/tell resume."
            )
        return state_payload

    def _cmaes_state_from_checkpoint(
        self,
        state_payload: Mapping[str, Any],
    ) -> _core.PyCMAESState:
        cmaes_state_payload = state_payload.get("cmaes_state")
        if not isinstance(cmaes_state_payload, Mapping):
            raise CheckpointError("checkpoint state.payload.cmaes_state must be an object.")
        try:
            return _core.PyCMAESState.from_dict(cmaes_state_payload)
        except ValueError as exc:
            raise CheckpointError(
                f"checkpoint state.payload.cmaes_state is invalid: {exc}"
            ) from exc

    def _candidates_from_checkpoint(
        self,
        state_payload: Mapping[str, Any],
    ) -> dict[str, Candidate]:
        raw_candidates = state_payload.get("candidates_by_id")
        if not isinstance(raw_candidates, Mapping):
            raise CheckpointError("checkpoint state.payload.candidates_by_id must be an object.")
        candidates = {
            str(candidate_id): candidate_from_checkpoint(candidate_payload)
            for candidate_id, candidate_payload in raw_candidates.items()
        }
        for candidate_id, candidate in candidates.items():
            if candidate.candidate_id != candidate_id:
                raise CheckpointError(
                    f"checkpoint candidate key {candidate_id!r} does not match "
                    f"candidate_id {candidate.candidate_id!r}."
                )
        return candidates

    def _batches_from_checkpoint(
        self,
        state_payload: Mapping[str, Any],
        candidates: Mapping[str, Candidate],
    ) -> dict[str, CandidateBatch]:
        raw_batches = state_payload.get("batches_by_id")
        if not isinstance(raw_batches, Mapping):
            raise CheckpointError("checkpoint state.payload.batches_by_id must be an object.")
        batches = {
            str(batch_id): batch_from_checkpoint(batch_payload)
            for batch_id, batch_payload in raw_batches.items()
        }
        for batch_id, batch in batches.items():
            if batch.batch_id != batch_id:
                raise CheckpointError(
                    f"checkpoint batch key {batch_id!r} does not match "
                    f"batch_id {batch.batch_id!r}."
                )
            for candidate_id in batch.candidate_ids:
                if candidate_id not in candidates:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} references unknown "
                        f"candidate_id {candidate_id!r}."
                    )
            if not batch.consumed:
                missing_samples = [
                    candidate_id
                    for candidate_id in batch.candidate_ids
                    if candidate_id not in batch.continuous_samples_by_id
                ]
                if missing_samples:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} is missing continuous samples "
                        f"for candidate_ids: {missing_samples!r}."
                    )
        return batches

    def _best_candidate_id_from_checkpoint(
        self,
        state_payload: Mapping[str, Any],
        candidates: Mapping[str, Candidate],
    ) -> str | None:
        best_candidate_id = state_payload.get("best_candidate_id")
        if best_candidate_id is not None and best_candidate_id not in candidates:
            raise CheckpointError(
                f"checkpoint best_candidate_id {best_candidate_id!r} is unknown."
            )
        return best_candidate_id

    def _event_index_from_checkpoint(self, state_payload: Mapping[str, Any]) -> int:
        try:
            event_index = int(state_payload.get("event_index", 0))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint state.payload.event_index must be a non-negative integer."
            ) from exc
        if event_index < 0:
            raise CheckpointError(
                "checkpoint state.payload.event_index must be a non-negative integer."
            )
        return event_index

    def _restore_ask_tell_state(self, state_payload: Mapping[str, Any]) -> None:
        cmaes_state = self._cmaes_state_from_checkpoint(state_payload)
        candidates = self._candidates_from_checkpoint(state_payload)
        batches = self._batches_from_checkpoint(state_payload, candidates)
        best_candidate_id = self._best_candidate_id_from_checkpoint(state_payload, candidates)
        event_index = self._event_index_from_checkpoint(state_payload)

        self._state = cmaes_state
        self._candidates_by_id = candidates
        self._batches_by_id = batches
        self.best_candidate = None if best_candidate_id is None else candidates[best_candidate_id]
        self.vnext_telemetry = telemetry_from_checkpoint(state_payload.get("telemetry") or {})
        self.events = event_history_from_checkpoint(state_payload.get("events") or [])
        self._event_index = event_index

    def resume_ask_tell_checkpoint(
        self,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    ):
        """Resume CMA-ES ask/tell runtime state from a stable checkpoint."""
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        state_payload = self._ask_tell_payload_from_checkpoint(payload)
        self._restore_ask_tell_state(state_payload)
        return self.state_summary()


__all__ = [
    "CMAES_ASK_TELL_STATE_KIND",
    "CMAES_CHECKPOINT_STATE_SCHEMA_VERSION",
    "CMAESCheckpointingMixin",
]
