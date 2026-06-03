from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.lifecycle import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    is_state_update_confidence,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
from evocore.optimizers.de.adaptive import (
    strategy_state_from_checkpoint,
    strategy_state_to_checkpoint,
)
from evocore.results import CheckpointSnapshot, validate_checkpoint_identity
from evocore.results import load_checkpoint as load_checkpoint_payload
from evocore.results import save_checkpoint as save_checkpoint_payload

DE_ASK_TELL_STATE_KIND = "de_ask_tell"
DE_CHECKPOINT_STATE_SCHEMA_VERSION = 1


def _required_payload_value(payload: Mapping[str, Any], key: str) -> object:
    if key not in payload:
        raise CheckpointError(f"checkpoint state.payload.{key} is required.")
    return payload[key]


def _required_mapping_payload(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required_payload_value(payload, key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an object.")
    return value


def _required_sequence_payload(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = _required_payload_value(payload, key)
    if not isinstance(value, list | tuple):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an array.")
    return list(value)


def _required_int_payload(payload: Mapping[str, Any], key: str) -> int:
    value = _required_payload_value(payload, key)
    if isinstance(value, bool):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint state.payload.{key} must be an integer.") from exc


class DifferentialEvolutionCheckpointingMixin:
    """Stable checkpoint helpers for DE ask/tell workflows."""

    @staticmethod
    def load_checkpoint(checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
        """Load a stable checkpoint file."""
        return load_checkpoint_payload(checkpoint)

    @staticmethod
    def save_checkpoint(
        checkpoint: str | os.PathLike[str],
        snapshot: CheckpointSnapshot | Mapping[str, Any],
    ) -> None:
        """Write a stable checkpoint file."""
        save_checkpoint_payload(checkpoint, snapshot)

    def _validate_stable_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        validate_checkpoint_identity(
            payload,
            optimizer_type="DifferentialEvolutionOptimizer",
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
        """Return a stable DE ask/tell runtime checkpoint snapshot."""
        best_candidate_id = (
            None if self.best_candidate is None else self.best_candidate.candidate_id
        )
        state_payload = {
            "state_kind": DE_ASK_TELL_STATE_KIND,
            "event_index": self._event_index,
            "generation": self.generation,
            "candidates_by_id": {
                candidate_id: candidate_to_checkpoint(candidate)
                for candidate_id, candidate in sorted(self._candidates_by_id.items())
            },
            "batches_by_id": {
                batch_id: batch_to_checkpoint(batch)
                for batch_id, batch in sorted(self._batches_by_id.items())
            },
            "target_candidate_ids": list(self._target_candidate_ids),
            "trial_target_slots": dict(sorted(self._trial_target_slots.items())),
            "trial_target_candidate_ids": dict(sorted(self._trial_target_candidate_ids.items())),
            "best_candidate_id": best_candidate_id,
            "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            "events": event_history_to_checkpoint(self.events),
            "strategy_state": strategy_state_to_checkpoint(self._de_strategy_state),
        }
        return CheckpointSnapshot(
            optimizer_type="DifferentialEvolutionOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "mode": "ask_tell",
                "event_index": self._event_index,
                "generation": self.generation,
                "pending_batch_ids": list(self._pending_batch_ids()),
                "best_candidate_id": best_candidate_id,
            },
            state={
                "optimizer_type": "DifferentialEvolutionOptimizer",
                "schema_version": DE_CHECKPOINT_STATE_SCHEMA_VERSION,
                "payload": state_payload,
            },
            audit={
                "events": event_history_to_checkpoint(self.events),
                "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            },
            metadata=dict(metadata or {}),
        )

    def _ask_tell_payload_from_checkpoint(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        state = payload["state"]
        if state.get("schema_version") != DE_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise CheckpointError("checkpoint state.schema_version must be 1.")
        state_payload = state["payload"]
        if state_payload.get("state_kind") != DE_ASK_TELL_STATE_KIND:
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by DE ask/tell resume."
            )
        return state_payload

    def _restore_ask_tell_state(self, state_payload: Mapping[str, Any]) -> None:  # noqa: PLR0912
        raw_candidates = _required_mapping_payload(state_payload, "candidates_by_id")
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

        raw_batches = _required_mapping_payload(state_payload, "batches_by_id")
        batches = {
            str(batch_id): batch_from_checkpoint(batch_payload)
            for batch_id, batch_payload in raw_batches.items()
        }
        for batch_id, batch in batches.items():
            if batch.batch_id != batch_id:
                raise CheckpointError(
                    f"checkpoint batch key {batch_id!r} does not match batch_id {batch.batch_id!r}."
                )
            for candidate_id in batch.candidate_ids:
                if candidate_id not in candidates:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} references unknown candidate_id {candidate_id!r}."
                    )
        terminal_record_candidate_ids = {
            record.candidate_id
            for batch in batches.values()
            for record in batch.records_by_key.values()
            if is_state_update_confidence(record.confidence) or record.confidence == "rejected"
        }

        target_candidate_ids = [
            str(value)
            for value in _required_sequence_payload(state_payload, "target_candidate_ids")
        ]
        if len(target_candidate_ids) > self.population_size:
            raise CheckpointError(
                "checkpoint state.payload.target_candidate_ids cannot be larger than "
                "the configured DE population_size."
            )
        for candidate_id in target_candidate_ids:
            if candidate_id not in candidates:
                raise CheckpointError(
                    f"checkpoint target_candidate_id {candidate_id!r} is unknown."
                )
        trial_target_slots = {
            str(candidate_id): int(slot)
            for candidate_id, slot in _required_mapping_payload(
                state_payload, "trial_target_slots"
            ).items()
        }
        trial_target_candidate_ids = {
            str(candidate_id): str(target_id)
            for candidate_id, target_id in (
                _required_mapping_payload(state_payload, "trial_target_candidate_ids")
            ).items()
        }
        for candidate_id in set(trial_target_slots) | set(trial_target_candidate_ids):
            if candidate_id not in candidates:
                raise CheckpointError(
                    f"checkpoint trial candidate_id {candidate_id!r} is unknown."
                )
            if (
                candidate_id not in trial_target_slots
                or candidate_id not in trial_target_candidate_ids
            ):
                raise CheckpointError(
                    f"checkpoint trial candidate_id {candidate_id!r} must have slot and target mappings."
                )
            target_id = trial_target_candidate_ids[candidate_id]
            target_slot = trial_target_slots[candidate_id]
            if target_slot < 0 or target_slot >= len(target_candidate_ids):
                raise CheckpointError(
                    f"checkpoint trial target slot {target_slot!r} is outside target_candidate_ids."
                )
            if (
                candidate_id not in terminal_record_candidate_ids
                and target_candidate_ids[target_slot] != target_id
            ):
                raise CheckpointError(
                    "checkpoint trial target_candidate_id does not match "
                    f"target_candidate_ids[{target_slot}]."
                )
            if target_id not in candidates:
                raise CheckpointError(
                    f"checkpoint trial target_candidate_id {target_id!r} is unknown."
                )
        best_candidate_id = state_payload.get("best_candidate_id")
        if best_candidate_id is not None and best_candidate_id not in candidates:
            raise CheckpointError(
                f"checkpoint best_candidate_id {best_candidate_id!r} is unknown."
            )

        self._candidates_by_id = candidates
        self._batches_by_id = batches
        self._target_candidate_ids = target_candidate_ids
        self._trial_target_slots = trial_target_slots
        self._trial_target_candidate_ids = trial_target_candidate_ids
        self.best_candidate = None if best_candidate_id is None else candidates[best_candidate_id]
        self.vnext_telemetry = telemetry_from_checkpoint(
            _required_mapping_payload(state_payload, "telemetry")
        )
        self.events = event_history_from_checkpoint(
            _required_sequence_payload(state_payload, "events")
        )
        self._event_index = _required_int_payload(state_payload, "event_index")
        self.generation = _required_int_payload(state_payload, "generation")
        self._de_strategy_state = strategy_state_from_checkpoint(
            strategy=self.strategy,
            payload=state_payload.get("strategy_state"),
            population_size=self.population_size,
            expected_pending_slots={
                candidate_id: slot
                for candidate_id, slot in trial_target_slots.items()
                if candidate_id not in terminal_record_candidate_ids
            },
        )

    def resume_ask_tell_checkpoint(self, checkpoint: str | os.PathLike[str] | Mapping[str, Any]):
        """Resume DE ask/tell runtime state from a stable checkpoint."""
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
    "DE_ASK_TELL_STATE_KIND",
    "DE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "DifferentialEvolutionCheckpointingMixin",
]
