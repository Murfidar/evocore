from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from evocore import _core
from evocore.core.errors import CheckpointError

JDE_STATE_SCHEMA_VERSION = 1
JDE_F_REFRESH_PROBABILITY = 0.1
JDE_CR_REFRESH_PROBABILITY = 0.1
JDE_F_LOW = 0.1
JDE_F_HIGH = 1.0


@dataclass(frozen=True)
class JDETrialParameters:
    """Per-trial jDE parameters attached to one pending trial candidate."""

    target_slot: int
    mutation_factor: float
    crossover_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe checkpoint representation."""
        return {
            "target_slot": self.target_slot,
            "mutation_factor": self.mutation_factor,
            "crossover_rate": self.crossover_rate,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> JDETrialParameters:
        """Load pending trial parameters from checkpoint payload data."""
        try:
            params = cls(
                target_slot=int(payload["target_slot"]),
                mutation_factor=float(payload["mutation_factor"]),
                crossover_rate=float(payload["crossover_rate"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint state.payload.strategy_state pending trial params are invalid."
            ) from exc
        _validate_mutation_factor(
            params.mutation_factor,
            "checkpoint state.payload.strategy_state.pending_trial_params.mutation_factor",
        )
        _validate_crossover_rate(
            params.crossover_rate,
            "checkpoint state.payload.strategy_state.pending_trial_params.crossover_rate",
        )
        return params


@dataclass
class JDEAdaptiveState:
    """Checkpointable jDE per-slot parameter state."""

    f_by_slot: list[float]
    cr_by_slot: list[float]
    pending_trial_params: dict[str, JDETrialParameters] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        *,
        population_size: int,
        mutation_factor: float,
        crossover_rate: float,
    ) -> JDEAdaptiveState:
        """Create initial per-slot state from constructor defaults."""
        return cls(
            f_by_slot=[float(mutation_factor)] * int(population_size),
            cr_by_slot=[float(crossover_rate)] * int(population_size),
        )

    def propose_parameters(
        self,
        *,
        seed: int,
        generation: int,
        target_slot: int,
    ) -> JDETrialParameters:
        """Deterministically propose jDE parameters for one target slot."""
        f_value = self.f_by_slot[target_slot]
        cr_value = self.cr_by_slot[target_slot]
        f_rng = _jde_rng(seed, generation, target_slot, offset=1)
        cr_rng = _jde_rng(seed, generation, target_slot, offset=2)
        if f_rng.random() < JDE_F_REFRESH_PROBABILITY:
            f_value = JDE_F_LOW + f_rng.random() * (JDE_F_HIGH - JDE_F_LOW)
        if cr_rng.random() < JDE_CR_REFRESH_PROBABILITY:
            cr_value = cr_rng.random()
        return JDETrialParameters(
            target_slot=target_slot,
            mutation_factor=f_value,
            crossover_rate=cr_value,
        )

    def register_pending(self, candidate_id: str, params: JDETrialParameters) -> None:
        """Attach proposed parameters to a pending trial candidate."""
        self.pending_trial_params[str(candidate_id)] = params

    def complete_pending(self, candidate_id: str, *, accepted: bool) -> None:
        """Clear a terminal trial and commit parameters only when accepted."""
        params = self.pending_trial_params.pop(str(candidate_id), None)
        if params is None:
            return
        if accepted:
            self.f_by_slot[params.target_slot] = params.mutation_factor
            self.cr_by_slot[params.target_slot] = params.crossover_rate

    def discard_pending(self, candidate_id: str) -> None:
        """Clear a pending trial candidate without committing parameters."""
        self.pending_trial_params.pop(str(candidate_id), None)

    def to_checkpoint(self) -> dict[str, object]:
        """Return a JSON-safe checkpoint payload for jDE state."""
        return {
            "strategy": "jde-rand1bin",
            "strategy_state_schema_version": JDE_STATE_SCHEMA_VERSION,
            "f_by_slot": list(self.f_by_slot),
            "cr_by_slot": list(self.cr_by_slot),
            "pending_trial_params": {
                candidate_id: params.to_dict()
                for candidate_id, params in sorted(self.pending_trial_params.items())
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        population_size: int,
        expected_pending_slots: Mapping[str, int] | None = None,
    ) -> JDEAdaptiveState:
        """Restore and validate jDE state from a checkpoint payload."""
        if payload.get("strategy") != "jde-rand1bin":
            raise CheckpointError(
                "checkpoint state.payload.strategy_state strategy must be 'jde-rand1bin'."
            )
        if payload.get("strategy_state_schema_version") != JDE_STATE_SCHEMA_VERSION:
            raise CheckpointError(
                "strategy_state_schema_version 1 is required for strategy='jde-rand1bin'."
            )
        f_by_slot = _float_list(payload, "f_by_slot")
        cr_by_slot = _float_list(payload, "cr_by_slot")
        if len(f_by_slot) != population_size or len(cr_by_slot) != population_size:
            raise CheckpointError(
                "checkpoint state.payload.strategy_state slot arrays must match population_size."
            )
        for value in f_by_slot:
            _validate_mutation_factor(value, "checkpoint state.payload.strategy_state.f_by_slot")
        for value in cr_by_slot:
            _validate_crossover_rate(value, "checkpoint state.payload.strategy_state.cr_by_slot")
        raw_pending = payload.get("pending_trial_params")
        if not isinstance(raw_pending, Mapping):
            raise CheckpointError(
                "checkpoint state.payload.strategy_state.pending_trial_params must be an object."
            )
        pending_trial_params = {
            str(candidate_id): JDETrialParameters.from_mapping(params)
            for candidate_id, params in raw_pending.items()
        }
        if expected_pending_slots is not None:
            _validate_pending_trial_params(
                pending_trial_params,
                expected_pending_slots=expected_pending_slots,
                population_size=population_size,
            )
        return cls(
            f_by_slot=f_by_slot,
            cr_by_slot=cr_by_slot,
            pending_trial_params=pending_trial_params,
        )


def _jde_rng(seed: int, generation: int, target_slot: int, *, offset: int) -> random.Random:
    derived = int(
        _core.py_derive_seed(
            int(seed),
            int(generation),
            int(target_slot) * 10 + int(offset),
            _core.OP_MUTATION,
        )
    )
    return random.Random(derived)  # noqa: S311 - deterministic optimizer sampling.


def _float_list(payload: Mapping[str, Any], key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise CheckpointError(f"checkpoint state.payload.strategy_state.{key} must be an array.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"checkpoint state.payload.strategy_state.{key} must contain floats."
        ) from exc


def _validate_mutation_factor(value: float, context: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise CheckpointError(f"{context} must be finite and >= 0.")


def _validate_crossover_rate(value: float, context: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CheckpointError(f"{context} must be finite and in [0, 1].")


def _validate_pending_trial_params(
    pending_trial_params: Mapping[str, JDETrialParameters],
    *,
    expected_pending_slots: Mapping[str, int],
    population_size: int,
) -> None:
    actual_ids = set(pending_trial_params)
    expected_ids = {str(candidate_id) for candidate_id in expected_pending_slots}
    if actual_ids != expected_ids:
        raise CheckpointError(
            "checkpoint state.payload.strategy_state.pending_trial_params must match "
            "pending trial candidates."
        )
    for candidate_id, expected_slot in expected_pending_slots.items():
        params = pending_trial_params[str(candidate_id)]
        if params.target_slot != int(expected_slot):
            raise CheckpointError(
                "checkpoint state.payload.strategy_state.pending_trial_params target_slot "
                "must match trial_target_slots."
            )
        if params.target_slot < 0 or params.target_slot >= population_size:
            raise CheckpointError(
                "checkpoint state.payload.strategy_state.pending_trial_params target_slot "
                "is outside population_size."
            )


def initial_strategy_state(
    *,
    strategy: str,
    population_size: int,
    mutation_factor: float,
    crossover_rate: float,
) -> JDEAdaptiveState | None:
    """Create strategy state for adaptive DE strategies."""
    if strategy == "jde-rand1bin":
        return JDEAdaptiveState.initial(
            population_size=population_size,
            mutation_factor=mutation_factor,
            crossover_rate=crossover_rate,
        )
    return None


def strategy_state_to_checkpoint(state: object | None) -> dict[str, object] | None:
    """Serialize known adaptive strategy state for checkpoints."""
    if isinstance(state, JDEAdaptiveState):
        return state.to_checkpoint()
    return None


def strategy_state_from_checkpoint(
    *,
    strategy: str,
    payload: object,
    population_size: int,
    expected_pending_slots: Mapping[str, int] | None = None,
) -> JDEAdaptiveState | None:
    """Restore adaptive strategy state for a checkpointed optimizer."""
    if strategy != "jde-rand1bin":
        return None
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            "strategy_state is required for strategy='jde-rand1bin' checkpoints."
        )
    return JDEAdaptiveState.from_checkpoint(
        payload,
        population_size=population_size,
        expected_pending_slots=expected_pending_slots,
    )


__all__ = [
    "JDEAdaptiveState",
    "JDETrialParameters",
    "initial_strategy_state",
    "strategy_state_from_checkpoint",
    "strategy_state_to_checkpoint",
]
