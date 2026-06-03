from __future__ import annotations

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
            return cls(
                target_slot=int(payload["target_slot"]),
                mutation_factor=float(payload["mutation_factor"]),
                crossover_rate=float(payload["crossover_rate"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint state.payload.strategy_state pending trial params are invalid."
            ) from exc


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
        raw_pending = payload.get("pending_trial_params")
        if not isinstance(raw_pending, Mapping):
            raise CheckpointError(
                "checkpoint state.payload.strategy_state.pending_trial_params must be an object."
            )
        return cls(
            f_by_slot=f_by_slot,
            cr_by_slot=cr_by_slot,
            pending_trial_params={
                str(candidate_id): JDETrialParameters.from_mapping(params)
                for candidate_id, params in raw_pending.items()
            },
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
) -> JDEAdaptiveState | None:
    """Restore adaptive strategy state for a checkpointed optimizer."""
    if strategy != "jde-rand1bin":
        return None
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            "strategy_state is required for strategy='jde-rand1bin' checkpoints."
        )
    return JDEAdaptiveState.from_checkpoint(payload, population_size=population_size)


__all__ = [
    "JDEAdaptiveState",
    "JDETrialParameters",
    "initial_strategy_state",
    "strategy_state_from_checkpoint",
    "strategy_state_to_checkpoint",
]
