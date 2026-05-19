from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.lifecycle import score_for_direction
from evocore.results import (
    CheckpointSnapshot,
    OptimizationResult,
    validate_checkpoint_identity,
)
from evocore.results import (
    load_checkpoint as load_checkpoint_payload,
)
from evocore.results import (
    save_checkpoint as save_checkpoint_payload,
)
from evocore.search_space import Solution


def _solution_to_checkpoint(solution: Solution) -> dict[str, Any]:
    """Export one solution into the GA checkpoint payload."""
    return {
        "values": list(solution.values),
        "score": solution.score,
        "score_valid": bool(solution.score_valid),
        "metadata": dict(solution.metadata),
    }


def _solution_from_checkpoint(payload: Mapping[str, Any]) -> Solution:
    """Restore one Solution from a GA checkpoint payload row."""
    values = payload.get("values")
    if not isinstance(values, list):
        raise CheckpointError("checkpoint solution.values must be a list.")
    return Solution(
        values,
        score=payload.get("score"),
        score_valid=bool(payload.get("score_valid", False)),
        metadata=dict(payload.get("metadata") or {}),
    )


class GeneticAlgorithmCheckpointingMixin:
    """Checkpoint loading and resume helpers for GA."""

    def checkpoint(
        self,
        *,
        generation: int | None = None,
        population: Sequence[Solution] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        """Return a stable GA generation-loop checkpoint snapshot."""
        if generation is None or population is None:
            raise CheckpointError(
                "GA stable checkpoint v1 requires generation and population. "
                "Use CheckpointCallback during generation-loop runs or pass both arguments."
            )
        population_payload = [_solution_to_checkpoint(solution) for solution in population]
        state_payload = {
            "state_kind": "ga_generation_loop",
            "generation": int(generation),
            "population": population_payload,
        }
        best_payload = None
        scored = [
            solution
            for solution in population
            if solution.score is not None and solution.score_valid
        ]
        if scored:
            best = max(
                scored,
                key=lambda solution: score_for_direction(float(solution.score), self.direction),
            )
            best_payload = _solution_to_checkpoint(best)
        return CheckpointSnapshot(
            optimizer_type="GeneticAlgorithmOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "generation": int(generation),
                "event_index": self.state_summary().event_index,
                "n_evaluations": None,
            },
            state={
                "optimizer_type": "GeneticAlgorithmOptimizer",
                "schema_version": 1,
                "payload": state_payload,
            },
            audit={
                "events": self.events.to_dict(),
                "telemetry": self.vnext_telemetry.to_dict(),
                "best": best_payload,
            },
            metadata=dict(metadata or {}),
        )

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
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=self.gene_space.hash(),
            optimizer_config_hash=self.config_hash(),
            seed=self.seed,
            direction=self.direction,
        )

    def _population_from_stable_checkpoint(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[list[Solution], int]:
        state = payload["state"]
        state_payload = state["payload"]
        if state_payload.get("state_kind") != "ga_generation_loop":
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by "
                "GA generation-loop resume."
            )
        raw_population = state_payload.get("population")
        if not isinstance(raw_population, list):
            raise CheckpointError("checkpoint state.payload.population must be a list.")
        population = []
        for row in raw_population:
            if not isinstance(row, Mapping):
                raise CheckpointError("checkpoint population entries must be objects.")
            population.append(_solution_from_checkpoint(row))
        generation = int(state_payload.get("generation", payload["position"]["generation"]))
        return population, generation

    def resume_from_checkpoint(
        self,
        objective_fn: Callable,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    ) -> OptimizationResult:
        """Resume a GA generation-loop run from a stable checkpoint."""
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        population, saved_generation = self._population_from_stable_checkpoint(payload)
        return self._run_from_population(
            population,
            objective_fn,
            start_generation=saved_generation + 1,
        )

    def _resume_legacy_pickle(self, objective_fn: Callable, checkpoint: str) -> OptimizationResult:
        """Resume from the legacy GA pickle checkpoint format."""
        if not os.path.exists(checkpoint):
            directory = os.path.dirname(checkpoint) or "."
            available = []
            if os.path.isdir(directory):
                available = sorted(
                    name for name in os.listdir(directory) if name.startswith("checkpoint_gen_")
                )
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} not found. Available checkpoints: "
                f"{', '.join(available) or 'none'}"
            )

        try:
            with open(checkpoint, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} is corrupt or incompatible: {exc}"
            ) from exc

        solutions = payload.get("population")
        if solutions is None:
            solutions = payload.get("SolutionSet")
        if not isinstance(solutions, list) or not all(
            isinstance(solution, Solution) for solution in solutions
        ):
            raise CheckpointError(
                "checkpoint payload must contain a list[Solution] under key 'population'."
            )

        saved_generation = int(payload.get("generation", -1))
        saved_seed = payload.get("seed")
        if saved_seed is not None and int(saved_seed) != self.seed:
            raise CheckpointError(
                f"checkpoint seed {saved_seed} does not match engine seed {self.seed}."
            )

        return self._run_from_population(
            solutions,
            objective_fn,
            start_generation=saved_generation + 1,
        )

    def resume(self, objective_fn: Callable, checkpoint: str) -> OptimizationResult:
        """Resume a GA run from a stable JSON checkpoint or legacy pickle checkpoint."""
        if str(checkpoint).endswith(".json"):
            return self.resume_from_checkpoint(objective_fn, checkpoint)
        return self._resume_legacy_pickle(objective_fn, checkpoint)


__all__ = ["GeneticAlgorithmCheckpointingMixin"]
