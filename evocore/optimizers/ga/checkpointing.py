from __future__ import annotations

import os
import pickle
from collections.abc import Callable

from evocore.core.errors import CheckpointError
from evocore.results import OptimizationResult
from evocore.search_space import Solution


class GeneticAlgorithmCheckpointingMixin:
    """Checkpoint loading and resume helpers for GA."""

    def resume(self, objective_fn: Callable, checkpoint: str) -> OptimizationResult:
        """Resume a GA run from a checkpoint file.

        Args:
            objective_fn: Objective callable used for remaining generations.
            checkpoint: Path to a checkpoint written by `CheckpointCallback`.

        Returns:
            Run result for the resumed optimization.

        Raises:
            CheckpointError: If the file is missing, corrupt, incompatible, or has a
                seed that differs from the engine seed.
        """
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


__all__ = ["GeneticAlgorithmCheckpointingMixin"]
