"""Checkpoint-writing callback."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from typing import Literal

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.core.errors import CheckpointError
from evocore.results import CheckpointSnapshot, save_checkpoint
from evocore.search_space import Solution, SolutionSet

CheckpointFormat = Literal["stable", "legacy_pickle"]


class CheckpointCallback(Callback):
    """Write optimizer checkpoints at a fixed generation interval."""

    def __init__(
        self,
        path: str = "./checkpoints",
        every: int = 10,
        format: CheckpointFormat = "legacy_pickle",  # noqa: A002
    ) -> None:
        if format not in ("stable", "legacy_pickle"):
            raise CheckpointError("CheckpointCallback format must be 'stable' or 'legacy_pickle'.")
        self.path = path
        self.every = every
        self.format = format
        self._seed: int | None = None
        self._checkpoint_factory: Callable[..., CheckpointSnapshot] | None = None

    def bind_context(self, **kwargs) -> None:
        """Capture engine checkpoint context."""
        self._seed = kwargs.get("seed")
        factory = kwargs.get("checkpoint_factory")
        self._checkpoint_factory = factory if callable(factory) else None

    def _generation_metadata(self, info: GenerationInfo) -> dict[str, object]:
        return {
            "callback": {
                "generation_info": {
                    "generation": info.generation,
                    "nan_score_count": info.nan_score_count,
                    "cached_count": info.cached_count,
                }
            }
        }

    def _write_legacy_pickle(self, gen: int, pop: SolutionSet) -> None:
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.pkl")
        with open(filename, "wb") as handle:
            pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, handle)

    def _write_stable(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        if self._checkpoint_factory is None:
            raise CheckpointError(
                "CheckpointCallback(format='stable') requires optimizer checkpoint support. "
                "Use format='legacy_pickle' for the legacy population pickle format."
            )
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.evocore-checkpoint.json")
        snapshot = self._checkpoint_factory(
            generation=gen,
            population=[
                solution.clone() if isinstance(solution, Solution) else solution
                for solution in pop
            ],
            metadata=self._generation_metadata(info),
        )
        save_checkpoint(filename, snapshot)

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Persist a checkpoint when the current generation matches the cadence."""
        if self.every <= 0 or gen % self.every != 0:
            return

        os.makedirs(self.path, exist_ok=True)
        if self.format == "legacy_pickle":
            self._write_legacy_pickle(gen, pop)
        else:
            self._write_stable(gen, pop, info)
