"""Checkpoint-writing callback."""

from __future__ import annotations

import os
import pickle

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.search_space import SolutionSet


class CheckpointCallback(Callback):
    """Write pickle checkpoints at a fixed generation interval."""

    def __init__(self, path: str = "./checkpoints", every: int = 10) -> None:
        self.path = path
        self.every = every
        self._seed: int | None = None

    def bind_context(self, **kwargs) -> None:
        """Capture the engine seed for checkpoint validation."""
        self._seed = kwargs.get("seed")

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Persist a checkpoint when the current generation matches the cadence."""
        if self.every <= 0 or gen % self.every != 0:
            return

        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.pkl")
        with open(filename, "wb") as handle:
            pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, handle)
