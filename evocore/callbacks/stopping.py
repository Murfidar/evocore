"""Built-in callback for score-based early stopping."""

from __future__ import annotations

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.search_space import SolutionSet


class EarlyStopping(Callback):
    """Stop a run after score stagnates for a patience window."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.should_stop = False
        self._best = float("-inf")
        self._no_improve_count = 0

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Track best score and request a stop when progress stalls."""
        best = pop.best(1)
        if not best or best[0].score is None:
            return

        score = float(best[0].score)
        if score - self._best > self.min_delta:
            self._best = score
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
            if self._no_improve_count >= self.patience:
                self.should_stop = True
