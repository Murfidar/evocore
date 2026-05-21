"""Optional tqdm progress callback."""

from __future__ import annotations

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.results import OptimizationResult
from evocore.search_space import SolutionSet


class ProgressBar(Callback):
    """Display a tqdm progress bar when tqdm is available."""

    def __init__(self) -> None:
        self._bar = None
        self._total = None

    def bind_context(self, **kwargs) -> None:
        """Store the total generation count for the progress bar."""
        self._total = kwargs.get("max_generations")

    def on_generation_start(self, gen: int, pop: SolutionSet) -> None:
        """Create the bar lazily on the first generation."""
        if self._bar is None:
            try:
                from tqdm import tqdm
            except ImportError:
                self._bar = False
                return
            self._bar = tqdm(total=self._total)

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Advance the bar and show the latest best score."""
        if not self._bar:
            return

        best = pop.best(1)
        postfix = {"best": best[0].score if best else None}
        if info.nan_score_count:
            postfix["nan"] = info.nan_score_count
        self._bar.set_postfix(**postfix)
        self._bar.update(1)

    def on_run_end(self, result: OptimizationResult) -> None:
        """Close the progress bar when the run ends."""
        if self._bar:
            self._bar.close()
