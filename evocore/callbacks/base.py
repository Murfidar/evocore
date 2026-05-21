"""Callback base types and generation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evocore.results import OptimizationResult
    from evocore.search_space import SolutionSet


@dataclass
class GenerationInfo:
    """Expose per-generation metadata to callbacks."""

    generation: int
    nan_score_count: int
    cached_count: int


class Callback:
    """Define lifecycle hooks for optimization runs."""

    should_stop: bool = False

    def bind_context(self, **kwargs) -> None:
        """Receive run context before optimization starts."""

    def on_generation_start(self, gen: int, pop: SolutionSet) -> None:
        """Run before one generation starts."""

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Run after one generation completes."""

    def on_run_end(self, result: OptimizationResult) -> None:
        """Run after the optimization finishes."""
