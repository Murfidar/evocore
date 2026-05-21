"""Metrics logging callback."""

from __future__ import annotations

import json

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.search_space import SolutionSet


class MetricsLogger(Callback):
    """Append per-generation metrics to a JSON Lines file."""

    def __init__(self, path: str = "./metrics.jsonl") -> None:
        self.path = path

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Write one JSON line containing generation metrics."""
        best = pop.best(1)
        record = {
            "generation": gen,
            "best_score": best[0].score if best else None,
            "nan_score_count": info.nan_score_count,
            "cached_count": info.cached_count,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
