"""Run GenerationHistory data structures and reporting helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from evocore.core.serialization import json_safe, stable_json_dumps


@dataclass
class GenerationRecord:
    """Capture per-generation statistics from an optimization engine."""

    gen: int
    best_score: float
    mean_score: float
    std_score: float
    wall_time_ms: float
    n_evaluations: int
    nan_score_count: int
    cached_count: int
    diversity: list[float] = field(default_factory=list)
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export this generation summary as a JSON-safe dictionary."""
        row: dict[str, Any] = {
            "gen": self.gen,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "wall_time_ms": self.wall_time_ms,
            "n_evaluations": self.n_evaluations,
            "nan_score_count": self.nan_score_count,
            "cached_count": self.cached_count,
            "diversity": list(self.diversity),
        }
        row.update(self.custom)
        return json_safe(row)


class GenerationHistory:
    """Store ordered `GenerationRecord` records with export helpers."""

    def __init__(self) -> None:
        self._entries: list[GenerationRecord] = []

    def append(self, entry: GenerationRecord) -> None:
        """Append a generation record to the GenerationHistory."""
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[GenerationRecord]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> GenerationRecord:
        return self._entries[index]

    def best_scores(self) -> list[float]:
        """Return the best score value from each generation."""
        return [entry.best_score for entry in self._entries]

    def nan_counts(self) -> list[int]:
        """Return the number of non-finite score values per generation."""
        return [entry.nan_score_count for entry in self._entries]

    def to_rows(self) -> list[dict[str, Any]]:
        """Convert log entries into JSON-serializable row dictionaries."""
        return [entry.to_dict() for entry in self._entries]

    def to_dict(self) -> list[dict[str, Any]]:
        """Export stable generation-summary dictionaries."""
        return self.to_rows()

    def to_json(self, *, indent: int | None = None) -> str:
        """Export generation summaries as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)

    def print(self) -> None:
        """Print row dictionaries for quick inspection."""
        for row in self.to_rows():
            print(row)

    def to_dataframe(self):
        """Convert the GenerationHistory into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "GenerationHistory.to_dataframe() requires pandas; pip install pandas."
            ) from exc
        return pd.DataFrame(self.to_rows())

    def plot(self, metrics: list[str] | None = None):
        """Plot selected GenerationHistory metrics with matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "GenerationHistory.plot() requires matplotlib; pip install matplotlib."
            ) from exc
        metric_names = metrics or ["best_score", "mean_score"]
        rows = self.to_rows()
        xs = [row["gen"] for row in rows]
        fig, ax = plt.subplots()
        for metric in metric_names:
            ax.plot(xs, [row.get(metric) for row in rows], label=metric)
        ax.set_xlabel("generation")
        ax.legend()
        return fig
