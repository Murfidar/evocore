"""Run logbook data structures and reporting helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    """Capture per-generation statistics from an optimization engine."""

    gen: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    wall_time_ms: float
    n_evaluations: int
    nan_fitness_count: int
    cached_count: int
    diversity: list[float] = field(default_factory=list)
    custom: dict = field(default_factory=dict)


class Logbook:
    """Store ordered `LogEntry` records with export helpers."""

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def append(self, entry: LogEntry) -> None:
        """Append a generation record to the logbook."""
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> LogEntry:
        return self._entries[index]

    def best_fitnesses(self) -> list[float]:
        """Return the best fitness value from each generation."""
        return [entry.best_fitness for entry in self._entries]

    def nan_counts(self) -> list[int]:
        """Return the number of non-finite fitness values per generation."""
        return [entry.nan_fitness_count for entry in self._entries]

    def to_rows(self) -> list[dict]:
        """Convert log entries into JSON-serializable row dictionaries."""
        rows: list[dict] = []
        for entry in self._entries:
            row = {
                "gen": entry.gen,
                "best_fitness": entry.best_fitness,
                "mean_fitness": entry.mean_fitness,
                "std_fitness": entry.std_fitness,
                "wall_time_ms": entry.wall_time_ms,
                "n_evaluations": entry.n_evaluations,
                "nan_fitness_count": entry.nan_fitness_count,
                "cached_count": entry.cached_count,
            }
            row.update(entry.custom)
            rows.append(row)
        return rows

    def print(self) -> None:
        """Print row dictionaries for quick inspection."""
        for row in self.to_rows():
            print(row)

    def to_dataframe(self):
        """Convert the logbook into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Logbook.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        return pd.DataFrame(self.to_rows())

    def plot(self, metrics: list[str] | None = None):
        """Plot selected logbook metrics with matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Logbook.plot() requires matplotlib. Install with: pip install matplotlib"
            ) from exc

        metric_names = metrics or ["best_fitness", "mean_fitness"]
        rows = self.to_rows()
        xs = [row["gen"] for row in rows]
        fig, ax = plt.subplots()
        for metric in metric_names:
            ax.plot(xs, [row.get(metric) for row in rows], label=metric)
        ax.set_xlabel("generation")
        ax.legend()
        return fig
