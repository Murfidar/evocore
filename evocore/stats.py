"""Run logbook data structures and reporting helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.evaluation import CandidateOrigin, CandidateStatus, Direction, EvaluationConfidence
from evocore.exceptions import ConfigurationError
from evocore.exporting import canonical_json_hash, json_safe, stable_json_dumps
from evocore.gene_space import GeneSpace
from evocore.individual import GeneValue


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

    def to_dict(self) -> dict[str, Any]:
        """Export this generation summary as a JSON-safe dictionary."""
        row: dict[str, Any] = {
            "gen": self.gen,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "wall_time_ms": self.wall_time_ms,
            "n_evaluations": self.n_evaluations,
            "nan_fitness_count": self.nan_fitness_count,
            "cached_count": self.cached_count,
            "diversity": list(self.diversity),
        }
        row.update(self.custom)
        return json_safe(row)


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


@dataclass(frozen=True)
class EventRecord:
    """Represent one append-only optimizer lifecycle observation."""

    event_index: int
    event_type: Literal["ask", "tell", "generation"]
    batch_id: str | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    generation: int | None = None
    rung: str | None = None
    confidence: EvaluationConfidence | None = None
    raw_score: float | None = None
    comparison_score: float | None = None
    cost: float = 0.0
    status: CandidateStatus | None = None
    origin: CandidateOrigin | None = None
    parents: tuple[str, ...] = ()
    genes: tuple[GeneValue, ...] = ()
    params: dict[str, GeneValue] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export this event as one JSON-safe row."""
        return json_safe(
            {
                "event_index": self.event_index,
                "event_type": self.event_type,
                "batch_id": self.batch_id,
                "candidate_id": self.candidate_id,
                "candidate_hash": self.candidate_hash,
                "generation": self.generation,
                "rung": self.rung,
                "confidence": self.confidence,
                "raw_score": self.raw_score,
                "comparison_score": self.comparison_score,
                "cost": self.cost,
                "status": self.status,
                "origin": self.origin,
                "parents": list(self.parents),
                "genes": list(self.genes),
                "params": self.params,
                "metrics": self.metrics,
                "metadata": self.metadata,
            }
        )


class EventHistory:
    """Store append-only optimizer lifecycle events."""

    def __init__(self) -> None:
        self._events: list[EventRecord] = []

    def append(self, event: EventRecord) -> None:
        """Append one event in strict sequence order."""
        if event.event_index != len(self._events):
            raise ConfigurationError(
                "EventHistory is append-only; event_index must match the next row index."
            )
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self._events)

    def __getitem__(self, index: int) -> EventRecord:
        return self._events[index]

    def to_rows(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return [event.to_dict() for event in self._events]

    def to_dict(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return self.to_rows()

    def to_dataframe(self):
        """Convert event rows into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "EventHistory.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        return pd.DataFrame(self.to_rows())


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Capture deterministic optimizer and environment identity for a result."""

    evocore_version: str
    engine_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    extension: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export reproducibility metadata as JSON-safe stable fields."""
        return json_safe(
            {
                "evocore_version": self.evocore_version,
                "engine_type": self.engine_type,
                "seed": self.seed,
                "direction": self.direction,
                "gene_space_signature": self.gene_space_signature,
                "gene_space_hash": self.gene_space_hash,
                "optimizer_config": self.optimizer_config,
                "extension": self.extension,
            }
        )


def gene_space_signature(gene_space: GeneSpace) -> dict[str, Any]:
    """Return the canonical signature for a gene space."""
    return gene_space.signature()


def gene_space_hash(signature: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a gene-space signature."""
    return canonical_json_hash(signature)
