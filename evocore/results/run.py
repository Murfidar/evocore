"""Completed optimization result envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

from evocore.core.serialization import json_safe, stable_json_dumps
from evocore.lifecycle import Direction, OptimizationTelemetry
from evocore.lifecycle.events import EventHistory, StopReason
from evocore.results.generation import GenerationHistory
from evocore.results.reproducibility import ReproducibilityMetadata
from evocore.search_space import Solution, SolutionSet


@dataclass
class OptimizationResult:
    """Store the outcome of one optimization run."""

    best_solution: Solution
    best_score: float
    final_solutions: SolutionSet
    generations: GenerationHistory
    wall_time_seconds: float
    n_evaluations: int
    elite_solutions: list[Solution]
    diversity_by_generation: list[list[float]]
    seed: int
    stop_reason: StopReason = "max_generations"
    max_generations: int | None = None
    max_evaluations: int | None = None
    telemetry: OptimizationTelemetry = field(default_factory=OptimizationTelemetry)
    direction: Direction = "maximize"
    optimizer_type: str = ""
    best_candidate_id: str | None = None
    best_observed_score: float | None = None
    events: EventHistory = field(default_factory=EventHistory)
    reproducibility: ReproducibilityMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export this optimization result as a stable JSON-safe dictionary."""
        best_score = (
            self.best_observed_score if self.best_observed_score is not None else self.best_score
        )
        payload: dict[str, Any] = {
            "schema_version": 2,
            "optimizer_type": self.optimizer_type,
            "direction": self.direction,
            "seed": self.seed,
            "best": {
                "score": self.best_score,
                "observed_score": best_score,
                "candidate_id": self.best_candidate_id,
                "values": list(self.best_solution.values),
                "params": self.best_solution.metadata.get("params"),
            },
            "stop": {
                "reason": self.stop_reason,
            },
            "budget": {
                "max_evaluations": self.max_evaluations,
                "max_generations": self.max_generations,
                "n_evaluations": self.n_evaluations,
            },
            "n_evaluations": self.n_evaluations,
            "reproducibility": (
                self.reproducibility.to_dict() if self.reproducibility is not None else None
            ),
            "telemetry": self.telemetry.to_dict(),
            "events": self.events.to_dict(),
            "generations": self.generations.to_dict(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export this optimization result as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return event history as a DataFrame, falling back to generation rows."""
        if len(self.events):
            return self.events.to_dataframe()
        return self.generations.to_dataframe()


@dataclass
class OptimizationBatchResult:
    """Store the aggregated outcome of multiple optimizer runs."""

    best: OptimizationResult
    all_runs: list[OptimizationResult]
    n_runs: int
    wall_time_seconds: float
    direction: Direction = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_n(self, n: int) -> list[OptimizationResult]:
        """Return the top `n` runs sorted by best score."""
        return self.all_runs[:n]

    def score_summary(self) -> dict[str, float]:
        """Return summary statistics across best score values."""
        values = [run.best_score for run in self.all_runs]
        return {
            "mean": mean(values) if values else float("nan"),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export aggregate optimization results as a stable JSON-safe dictionary."""
        payload: dict[str, Any] = {
            "schema_version": 2,
            "direction": self.direction,
            "n_runs": self.n_runs,
            "best": self.best.to_dict(include_runtime=include_runtime),
            "runs": [run.to_dict(include_runtime=include_runtime) for run in self.all_runs],
            "score_summary": self.score_summary(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export aggregate optimization results as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return one pandas DataFrame row per child run."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "OptimizationBatchResult.to_dataframe() requires pandas; pip install pandas."
            ) from exc
        rows = [
            {
                "run_index": index,
                "seed": run.seed,
                "best_score": run.best_score,
                "best_observed_score": (
                    run.best_observed_score
                    if run.best_observed_score is not None
                    else run.best_score
                ),
                "n_evaluations": run.n_evaluations,
            }
            for index, run in enumerate(self.all_runs)
        ]
        return pd.DataFrame(rows)
