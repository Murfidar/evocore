from __future__ import annotations

import copy
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Sequence

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.exceptions import CheckpointError, ConfigurationError, FitnessError, FitnessWarning
from evocore.gene_space import GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.stats import LogEntry, Logbook


@dataclass
class RunResult:
    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
    n_evaluations: int
    elite_history: list[Individual]
    diversity_history: list[list[float]]
    seed: int
    stopped_early: bool


@dataclass
class MultiRunResult:
    best: RunResult
    all_runs: list[RunResult]
    n_runs: int
    wall_time_seconds: float

    def best_n(self, n: int) -> list[RunResult]:
        return self.all_runs[:n]

    def fitness_summary(self) -> dict[str, float]:
        values = [run.best_fitness for run in self.all_runs]
        return {
            "mean": mean(values) if values else float("nan"),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }
