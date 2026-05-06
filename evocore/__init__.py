"""Top-level evocore package exports for Part 1."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("evocore")
except _metadata.PackageNotFoundError:
    __version__ = "0.6.0"

from evocore._core import (
    OP_CMAES_ASK,
    OP_CROSSOVER,
    OP_CROSSOVER_PROB,
    OP_INIT,
    OP_MULTI_RUN,
    OP_MUTATION,
    OP_SELECTION,
    BinaryIndividual,
    FloatIndividual,
    IntegerIndividual,
    py_derive_seed,
)
from evocore.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStopping,
    GenerationInfo,
    MetricsLogger,
    ProgressBar,
)
from evocore.cmaes import CMAESEngine
from evocore.exceptions import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    ConvergenceError,
    EvocoreError,
    FitnessError,
    FitnessWarning,
    ParallelError,
)
from evocore.ga import GAEngine, MultiRunResult, RunResult
from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ProcessParallel, ThreadParallel
from evocore.stats import Logbook, LogEntry

__all__ = [
    "OP_CMAES_ASK",
    "OP_CROSSOVER",
    "OP_CROSSOVER_PROB",
    "OP_INIT",
    "OP_MULTI_RUN",
    "OP_MUTATION",
    "OP_SELECTION",
    "BinaryIndividual",
    "CMAESEngine",
    "Callback",
    "CheckpointCallback",
    "CheckpointError",
    "ConfigurationError",
    "ConfigurationWarning",
    "ConvergenceError",
    "EarlyStopping",
    "EvocoreError",
    "FitnessError",
    "FitnessWarning",
    "FloatIndividual",
    "GAEngine",
    "GeneDef",
    "GeneSpace",
    "GenerationInfo",
    "Individual",
    "IntegerIndividual",
    "LogEntry",
    "Logbook",
    "MetricsLogger",
    "MultiRunResult",
    "OperatorSet",
    "ParallelError",
    "Population",
    "ProcessParallel",
    "ProgressBar",
    "RunResult",
    "ThreadParallel",
    "__version__",
    "py_derive_seed",
]
