"""Top-level evocore package exports."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("evocore")
except _metadata.PackageNotFoundError:
    __version__ = "0.6.1"

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
from evocore.evaluation import (
    Candidate,
    CandidateScore,
    EvaluationRecord,
    Evaluator,
    OptimizationTelemetry,
    Rung,
)
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
    # 1. SCREAMING_SNAKE_CASE (Constants)
    "OP_CMAES_ASK",
    "OP_CROSSOVER",
    "OP_CROSSOVER_PROB",
    "OP_INIT",
    "OP_MULTI_RUN",
    "OP_MUTATION",
    "OP_SELECTION",
    # 2. CamelCase (Classes)
    "BinaryIndividual",
    "CMAESEngine",
    "Callback",
    "Candidate",
    "CandidateScore",
    "CheckpointCallback",
    "CheckpointError",
    "ConfigurationError",
    "ConfigurationWarning",
    "ConvergenceError",
    "EarlyStopping",
    "EvaluationRecord",
    "Evaluator",
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
    "OptimizationTelemetry",
    "ParallelError",
    "Population",
    "ProcessParallel",
    "ProgressBar",
    "RunResult",
    "Rung",
    "ThreadParallel",
    # 3. snake_case / dunders
    "__version__",
    "py_derive_seed",
]
