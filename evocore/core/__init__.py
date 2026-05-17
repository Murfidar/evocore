"""Core EvoCore utilities."""

from evocore.core.errors import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    ConvergenceError,
    EvocoreError,
    FitnessError,
    FitnessWarning,
    ParallelError,
)
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.core.serialization import (
    canonical_json_hash,
    json_safe,
    package_version,
    stable_json_dumps,
)

__all__ = [
    "CheckpointError",
    "ConfigurationError",
    "ConfigurationWarning",
    "ConvergenceError",
    "EvocoreError",
    "FitnessError",
    "FitnessWarning",
    "ParallelError",
    "ProcessParallel",
    "ThreadParallel",
    "canonical_json_hash",
    "ensure_picklable",
    "json_safe",
    "package_version",
    "stable_json_dumps",
]
