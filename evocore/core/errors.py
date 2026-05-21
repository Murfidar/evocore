"""evocore exception and warning hierarchy."""


class EvocoreError(Exception):
    """Base class for all evocore exceptions."""


class ConfigurationError(EvocoreError):
    """Raised when engine or gene-space configuration is invalid."""


class FitnessError(EvocoreError):
    """Raised when an objective function fails or returns an invalid type."""


class ConvergenceError(EvocoreError):
    """Raised when a numerical failure makes continuation impossible."""


class ParallelError(EvocoreError):
    """Raised when a parallel worker pool fails unrecoverably."""


class CheckpointError(EvocoreError):
    """Raised when a checkpoint file is missing, corrupt, or incompatible."""


class FitnessWarning(UserWarning):
    """Warning emitted when NaN or Inf score values are encountered."""


class ConfigurationWarning(UserWarning):
    """Warning emitted for valid but likely unintended configuration."""
