def test_evocore_imports_without_error():
    import evocore  # noqa: F401


def test_exceptions_accessible_from_top_level():
    from evocore import (
        CheckpointError,
        ConfigurationError,
        ConfigurationWarning,
        ConvergenceError,
        EvocoreError,
        FitnessError,
        FitnessWarning,
        ParallelError,
    )

    assert issubclass(ConfigurationError, EvocoreError)
    assert issubclass(FitnessWarning, Warning)
    assert issubclass(ConfigurationWarning, Warning)
    assert issubclass(CheckpointError, EvocoreError)
    assert issubclass(ConvergenceError, EvocoreError)
    assert issubclass(FitnessError, EvocoreError)
    assert issubclass(ParallelError, EvocoreError)


def test_core_extension_accessible():
    from evocore import _core

    assert hasattr(_core, "FloatIndividual")
    assert hasattr(_core, "IntegerIndividual")
    assert hasattr(_core, "BinaryIndividual")
    assert hasattr(_core, "py_derive_seed")


def test_part5_exports_accessible_from_top_level():
    from evocore import (
        Callback,
        CheckpointCallback,
        EarlyStopping,
        GeneDef,
        GenerationInfo,
        GeneSpace,
        Individual,
        Logbook,
        LogEntry,
        MetricsLogger,
        OperatorSet,
        Population,
        ProcessParallel,
        ThreadParallel,
    )

    exports = (
        Callback,
        CheckpointCallback,
        EarlyStopping,
        GeneDef,
        GenerationInfo,
        LogEntry,
        Logbook,
        MetricsLogger,
        OperatorSet,
        Population,
        ProcessParallel,
        ThreadParallel,
    )
    assert all(export is not None for export in exports)
    assert GeneSpace.uniform(-1.0, 1.0, 2).length == 2
    assert Individual([1.0]).genes == [1.0]


def test_ga_exports_accessible_from_top_level():
    from evocore import GAEngine, MultiRunResult, RunResult

    assert GAEngine is not None
    assert RunResult is not None
    assert MultiRunResult is not None


def test_cmaes_export_accessible_from_top_level():
    from evocore import CMAESEngine

    assert CMAESEngine is not None


def test_vnext_public_exports_are_available() -> None:
    import evocore

    assert evocore.Candidate.__name__ == "Candidate"
    assert evocore.EvaluationRecord.__name__ == "EvaluationRecord"
    assert evocore.Rung.__name__ == "Rung"
    assert evocore.OptimizationTelemetry.__name__ == "OptimizationTelemetry"
    assert evocore.EventRecord.__name__ == "EventRecord"
    assert evocore.EventHistory.__name__ == "EventHistory"
    assert evocore.ReproducibilityMetadata.__name__ == "ReproducibilityMetadata"
