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
