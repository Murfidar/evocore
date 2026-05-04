import warnings

import pytest

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


def test_all_errors_subclass_evocore_error():
    for cls in [
        ConfigurationError,
        FitnessError,
        ConvergenceError,
        ParallelError,
        CheckpointError,
    ]:
        assert issubclass(cls, EvocoreError), f"{cls.__name__} must subclass EvocoreError"


def test_evocore_error_subclasses_exception():
    assert issubclass(EvocoreError, Exception)


def test_fitness_warning_subclasses_user_warning():
    assert issubclass(FitnessWarning, UserWarning)


def test_configuration_warning_subclasses_user_warning():
    assert issubclass(ConfigurationWarning, UserWarning)


def test_errors_are_distinct():
    classes = [
        ConfigurationError,
        FitnessError,
        ConvergenceError,
        ParallelError,
        CheckpointError,
    ]
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            assert not issubclass(a, b), f"{a.__name__} must not subclass {b.__name__}"
            assert not issubclass(b, a), f"{b.__name__} must not subclass {a.__name__}"


def test_configuration_error_carries_message():
    with pytest.raises(ConfigurationError, match="gene_bounds"):
        raise ConfigurationError("gene_bounds required for individual_type='float'.")


def test_fitness_warning_can_be_promoted_to_error():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FitnessWarning)
        with pytest.raises(FitnessWarning):
            warnings.warn("8 individuals returned NaN fitness.", FitnessWarning)


def test_configuration_warning_can_be_promoted_to_error():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConfigurationWarning)
        with pytest.raises(ConfigurationWarning):
            warnings.warn("Large int range without sigma.", ConfigurationWarning)
