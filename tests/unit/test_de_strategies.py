import pytest

from evocore.core.errors import ConfigurationError
from evocore.optimizers.de.strategies import (
    strategy_spec_for,
    supported_strategy_names,
    validate_strategy_population_size,
)


def test_supported_strategy_names_include_all_builtins() -> None:
    assert supported_strategy_names() == (
        "rand1bin",
        "best1bin",
        "rand2bin",
        "current-to-best1bin",
        "jde-rand1bin",
    )


def test_strategy_spec_for_returns_rand1bin_contract() -> None:
    spec = strategy_spec_for("rand1bin")

    assert spec.name == "rand1bin"
    assert spec.min_population_size == 4
    assert spec.is_adaptive is False
    assert spec.checkpoint_state_schema is None


def test_strategy_spec_for_returns_jde_contract() -> None:
    spec = strategy_spec_for("jde-rand1bin")

    assert spec.name == "jde-rand1bin"
    assert spec.min_population_size == 4
    assert spec.is_adaptive is True
    assert spec.checkpoint_state_schema == 1


def test_strategy_spec_for_rejects_unknown_strategy() -> None:
    with pytest.raises(
        ConfigurationError,
        match=(
            "strategy must be one of 'rand1bin', 'best1bin', "
            "'rand2bin', 'current-to-best1bin', 'jde-rand1bin'"
        ),
    ):
        strategy_spec_for("jade")


@pytest.mark.parametrize(
    ("strategy", "minimum"),
    [
        ("rand1bin", 4),
        ("best1bin", 4),
        ("rand2bin", 6),
        ("current-to-best1bin", 4),
        ("jde-rand1bin", 4),
    ],
)
def test_validate_strategy_population_size(strategy: str, minimum: int) -> None:
    validate_strategy_population_size(strategy, minimum)
    with pytest.raises(ConfigurationError, match="population_size must be at least"):
        validate_strategy_population_size(strategy, minimum - 1)
