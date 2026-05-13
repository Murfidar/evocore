import pytest

from evocore.mixed_cma import CategoricalState, IntegerMargin


def test_integer_margin_keeps_probability_mass_inside_bounds() -> None:
    margin = IntegerMargin(low=0, high=3, min_probability=0.10)

    probabilities = margin.probabilities(mean=1.4, sigma=0.2)

    assert set(probabilities) == {0, 1, 2, 3}
    assert all(value >= 0.10 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_categorical_state_updates_toward_better_category() -> None:
    state = CategoricalState(categories=(0, 1, 2), learning_rate=0.5)

    state.update(weighted_observations=[(2, 1.0), (1, 0.0)])

    assert state.probabilities[2] > state.probabilities[0]
    assert state.probabilities[2] > state.probabilities[1]
    assert sum(state.probabilities.values()) == pytest.approx(1.0)
