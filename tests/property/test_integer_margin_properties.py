from hypothesis import given, strategies as st

from evocore.optimizers.cmaes import IntegerMarginDistribution


@given(
    low=st.integers(-5, 0),
    high=st.integers(1, 8),
    mean=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    sigma=st.floats(0.05, 10, allow_nan=False, allow_infinity=False),
)
def test_integer_margin_probabilities_are_bounded_and_normalized(
    low: int,
    high: int,
    mean: float,
    sigma: float,
) -> None:
    margin = IntegerMarginDistribution(low=low, high=high, min_probability=0.01)
    probabilities = margin.probabilities(mean=mean, sigma=sigma)

    assert set(probabilities) == set(range(low, high + 1))
    assert abs(sum(probabilities.values()) - 1.0) < 1.0e-12
    assert all(value >= 0.01 for value in probabilities.values())
