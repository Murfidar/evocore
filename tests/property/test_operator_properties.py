from hypothesis import given
from hypothesis import strategies as st

from evocore._core import (
    bit_flip_mutation,
    gaussian_mutation,
    int_uniform_mutation,
    one_point_crossover,
    py_derive_seed,
    two_point_crossover,
    uniform_crossover,
    uniform_mutation,
)

bounded_float = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

float_lists = st.lists(
    bounded_float,
    min_size=1,
    max_size=25,
)

binary_lists = st.lists(st.sampled_from([0.0, 1.0]), min_size=2, max_size=30)


@given(float_lists, st.integers(min_value=0, max_value=2**32 - 1))
def test_gaussian_mutation_is_deterministic(genes, seed):
    left = gaussian_mutation(genes, 0.5, 0.75, seed, 3, 4)
    right = gaussian_mutation(genes, 0.5, 0.75, seed, 3, 4)

    assert left == right


@given(
    float_lists,
    st.floats(min_value=-50.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_uniform_mutation_respects_bounds(genes, low, span):
    high = low + span

    mutated = uniform_mutation(genes, low, high, 1.0, 42, 0, 0)

    assert all(low <= value < high for value in mutated)


@given(
    st.lists(bounded_float, min_size=1, max_size=25),
    st.integers(min_value=-100, max_value=99),
    st.integers(min_value=1, max_value=100),
)
def test_int_uniform_mutation_outputs_integer_values(genes, low, span):
    high = low + span

    mutated = int_uniform_mutation(genes, float(low), float(high), 1.0, 42, 0, 0)

    assert all(float(low) <= value <= float(high) for value in mutated)
    assert all(value == int(value) for value in mutated)


@given(binary_lists)
def test_bit_flip_mutation_returns_binary_values(genes):
    mutated = bit_flip_mutation(genes, 0.5, 42, 0, 0)

    assert all(value in {0.0, 1.0} for value in mutated)


@given(binary_lists, binary_lists)
def test_binary_crossovers_return_binary_values(left, right):
    size = min(len(left), len(right))
    left = left[:size]
    right = right[:size]

    for first, second in [
        one_point_crossover(left, right, 42, 0, 0),
        two_point_crossover(left, right, 42, 0, 0),
        uniform_crossover(left, right, 0.5, 42, 0, 0),
    ]:
        assert len(first) == size
        assert len(second) == size
        assert all(value in {0.0, 1.0} for value in first)
        assert all(value in {0.0, 1.0} for value in second)


@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=10),
)
def test_derive_seed_is_stable(master_seed, generation, individual_idx, op):
    assert py_derive_seed(master_seed, generation, individual_idx, op) == py_derive_seed(
        master_seed,
        generation,
        individual_idx,
        op,
    )
