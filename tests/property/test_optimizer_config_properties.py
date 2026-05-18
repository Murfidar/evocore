import json

from hypothesis import given
from hypothesis import strategies as st

from evocore import CMAESOptimizer, GeneSpace, GeneticAlgorithmOptimizer


@given(
    population_size=st.integers(min_value=2, max_value=20),
    max_generations=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_ga_config_signature_round_trips_through_json(
    population_size: int,
    max_generations: int,
    seed: int,
) -> None:
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
    )

    payload = engine.config_signature()

    assert json.loads(engine.config().to_json()) == payload
    assert engine.config_hash() == engine.config_hash()


@given(
    population_size=st.integers(min_value=2, max_value=20),
    max_generations=st.integers(min_value=0, max_value=20),
    initial_sigma=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_cmaes_config_signature_round_trips_through_json(
    population_size: int,
    max_generations: int,
    initial_sigma: float,
) -> None:
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=population_size,
        max_generations=max_generations,
        initial_sigma=initial_sigma,
    )

    payload = engine.config_signature()

    assert json.loads(engine.config().to_json()) == payload
    assert engine.config_hash() == engine.config_hash()


@given(seed=st.integers(min_value=0, max_value=2**32 - 2))
def test_ga_seed_change_alters_hash(seed: int) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    left = GeneticAlgorithmOptimizer(space, seed=seed)
    right = GeneticAlgorithmOptimizer(space, seed=seed + 1)

    assert left.config_hash() != right.config_hash()


@given(
    initial_sigma=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_cmaes_initial_sigma_change_alters_hash(initial_sigma: float) -> None:
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    left = CMAESOptimizer(space, initial_sigma=initial_sigma)
    right = CMAESOptimizer(space, initial_sigma=initial_sigma + 0.25)

    assert left.config_hash() != right.config_hash()
