import json

from hypothesis import given
from hypothesis import strategies as st

from evocore import BoundsPolicy, Gene, GeneSpace
from evocore.core.serialization import stable_json_dumps
from evocore.optimizers.operators import (
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    apply_bounds_policy,
)


@given(
    eta=st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_crossover_signature_is_json_safe(eta, probability):
    payload = CrossoverOperator.sbx(eta=eta, probability=probability).signature()

    assert json.loads(stable_json_dumps(payload)) == payload


@given(
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    individual_probability=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    sigma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_mutation_signature_is_json_safe(probability, individual_probability, sigma):
    payload = MutationOperator.gaussian(
        probability=probability,
        individual_probability=individual_probability,
        sigma=sigma,
    ).signature()

    assert json.loads(stable_json_dumps(payload)) == payload


@given(
    x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    period=st.integers(min_value=-100, max_value=100),
    flag=st.booleans(),
)
def test_bounds_policy_outputs_valid_decoded_values(x, period, flag):
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
        ]
    )

    bounded = apply_bounds_policy([x, period, flag], space, BoundsPolicy.clamp())

    space.validate_genes(bounded)


def test_selection_and_bounds_signatures_are_json_safe():
    for payload in [
        SelectionOperator.tournament(size=3).signature(),
        SelectionOperator.roulette().signature(),
        SelectionOperator.rank().signature(),
        BoundsPolicy.clamp().signature(),
    ]:
        assert json.loads(stable_json_dumps(payload)) == payload
