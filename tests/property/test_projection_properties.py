from hypothesis import given
from hypothesis import strategies as st

from evocore import Gene, GeneSpace
from evocore.search_space import ActiveGeneProjection


@given(
    fast=st.floats(min_value=2.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    slow=st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False),
)
def test_projection_round_trip_active_values(fast: float, slow: float) -> None:
    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
            ]
        ),
        active_names=["fast", "slow"],
        schema_id="round-trip",
        schema_version="1",
    )

    projected = projection.project({"fast": fast, "slow": slow})
    reconstructed = projection.reconstruct(projected.optimizer_values)

    assert reconstructed.parameters == projected.parameters
    assert reconstructed.projection_hash == projected.projection_hash


@given(
    inactive_left=st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    inactive_right=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_inactive_hash_invariance(inactive_left: float, inactive_right: float) -> None:
    space = GeneSpace(
        [
            Gene("active", "float", 0.0, 1.0),
            Gene("inactive", "float", -1.0, 1.0),
        ]
    )
    left = ActiveGeneProjection(
        source_space=space,
        active_names=["active"],
        structural_bindings={"inactive": inactive_left},
        schema_id="hash",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=space,
        active_names=["active"],
        structural_bindings={"inactive": inactive_right},
        schema_id="hash",
        schema_version="1",
    )

    assert left.value_hash({"active": 0.5}) == right.value_hash({"active": 0.5})
