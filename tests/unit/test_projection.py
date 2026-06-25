import pytest

from evocore import Gene, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ExponentialIntegerTransform,
    IdentityTransform,
    RepairRecord,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("family", "int", 0, 2),
            Gene("fast", "float", 2.0, 20.0),
            Gene("slow", "float", 10.0, 80.0),
            Gene("use_filter", "float", 0.0, 1.0),
            Gene("inactive", "float", -1.0, 1.0),
        ]
    )


def test_projection_requires_named_source_space() -> None:
    with pytest.raises(ConfigurationError, match="named GeneSpace"):
        ActiveGeneProjection(
            source_space=GeneSpace.uniform(-1.0, 1.0, 2),
            active_names=["gene_0"],
        )


def test_projection_canonicalizes_active_order_and_reconstructs() -> None:
    projection = ActiveGeneProjection(
        source_space=_space(),
        active_names=["slow", "fast", "use_filter"],
        structural_bindings={"family": 1, "inactive": 0.25},
        transforms={"use_filter": BinaryThresholdTransform(threshold=0.5)},
        schema_id="template-a",
        schema_version="1",
    )

    result = projection.reconstruct([5.5, 34.0, 0.75])

    assert projection.optimizer_space.names == ["fast", "slow", "use_filter"]
    assert result.parameters == {
        "family": 1,
        "fast": 5.5,
        "slow": 34.0,
        "use_filter": True,
        "inactive": 0.25,
    }
    assert result.optimizer_values == (5.5, 34.0, 0.75)
    assert result.active_names == ("fast", "slow", "use_filter")
    assert result.repairs == ()
    assert result.violations == ()
    assert result.valid is True


def test_inactive_changes_do_not_change_projected_hash() -> None:
    left = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1, "inactive": -1.0},
        schema_id="template-a",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1, "inactive": 1.0},
        schema_id="template-a",
        schema_version="1",
    )

    values = {"family": 1, "fast": 4.0, "slow": 40.0, "inactive": 0.0}

    assert left.value_hash(values) == right.value_hash(values)


def test_structural_changes_do_change_projected_hash() -> None:
    left = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1},
        identity_keys=("family",),
        schema_id="template-a",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 2},
        identity_keys=("family",),
        schema_id="template-a",
        schema_version="1",
    )

    assert left.value_hash({"fast": 4.0, "slow": 40.0}) != right.value_hash(
        {"fast": 4.0, "slow": 40.0}
    )


def test_project_rejects_structural_binding_mismatch() -> None:
    projection = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1},
        identity_keys=("family",),
        schema_id="template-a",
        schema_version="1",
    )

    with pytest.raises(ConfigurationError, match="structural binding"):
        projection.project({"family": 2, "fast": 4.0, "slow": 40.0})


def test_project_reencodes_active_values_after_repair() -> None:
    class ForceFastRepair:
        checkpointable = True

        def repair(self, parameters):
            repaired = dict(parameters)
            previous = repaired["fast"]
            repaired["fast"] = 9.0
            return repaired, [
                RepairRecord(
                    name="fast",
                    previous=previous,
                    repaired=9.0,
                    reason="force test repair",
                )
            ]

        def signature(self):
            return {"type": "force_fast", "version": 1}

    projection = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        repairs=[ForceFastRepair()],
        schema_id="template-a",
        schema_version="1",
    )

    result = projection.project({"fast": 4.0, "slow": 40.0})

    assert result.parameters["fast"] == 9.0
    assert result.optimizer_values == (9.0, 40.0)


def test_transform_versions_participate_in_snapshot_hash() -> None:
    first = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast"],
        transforms={"fast": IdentityTransform()},
        schema_id="template-a",
        schema_version="1",
    )
    second = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast"],
        transforms={"fast": ExponentialIntegerTransform(base=2.0)},
        schema_id="template-a",
        schema_version="1",
    )

    assert first.snapshot().signature_hash != second.snapshot().signature_hash
