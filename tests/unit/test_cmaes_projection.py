import pytest

from evocore import Gene, GeneSpace, WarmStartRecord
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import build_projected_cma_mean
from evocore.search_space import ActiveGeneProjection, BinaryThresholdTransform, RepairRecord


def _projection() -> ActiveGeneProjection:
    return ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": BinaryThresholdTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )


def test_build_projected_cma_mean_from_best_matching_record() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[
            WarmStartRecord(
                params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True},
                score=12.0,
            ),
            WarmStartRecord(
                params={"template": 1, "fast": 8.0, "slow": 50.0, "flag": False},
                score=10.0,
            ),
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean == [5.0, 40.0, 1.0]
    assert result.accepted_count == 2


def test_projected_mean_rejects_template_mismatch() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[
            WarmStartRecord(
                params={"template": 2, "fast": 5.0, "slow": 40.0, "flag": True},
                score=12.0,
            )
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean is None
    assert result.rejected[0]["reason"] == "projection_mismatch"


def test_projected_mean_reports_non_invertible_transform() -> None:
    class DecodeOnlyTransform:
        checkpointable = True

        def decode(self, value):
            return bool(value)

        def encode(self, value):
            raise ConfigurationError("DecodeOnlyTransform cannot encode historical values.")

        def signature(self):
            return {"type": "decode_only", "version": 1}

    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": DecodeOnlyTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )

    with pytest.raises(ConfigurationError, match="encode"):
        build_projected_cma_mean(
            projection=projection,
            records=[
                WarmStartRecord(
                    params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True},
                    score=12.0,
                )
            ],
            direction="maximize",
            strategy="best",
        )


def test_projected_mean_uses_repaired_optimizer_values() -> None:
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
        source_space=GeneSpace(
            [
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
            ]
        ),
        active_names=["fast", "slow"],
        repairs=[ForceFastRepair()],
        schema_id="template-1",
        schema_version="1",
    )

    result = build_projected_cma_mean(
        projection=projection,
        records=[
            WarmStartRecord(
                params={"fast": 5.0, "slow": 40.0},
                score=12.0,
            )
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean == [9.0, 40.0]
