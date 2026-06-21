import math

import pytest

from evocore import CandidateSnapshot, FamilyQuota, SpecialistCap, select_candidates
from evocore.core import ConfigurationError


def _candidate(
    candidate_id: str,
    candidate_hash: str,
    score: float | None,
    *,
    family: str | None = "core",
    specialist: str | None = None,
) -> CandidateSnapshot:
    metadata = {}
    if family is not None:
        metadata["family"] = family
    if specialist is not None:
        metadata["specialist"] = specialist
    return CandidateSnapshot(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        values=(score or 0.0,),
        params=None,
        origin="memory_seed",
        batch_id="batch-1",
        event_index=1,
        generation=None,
        status="trusted",
        stage="full",
        confidence="trusted_full",
        score=score,
        scores={},
        cost=0.0,
        metadata=metadata,
    )


def test_select_candidates_top_k_is_direction_aware_and_deterministic() -> None:
    result = select_candidates(
        [
            _candidate("c-low", "h-low", 1.0),
            _candidate("c-high", "h-high", 9.0),
            _candidate("c-tie", "h-tie", 9.0),
        ],
        k=2,
        score_direction="maximize",
    )

    assert [item.candidate_id for item in result.selected] == ["c-high", "c-tie"]
    assert result.summary["selected"] == 2
    assert [decision.reason for decision in result.decisions if not decision.selected] == [
        "overflow"
    ]


def test_select_candidates_minimize_prefers_lower_scores() -> None:
    result = select_candidates(
        [
            _candidate("c-low", "h-low", 1.0),
            _candidate("c-high", "h-high", 9.0),
        ],
        k=1,
        score_direction="minimize",
    )

    assert [item.candidate_id for item in result.selected] == ["c-low"]


def test_select_candidates_suppresses_duplicate_hashes() -> None:
    result = select_candidates(
        [
            _candidate("c-first", "same", 8.0),
            _candidate("c-second", "same", 7.0),
            _candidate("c-third", "other", 6.0),
        ],
        k=3,
        score_direction="maximize",
        duplicate_policy="suppress",
    )

    assert [item.candidate_id for item in result.selected] == ["c-first", "c-third"]
    assert result.rejected[0].candidate_id == "c-second"
    assert result.decisions[1].reason == "duplicate"


def test_select_candidates_enforces_family_quota() -> None:
    result = select_candidates(
        [
            _candidate("c-a1", "h-a1", 9.0, family="a"),
            _candidate("c-a2", "h-a2", 8.0, family="a"),
            _candidate("c-b1", "h-b1", 7.0, family="b"),
        ],
        k=3,
        score_direction="maximize",
        quotas=[FamilyQuota(metadata_key="family", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-a1", "c-b1"]
    assert result.rejected[0].candidate_id == "c-a2"
    assert result.decisions[1].reason == "quota:family"


def test_select_candidates_enforces_specialist_cap() -> None:
    result = select_candidates(
        [
            _candidate("c-s1", "h-s1", 9.0, specialist="fast"),
            _candidate("c-s2", "h-s2", 8.0, specialist="fast"),
            _candidate("c-s3", "h-s3", 7.0, specialist="slow"),
        ],
        k=3,
        score_direction="maximize",
        caps=[SpecialistCap(metadata_key="specialist", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-s1", "c-s3"]
    assert result.rejected[0].candidate_id == "c-s2"
    assert result.decisions[1].reason == "cap:specialist"


def test_select_candidates_missing_metadata_defaults_to_unknown_bucket() -> None:
    result = select_candidates(
        [
            _candidate("c-1", "h-1", 9.0, family=None),
            _candidate("c-2", "h-2", 8.0, family=None),
        ],
        k=2,
        score_direction="maximize",
        quotas=[FamilyQuota(metadata_key="family", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-1"]
    assert result.rejected[0].candidate_id == "c-2"


def test_select_candidates_strict_missing_metadata_raises() -> None:
    with pytest.raises(ConfigurationError, match="missing metadata key"):
        select_candidates(
            [_candidate("c-1", "h-1", 9.0, family=None)],
            k=1,
            score_direction="maximize",
            quotas=[FamilyQuota(metadata_key="family", max_count=1)],
            missing_metadata="error",
        )


def test_select_candidates_rejects_candidates_without_score() -> None:
    result = select_candidates(
        [_candidate("c-1", "h-1", None)],
        k=1,
        score_direction="maximize",
    )

    assert result.selected == ()
    assert result.rejected[0].candidate_id == "c-1"
    assert result.decisions[0].reason == "no_score"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_select_candidates_rejects_non_finite_scores(score: float) -> None:
    result = select_candidates(
        [_candidate("c-1", "h-1", score)],
        k=1,
        score_direction="maximize",
    )

    assert result.selected == ()
    assert result.rejected[0].candidate_id == "c-1"
    assert result.decisions[0].reason == "non_finite_score"
