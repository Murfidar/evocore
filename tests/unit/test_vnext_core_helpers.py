from evocore import _core


def test_candidate_id_is_deterministic_and_distinguishes_index() -> None:
    left = _core.candidate_id(42, 3, 0)
    right = _core.candidate_id(42, 3, 0)
    other = _core.candidate_id(42, 3, 1)

    assert left == right
    assert left != other
    assert left.startswith("c-")


def test_rank_top_k_prefers_trusted_then_score() -> None:
    indices = _core.rank_top_k(
        scores=[0.9, 10.0, 0.7, 0.5],
        trusted_mask=[True, False, True, True],
        k=2,
    )

    assert indices == [0, 2]


def test_rank_top_k_uses_score_when_trust_matches() -> None:
    indices = _core.rank_top_k(
        scores=[1.0, 3.0, 2.0],
        trusted_mask=[True, True, True],
        k=3,
    )

    assert indices == [1, 2, 0]
