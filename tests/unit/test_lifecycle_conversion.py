import pytest

from evocore.lifecycle import (
    Candidate,
    EvaluationRecord,
    candidate_to_solution,
    solution_to_candidate,
)
from evocore.search_space import Gene, GeneSpace, Solution


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def test_candidate_to_solution_copies_state_eligible_score_and_provenance() -> None:
    space = _space()
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-1",
        genes=[1.0, 5, True],
        params={"x": 1.0, "period": 5, "enabled": True},
        origin="random",
        event_index=3,
        generation=2,
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=10.0,
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
    )

    solution = candidate_to_solution(candidate, direction="maximize", gene_space=space)

    assert solution.values == [1.0, 5, True]
    assert solution.score == pytest.approx(10.0)
    assert solution.score_valid is True
    assert solution.metadata["params"] == {"x": 1.0, "period": 5, "enabled": True}
    assert solution.metadata["candidate_id"] == "c-1"
    assert solution.metadata["candidate_hash"] == space.value_hash([1.0, 5, True])
    assert solution.metadata["batch_id"] == "b-1"
    assert solution.metadata["origin"] == "random"
    assert solution.metadata["generation"] == 2
    assert "stage" not in solution.metadata
    assert "status" not in solution.metadata
    assert "scores" not in solution.metadata
    assert not hasattr(solution, "stage")
    assert not hasattr(solution, "status")


def test_candidate_to_solution_uses_raw_minimize_state_score() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[0.0], event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=5.0,
            confidence="cached",
            stage="full",
            cost=0.0,
        )
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=2.0,
            confidence="trusted_full",
            stage="rerun",
            cost=1.0,
        )
    )

    solution = candidate_to_solution(candidate, direction="minimize")

    assert solution.score == pytest.approx(2.0)
    assert solution.score_valid is True


def test_candidate_to_solution_leaves_non_state_observation_invalid() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[0.0], event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=99.0,
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
    )

    solution = candidate_to_solution(candidate, direction="maximize")

    assert solution.values == [0.0]
    assert solution.score is None
    assert solution.score_valid is False
    assert solution.metadata["candidate_id"] == "c-1"


def test_candidate_to_solution_can_omit_provenance() -> None:
    candidate = Candidate(candidate_id="c-1", batch_id="b-1", genes=[0.0], event_index=0)

    solution = candidate_to_solution(
        candidate,
        direction="maximize",
        include_provenance=False,
    )

    assert solution.metadata == {}


def test_solution_to_candidate_recomputes_params_and_does_not_copy_score() -> None:
    space = _space()
    solution = Solution(
        [1.0, 5, True],
        score=123.0,
        score_valid=True,
        metadata={"params": {"x": "stale"}},
    )

    candidate = solution_to_candidate(
        solution,
        gene_space=space,
        candidate_id="c-2",
        batch_id="b-2",
        origin="memory_seed",
        event_index=4,
        parents=("c-parent",),
        generation=3,
        metadata={"source": "unit"},
    )

    assert candidate.candidate_id == "c-2"
    assert candidate.batch_id == "b-2"
    assert candidate.genes == [1.0, 5, True]
    assert candidate.params == {"x": 1.0, "period": 5, "enabled": True}
    assert candidate.origin == "memory_seed"
    assert candidate.parents == ("c-parent",)
    assert candidate.event_index == 4
    assert candidate.generation == 3
    assert candidate.metadata == {"source": "unit"}
    assert candidate.scores == {}
    assert candidate.confidence is None
    assert candidate.status == "proposed"
