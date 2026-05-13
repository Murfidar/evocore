from evocore.advisors import InverseDistanceSurrogateAdvisor
from evocore.evaluation import Candidate, EvaluationRecord
from evocore.gene_space import GeneDef, GeneSpace


def _candidate(candidate_id: str, x: float) -> Candidate:
    return Candidate(candidate_id=candidate_id, genes=[x], origin="random", event_index=0)


def test_surrogate_advisor_scores_near_known_good_candidate_higher() -> None:
    advisor = InverseDistanceSurrogateAdvisor()
    good = _candidate("good", 1.0)
    bad = _candidate("bad", 5.0)
    advisor.observe(
        [
            EvaluationRecord("good", score=10.0, confidence="trusted_full", rung="full", cost=1.0),
            EvaluationRecord("bad", score=-10.0, confidence="trusted_full", rung="full", cost=1.0),
        ],
        candidates={"good": good, "bad": bad},
    )

    near_good = _candidate("near_good", 1.1)
    near_bad = _candidate("near_bad", 4.9)
    rankings = advisor.rank([near_bad, near_good])

    assert rankings[0].candidate_id == "near_good"
    assert rankings[0].confidence == "surrogate"


def test_surrogate_advisor_returns_zero_scores_before_observations() -> None:
    advisor = InverseDistanceSurrogateAdvisor()
    rankings = advisor.rank([_candidate("x", 1.0)])

    assert rankings[0].score == 0.0
    assert rankings[0].reason == "no_training_data"


def test_surrogate_advisor_normalizes_mixed_gene_distances() -> None:
    space = GeneSpace(
        [
            GeneDef("wide", "float", 0.0, 1000.0),
            GeneDef("narrow", "float", 0.0, 1.0),
        ]
    )
    advisor = InverseDistanceSurrogateAdvisor(gene_space=space)
    good = Candidate(candidate_id="good", genes=[0.0, 1.0], origin="random", event_index=0)
    bad = Candidate(candidate_id="bad", genes=[1000.0, 0.0], origin="random", event_index=0)
    advisor.observe(
        [
            EvaluationRecord("good", score=10.0, confidence="trusted_full", rung="full", cost=1.0),
            EvaluationRecord("bad", score=-10.0, confidence="trusted_full", rung="full", cost=1.0),
        ],
        candidates={"good": good, "bad": bad},
    )

    near_good_on_normalized_scale = Candidate(
        candidate_id="near_good",
        genes=[800.0, 0.9],
        origin="random",
        event_index=0,
    )
    near_bad_on_normalized_scale = Candidate(
        candidate_id="near_bad",
        genes=[200.0, 0.1],
        origin="random",
        event_index=0,
    )
    rankings = advisor.rank([near_bad_on_normalized_scale, near_good_on_normalized_scale])

    assert rankings[0].candidate_id == "near_good"
