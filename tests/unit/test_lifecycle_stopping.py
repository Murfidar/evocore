from evocore import (
    CompositeStopPolicy,
    ConvergencePolicy,
    EvaluationLimitPolicy,
    NoImprovementPolicy,
    OptimizationTelemetry,
    UpdateResult,
)


def _update(
    *,
    best_score: float | None = None,
    trusted: int = 0,
    cached: int = 0,
    partial: int = 0,
) -> UpdateResult:
    return UpdateResult(
        accepted_count=trusted + cached + partial,
        trusted_count=trusted,
        partial_count=partial,
        surrogate_count=0,
        cached_count=cached,
        rejected_count=0,
        best_candidate_id="best" if best_score is not None else None,
        best_score=best_score,
    )


def test_evaluation_limit_policy_uses_update_counts() -> None:
    policy = EvaluationLimitPolicy(max_evaluations=3)

    first = policy.observe(_update(trusted=1, cached=1))
    second = policy.observe(_update(trusted=1))

    assert first.stop is False
    assert first.metadata["observed_evaluations"] == 2
    assert second.stop is True
    assert second.reason == "evaluation_limit"
    assert second.metadata["observed_evaluations"] == 3


def test_evaluation_limit_policy_can_ignore_cached_records() -> None:
    policy = EvaluationLimitPolicy(max_evaluations=2, include_cached=False)

    decision = policy.observe(_update(trusted=1, cached=10))

    assert decision.stop is False
    assert decision.metadata["observed_evaluations"] == 1


def test_evaluation_limit_policy_prefers_explicit_telemetry_snapshot() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_full(2, stage="full", cost=2.0)
    telemetry.record_cached(3, stage="cache", cost=0.0)
    policy = EvaluationLimitPolicy(max_evaluations=5)

    decision = policy.observe(telemetry=telemetry)

    assert decision.stop is True
    assert decision.metadata["observed_evaluations"] == 5


def test_no_improvement_policy_stops_after_window_without_improvement() -> None:
    policy = NoImprovementPolicy(window=2, min_delta=0.5, score_direction="maximize")

    assert policy.observe(_update(best_score=10.0)).stop is False
    assert policy.observe(_update(best_score=10.2)).stop is False
    decision = policy.observe(_update(best_score=10.3))

    assert decision.stop is True
    assert decision.reason == "no_improvement"
    assert decision.metadata["best_score"] == 10.0


def test_no_improvement_policy_resets_on_improvement() -> None:
    policy = NoImprovementPolicy(window=2, min_delta=0.5, score_direction="maximize")

    policy.observe(_update(best_score=10.0))
    policy.observe(_update(best_score=10.2))
    decision = policy.observe(_update(best_score=10.8))

    assert decision.stop is False
    assert decision.metadata["stale_count"] == 0


def test_convergence_policy_stops_when_target_reached_for_maximize() -> None:
    policy = ConvergencePolicy(target_score=5.0, score_direction="maximize")

    assert policy.observe(_update(best_score=4.9)).stop is False
    decision = policy.observe(_update(best_score=5.0))

    assert decision.stop is True
    assert decision.reason == "convergence"


def test_convergence_policy_stops_when_target_reached_for_minimize() -> None:
    policy = ConvergencePolicy(target_score=1.0, score_direction="minimize")

    assert policy.observe(_update(best_score=1.1)).stop is False
    assert policy.observe(_update(best_score=1.0)).stop is True


def test_composite_stop_policy_returns_first_stop_decision() -> None:
    policy = CompositeStopPolicy(
        [
            EvaluationLimitPolicy(max_evaluations=10),
            ConvergencePolicy(target_score=5.0, score_direction="maximize"),
        ]
    )

    decision = policy.observe(_update(best_score=5.0, trusted=1))

    assert decision.stop is True
    assert decision.reason == "convergence"
