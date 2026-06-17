from evocore import EvaluationLimitPolicy, EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer


def test_stop_policy_can_drive_manual_ask_tell_loop() -> None:
    optimizer = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 2),
        population_size=4,
        seed=42,
    )
    stop_policy = EvaluationLimitPolicy(max_evaluations=4)
    candidates = optimizer.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=float(index),
            confidence="trusted_full",
            stage="full",
        )
        for index, candidate in enumerate(candidates)
    ]

    update = optimizer.tell(records)
    decision = stop_policy.observe(update, snapshot=optimizer.candidate_snapshot(scope="trusted"))

    assert decision.stop is True
    assert decision.reason == "evaluation_limit"
