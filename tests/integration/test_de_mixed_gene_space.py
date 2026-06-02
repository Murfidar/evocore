from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    Gene,
    GeneSpace,
)


class NumericSphereEvaluator:
    def evaluate(self, candidates, context: EvaluationContext):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class MixedSwitchEvaluator:
    def evaluate(self, candidates, context: EvaluationContext):
        records = []
        for candidate in candidates:
            x, period, enabled = candidate.genes
            score = -abs(float(x) - 0.25) - abs(int(period) - 7)
            if enabled:
                score += 2.0
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence=context.stage.confidence,
                    stage=context.stage.name,
                    cost=context.stage.budget,
                )
            )
        return records


def test_de_improves_numeric_sphere_smoke() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 4),
        population_size=12,
        max_generations=5,
        seed=42,
    )

    result = optimizer.run(NumericSphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.best_score > -100.0
    assert result.n_evaluations > 0


def test_de_runs_mixed_bool_numeric_space_smoke() -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -2.0, 2.0),
            Gene("period", "int", 2, 12),
            Gene("enabled", "bool"),
        ]
    )
    optimizer = DifferentialEvolutionOptimizer(
        space, population_size=10, max_generations=4, seed=7
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert type(result.best_solution.values[2]) is bool
    assert result.best_solution.metadata["params"]["enabled"] in (True, False)
    assert result.best_score > -20.0
