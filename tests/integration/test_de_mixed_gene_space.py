from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
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


class TwoStageMixedEvaluator:
    def evaluate(self, candidates, context):
        assert context.stage is not None
        scale = 0.5 if context.stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale
                * (
                    float(candidate.params["x"]) ** 2
                    + float(candidate.params["period"] - 8) ** 2
                    + (0.0 if candidate.params["enabled"] else 1.0)
                ),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -2.0, 2.0),
            Gene("period", "int", 2, 12),
            Gene("enabled", "bool"),
        ]
    )


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
    space = _mixed_space()
    optimizer = DifferentialEvolutionOptimizer(
        space, population_size=10, max_generations=4, seed=7
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert type(result.best_solution.values[2]) is bool
    assert result.best_solution.metadata["params"]["enabled"] in (True, False)
    assert result.best_score > -20.0


def test_de_non_default_strategy_runs_on_mixed_gene_space() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=8,
        max_generations=3,
        strategy="current-to-best1bin",
        seed=123,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.final_solutions
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)


def test_de_jde_runs_mixed_bool_numeric_space_smoke() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=8,
        max_generations=3,
        strategy="jde-rand1bin",
        seed=123,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.reproducibility.optimizer_config["parameters"]["strategy"] == "jde-rand1bin"
    assert result.final_solutions
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)


def test_de_jde_mixed_space_run_uses_valid_gene_types() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        max_generations=2,
        strategy="jde-rand1bin",
        seed=42,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    for solution in result.final_solutions:
        assert isinstance(solution.values[0], float)
        assert isinstance(solution.values[1], int)
        assert isinstance(solution.values[2], bool)


def test_de_budgeted_run_supports_mixed_gene_space() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        max_evaluations=18,
        batch_size=6,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        max_generations=4,
        seed=7,
    )

    result = optimizer.run(TwoStageMixedEvaluator(), policy=policy)

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.n_evaluations == 18
    assert result.telemetry.candidates_partial_evaluated > 0
    assert result.telemetry.candidates_full_evaluated == 18
    assert len(result.final_solutions) == 6
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)
