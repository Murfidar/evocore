"""Budget-aware EvoCore vNext GA example."""

from __future__ import annotations

from evocore import (
    BudgetPolicy,
    EvaluationRecord,
    EvaluationStage,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
)


class TwoStageSphere:
    def evaluate(self, candidates, context):
        stage = context.stage
        if stage is None:
            raise ValueError("TwoStageSphere requires a scheduled stage.")
        scale = 0.5 if stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
                metrics={"stage": stage.name},
            )
            for candidate in candidates
        ]


def main() -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        max_evaluations=32,
        batch_size=8,
        audit_fraction=0.10,
    )
    result = GeneticAlgorithmOptimizer(space, population_size=8, max_generations=20, seed=42).run(
        TwoStageSphere(),
        policy=policy,
    )
    print(f"best={result.best_score:.6f}")
    print(f"full_evals={result.telemetry.candidates_full_evaluated}")
    print(f"partial_evals={result.telemetry.candidates_partial_evaluated}")


if __name__ == "__main__":
    main()
