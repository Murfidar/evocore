"""Budget-aware EvoCore vNext GA example."""

from __future__ import annotations

from evocore import (
    EvaluationRecord,
    GAEngine,
    GeneDef,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)


class TwoRungSphere:
    def evaluate(self, candidates, context):
        rung = context.rung
        if rung is None:
            raise ValueError("TwoRungSphere requires a scheduled rung.")
        scale = 0.5 if rung.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
                metrics={"rung": rung.name},
            )
            for candidate in candidates
        ]


def main() -> None:
    space = GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("y", "float", -5.0, 5.0),
        ]
    )
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            Rung("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        full_evaluation_budget=32,
        batch_size=8,
        audit_fraction=0.10,
    )
    result = GAEngine(space, population_size=8, generations=20, seed=42).run(
        TwoRungSphere(),
        policy=policy,
    )
    print(f"best={result.best_fitness:.6f}")
    print(f"full_evals={result.telemetry.candidates_full_evaluated}")
    print(f"partial_evals={result.telemetry.candidates_partial_evaluated}")


if __name__ == "__main__":
    main()
