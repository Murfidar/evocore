from evocore import (
    CandidateArchive,
    CMAESOptimizer,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
    derive_child_seed,
)
from evocore.lifecycle import constraint_penalty_record
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ConstraintViolation,
    ExponentialIntegerTransform,
)


def _template_projection(family: int) -> ActiveGeneProjection:
    return ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("family", "int", 0, 2),
                Gene("fast_log", "float", 1.0, 4.0),
                Gene("use_filter", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast_log", "use_filter"],
        structural_bindings={"family": family},
        transforms={
            "fast_log": ExponentialIntegerTransform(base=2.0),
            "use_filter": BinaryThresholdTransform(),
        },
        identity_keys=("family",),
        schema_id="synthetic-template",
        schema_version="1",
    )


def test_template_outer_ga_inner_cma_projection_recipe_is_deterministic() -> None:
    outer_space = GeneSpace([Gene("family", "int", 0, 2), Gene("mode", "int", 0, 1)])
    outer = GeneticAlgorithmOptimizer(outer_space, population_size=6, seed=44)
    archive = CandidateArchive(score_direction="maximize")
    outer_candidate = outer.ask(1)[0]
    family = int(outer_candidate.genes[0])
    projection = _template_projection(family)
    inner_seed = derive_child_seed(
        parent_seed=44,
        candidate_hash=outer_candidate.candidate_hash(outer_space),
        stage="inner_cma",
    )
    inner = CMAESOptimizer(
        projection.optimizer_space,
        population_size=4,
        seed=inner_seed,
        integer_strategy="margin",
    )

    prior = WarmStartRecord(values=(2.0, 1.0), score=4.0, confidence="cached")
    inner.warm_start([prior], mode="state")
    inner_batch = inner.ask()
    records = []
    for candidate in inner_batch:
        decoded = projection.reconstruct(candidate.genes)
        if decoded.parameters["fast_log"] < 4:
            records.append(
                constraint_penalty_record(
                    candidate=candidate,
                    stage="projection",
                    direction="maximize",
                    violations=[
                        ConstraintViolation(
                            code="min_fast_period",
                            message="fast period must be at least 4",
                            names=("fast_log",),
                        )
                    ],
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )
        else:
            score = float(decoded.parameters["fast_log"]) + float(decoded.parameters["family"])
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence="trusted_full",
                    stage="full",
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )

    update = inner.tell(records)
    trusted = inner.top_candidates(2)
    archive.add_population(inner.candidate_snapshot(scope="trusted"), source="inner_cma")

    assert update.state_accepted_count == 4
    assert inner.state_summary().pending_batch_ids == ()
    assert any(record.confidence == "constraint_penalty" for record in records)
    assert all(snapshot.confidence != "constraint_penalty" for snapshot in trusted)
    assert archive.to_warm_start_records(k=4)
