from evocore import (
    CMAESOptimizer,
    EvaluationRecord,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)


def test_outer_ga_inner_cma_helper_flow() -> None:
    outer_space = GeneSpace.uniform(-5.0, 5.0, 2)
    outer = GeneticAlgorithmOptimizer(outer_space, population_size=4, seed=100)
    outer_candidate = outer.ask(1)[0]
    outer_hash = outer_candidate.candidate_hash(outer_space)

    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=outer_hash,
        stage="inner_cma",
    )
    inner = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=inner_seed)

    inner_candidates = inner.ask(4)
    inner_update = inner.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="inner_full",
            )
            for index, candidate in enumerate(inner_candidates)
        ]
    )
    metadata = lineage_metadata(
        outer_candidate=outer_candidate,
        gene_space=outer_space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=inner_seed,
        stage="inner_cma",
    )
    record = inner_result_record(
        outer_candidate=outer_candidate,
        gene_space=outer_space,
        score=inner_update.best_score,
        confidence="trusted_full",
        stage="inner_cma",
        metadata=metadata,
    )

    update = outer.tell([record])

    assert update.trusted_count == 1
    assert update.best_candidate_id == outer_candidate.candidate_id
