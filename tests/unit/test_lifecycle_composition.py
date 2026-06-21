from evocore import (
    Candidate,
    CandidateSnapshot,
    EvaluationRecord,
    GeneSpace,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)


def _snapshot() -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id="outer-1",
        candidate_hash="hash-outer",
        values=(1.0, 2.0),
        params={"x": 1.0, "y": 2.0},
        origin="memory_seed",
        batch_id="outer-batch",
        event_index=3,
        generation=None,
        status="trusted",
        stage="template",
        confidence="cached",
        score=10.0,
        scores={},
        cost=0.0,
        metadata={"family": "template-a"},
    )


def test_derive_child_seed_is_deterministic() -> None:
    first = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")
    second = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")

    assert first == second
    assert isinstance(first, int)
    assert 0 <= first < 2**32


def test_derive_child_seed_changes_with_hash_or_stage() -> None:
    base = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")

    assert derive_child_seed(parent_seed=42, candidate_hash="def", stage="inner_cma") != base
    assert derive_child_seed(parent_seed=42, candidate_hash="abc", stage="audit") != base


def test_lineage_metadata_from_candidate_snapshot_is_json_safe() -> None:
    metadata = lineage_metadata(
        outer_candidate=_snapshot(),
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=123,
        stage="inner_cma",
        checkpoint_path="runs/inner-1.json",
        metadata={"archive_id": "template-a"},
    )

    assert metadata == {
        "outer_candidate_id": "outer-1",
        "outer_candidate_hash": "hash-outer",
        "outer_batch_id": "outer-batch",
        "inner_optimizer_type": "CMAESOptimizer",
        "inner_seed": 123,
        "composition_stage": "inner_cma",
        "inner_checkpoint_path": "runs/inner-1.json",
        "archive_id": "template-a",
    }


def test_lineage_metadata_from_candidate_requires_gene_space_for_hash() -> None:
    space = GeneSpace.uniform(-5.0, 5.0, 2)
    candidate = Candidate(
        candidate_id="outer-raw",
        genes=[1.0, 2.0],
        batch_id="outer-batch",
    )

    metadata = lineage_metadata(
        outer_candidate=candidate,
        gene_space=space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=123,
        stage="inner_cma",
    )

    assert metadata["outer_candidate_id"] == "outer-raw"
    assert metadata["outer_candidate_hash"] == space.value_hash([1.0, 2.0])


def test_lineage_metadata_protects_canonical_fields_from_user_metadata() -> None:
    metadata = lineage_metadata(
        outer_candidate=_snapshot(),
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=123,
        stage="inner_cma",
        metadata={
            "outer_candidate_id": "forged",
            "outer_candidate_hash": "forged-hash",
            "outer_batch_id": "forged-batch",
            "inner_optimizer_type": "ForgedOptimizer",
            "inner_seed": 999,
            "composition_stage": "forged-stage",
            "inner_checkpoint_path": "forged.json",
            "template_name": "template-a",
        },
    )

    assert metadata == {
        "outer_candidate_id": "outer-1",
        "outer_candidate_hash": "hash-outer",
        "outer_batch_id": "outer-batch",
        "inner_optimizer_type": "CMAESOptimizer",
        "inner_seed": 123,
        "composition_stage": "inner_cma",
        "template_name": "template-a",
    }


def test_inner_result_record_targets_outer_candidate() -> None:
    snapshot = _snapshot()
    record = inner_result_record(
        outer_candidate=snapshot,
        score=17.5,
        confidence="trusted_full",
        stage="inner_cma",
        cost=32.0,
        metrics={"inner_generations": 4},
        metadata={"inner_seed": 123},
    )

    assert record == EvaluationRecord(
        candidate_id="outer-1",
        batch_id="outer-batch",
        score=17.5,
        confidence="trusted_full",
        stage="inner_cma",
        cost=32.0,
        metrics={"inner_generations": 4},
        metadata={"inner_seed": 123},
    )
