import pytest

from evocore import Candidate, EvaluationRecord, Gene, GeneSpace, WarmStartRecord
from evocore.core import ConfigurationError, FitnessError
from evocore.lifecycle.external import (
    build_candidate_snapshot,
    build_population_snapshot,
    resolve_warm_start_values,
    top_candidate_snapshots,
)
from evocore.lifecycle.telemetry import OptimizationTelemetry


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("fast", "int", 1, 10),
            Gene("slow", "int", 5, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_warm_start_record_resolves_params_in_gene_order() -> None:
    record = WarmStartRecord(
        params={"slow": 21, "enabled": True, "fast": 3},
        score=12.5,
        metadata={"source": "search_memory"},
    )

    assert resolve_warm_start_values(record, _space()) == (3, 21, True)


def test_warm_start_record_rejects_missing_values_and_params() -> None:
    with pytest.raises(ConfigurationError, match="values or params"):
        WarmStartRecord(score=1.0)


def test_warm_start_record_rejects_values_and_params_together() -> None:
    with pytest.raises(ConfigurationError, match="not both"):
        WarmStartRecord(values=(1, 5, False), params={"fast": 1}, score=1.0)


def test_warm_start_record_rejects_non_state_confidence() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full or cached"):
        WarmStartRecord(values=(1, 5, False), score=1.0, confidence="partial")


def test_resolve_warm_start_values_rejects_unknown_param() -> None:
    record = WarmStartRecord(
        params={"fast": 3, "slow": 21, "enabled": True, "extra": 1},
        score=1.0,
    )

    with pytest.raises(ConfigurationError, match="unknown parameter"):
        resolve_warm_start_values(record, _space())


def test_cached_records_converts_hash_mapping_to_evaluation_records() -> None:
    from evocore import cached_records

    space = _space()
    candidate = Candidate(
        candidate_id="c-1",
        genes=[3, 21, True],
        params=space.params_for([3, 21, True]),
        batch_id="b-1",
        event_index=0,
        metadata={"candidate_source": "ask"},
    )
    cache_key = space.value_hash(candidate.genes)

    records = cached_records(
        [candidate],
        gene_space=space,
        cache={
            cache_key: {
                "score": 99.0,
                "metrics": {"fold": 2},
                "metadata": {"cache_reason": "exact_hash"},
            }
        },
        stage="search_memory",
        cost=0.0,
        metadata={"cache_table": "trusted_elites"},
    )

    assert records == (
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=99.0,
            confidence="cached",
            stage="search_memory",
            cost=0.0,
            metrics={"fold": 2},
            metadata={
                "cache_key": cache_key,
                "cache_table": "trusted_elites",
                "cache_reason": "exact_hash",
            },
        ),
    )


def test_cached_records_rejects_non_finite_cached_score() -> None:
    from evocore import cached_records

    space = _space()
    candidate = Candidate(candidate_id="c-1", genes=[3, 21, True], batch_id="b-1")
    cache_key = space.value_hash(candidate.genes)

    with pytest.raises(FitnessError, match="finite score"):
        cached_records(
            [candidate],
            gene_space=space,
            cache={cache_key: {"score": float("nan")}},
            stage="search_memory",
        )


def _scored_candidate(candidate_id: str, values: list[object], score: float) -> Candidate:
    candidate = Candidate(candidate_id=candidate_id, genes=list(values), batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id=candidate_id,
            batch_id="b-1",
            score=score,
            confidence="cached",
            stage="warm_start",
            metadata={"fold": 1},
        )
    )
    return candidate


def test_candidate_snapshot_is_detached_from_candidate_mutation() -> None:
    space = _space()
    candidate = _scored_candidate("c-1", [3, 21, True], 10.0)

    snapshot = build_candidate_snapshot(candidate, gene_space=space, direction="maximize")
    candidate.metadata["record_metadata"]["fold"] = 9

    assert snapshot.candidate_id == "c-1"
    assert snapshot.candidate_hash == space.value_hash([3, 21, True])
    assert snapshot.metadata["record_metadata"]["fold"] == 1
    assert snapshot.score == 10.0


def test_population_snapshot_copies_telemetry_and_pending_batches() -> None:
    space = _space()
    telemetry = OptimizationTelemetry()
    telemetry.record_cached(1, stage="warm_start", cost=0.0)
    candidate = _scored_candidate("c-1", [3, 21, True], 10.0)

    snapshot = build_population_snapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        direction="maximize",
        event_index=4,
        pending_batch_ids=("b-open",),
        trusted_count=1,
        candidates=[candidate],
        gene_space=space,
        telemetry=telemetry,
    )
    telemetry.record_cached(1, stage="after_snapshot", cost=0.0)

    assert snapshot.optimizer_type == "GeneticAlgorithmOptimizer"
    assert snapshot.pending_batch_ids == ("b-open",)
    assert snapshot.telemetry.candidates_cached == 1


def test_top_candidate_snapshots_respects_direction_and_confidence() -> None:
    space = _space()
    low = _scored_candidate("c-low", [3, 21, True], 1.0)
    high = _scored_candidate("c-high", [4, 22, False], 9.0)

    selected = top_candidate_snapshots(
        [low, high],
        k=1,
        gene_space=space,
        direction="maximize",
        confidence=("trusted_full", "cached"),
    )

    assert [item.candidate_id for item in selected] == ["c-high"]
