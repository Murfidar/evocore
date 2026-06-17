import pytest

from evocore import (
    CandidateArchive,
    CandidateSnapshot,
    PopulationSnapshot,
    WarmStartRecord,
)
from evocore.core import ConfigurationError
from evocore.lifecycle import OptimizationTelemetry
from evocore.lifecycle.archives import ARCHIVE_SCHEMA_VERSION


def _snapshot(
    candidate_id: str,
    candidate_hash: str,
    score: float,
    *,
    family: str = "baseline",
    confidence: str = "cached",
) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        values=(float(score),),
        params={"x": float(score)},
        origin="memory_seed",
        batch_id="batch-1",
        event_index=1,
        generation=None,
        status="trusted",
        stage="archive",
        confidence=confidence,
        score=score,
        scores={},
        cost=0.0,
        metadata={"family": family, "record_metadata": {"fold": 1}},
    )


def test_candidate_archive_keep_first_duplicate_policy() -> None:
    archive = CandidateArchive(duplicate_policy="keep_first", score_direction="maximize")

    archive.add_candidates(
        [
            _snapshot("c-1", "same", 1.0),
            _snapshot("c-2", "same", 9.0),
        ],
        source="stage1",
    )

    assert [entry.candidate_id for entry in archive.entries] == ["c-1"]
    assert archive.entries[0].score == 1.0


def test_candidate_archive_keep_latest_duplicate_policy() -> None:
    archive = CandidateArchive(duplicate_policy="keep_latest", score_direction="maximize")

    archive.add_candidates([_snapshot("c-1", "same", 1.0)], source="stage1")
    archive.add_candidates([_snapshot("c-2", "same", 9.0)], source="stage2")

    assert [entry.candidate_id for entry in archive.entries] == ["c-2"]
    assert archive.entries[0].source == "stage2"


def test_candidate_archive_keep_best_duplicate_policy_respects_direction() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="minimize")

    archive.add_candidates(
        [
            _snapshot("c-1", "same", 9.0),
            _snapshot("c-2", "same", 1.0),
        ],
        source="stage1",
    )

    assert [entry.candidate_id for entry in archive.entries] == ["c-2"]
    assert archive.entries[0].score == 1.0


def test_candidate_archive_exports_warm_start_records() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
    archive.add_candidates(
        [
            _snapshot("c-1", "hash-a", 1.0, family="a"),
            _snapshot("c-2", "hash-b", 9.0, family="b"),
        ],
        source="stage1",
    )

    records = archive.to_warm_start_records(k=1, stage="refine", confidence="cached")

    assert records == (
        WarmStartRecord(
            params={"x": 9.0},
            score=9.0,
            confidence="cached",
            stage="refine",
            cost=0.0,
            metrics={},
            metadata={
                "archive_candidate_id": "c-2",
                "archive_candidate_hash": "hash-b",
                "archive_source": "stage1",
                "family": "b",
                "record_metadata": {"fold": 1},
            },
        ),
    )


def test_candidate_archive_add_population_inherits_direction() -> None:
    population = PopulationSnapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        direction="minimize",
        event_index=4,
        pending_batch_ids=(),
        trusted_count=2,
        candidates=(
            _snapshot("c-1", "hash-a", 5.0),
            _snapshot("c-2", "hash-b", 1.0),
        ),
        telemetry=OptimizationTelemetry(),
    )
    archive = CandidateArchive()

    archive.add_population(population, source="trusted")
    records = archive.to_warm_start_records(k=1)

    assert records[0].score == 1.0


def test_candidate_archive_round_trips_json_safe_dict() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
    archive.add_candidates([_snapshot("c-1", "hash-a", 2.0)], source="stage1")

    payload = archive.to_dict()
    restored = CandidateArchive.from_dict(payload)

    assert payload["schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert restored.duplicate_policy == "keep_best"
    assert restored.score_direction == "maximize"
    assert restored.entries == archive.entries


def test_candidate_archive_rejects_snapshot_without_score() -> None:
    snapshot = _snapshot("c-1", "hash-a", 1.0)
    snapshot = CandidateSnapshot(
        candidate_id=snapshot.candidate_id,
        candidate_hash=snapshot.candidate_hash,
        values=snapshot.values,
        params=snapshot.params,
        origin=snapshot.origin,
        batch_id=snapshot.batch_id,
        event_index=snapshot.event_index,
        generation=snapshot.generation,
        status=snapshot.status,
        stage=snapshot.stage,
        confidence=snapshot.confidence,
        score=None,
        scores=snapshot.scores,
        cost=snapshot.cost,
        metadata=snapshot.metadata,
    )
    archive = CandidateArchive()

    with pytest.raises(ConfigurationError, match="finite score"):
        archive.add_candidates([snapshot], source="stage1")
