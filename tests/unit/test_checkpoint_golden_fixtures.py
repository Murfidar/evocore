from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from evocore import (
    CheckpointError,
    CMAESOptimizer,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
)
from evocore.results import CHECKPOINT_SCHEMA_VERSION, load_checkpoint
from evocore.search_space import Solution

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "checkpoints" / "v0.8.0"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

EXPECTED_FILES = {
    "ga-generation-loop.evocore-checkpoint.json",
    "ga-ask-tell-after-ask.evocore-checkpoint.json",
    "ga-ask-tell-after-partial-tell.evocore-checkpoint.json",
    "cmaes-ask-tell-after-ask.evocore-checkpoint.json",
    "cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json",
}


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.exists(), f"missing checkpoint fixture manifest: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _fixture_entries() -> list[dict[str, Any]]:
    return list(_load_manifest()["fixtures"])


def _entry(name: str) -> dict[str, Any]:
    for fixture in _fixture_entries():
        if fixture["name"] == name:
            return fixture
    raise AssertionError(f"fixture entry {name!r} is missing from manifest")


def _fixture_path(entry: dict[str, Any]) -> Path:
    return FIXTURE_DIR / entry["file"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _walk_strings(item)


def _score_from_genes(genes: Iterable[object]) -> float:
    return -sum(float(value) ** 2 for value in genes)


def _ga_sphere(solution: Solution) -> float:
    return _score_from_genes(solution.values)


def _ga_generation_optimizer() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=6,
        max_generations=3,
        seed=123,
    )


def _ga_ask_tell_optimizer(**overrides: object) -> GeneticAlgorithmOptimizer:
    params: dict[str, object] = {
        "population_size": 4,
        "max_generations": 5,
        "seed": 123,
    }
    params.update(overrides)
    return GeneticAlgorithmOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), **params)


def _cmaes_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


def _cmaes_optimizer(**overrides: object) -> CMAESOptimizer:
    params: dict[str, object] = {
        "population_size": 4,
        "max_generations": 5,
        "seed": 7,
    }
    params.update(overrides)
    return CMAESOptimizer(_cmaes_space(), **params)


def _first_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    batches = payload["state"]["payload"]["batches_by_id"]
    assert len(batches) == 1
    return next(iter(batches.values()))


def _trusted_records_from_payload(
    payload: dict[str, Any],
    *,
    skip_existing: bool = False,
) -> list[EvaluationRecord]:
    state_payload = payload["state"]["payload"]
    batch_payload = _first_batch_payload(payload)
    existing_candidate_ids = {record["candidate_id"] for record in batch_payload["records"]}
    records: list[EvaluationRecord] = []
    for candidate_id in batch_payload["candidate_ids"]:
        if skip_existing and candidate_id in existing_candidate_ids:
            continue
        candidate_payload = state_payload["candidates_by_id"][candidate_id]
        records.append(
            EvaluationRecord(
                candidate_id=candidate_id,
                batch_id=batch_payload["batch_id"],
                score=_score_from_genes(candidate_payload["genes"]),
                confidence="trusted_full",
                stage="full",
                cost=1.0,
                metadata={"source": "golden-fixture-test"},
            )
        )
    return records


def test_manifest_lists_expected_v080_checkpoint_fixtures() -> None:
    manifest = _load_manifest()

    assert manifest["fixture_format_version"] == 1
    assert manifest["source_evocore_version"] == "0.8.0"
    assert manifest["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert {entry["file"] for entry in manifest["fixtures"]} == EXPECTED_FILES

    for entry in manifest["fixtures"]:
        assert _fixture_path(entry).exists()


def test_fixture_file_hashes_match_manifest() -> None:
    for entry in _fixture_entries():
        assert _sha256(_fixture_path(entry)) == entry["sha256"]


def test_valid_fixtures_load_and_match_manifest_identity() -> None:
    for entry in _fixture_entries():
        payload = load_checkpoint(_fixture_path(entry))

        assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert payload["created_by"]["evocore_version"] == "0.8.0"
        assert payload["optimizer"]["optimizer_type"] == entry["optimizer_type"]
        assert payload["optimizer"]["seed"] == entry["seed"]
        assert payload["optimizer"]["direction"] == entry["direction"]
        assert payload["optimizer"]["gene_space_hash"] == entry["gene_space_hash"]
        assert payload["optimizer"]["optimizer_config_hash"] == entry["optimizer_config_hash"]
        assert payload["state"]["payload"]["state_kind"] == entry["state_kind"]


def test_fixture_payloads_do_not_embed_machine_local_paths() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    forbidden_fragments = (repo_root, "C:\\", "D:\\", "/home/", "/Users/")

    for entry in _fixture_entries():
        payload = load_checkpoint(_fixture_path(entry))
        for text in _walk_strings(payload):
            assert not any(fragment in text for fragment in forbidden_fragments)


def test_ga_generation_loop_fixture_resumes_to_manifest_continuation() -> None:
    entry = _entry("ga_generation_loop")
    expected = entry["continuation"]

    result = _ga_generation_optimizer().resume_from_checkpoint(
        _ga_sphere,
        _fixture_path(entry),
    )

    assert result.best_score == pytest.approx(expected["best_score"])
    assert [solution.values for solution in result.final_solutions] == expected[
        "final_values"
    ]
    assert result.stop_reason == expected["stop_reason"]


def test_ga_ask_tell_after_ask_fixture_accepts_pending_records() -> None:
    entry = _entry("ga_ask_tell_after_ask")
    payload = load_checkpoint(_fixture_path(entry))

    restored = _ga_ask_tell_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_fixture_path(entry))
    result = restored.tell(_trusted_records_from_payload(payload))

    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert result.trusted_count == entry["continuation"]["trusted_count_after_tell"]
    assert result.pending_batch_ids == ()
    assert restored.state_summary().trusted_count == 4
    assert restored.best_candidate is not None


def test_ga_ask_tell_partial_fixture_accepts_missing_records() -> None:
    entry = _entry("ga_ask_tell_after_partial_tell")
    payload = load_checkpoint(_fixture_path(entry))

    restored = _ga_ask_tell_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_fixture_path(entry))
    result = restored.tell(_trusted_records_from_payload(payload, skip_existing=True))

    assert summary.best_candidate_id == entry["continuation"]["best_candidate_id"]
    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert result.accepted_count == entry["continuation"]["accepted_count_after_tell"]
    assert result.pending_batch_ids == ()
    assert restored.state_summary().trusted_count == 4


def test_cmaes_ask_tell_after_ask_fixture_accepts_pending_records() -> None:
    entry = _entry("cmaes_ask_tell_after_ask")
    payload = load_checkpoint(_fixture_path(entry))

    restored = _cmaes_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_fixture_path(entry))
    result = restored.tell(_trusted_records_from_payload(payload))

    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert result.trusted_count == entry["continuation"]["trusted_count_after_tell"]
    assert result.consumed_batch_ids == tuple(entry["continuation"]["consumed_batch_ids"])
    assert restored.generation == entry["continuation"]["generation_after_tell"]
    assert restored.state_summary().trusted_count == 4


def test_cmaes_consumed_batch_fixture_next_ask_matches_manifest() -> None:
    entry = _entry("cmaes_ask_tell_after_consumed_batch")
    expected = entry["continuation"]["next_ask"]

    restored = _cmaes_optimizer()
    restored.resume_ask_tell_checkpoint(_fixture_path(entry))
    candidates = restored.ask()

    assert [candidate.candidate_id for candidate in candidates] == expected["candidate_ids"]
    assert [candidate.batch_id for candidate in candidates] == expected["batch_ids"]
    assert [candidate.genes for candidate in candidates] == expected["genes"]


def test_fixture_derived_envelope_incompatibilities_raise_checkpoint_error() -> None:
    entry = _entry("ga_ask_tell_after_ask")
    payload = load_checkpoint(_fixture_path(entry))
    cases = [
        (
            lambda item: item.__setitem__("checkpoint_schema_version", 999),
            "checkpoint_schema_version",
        ),
        (
            lambda item: item.__setitem__("checkpoint_kind", "result_export"),
            "checkpoint_kind",
        ),
        (
            lambda item: item["optimizer"].__setitem__(
                "optimizer_type",
                "CMAESOptimizer",
            ),
            "optimizer_type",
        ),
        (
            lambda item: item["optimizer"].__setitem__("seed", 999),
            "seed",
        ),
        (
            lambda item: item["optimizer"].__setitem__("direction", "minimize"),
            "direction",
        ),
        (
            lambda item: item["state"]["payload"].__setitem__(
                "state_kind",
                "cmaes_ask_tell",
            ),
            "ask/tell resume",
        ),
    ]

    for mutate, message in cases:
        bad_payload = copy.deepcopy(payload)
        mutate(bad_payload)
        with pytest.raises(CheckpointError, match=message):
            _ga_ask_tell_optimizer().resume_ask_tell_checkpoint(bad_payload)


def test_fixture_derived_identity_mismatches_raise_checkpoint_error() -> None:
    entry = _entry("ga_ask_tell_after_ask")

    with pytest.raises(CheckpointError, match="gene_space_hash"):
        GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-2.0, 2.0, 2),
            population_size=4,
            max_generations=5,
            seed=123,
        ).resume_ask_tell_checkpoint(_fixture_path(entry))

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        _ga_ask_tell_optimizer(population_size=6).resume_ask_tell_checkpoint(
            _fixture_path(entry)
        )


def test_fixture_derived_malformed_cmaes_state_raises_checkpoint_error() -> None:
    entry = _entry("cmaes_ask_tell_after_ask")
    payload = load_checkpoint(_fixture_path(entry))
    bad_payload = copy.deepcopy(payload)
    bad_payload["state"]["payload"]["cmaes_state"]["schema_version"] = 999

    with pytest.raises(CheckpointError, match="cmaes_state"):
        _cmaes_optimizer().resume_ask_tell_checkpoint(bad_payload)
