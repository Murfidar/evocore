import json

import pytest

from evocore import CheckpointError, GeneSpace, GeneticAlgorithmOptimizer
from evocore.results import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSnapshot,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_identity,
)


def _engine():
    return GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=3,
        seed=42,
    )


def _snapshot() -> CheckpointSnapshot:
    engine = _engine()
    return CheckpointSnapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        optimizer_config=engine.config_signature(),
        optimizer_config_hash=engine.config_hash(),
        gene_space_signature=engine.gene_space.signature(),
        gene_space_hash=engine.gene_space.hash(),
        direction=engine.direction,
        seed=engine.seed,
        position={"generation": 1, "event_index": 0, "n_evaluations": 8},
        state={
            "optimizer_type": "GeneticAlgorithmOptimizer",
            "schema_version": 1,
            "payload": {"state_kind": "ga_generation_loop", "population": []},
        },
        audit={"events": [], "telemetry": {}, "best": None},
        metadata={"source": "unit"},
    )


def test_checkpoint_snapshot_exports_required_envelope() -> None:
    payload = _snapshot().to_dict()

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["checkpoint_kind"] == CHECKPOINT_KIND
    assert payload["created_by"]["evocore_version"]
    assert payload["optimizer"]["optimizer_type"] == "GeneticAlgorithmOptimizer"
    assert payload["optimizer"]["seed_derivation"] == {
        "algorithm": "py_derive_seed",
        "version": 1,
    }
    assert payload["position"] == {
        "event_index": 0,
        "generation": 1,
        "n_evaluations": 8,
    }
    assert payload["state"]["payload"]["state_kind"] == "ga_generation_loop"
    assert payload["metadata"] == {"source": "unit"}


def test_checkpoint_save_and_load_round_trip_json(tmp_path) -> None:
    path = tmp_path / "checkpoint_gen_1.evocore-checkpoint.json"

    save_checkpoint(path, _snapshot())
    loaded = load_checkpoint(path)

    assert loaded == _snapshot().to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_checkpoint_load_missing_file_lists_available(tmp_path) -> None:
    (tmp_path / "checkpoint_gen_1.evocore-checkpoint.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="Available checkpoints"):
        load_checkpoint(tmp_path / "checkpoint_gen_9.evocore-checkpoint.json")


def test_checkpoint_load_rejects_wrong_kind(tmp_path) -> None:
    payload = _snapshot().to_dict()
    payload["checkpoint_kind"] = "result_export"
    path = tmp_path / "bad.evocore-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint_kind"):
        load_checkpoint(path)


def test_checkpoint_load_rejects_unsupported_schema(tmp_path) -> None:
    payload = _snapshot().to_dict()
    payload["checkpoint_schema_version"] = 999
    path = tmp_path / "bad.evocore-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint_schema_version"):
        load_checkpoint(path)


def test_checkpoint_identity_validation_rejects_gene_space_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="gene_space_hash"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash="different",
            optimizer_config_hash=engine.config_hash(),
            seed=engine.seed,
            direction=engine.direction,
        )


def test_checkpoint_identity_validation_rejects_config_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash="different",
            seed=engine.seed,
            direction=engine.direction,
        )


def test_checkpoint_identity_validation_rejects_seed_and_direction_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="seed"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash=engine.config_hash(),
            seed=999,
            direction=engine.direction,
        )

    with pytest.raises(CheckpointError, match="direction"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash=engine.config_hash(),
            seed=engine.seed,
            direction="minimize",
        )
