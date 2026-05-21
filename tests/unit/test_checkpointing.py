import json

import pytest

from evocore import CheckpointError, Gene, GeneSpace, GeneticAlgorithmOptimizer
from evocore.results import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSnapshot,
    GenerationHistory,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_identity,
)
from evocore.search_space import Solution


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


def _sphere(solution: Solution) -> float:
    return -sum(float(value) ** 2 for value in solution.values)


def _run_generation_loop(engine: GeneticAlgorithmOptimizer):
    return engine._run_from_population(
        engine._initial_population(),
        _sphere,
        start_generation=0,
    )


def _population_after_generation_zero(engine: GeneticAlgorithmOptimizer) -> list[Solution]:
    working_population, fitnesses, evaluated_now, _ = engine._evaluate_with_budget(
        engine._initial_population(),
        _sphere,
        gen=-1,
        n_evaluations=0,
    )
    generation_history = GenerationHistory()
    working_population, _, _, stopped, _ = engine._run_generation(
        working_population=working_population,
        fitnesses=fitnesses,
        objective_fn=_sphere,
        gen=0,
        n_evaluations=evaluated_now,
        elite_history=[],
        diversity_history=[],
        generation_history=generation_history,
    )
    assert stopped is False
    return working_population


def test_ga_checkpoint_generation_snapshot_contains_population_state() -> None:
    engine = _engine()
    population = [
        Solution([0.25, -0.5], score=-0.3125, score_valid=True, metadata={"rank": 1}),
        Solution([0.5, -0.25], score=-0.3125, score_valid=True, metadata={"rank": 2}),
    ]

    snapshot = engine.checkpoint(generation=2, population=population)
    payload = snapshot.to_dict()
    state_payload = payload["state"]["payload"]

    assert state_payload["state_kind"] == "ga_generation_loop"
    assert payload["position"]["generation"] == 2
    assert payload["position"]["event_index"] == engine.state_summary().event_index
    assert state_payload["population"] == [
        {
            "values": [0.25, -0.5],
            "score": -0.3125,
            "score_valid": True,
            "metadata": {"rank": 1},
        },
        {
            "values": [0.5, -0.25],
            "score": -0.3125,
            "score_valid": True,
            "metadata": {"rank": 2},
        },
    ]


def test_ga_resume_from_stable_checkpoint_matches_uninterrupted_generation_loop(tmp_path) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    checkpoint_path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"

    partial = GeneticAlgorithmOptimizer(
        space,
        population_size=6,
        max_generations=3,
        seed=123,
    )
    generation_zero_population = _population_after_generation_zero(partial)
    partial.save_checkpoint(
        checkpoint_path,
        partial.checkpoint(generation=0, population=generation_zero_population),
    )

    resumed = GeneticAlgorithmOptimizer(
        space,
        population_size=6,
        max_generations=3,
        seed=123,
    ).resume_from_checkpoint(_sphere, checkpoint_path)
    uninterrupted = _run_generation_loop(
        GeneticAlgorithmOptimizer(
            space,
            population_size=6,
            max_generations=3,
            seed=123,
        )
    )

    assert resumed.best_score == pytest.approx(uninterrupted.best_score)
    assert [solution.values for solution in resumed.final_solutions] == [
        solution.values for solution in uninterrupted.final_solutions
    ]
    assert resumed.seed == 123
    assert resumed.stop_reason == uninterrupted.stop_reason


def test_ga_resume_from_stable_checkpoint_rejects_config_mismatch(tmp_path) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    source = GeneticAlgorithmOptimizer(space, population_size=6, max_generations=3, seed=123)
    generation_zero_population = _population_after_generation_zero(source)
    checkpoint_path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"
    source.save_checkpoint(
        checkpoint_path,
        source.checkpoint(generation=0, population=generation_zero_population),
    )

    mismatched = GeneticAlgorithmOptimizer(space, population_size=8, max_generations=3, seed=123)

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        mismatched.resume_from_checkpoint(_sphere, checkpoint_path)


def test_ga_resume_keeps_legacy_pickle_path(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint_gen_0.pkl"
    population = [
        Solution([0.0, 0.0], score=0.0, score_valid=True),
        Solution([0.1, 0.0], score=-0.01, score_valid=True),
        Solution([0.0, 0.1], score=-0.01, score_valid=True),
        Solution([0.1, 0.1], score=-0.02, score_valid=True),
    ]

    import pickle

    checkpoint_path.write_bytes(
        pickle.dumps({"population": population, "generation": 0, "seed": 42})
    )

    result = _engine().resume(_sphere, str(checkpoint_path))

    assert result.seed == 42
    assert result.best_solution.score_valid


def test_ga_generation_checkpoint_round_trip_preserves_mixed_bool_values(tmp_path) -> None:
    space = GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )
    engine = GeneticAlgorithmOptimizer(space, population_size=4, max_generations=2, seed=42)
    population = [
        Solution(
            [0.25, 10, True],
            score=1.0,
            score_valid=True,
            metadata={"params": {"threshold": 0.25, "period": 10, "enabled": True}},
        )
    ]
    path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"

    engine.save_checkpoint(path, engine.checkpoint(generation=0, population=population))
    loaded = load_checkpoint(path)

    row = loaded["state"]["payload"]["population"][0]
    assert row["values"] == [0.25, 10, True]
    assert type(row["values"][2]) is bool
    assert row["metadata"]["params"]["enabled"] is True
