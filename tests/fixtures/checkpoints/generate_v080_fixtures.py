from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evocore import (  # noqa: E402
    CMAESOptimizer,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
)
from evocore.core.serialization import stable_json_dumps  # noqa: E402
from evocore.results import CHECKPOINT_SCHEMA_VERSION, GenerationHistory  # noqa: E402
from evocore.search_space import Solution  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "v0.8.0"
FIXTURE_CREATED_BY = {
    "evocore_version": "0.8.0",
    "python_version": "fixture-python",
    "platform": "fixture-platform",
}


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


def _ga_ask_tell_optimizer() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=5,
        seed=123,
    )


def _cmaes_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


def _cmaes_optimizer() -> CMAESOptimizer:
    return CMAESOptimizer(
        _cmaes_space(),
        population_size=4,
        max_generations=5,
        seed=7,
    )


def _trusted_records_for_candidates(candidates) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=_score_from_genes(candidate.genes),
            confidence="trusted_full",
            stage="full",
            cost=1.0,
            metadata={"source": "golden-fixture"},
        )
        for candidate in candidates
    ]


def _first_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    batches = payload["state"]["payload"]["batches_by_id"]
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
                metadata={"source": "golden-fixture"},
            )
        )
    return records


def _population_after_generation_zero(
    engine: GeneticAlgorithmOptimizer,
) -> list[Solution]:
    working_population, fitnesses, evaluated_now, _ = engine._evaluate_with_budget(
        engine._initial_population(),
        _ga_sphere,
        gen=-1,
        n_evaluations=0,
    )
    generation_history = GenerationHistory()
    working_population, _, _, stopped, _ = engine._run_generation(
        working_population=working_population,
        fitnesses=fitnesses,
        objective_fn=_ga_sphere,
        gen=0,
        n_evaluations=evaluated_now,
        elite_history=[],
        diversity_history=[],
        generation_history=generation_history,
    )
    if stopped:
        raise RuntimeError("generation-zero fixture setup stopped unexpectedly")
    return working_population


def _fixture_payload(snapshot) -> dict[str, Any]:
    return replace(snapshot, created_by=FIXTURE_CREATED_BY).to_dict()


def _write_json(path: Path, payload: object) -> str:
    text = stable_json_dumps(payload, indent=2) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _entry(
    *,
    name: str,
    file_name: str,
    payload: dict[str, Any],
    continuation: dict[str, Any],
) -> dict[str, Any]:
    sha256 = _write_json(FIXTURE_DIR / file_name, payload)
    optimizer = payload["optimizer"]
    state_payload = payload["state"]["payload"]
    return {
        "name": name,
        "file": file_name,
        "source_evocore_version": payload["created_by"]["evocore_version"],
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "optimizer_type": optimizer["optimizer_type"],
        "state_kind": state_payload["state_kind"],
        "seed": optimizer["seed"],
        "direction": optimizer["direction"],
        "gene_space_hash": optimizer["gene_space_hash"],
        "optimizer_config_hash": optimizer["optimizer_config_hash"],
        "sha256": sha256,
        "continuation": continuation,
    }


def _ga_generation_fixture() -> dict[str, Any]:
    engine = _ga_generation_optimizer()
    population = _population_after_generation_zero(engine)
    payload = _fixture_payload(engine.checkpoint(generation=0, population=population))
    result = _ga_generation_optimizer().resume_from_checkpoint(_ga_sphere, payload)
    return _entry(
        name="ga_generation_loop",
        file_name="ga-generation-loop.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "best_score": result.best_score,
            "final_values": [solution.values for solution in result.final_solutions],
            "stop_reason": result.stop_reason,
        },
    )


def _ga_after_ask_fixture() -> dict[str, Any]:
    source = _ga_ask_tell_optimizer()
    source.ask(4)
    payload = _fixture_payload(source.ask_tell_checkpoint())
    restored = _ga_ask_tell_optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(_trusted_records_from_payload(payload))
    return _entry(
        name="ga_ask_tell_after_ask",
        file_name="ga-ask-tell-after-ask.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "pending_batch_ids": list(summary.pending_batch_ids),
            "trusted_count_after_tell": result.trusted_count,
        },
    )


def _ga_after_partial_tell_fixture() -> dict[str, Any]:
    source = _ga_ask_tell_optimizer()
    candidates = source.ask(4)
    source.tell(_trusted_records_for_candidates(candidates)[:1])
    payload = _fixture_payload(source.ask_tell_checkpoint())
    restored = _ga_ask_tell_optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(_trusted_records_from_payload(payload, skip_existing=True))
    return _entry(
        name="ga_ask_tell_after_partial_tell",
        file_name="ga-ask-tell-after-partial-tell.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "best_candidate_id": summary.best_candidate_id,
            "pending_batch_ids": list(summary.pending_batch_ids),
            "accepted_count_after_tell": result.accepted_count,
        },
    )


def _cmaes_after_ask_fixture() -> dict[str, Any]:
    source = _cmaes_optimizer()
    source.ask()
    payload = _fixture_payload(source.ask_tell_checkpoint())
    restored = _cmaes_optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(_trusted_records_from_payload(payload))
    return _entry(
        name="cmaes_ask_tell_after_ask",
        file_name="cmaes-ask-tell-after-ask.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "pending_batch_ids": list(summary.pending_batch_ids),
            "trusted_count_after_tell": result.trusted_count,
            "consumed_batch_ids": list(result.consumed_batch_ids),
            "generation_after_tell": restored.generation,
        },
    )


def _cmaes_after_consumed_batch_fixture() -> dict[str, Any]:
    source = _cmaes_optimizer()
    candidates = source.ask()
    source.tell(_trusted_records_for_candidates(candidates))
    payload = _fixture_payload(source.ask_tell_checkpoint())
    restored = _cmaes_optimizer()
    restored.resume_ask_tell_checkpoint(payload)
    next_candidates = restored.ask()
    return _entry(
        name="cmaes_ask_tell_after_consumed_batch",
        file_name="cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "next_ask": {
                "candidate_ids": [candidate.candidate_id for candidate in next_candidates],
                "batch_ids": [candidate.batch_id for candidate in next_candidates],
                "genes": [candidate.genes for candidate in next_candidates],
            },
        },
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIXTURE_DIR.glob("*.evocore-checkpoint.json"):
        path.unlink()

    entries = [
        _ga_generation_fixture(),
        _ga_after_ask_fixture(),
        _ga_after_partial_tell_fixture(),
        _cmaes_after_ask_fixture(),
        _cmaes_after_consumed_batch_fixture(),
    ]
    manifest = {
        "fixture_format_version": 1,
        "source_evocore_version": "0.8.0",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fixtures": entries,
    }
    _write_json(FIXTURE_DIR / "manifest.json", manifest)
    print(f"Wrote {len(entries)} checkpoint fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
