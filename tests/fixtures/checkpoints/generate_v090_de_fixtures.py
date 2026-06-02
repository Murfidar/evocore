from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace  # noqa: E402
from evocore.core.serialization import stable_json_dumps  # noqa: E402
from evocore.results import CHECKPOINT_SCHEMA_VERSION  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "v0.9.0"
FIXTURE_CREATED_BY = {
    "evocore_version": "0.9.0",
    "python_version": "fixture-python",
    "platform": "fixture-platform",
}


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _optimizer(**overrides: object) -> DifferentialEvolutionOptimizer:
    params: dict[str, object] = {
        "population_size": 6,
        "max_generations": 5,
        "seed": 42,
    }
    params.update(overrides)
    return DifferentialEvolutionOptimizer(_space(), **params)


def _score_from_genes(genes: Iterable[object]) -> float:
    x, period, enabled, fixed = genes
    score = -abs(float(x) - 0.25) - abs(int(period) - 7) + float(fixed)
    if bool(enabled):
        score += 2.0
    return float(score)


def _trusted_records_for_candidates(
    candidates: Sequence[object],
    *,
    scores: Sequence[float] | None = None,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for index, candidate in enumerate(candidates):
        score = _score_from_genes(candidate.genes) if scores is None else float(scores[index])
        records.append(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=score,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
                metadata={"source": "de-golden-fixture"},
            )
        )
    return records


def _trial_records(candidates: Sequence[object]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for index, candidate in enumerate(candidates):
        if index in (2, 5):
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=None,
                    confidence="rejected",
                    stage="full",
                    cost=1.0,
                    metadata={"source": "de-golden-fixture", "reason": "constraint"},
                )
            )
            continue
        score = 100.0 - float(index) if index in (0, 3) else -100.0 - float(index)
        records.append(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=score,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
                metadata={"source": "de-golden-fixture"},
            )
        )
    return records


def _fixture_payload(snapshot) -> dict[str, Any]:
    return replace(snapshot, created_by=FIXTURE_CREATED_BY).to_dict()


def _write_json(path: Path, payload: object) -> str:
    text = stable_json_dumps(payload, indent=2) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _target_genes(engine: DifferentialEvolutionOptimizer) -> list[list[object]]:
    return [
        list(engine._candidates_by_id[candidate_id].genes)
        for candidate_id in engine._target_candidate_ids
    ]


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


def _after_initial_ask_fixture() -> dict[str, Any]:
    source = _optimizer()
    candidates = source.ask()
    payload = _fixture_payload(source.ask_tell_checkpoint(metadata={"phase": "initial_ask"}))
    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(_trusted_records_for_candidates(candidates))
    return _entry(
        name="de_after_initial_ask",
        file_name="de-after-initial-ask.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "pending_batch_ids": list(summary.pending_batch_ids),
            "trusted_count_after_tell": result.trusted_count,
            "state_accepted_count_after_tell": result.state_accepted_count,
            "best_score_after_tell": result.best_score,
        },
    )


def _after_partial_initial_tell_fixture() -> dict[str, Any]:
    source = _optimizer()
    candidates = source.ask()
    source.tell(_trusted_records_for_candidates(candidates[:2], scores=[1.0, 2.0]))
    payload = _fixture_payload(
        source.ask_tell_checkpoint(metadata={"phase": "partial_initial_tell"})
    )
    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(
        _trusted_records_for_candidates(candidates[2:], scores=[3.0, 4.0, 5.0, 6.0])
    )
    return _entry(
        name="de_after_partial_initial_tell",
        file_name="de-after-partial-initial-tell.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "best_candidate_id": summary.best_candidate_id,
            "pending_batch_ids": list(summary.pending_batch_ids),
            "accepted_count_after_tell": result.accepted_count,
            "trusted_count_after_tell": restored.state_summary().trusted_count,
            "best_score_after_tell": result.best_score,
        },
    )


def _initialized_source() -> tuple[DifferentialEvolutionOptimizer, list[object]]:
    source = _optimizer()
    targets = source.ask()
    source.tell(
        _trusted_records_for_candidates(targets, scores=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    )
    return source, targets


def _after_initialized_population_fixture() -> dict[str, Any]:
    source, _ = _initialized_source()
    payload = _fixture_payload(
        source.ask_tell_checkpoint(metadata={"phase": "initialized_population"})
    )
    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(payload)
    trials = restored.ask()
    return _entry(
        name="de_after_initialized_population",
        file_name="de-after-initialized-population.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "next_ask": {
                "candidate_ids": [candidate.candidate_id for candidate in trials],
                "batch_ids": [candidate.batch_id for candidate in trials],
                "genes": [candidate.genes for candidate in trials],
                "target_slots": [candidate.metadata["target_slot"] for candidate in trials],
                "target_candidate_ids": [
                    candidate.metadata["target_candidate_id"] for candidate in trials
                ],
            },
        },
    )


def _after_trial_ask_fixture() -> dict[str, Any]:
    source, _ = _initialized_source()
    trials = source.ask()
    payload = _fixture_payload(source.ask_tell_checkpoint(metadata={"phase": "trial_ask"}))
    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(payload)
    result = restored.tell(_trusted_records_for_candidates(trials[:1], scores=[100.0]))
    decision = result.acceptance_decisions[0]
    return _entry(
        name="de_after_trial_ask",
        file_name="de-after-trial-ask.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "pending_batch_ids": list(summary.pending_batch_ids),
            "first_decision": {
                "candidate_id": decision.candidate_id,
                "accepted_for_state": decision.accepted_for_state,
                "reason": decision.reason,
                "target_candidate_id": decision.target_candidate_id,
                "target_slot": decision.target_slot,
            },
        },
    )


def _after_mixed_trial_tell_fixture() -> dict[str, Any]:
    source, _ = _initialized_source()
    trials = source.ask()
    result = source.tell(_trial_records(trials))
    payload = _fixture_payload(source.ask_tell_checkpoint(metadata={"phase": "mixed_trial_tell"}))
    return _entry(
        name="de_after_mixed_trial_tell",
        file_name="de-after-mixed-trial-tell.evocore-checkpoint.json",
        payload=payload,
        continuation={
            "generation": source.generation,
            "trusted_count": source.state_summary().trusted_count,
            "consumed_batch_ids": list(result.consumed_batch_ids),
            "state_accepted_count": result.state_accepted_count,
            "target_candidate_ids": list(source._target_candidate_ids),
            "target_genes": _target_genes(source),
            "best_candidate_id": source.state_summary().best_candidate_id,
            "best_score": source.state_summary().best_score,
        },
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIXTURE_DIR.glob("*.evocore-checkpoint.json"):
        path.unlink()

    entries = [
        _after_initial_ask_fixture(),
        _after_partial_initial_tell_fixture(),
        _after_initialized_population_fixture(),
        _after_trial_ask_fixture(),
        _after_mixed_trial_tell_fixture(),
    ]
    manifest = {
        "fixture_format_version": 1,
        "source_evocore_version": "0.9.0",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fixtures": entries,
    }
    _write_json(FIXTURE_DIR / "manifest.json", manifest)
    print(f"Wrote {len(entries)} DE checkpoint fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
