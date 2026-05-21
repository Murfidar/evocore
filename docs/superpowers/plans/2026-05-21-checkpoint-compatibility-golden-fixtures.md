# Checkpoint Compatibility Golden Fixtures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixture-backed checkpoint compatibility suite that makes EvoCore 0.8.0 the stable JSON checkpoint baseline for GA generation-loop, GA ask/tell, and CMA-ES ask/tell checkpoints.

**Architecture:** Keep runtime checkpoint code unchanged unless tests expose a real compatibility bug. Add deterministic fixture generation beside committed fixture files, then add unit tests that load those files and validate shape, hashes, resume behavior, and derived incompatibility errors. Document the 0.8.0 baseline in public docs and the changelog without turning this into a whole-framework release-readiness pass.

**Tech Stack:** Python 3.11+, pytest, EvoCore checkpoint APIs, MkDocs Markdown docs, SHA-256 fixture hashes, repository-local `.venv`.

---

## File Structure

- Create `tests/fixtures/checkpoints/generate_v080_fixtures.py`: deterministic fixture generator for v0.8.0 checkpoint files and their manifest.
- Create `tests/fixtures/checkpoints/v0.8.0/manifest.json`: generated manifest with fixture metadata, SHA-256 hashes, and expected continuation assertions.
- Create `tests/fixtures/checkpoints/v0.8.0/ga-generation-loop.evocore-checkpoint.json`: stable GA generation-loop checkpoint fixture.
- Create `tests/fixtures/checkpoints/v0.8.0/ga-ask-tell-after-ask.evocore-checkpoint.json`: stable GA ask/tell checkpoint after `ask(...)`.
- Create `tests/fixtures/checkpoints/v0.8.0/ga-ask-tell-after-partial-tell.evocore-checkpoint.json`: stable GA ask/tell checkpoint after a partial `tell(...)`.
- Create `tests/fixtures/checkpoints/v0.8.0/cmaes-ask-tell-after-ask.evocore-checkpoint.json`: stable CMA-ES ask/tell checkpoint after `ask()`.
- Create `tests/fixtures/checkpoints/v0.8.0/cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json`: stable CMA-ES ask/tell checkpoint after a consumed batch.
- Create `tests/unit/test_checkpoint_golden_fixtures.py`: fixture shape, hash, behavior, and incompatibility tests.
- Modify `docs/site/callbacks-checkpointing.md`: document the 0.8.0 stable JSON checkpoint compatibility baseline and exclusions.
- Modify `docs/site/ga.md`: add the 0.8.0 baseline note to GA checkpoint wording.
- Modify `docs/site/cmaes.md`: add the 0.8.0 baseline note to CMA-ES checkpoint wording.
- Modify `CHANGELOG.md`: add an `[Unreleased]` entry for fixture-backed checkpoint compatibility.

## Scope Check

This plan targets one subsystem: stable checkpoint compatibility. It does not include release artifact builds, version bumps, PyPI publication, package-wide API audits, or broad release readiness.

### Task 1: Golden Fixture Tests And Fixtures

**Files:**
- Create: `tests/unit/test_checkpoint_golden_fixtures.py`
- Create: `tests/fixtures/checkpoints/generate_v080_fixtures.py`
- Create: `tests/fixtures/checkpoints/v0.8.0/manifest.json`
- Create: `tests/fixtures/checkpoints/v0.8.0/ga-generation-loop.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.8.0/ga-ask-tell-after-ask.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.8.0/ga-ask-tell-after-partial-tell.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.8.0/cmaes-ask-tell-after-ask.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.8.0/cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json`

- [ ] **Step 1: Write the failing golden fixture tests**

Create `tests/unit/test_checkpoint_golden_fixtures.py` with this content:

```python
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

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "checkpoints" / "v0.8.0"
)
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

EXPECTED_FILES = {
    "ga-generation-loop.evocore-checkpoint.json",
    "ga-ask-tell-after-ask.evocore-checkpoint.json",
    "ga-ask-tell-after-partial-tell.evocore-checkpoint.json",
    "cmaes-ask-tell-after-ask.evocore-checkpoint.json",
    "cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json",
}


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.exists(), (
        f"missing checkpoint fixture manifest: {MANIFEST_PATH}"
    )
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
    existing_candidate_ids = {
        record["candidate_id"] for record in batch_payload["records"]
    }
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
        assert (
            payload["optimizer"]["optimizer_config_hash"]
            == entry["optimizer_config_hash"]
        )
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
    result = restored.tell(
        _trusted_records_from_payload(payload, skip_existing=True)
    )

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
    assert result.consumed_batch_ids == tuple(
        entry["continuation"]["consumed_batch_ids"]
    )
    assert restored.generation == entry["continuation"]["generation_after_tell"]
    assert restored.state_summary().trusted_count == 4


def test_cmaes_consumed_batch_fixture_next_ask_matches_manifest() -> None:
    entry = _entry("cmaes_ask_tell_after_consumed_batch")
    expected = entry["continuation"]["next_ask"]

    restored = _cmaes_optimizer()
    restored.resume_ask_tell_checkpoint(_fixture_path(entry))
    candidates = restored.ask()

    assert [candidate.candidate_id for candidate in candidates] == expected[
        "candidate_ids"
    ]
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
```

- [ ] **Step 2: Run the golden fixture tests and verify the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -v
```

Expected: FAIL on `test_manifest_lists_expected_v080_checkpoint_fixtures` with a message containing `missing checkpoint fixture manifest`.

- [ ] **Step 3: Add the deterministic fixture generator**

Create `tests/fixtures/checkpoints/generate_v080_fixtures.py` with this content:

```python
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
    existing_candidate_ids = {
        record["candidate_id"] for record in batch_payload["records"]
    }
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
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    result = restored.tell(
        _trusted_records_from_payload(payload, skip_existing=True)
    )
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
                "candidate_ids": [
                    candidate.candidate_id for candidate in next_candidates
                ],
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
```

- [ ] **Step 4: Generate the committed fixture files**

Run:

```powershell
.\.venv\Scripts\python.exe tests\fixtures\checkpoints\generate_v080_fixtures.py
```

Expected: PASS with output containing `Wrote 5 checkpoint fixtures`.

- [ ] **Step 5: Run the golden fixture tests and verify they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -v
```

Expected: PASS. All golden fixture shape, hash, resume, and derived incompatibility tests pass.

- [ ] **Step 6: Run adjacent checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: PASS. Existing generated-checkpoint tests still pass.

- [ ] **Step 7: Commit fixture tests and generated fixtures**

Run:

```powershell
git add tests/unit/test_checkpoint_golden_fixtures.py tests/fixtures/checkpoints/generate_v080_fixtures.py tests/fixtures/checkpoints/v0.8.0
git commit -m "test: add checkpoint golden fixtures"
```

Expected: commit succeeds and includes only the test module, generator, manifest, and checkpoint fixture JSON files.

### Task 2: Public Docs And Changelog Baseline

**Files:**
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update checkpoint documentation with the compatibility baseline**

In `docs/site/callbacks-checkpointing.md`, add this section immediately after the paragraph ending with `Resume fails with CheckpointError when the receiving optimizer does not match the checkpoint identity.`:

```markdown
## Compatibility Baseline

Stable JSON checkpoints produced by EvoCore 0.8.0 are the forward compatibility
baseline for checkpoint schema v1. Compatible patch and minor releases should
continue to load 0.8.0 stable checkpoint files for GA generation-loop, GA
ask/tell, and CMA-ES ask/tell workflows, or fail with an explicit
`CheckpointError` when a documented incompatibility is introduced.

The guarantee covers stable JSON checkpoint files only. Legacy GA pickle
checkpoints remain legacy support, but they are not part of the forward
compatibility guarantee. `OptimizationResult.to_dict()` exports and
`EventHistory.to_rows()` exports are not checkpoint files and are not replayed
to rebuild optimizer state.
```

Expected: the docs now identify v0.8.0 as the stable JSON checkpoint baseline and keep legacy pickle outside the guarantee.

- [ ] **Step 2: Update GA docs with the baseline note**

In `docs/site/ga.md`, replace this paragraph:

```markdown
The receiving optimizer must match the checkpoint seed, direction, gene space,
and optimizer configuration. Policy-driven `run(evaluator, policy=...)`
mid-loop resume and CMA-ES resume are not part of checkpoint v1. Manual GA
ask/tell checkpoints are supported with `ask_tell_checkpoint()` and
`resume_ask_tell_checkpoint(...)`. Result JSON and event rows are not checkpoint
files.
```

with:

```markdown
The receiving optimizer must match the checkpoint seed, direction, gene space,
and optimizer configuration. Stable JSON checkpoints produced by EvoCore 0.8.0
are the checkpoint schema v1 compatibility baseline for GA generation-loop and
manual GA ask/tell resume. Policy-driven `run(evaluator, policy=...)` mid-loop
resume remains outside checkpoint v1. Result JSON and event rows are not
checkpoint files.
```

Expected: GA docs no longer imply CMA-ES resume is unsupported and clearly name the v0.8.0 baseline.

- [ ] **Step 3: Update CMA-ES docs with the baseline note**

In `docs/site/cmaes.md`, replace this paragraph:

```markdown
The checkpoint combines the Rust CMA-ES state snapshot with Python candidate
ledgers, pending batches, telemetry, and audit events. Generation-loop and
policy-driven CMA-ES resume remain unsupported.
```

with:

```markdown
The checkpoint combines the Rust CMA-ES state snapshot with Python candidate
ledgers, pending batches, telemetry, and audit events. Stable JSON checkpoints
produced by EvoCore 0.8.0 are the checkpoint schema v1 compatibility baseline
for manual CMA-ES ask/tell resume. Generation-loop and policy-driven CMA-ES
resume remain unsupported.
```

Expected: CMA-ES docs identify v0.8.0 as the ask/tell checkpoint baseline while preserving generation-loop and policy-run exclusions.

- [ ] **Step 4: Add the changelog entry**

In `CHANGELOG.md`, replace:

```markdown
## [Unreleased]
```

with:

```markdown
## [Unreleased]

### Added

- Checkpoint golden fixtures documenting EvoCore 0.8.0 as the stable JSON
  checkpoint compatibility baseline for GA generation-loop, GA ask/tell, and
  CMA-ES ask/tell resume.
```

Expected: the user-visible compatibility guarantee is recorded under `[Unreleased]`.

- [ ] **Step 5: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS. MkDocs builds without broken navigation or Markdown errors.

- [ ] **Step 6: Run focused checkpoint docs-adjacent tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py tests/unit/test_checkpointing.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: PASS. Docs changes did not mask a fixture or checkpoint regression.

- [ ] **Step 7: Commit docs and changelog**

Run:

```powershell
git add docs/site/callbacks-checkpointing.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "docs: document checkpoint compatibility baseline"
```

Expected: commit succeeds and includes only docs and changelog changes.

### Task 3: Final Verification

**Files:**
- Verify: all files changed in Tasks 1 and 2

- [ ] **Step 1: Check Python formatting**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: PASS. No formatting changes required.

- [ ] **Step 2: Run Python linting**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS. The generator and fixture tests satisfy the repository lint rules.

- [ ] **Step 3: Run the focused checkpoint verification set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: PASS. Golden fixtures and existing checkpoint tests pass together.

- [ ] **Step 4: Run docs verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS. Public docs build with the new compatibility baseline wording.

- [ ] **Step 5: Check the worktree diff for unintended changes**

Run:

```powershell
git status --short
git diff --check
```

Expected: `git diff --check` prints no whitespace errors. `git status --short` shows only the intended task files if any final verification edits are still unstaged.

- [ ] **Step 6: Commit verification fixes if needed**

If Steps 1-5 required formatting or documentation fixes, run:

```powershell
git add tests/unit/test_checkpoint_golden_fixtures.py tests/fixtures/checkpoints/generate_v080_fixtures.py tests/fixtures/checkpoints/v0.8.0 docs/site/callbacks-checkpointing.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "test: verify checkpoint compatibility fixtures"
```

Expected: a commit is created only if final verification required additional edits. If no files changed after Task 2, skip this commit and record that no final fix commit was needed.

## Self-Review Notes

- Spec coverage: the plan adds committed v0.8.0 fixtures, a manifest with hashes, shape tests, behavior tests, negative incompatibility tests, docs, changelog, and focused verification.
- Scope boundary: the plan does not include version bumps, wheel builds, PyPI publishing, whole-framework release readiness, legacy pickle guarantees, event replay, or `CMAESOptimizer.run()` resume.
- Type consistency: tests and generator use existing public names `GeneticAlgorithmOptimizer`, `CMAESOptimizer`, `Gene`, `GeneSpace`, `EvaluationRecord`, `Solution`, and `CheckpointError`.
- Verification: Rust checks and `maturin develop` are not required unless implementation unexpectedly touches Rust, `_core.pyi`, or PyO3-facing behavior.
