# Differential Evolution Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `DifferentialEvolutionOptimizer` to a first-class, release-stable optimizer by adding DE golden checkpoint fixtures, deterministic resume coverage, lifecycle hardening, docs, and changelog coverage.

**Architecture:** Keep DE runtime code in `evocore/optimizers/de/` and add maturity around the existing public surface instead of expanding features. Use committed v0.9.0 fixture files under `tests/fixtures/checkpoints/v0.9.0/`, extend the existing golden fixture tests, harden checkpoint restore validation only where tests expose silent acceptance, and document DE as a stable ask/tell checkpoint surface. Do not add `run_multiple(...)`, policy-aware DE execution, custom DE strategies, or Rust DE kernels in this slice.

**Tech Stack:** Python 3.11+, pytest, EvoCore checkpoint APIs, stable JSON serialization, SHA-256 fixture hashes, MkDocs Markdown docs, repository-local `.venv`.

---

## File Structure

- Create `tests/fixtures/checkpoints/generate_v090_de_fixtures.py`: deterministic DE-only fixture generator for v0.9.0 checkpoint files and their manifest.
- Create `tests/fixtures/checkpoints/v0.9.0/manifest.json`: generated DE fixture manifest with metadata, SHA-256 hashes, and continuation assertions.
- Create `tests/fixtures/checkpoints/v0.9.0/de-after-initial-ask.evocore-checkpoint.json`: DE checkpoint after an initialization `ask()`.
- Create `tests/fixtures/checkpoints/v0.9.0/de-after-partial-initial-tell.evocore-checkpoint.json`: DE checkpoint after partial initialization `tell(...)`.
- Create `tests/fixtures/checkpoints/v0.9.0/de-after-initialized-population.evocore-checkpoint.json`: DE checkpoint after the initial target population is full.
- Create `tests/fixtures/checkpoints/v0.9.0/de-after-trial-ask.evocore-checkpoint.json`: DE checkpoint after trial candidate generation.
- Create `tests/fixtures/checkpoints/v0.9.0/de-after-mixed-trial-tell.evocore-checkpoint.json`: DE checkpoint after accepted and rejected trial records.
- Modify `tests/unit/test_checkpoint_golden_fixtures.py`: add v0.9.0 DE fixture shape, hash, identity, and continuation tests.
- Modify `tests/unit/test_de_checkpointing.py`: add DE restore validation, resume equivalence, and deterministic checkpoint continuation tests.
- Modify `tests/unit/test_de_ask_tell.py`: add lifecycle edge-case tests for stale tells, cached records, surrogate/partial records, and minimize rejection.
- Modify `tests/unit/test_de_engine.py`: add a public checkpoint example smoke test and seeded run reproducibility test.
- Modify `evocore/optimizers/de/checkpointing.py`: require all DE checkpoint payload fields that are necessary for deterministic resume.
- Modify `evocore/optimizers/de/ask_tell.py`: reject state-updating records for consumed DE batches so stale trial candidates cannot mutate population state.
- Modify `docs/site/de.md`: expand DE usage, reproducibility, checkpointing, acceptance-decision, and limitation docs.
- Modify `docs/site/callbacks-checkpointing.md`: list DE as a v0.9.0 stable ask/tell checkpoint baseline and add a complete DE checkpoint example.
- Modify `CHANGELOG.md`: add user-facing DE stabilization and checkpoint compatibility bullets.

## Scope Check

This plan targets one subsystem: Differential Evolution release-maturity parity. It does not add new DE features such as `run_multiple(...)`, policy-aware `run(...)`, strategy plugins, benchmarks, or Rust acceleration.

## Execution Setup

- [ ] **Step 1: Confirm the starting branch and working tree**

Run:

```powershell
git status --short --branch
```

Expected if the plan/spec docs have already merged:

```text
## main...origin/main
```

If the output shows uncommitted files, inspect them before editing:

```powershell
git diff --name-only
git diff --cached --name-only
```

- [ ] **Step 2: Create the implementation branch**

Run:

```powershell
git switch -c feature/de-stabilization
```

Expected:

```text
Switched to a new branch 'feature/de-stabilization'
```

If the plan/spec branch is still under review and the user explicitly asks to implement before it merges, create `feature/de-stabilization` from the current branch so the spec and plan remain present.

- [ ] **Step 3: Confirm the local Python interpreter**

Run:

```powershell
Test-Path .\.venv\Scripts\python.exe
```

Expected:

```text
True
```

If this prints `False`, stop and report that the repository-local virtual environment is missing.

### Task 1: DE Golden Fixture Shape Tests

**Files:**
- Modify: `tests/unit/test_checkpoint_golden_fixtures.py`

- [ ] **Step 1: Add DE imports and constants**

Modify the existing import from `evocore` so it includes `DifferentialEvolutionOptimizer`:

```python
from evocore import (
    CheckpointError,
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
)
```

Add these constants below the existing v0.8.0 constants:

```python
DE_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "checkpoints" / "v0.9.0"
DE_MANIFEST_PATH = DE_FIXTURE_DIR / "manifest.json"

EXPECTED_DE_FILES = {
    "de-after-initial-ask.evocore-checkpoint.json",
    "de-after-partial-initial-tell.evocore-checkpoint.json",
    "de-after-initialized-population.evocore-checkpoint.json",
    "de-after-trial-ask.evocore-checkpoint.json",
    "de-after-mixed-trial-tell.evocore-checkpoint.json",
}
```

- [ ] **Step 2: Add DE fixture helper functions**

Add these helpers after `_cmaes_optimizer(...)`:

```python
def _load_de_manifest() -> dict[str, Any]:
    assert DE_MANIFEST_PATH.exists(), f"missing DE checkpoint fixture manifest: {DE_MANIFEST_PATH}"
    return json.loads(DE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _de_fixture_entries() -> list[dict[str, Any]]:
    return list(_load_de_manifest()["fixtures"])


def _de_entry(name: str) -> dict[str, Any]:
    for fixture in _de_fixture_entries():
        if fixture["name"] == name:
            return fixture
    raise AssertionError(f"DE fixture entry {name!r} is missing from manifest")


def _de_fixture_path(entry: dict[str, Any]) -> Path:
    return DE_FIXTURE_DIR / entry["file"]


def _de_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _de_optimizer(**overrides: object) -> DifferentialEvolutionOptimizer:
    params: dict[str, object] = {
        "population_size": 6,
        "max_generations": 5,
        "seed": 42,
    }
    params.update(overrides)
    return DifferentialEvolutionOptimizer(_de_space(), **params)


def _de_score_from_genes(genes: Iterable[object]) -> float:
    x, period, enabled, fixed = genes
    score = -abs(float(x) - 0.25) - abs(int(period) - 7) + float(fixed)
    if bool(enabled):
        score += 2.0
    return float(score)


def _de_first_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    batches = payload["state"]["payload"]["batches_by_id"]
    assert len(batches) == 1
    return next(iter(batches.values()))


def _de_records_from_payload(
    payload: dict[str, Any],
    *,
    skip_existing: bool = False,
    scores: dict[str, float | None] | None = None,
    rejected_candidate_ids: set[str] | None = None,
) -> list[EvaluationRecord]:
    state_payload = payload["state"]["payload"]
    batch_payload = _de_first_batch_payload(payload)
    existing_candidate_ids = {record["candidate_id"] for record in batch_payload["records"]}
    rejected_candidate_ids = set(rejected_candidate_ids or set())
    records: list[EvaluationRecord] = []
    for candidate_id in batch_payload["candidate_ids"]:
        if skip_existing and candidate_id in existing_candidate_ids:
            continue
        candidate_payload = state_payload["candidates_by_id"][candidate_id]
        score = (
            scores[candidate_id]
            if scores is not None and candidate_id in scores
            else _de_score_from_genes(candidate_payload["genes"])
        )
        confidence = "rejected" if candidate_id in rejected_candidate_ids else "trusted_full"
        records.append(
            EvaluationRecord(
                candidate_id=candidate_id,
                batch_id=batch_payload["batch_id"],
                score=None if confidence == "rejected" else score,
                confidence=confidence,
                stage="full",
                cost=1.0,
                metadata={"source": "de-golden-fixture-test"},
            )
        )
    return records
```

- [ ] **Step 3: Add DE fixture shape tests**

Add these tests near the existing manifest/hash tests:

```python
def test_manifest_lists_expected_v090_de_checkpoint_fixtures() -> None:
    manifest = _load_de_manifest()

    assert manifest["fixture_format_version"] == 1
    assert manifest["source_evocore_version"] == "0.9.0"
    assert manifest["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert {entry["file"] for entry in manifest["fixtures"]} == EXPECTED_DE_FILES

    for entry in manifest["fixtures"]:
        assert _de_fixture_path(entry).exists()


def test_de_fixture_file_hashes_match_manifest() -> None:
    for entry in _de_fixture_entries():
        assert _sha256(_de_fixture_path(entry)) == entry["sha256"]


def test_valid_de_fixtures_load_and_match_manifest_identity() -> None:
    for entry in _de_fixture_entries():
        payload = load_checkpoint(_de_fixture_path(entry))

        assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert payload["created_by"]["evocore_version"] == "0.9.0"
        assert payload["optimizer"]["optimizer_type"] == "DifferentialEvolutionOptimizer"
        assert payload["optimizer"]["optimizer_type"] == entry["optimizer_type"]
        assert payload["optimizer"]["seed"] == entry["seed"]
        assert payload["optimizer"]["direction"] == entry["direction"]
        assert payload["optimizer"]["gene_space_hash"] == entry["gene_space_hash"]
        assert payload["optimizer"]["optimizer_config_hash"] == entry["optimizer_config_hash"]
        assert payload["state"]["payload"]["state_kind"] == entry["state_kind"]


def test_de_fixture_payloads_do_not_embed_machine_local_paths() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    forbidden_fragments = (repo_root, "C:\\", "D:\\", "/home/", "/Users/")

    for entry in _de_fixture_entries():
        payload = load_checkpoint(_de_fixture_path(entry))
        for text in _walk_strings(payload):
            assert not any(fragment in text for fragment in forbidden_fragments)
```

- [ ] **Step 4: Run the new shape test and verify it fails because fixtures are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py::test_manifest_lists_expected_v090_de_checkpoint_fixtures -v
```

Expected: FAIL with an assertion mentioning the missing DE checkpoint fixture manifest at `tests\fixtures\checkpoints\v0.9.0\manifest.json`.

### Task 2: DE Golden Fixture Generator And Files

**Files:**
- Create: `tests/fixtures/checkpoints/generate_v090_de_fixtures.py`
- Create: `tests/fixtures/checkpoints/v0.9.0/manifest.json`
- Create: `tests/fixtures/checkpoints/v0.9.0/de-after-initial-ask.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.9.0/de-after-partial-initial-tell.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.9.0/de-after-initialized-population.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.9.0/de-after-trial-ask.evocore-checkpoint.json`
- Create: `tests/fixtures/checkpoints/v0.9.0/de-after-mixed-trial-tell.evocore-checkpoint.json`

- [ ] **Step 1: Create the DE fixture generator**

Create `tests/fixtures/checkpoints/generate_v090_de_fixtures.py` with this content:

```python
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
    result = restored.tell(_trusted_records_for_candidates(candidates[2:], scores=[3.0, 4.0, 5.0, 6.0]))
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
    source.tell(_trusted_records_for_candidates(targets, scores=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
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
```

- [ ] **Step 2: Generate the committed DE fixtures**

Run:

```powershell
.\.venv\Scripts\python.exe tests\fixtures\checkpoints\generate_v090_de_fixtures.py
```

Expected:

```text
Wrote 5 DE checkpoint fixtures to D:\Kerja\pribadi\evocore\tests\fixtures\checkpoints\v0.9.0
```

- [ ] **Step 3: Run the DE fixture shape tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py::test_manifest_lists_expected_v090_de_checkpoint_fixtures tests/unit/test_checkpoint_golden_fixtures.py::test_de_fixture_file_hashes_match_manifest tests/unit/test_checkpoint_golden_fixtures.py::test_valid_de_fixtures_load_and_match_manifest_identity tests/unit/test_checkpoint_golden_fixtures.py::test_de_fixture_payloads_do_not_embed_machine_local_paths -v
```

Expected: PASS.

- [ ] **Step 4: Commit the DE fixture generator and shape tests**

Run:

```powershell
git add tests\unit\test_checkpoint_golden_fixtures.py tests\fixtures\checkpoints\generate_v090_de_fixtures.py tests\fixtures\checkpoints\v0.9.0
git commit -m "test(de): add golden checkpoint fixtures"
```

Expected: commit succeeds with the fixture generator, manifest, five DE checkpoint files, and the updated fixture tests.

### Task 3: DE Golden Fixture Continuation Tests

**Files:**
- Modify: `tests/unit/test_checkpoint_golden_fixtures.py`

- [ ] **Step 1: Add DE fixture continuation tests**

Add these tests after the existing CMA-ES fixture behavior tests:

```python
def test_de_after_initial_ask_fixture_accepts_pending_records() -> None:
    entry = _de_entry("de_after_initial_ask")
    payload = load_checkpoint(_de_fixture_path(entry))

    restored = _de_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_de_fixture_path(entry))
    result = restored.tell(_de_records_from_payload(payload))

    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert result.trusted_count == entry["continuation"]["trusted_count_after_tell"]
    assert result.state_accepted_count == entry["continuation"]["state_accepted_count_after_tell"]
    assert result.best_score == pytest.approx(entry["continuation"]["best_score_after_tell"])
    assert restored.state_summary().trusted_count == 6
    assert result.pending_batch_ids == ()


def test_de_partial_initial_fixture_accepts_missing_records() -> None:
    entry = _de_entry("de_after_partial_initial_tell")
    payload = load_checkpoint(_de_fixture_path(entry))

    restored = _de_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_de_fixture_path(entry))
    result = restored.tell(
        _de_records_from_payload(
            payload,
            skip_existing=True,
            scores={
                candidate_id: float(index + 3)
                for index, candidate_id in enumerate(
                    _de_first_batch_payload(payload)["candidate_ids"][2:]
                )
            },
        )
    )

    assert summary.best_candidate_id == entry["continuation"]["best_candidate_id"]
    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert result.accepted_count == entry["continuation"]["accepted_count_after_tell"]
    assert restored.state_summary().trusted_count == entry["continuation"]["trusted_count_after_tell"]
    assert result.best_score == pytest.approx(entry["continuation"]["best_score_after_tell"])
    assert result.pending_batch_ids == ()


def test_de_initialized_population_fixture_next_ask_matches_manifest() -> None:
    entry = _de_entry("de_after_initialized_population")
    expected = entry["continuation"]["next_ask"]

    restored = _de_optimizer()
    restored.resume_ask_tell_checkpoint(_de_fixture_path(entry))
    candidates = restored.ask()

    assert [candidate.candidate_id for candidate in candidates] == expected["candidate_ids"]
    assert [candidate.batch_id for candidate in candidates] == expected["batch_ids"]
    assert [candidate.genes for candidate in candidates] == expected["genes"]
    assert [candidate.metadata["target_slot"] for candidate in candidates] == expected["target_slots"]
    assert [
        candidate.metadata["target_candidate_id"] for candidate in candidates
    ] == expected["target_candidate_ids"]


def test_de_trial_ask_fixture_accepts_first_trial_record() -> None:
    entry = _de_entry("de_after_trial_ask")
    payload = load_checkpoint(_de_fixture_path(entry))
    first_candidate_id = _de_first_batch_payload(payload)["candidate_ids"][0]

    restored = _de_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_de_fixture_path(entry))
    result = restored.tell(
        [
            EvaluationRecord(
                candidate_id=first_candidate_id,
                batch_id=_de_first_batch_payload(payload)["batch_id"],
                score=100.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
                metadata={"source": "de-golden-fixture-test"},
            )
        ]
    )

    decision = result.acceptance_decisions[0]
    assert summary.pending_batch_ids == tuple(entry["continuation"]["pending_batch_ids"])
    assert decision.candidate_id == entry["continuation"]["first_decision"]["candidate_id"]
    assert decision.accepted_for_state is entry["continuation"]["first_decision"]["accepted_for_state"]
    assert decision.reason == entry["continuation"]["first_decision"]["reason"]
    assert decision.target_candidate_id == entry["continuation"]["first_decision"]["target_candidate_id"]
    assert decision.target_slot == entry["continuation"]["first_decision"]["target_slot"]


def test_de_mixed_trial_tell_fixture_restores_manifest_state() -> None:
    entry = _de_entry("de_after_mixed_trial_tell")

    restored = _de_optimizer()
    summary = restored.resume_ask_tell_checkpoint(_de_fixture_path(entry))

    assert restored.generation == entry["continuation"]["generation"]
    assert summary.trusted_count == entry["continuation"]["trusted_count"]
    assert summary.best_candidate_id == entry["continuation"]["best_candidate_id"]
    assert summary.best_score == pytest.approx(entry["continuation"]["best_score"])
    assert list(restored._target_candidate_ids) == entry["continuation"]["target_candidate_ids"]
    assert [
        restored._candidates_by_id[candidate_id].genes
        for candidate_id in restored._target_candidate_ids
    ] == entry["continuation"]["target_genes"]
```

- [ ] **Step 2: Run the DE continuation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -k "de_" -v
```

Expected: PASS.

- [ ] **Step 3: Commit continuation tests**

Run:

```powershell
git add tests\unit\test_checkpoint_golden_fixtures.py
git commit -m "test(de): verify golden checkpoint continuation"
```

Expected: commit succeeds with only `tests/unit/test_checkpoint_golden_fixtures.py` staged.

### Task 4: DE Checkpoint Restore Validation

**Files:**
- Modify: `tests/unit/test_de_checkpointing.py`
- Modify: `evocore/optimizers/de/checkpointing.py`

- [ ] **Step 1: Add failing restore validation tests**

Add `import copy` to the top of `tests/unit/test_de_checkpointing.py`:

```python
import copy
```

Add these tests after `test_de_checkpoint_rejects_wrong_optimizer_identity`:

```python
@pytest.mark.parametrize(
    "field",
    [
        "event_index",
        "generation",
        "candidates_by_id",
        "batches_by_id",
        "target_candidate_ids",
        "trial_target_slots",
        "trial_target_candidate_ids",
        "telemetry",
        "events",
    ],
)
def test_de_checkpoint_rejects_missing_required_payload_fields(field: str) -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    engine.ask()
    payload = engine.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"][field]

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match=field):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_target_population_larger_than_config() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["target_candidate_ids"].append(candidates[0].candidate_id)

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target_candidate_ids"):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_trial_mapping_with_wrong_target_slot() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    trial = engine.ask()[0]
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["trial_target_slots"][trial.candidate_id] = 99

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target slot"):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_trial_mapping_with_mismatched_target_id() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    trial = engine.ask()[0]
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["trial_target_candidate_ids"][trial.candidate_id] = candidates[1].candidate_id

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target_candidate_id"):
        restored.resume_ask_tell_checkpoint(payload)
```

- [ ] **Step 2: Run the new tests and verify at least one fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -k "missing_required_payload_fields or larger_than_config or wrong_target_slot or mismatched_target_id" -v
```

Expected before implementation: FAIL because DE restore currently defaults some missing fields and does not validate target-slot consistency deeply enough.

- [ ] **Step 3: Add checkpoint payload validation helpers**

In `evocore/optimizers/de/checkpointing.py`, add these helpers below the constants:

```python
def _required_payload_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise CheckpointError(f"checkpoint state.payload.{key} is required.")
    return payload[key]


def _required_mapping_payload(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required_payload_value(payload, key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an object.")
    return value


def _required_sequence_payload(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = _required_payload_value(payload, key)
    if not isinstance(value, list | tuple):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an array.")
    return list(value)


def _required_int_payload(payload: Mapping[str, Any], key: str) -> int:
    value = _required_payload_value(payload, key)
    if isinstance(value, bool):
        raise CheckpointError(f"checkpoint state.payload.{key} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint state.payload.{key} must be an integer.") from exc
```

- [ ] **Step 4: Use required-field helpers in `_restore_ask_tell_state`**

In `_restore_ask_tell_state`, replace the existing reads for candidates, batches, target IDs, trial mappings, telemetry/events, event index, and generation with this structure:

```python
        raw_candidates = _required_mapping_payload(state_payload, "candidates_by_id")
        candidates = {
            str(candidate_id): candidate_from_checkpoint(candidate_payload)
            for candidate_id, candidate_payload in raw_candidates.items()
        }
```

```python
        raw_batches = _required_mapping_payload(state_payload, "batches_by_id")
        batches = {
            str(batch_id): batch_from_checkpoint(batch_payload)
            for batch_id, batch_payload in raw_batches.items()
        }
```

```python
        target_candidate_ids = [
            str(value)
            for value in _required_sequence_payload(state_payload, "target_candidate_ids")
        ]
        if len(target_candidate_ids) > self.population_size:
            raise CheckpointError(
                "checkpoint state.payload.target_candidate_ids cannot be larger than "
                "the configured DE population_size."
            )
```

```python
        trial_target_slots = {
            str(candidate_id): int(slot)
            for candidate_id, slot in _required_mapping_payload(
                state_payload, "trial_target_slots"
            ).items()
        }
        trial_target_candidate_ids = {
            str(candidate_id): str(target_id)
            for candidate_id, target_id in _required_mapping_payload(
                state_payload, "trial_target_candidate_ids"
            ).items()
        }
```

Inside the existing loop that validates trial mappings, add these consistency checks after `target_id = trial_target_candidate_ids[candidate_id]`:

```python
            target_slot = trial_target_slots[candidate_id]
            if target_slot < 0 or target_slot >= len(target_candidate_ids):
                raise CheckpointError(
                    f"checkpoint trial target slot {target_slot!r} is outside target_candidate_ids."
                )
            if target_candidate_ids[target_slot] != target_id:
                raise CheckpointError(
                    "checkpoint trial target_candidate_id does not match "
                    f"target_candidate_ids[{target_slot}]."
                )
```

Replace the final telemetry/events/event-index/generation assignments with:

```python
        self.vnext_telemetry = telemetry_from_checkpoint(
            _required_mapping_payload(state_payload, "telemetry")
        )
        self.events = event_history_from_checkpoint(
            _required_sequence_payload(state_payload, "events")
        )
        self._event_index = _required_int_payload(state_payload, "event_index")
        self.generation = _required_int_payload(state_payload, "generation")
```

- [ ] **Step 5: Run DE checkpointing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit restore validation hardening**

Run:

```powershell
git add tests\unit\test_de_checkpointing.py evocore\optimizers\de\checkpointing.py
git commit -m "fix(de): validate checkpoint restore payloads"
```

Expected: commit succeeds with the DE checkpointing test and implementation changes.

### Task 5: DE Resume Equivalence And Determinism Tests

**Files:**
- Modify: `tests/unit/test_de_checkpointing.py`
- Modify: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add checkpoint resume equivalence tests**

Add these helpers to `tests/unit/test_de_checkpointing.py` after `_records(...)`:

```python
def _target_genes(engine: DifferentialEvolutionOptimizer) -> list[list[object]]:
    return [
        list(engine._candidates_by_id[candidate_id].genes)
        for candidate_id in engine._target_candidate_ids
    ]


def _state_tuple(engine: DifferentialEvolutionOptimizer) -> tuple[object, ...]:
    summary = engine.state_summary()
    return (
        engine.generation,
        summary.best_candidate_id,
        summary.best_score,
        summary.trusted_count,
        tuple(summary.pending_batch_ids),
        tuple(engine._target_candidate_ids),
        tuple(tuple(values) for values in _target_genes(engine)),
    )
```

Add these tests after `test_de_checkpoint_restores_after_partial_initial_tell`:

```python
def test_de_checkpoint_resume_matches_uninterrupted_initialization_completion() -> None:
    original = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = original.ask()
    original.tell(_records(candidates[:2], [1.0, 2.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(candidates[2:], [3.0, 4.0, 5.0, 6.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.best_score == pytest.approx(uninterrupted_result.best_score)
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count
    assert [
        decision.accepted_for_state for decision in restored_result.acceptance_decisions
    ] == [
        decision.accepted_for_state for decision in uninterrupted_result.acceptance_decisions
    ]


def test_de_checkpoint_resume_matches_uninterrupted_trial_completion() -> None:
    original = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    targets = original.ask()
    original.tell(_records(targets, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    trials = original.ask()
    original.tell(_records(trials[:2], [100.0, -100.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(trials[2:], [-101.0, 99.0, -102.0, 98.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.consumed_batch_ids == uninterrupted_result.consumed_batch_ids
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count
    assert [
        (decision.accepted_for_state, decision.reason, decision.target_slot)
        for decision in restored_result.acceptance_decisions
    ] == [
        (decision.accepted_for_state, decision.reason, decision.target_slot)
        for decision in uninterrupted_result.acceptance_decisions
    ]
```

- [ ] **Step 2: Add seeded run reproducibility test**

Add this test to `tests/unit/test_de_engine.py` after `test_de_run_returns_optimization_result_with_events_and_generations`:

```python
def test_de_run_is_reproducible_for_same_seed_and_config() -> None:
    left = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        seed=42,
    )
    right = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        seed=42,
    )

    left_result = left.run(SphereEvaluator())
    right_result = right.run(SphereEvaluator())

    assert left_result.best_score == pytest.approx(right_result.best_score)
    assert left_result.best_candidate_id == right_result.best_candidate_id
    assert [record.best_score for record in left_result.generations] == [
        record.best_score for record in right_result.generations
    ]
    assert [tuple(event.genes or ()) for event in left_result.events] == [
        tuple(event.genes or ()) for event in right_result.events
    ]
```

- [ ] **Step 3: Run DE checkpointing and engine tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py tests/unit/test_de_engine.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit resume equivalence and determinism tests**

Run:

```powershell
git add tests\unit\test_de_checkpointing.py tests\unit\test_de_engine.py
git commit -m "test(de): cover deterministic checkpoint resume"
```

Expected: commit succeeds with the two updated test files.

### Task 6: DE Ask/Tell Lifecycle Hardening

**Files:**
- Modify: `tests/unit/test_de_ask_tell.py`
- Modify: `evocore/optimizers/de/ask_tell.py`

- [ ] **Step 1: Add lifecycle edge-case tests**

Add these tests to `tests/unit/test_de_ask_tell.py` after `test_de_minimize_replaces_when_trial_score_is_lower`:

```python
def test_de_minimize_keeps_target_when_trial_score_is_higher() -> None:
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction="minimize",
    )
    targets = engine.ask()
    engine.tell(_records(targets, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]))
    trial = engine.ask()[0]
    target_slot = trial.metadata["target_slot"]

    result = engine.tell(_records([trial], [100.0]))

    assert result.state_accepted_count == 0
    assert result.acceptance_decisions[0].accepted_for_state is False
    assert result.acceptance_decisions[0].reason == "trial_kept_target"
    assert engine._target_candidate_ids[target_slot] == trial.metadata["target_candidate_id"]


def test_de_cached_records_are_state_eligible_for_initialization_and_trial_replacement() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    targets = engine.ask()
    init_result = engine.tell(_records(targets, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], confidence="cached"))
    trial = engine.ask()[0]

    replacement_result = engine.tell(_records([trial], [100.0], confidence="cached"))

    assert init_result.cached_count == 6
    assert init_result.state_accepted_count == 6
    assert replacement_result.cached_count == 1
    assert replacement_result.state_accepted_count == 1
    assert replacement_result.acceptance_decisions[0].reason == "trial_replaced_target"


def test_de_partial_and_surrogate_records_do_not_replace_trial_targets() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=trials[0].candidate_id,
                batch_id=trials[0].batch_id,
                score=100.0,
                confidence="partial",
                stage="screen",
            ),
            EvaluationRecord(
                candidate_id=trials[1].candidate_id,
                batch_id=trials[1].batch_id,
                score=100.0,
                confidence="surrogate",
                stage="surrogate",
            ),
        ]
    )

    assert result.partial_count == 1
    assert result.surrogate_count == 1
    assert result.state_accepted_count == 0
    assert result.acceptance_decisions == ()
    assert engine._target_candidate_ids == [target.candidate_id for target in targets]


def test_de_rejects_state_record_for_consumed_trial_batch() -> None:
    engine, _ = _trusted_engine()
    trials = engine.ask()
    rejected = [
        EvaluationRecord(
            candidate_id=trial.candidate_id,
            batch_id=trial.batch_id,
            score=None,
            confidence="rejected",
            stage="full",
            metadata={"reason": "constraint"},
        )
        for trial in trials
    ]
    engine.tell(rejected)

    with pytest.raises(FitnessError, match="already been consumed"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=trials[0].candidate_id,
                    batch_id=trials[0].batch_id,
                    score=100.0,
                    confidence="trusted_full",
                    stage="retry",
                )
            ]
        )
```

- [ ] **Step 2: Run the stale consumed-batch test and verify it fails before implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_rejects_state_record_for_consumed_trial_batch -v
```

Expected before implementation: FAIL because the consumed batch currently accepts a subsequent state-updating record for the stale candidate.

- [ ] **Step 3: Reject state records for consumed DE batches**

In `evocore/optimizers/de/ask_tell.py`, change this line inside `tell(...)`:

```python
            batch.accept_record(record)
```

to:

```python
            batch.accept_record(record, reject_consumed_state_record=True)
```

- [ ] **Step 4: Run DE ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit lifecycle hardening**

Run:

```powershell
git add tests\unit\test_de_ask_tell.py evocore\optimizers\de\ask_tell.py
git commit -m "fix(de): reject stale consumed batch records"
```

Expected: commit succeeds with the DE ask/tell test and implementation changes.

### Task 7: DE Public Checkpoint Smoke Test And Documentation

**Files:**
- Modify: `tests/unit/test_de_engine.py`
- Modify: `docs/site/de.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add public checkpoint smoke test**

Add this test to `tests/unit/test_de_engine.py` after `test_de_run_honors_max_evaluations`:

```python
def test_de_public_checkpoint_example_smoke(tmp_path) -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )
    optimizer = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
    candidates = optimizer.ask()
    checkpoint_path = tmp_path / "de-ask-tell.evocore-checkpoint.json"
    optimizer.save_checkpoint(
        checkpoint_path,
        optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
    )

    restored = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            stage="full",
        )
        for candidate in candidates
    ]
    result = restored.tell(records)

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.trusted_count == 6
    assert result.pending_batch_ids == ()
```

- [ ] **Step 2: Run the public smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_public_checkpoint_example_smoke -v
```

Expected: PASS.

- [ ] **Step 3: Expand DE docs**

Update `docs/site/de.md` so it includes these sections after the opening run example:

````markdown
## When To Choose DE

DE is a good fit for continuous or mostly numeric search spaces where objective
evaluations are expensive enough that stable candidate proposals matter more
than gradient information. It is often easier to tune than GA for numeric
parameters because the default `rand1bin` strategy uses scaled population
differences instead of custom crossover and mutation operators.

Use GA when the search is heavily discrete, operator design is central, or
multi-run utilities are required today. Use CMA-ES when the space is continuous
and covariance adaptation is the main advantage. DE currently supports flat
`float`, `int`, and `bool` `GeneSpace` values; CMA-ES continues to be the more
specialized continuous optimizer.

## Reproducibility

DE candidate IDs, batch IDs, initialization samples, trial target mappings, and
replacement decisions are deterministic for the same `GeneSpace`, optimizer
configuration, direction, and seed. Stable checkpoint files also include the
gene-space hash, optimizer config hash, seed, and direction so resume fails
early when the receiving optimizer does not match the saved state.

## Ask/Tell Checkpointing

Manual ask/tell checkpoints are the stable continuation boundary for DE.
Checkpointing preserves pending initialization candidates, target population
state, pending trial-to-target mappings, telemetry, and audit events.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace

space = GeneSpace(
    [
        Gene("x", "float", -5.0, 5.0),
        Gene("period", "int", 2, 20),
        Gene("enabled", "bool"),
    ]
)
optimizer = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "de-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
summary = restored.resume_ask_tell_checkpoint("de-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

## Current Limitations

DE does not yet expose `run_multiple(...)`, policy-aware `run(...)`, custom
strategy plugins, or a Rust-backed variation kernel. Those are future feature
and performance parity tracks; the current DE checkpoint contract focuses on
manual ask/tell continuation and synchronous evaluator-driven `run()`.
````

Keep the existing "Mixed Bool And Numeric Spaces" and "Ask/Tell Acceptance Decisions" sections. If the new sections make wording duplicate, merge sentences without removing the checkpoint example or limitation list.

- [ ] **Step 4: Update checkpoint docs for the DE baseline**

In `docs/site/callbacks-checkpointing.md`, update "Compatibility Baseline" to this wording:

```markdown
Stable JSON checkpoints produced by EvoCore 0.8.0 are the forward compatibility
baseline for checkpoint schema v1 across GA generation-loop, GA ask/tell, and
CMA-ES ask/tell workflows. Differential Evolution ask/tell checkpoints join the
stable checkpoint surface with the EvoCore 0.9.0 DE fixture baseline.
Compatible patch and minor releases should continue to load these stable
checkpoint files, or fail with an explicit `CheckpointError` when a documented
incompatibility is introduced.
```

Replace the short "Differential Evolution Ask/Tell Checkpoints" section with:

````markdown
## Differential Evolution Ask/Tell Checkpoints

Differential Evolution ask/tell checkpoints store target slots and pending trial
mappings in addition to candidates, batches, telemetry, and events. This lets a
restored optimizer compare returned trial records against the same target
candidate after resume.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, GeneSpace

gene_space = GeneSpace.uniform(-5.0, 5.0, 3)
optimizer = DifferentialEvolutionOptimizer(gene_space, population_size=6, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "de-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = DifferentialEvolutionOptimizer(gene_space, population_size=6, seed=42)
summary = restored.resume_ask_tell_checkpoint("de-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

DE checkpoint identity validation covers optimizer type, seed, direction,
gene-space hash, optimizer config hash, checkpoint state kind, schema version,
and trial target mappings.
````

- [ ] **Step 5: Update changelog**

Add these bullets under `## [Unreleased]` -> `### Added` in `CHANGELOG.md`:

```markdown
- Added committed Differential Evolution v0.9.0 golden checkpoint fixtures with
  manifest hashes and deterministic continuation coverage.
- Documented Differential Evolution as a stable ask/tell checkpoint surface,
  including reproducibility guarantees, target replacement decisions, and
  current feature limitations.
```

If the existing `DifferentialEvolutionOptimizer` bullet is still present in `[Unreleased]`, keep it and place these bullets below it.

- [ ] **Step 6: Run docs-related tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_public_checkpoint_example_smoke tests/unit/test_de_engine.py::test_de_run_is_reproducible_for_same_seed_and_config -v
```

Expected: PASS.

- [ ] **Step 7: Commit docs and changelog**

Run:

```powershell
git add tests\unit\test_de_engine.py docs\site\de.md docs\site\callbacks-checkpointing.md CHANGELOG.md
git commit -m "docs(de): document stabilization guarantees"
```

Expected: commit succeeds with docs, changelog, and DE public smoke tests.

### Task 8: Final Verification And Pull Request

**Files:**
- No planned source edits. This task verifies all prior commits and opens the PR.

- [ ] **Step 1: Run formatting check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected:

```text
All checks passed!
```

or:

```text
<N> files already formatted
```

If formatting fails, run:

```powershell
.\.venv\Scripts\python.exe -m ruff format
```

Then commit formatting-only changes:

```powershell
git add tests\unit\test_checkpoint_golden_fixtures.py tests\unit\test_de_checkpointing.py tests\unit\test_de_ask_tell.py tests\unit\test_de_engine.py evocore\optimizers\de\checkpointing.py evocore\optimizers\de\ask_tell.py
git commit -m "style: format differential evolution stabilization"
```

- [ ] **Step 2: Run lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected:

```text
All checks passed!
```

If lint finds fixable issues, apply the smallest local edits, rerun the command, and commit with:

```powershell
git add tests\unit\test_checkpoint_golden_fixtures.py tests\unit\test_de_checkpointing.py tests\unit\test_de_ask_tell.py tests\unit\test_de_engine.py evocore\optimizers\de\checkpointing.py evocore\optimizers\de\ask_tell.py
git commit -m "fix(de): address stabilization lint"
```

- [ ] **Step 3: Rebuild the Python extension**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: command exits 0 and installs the local `evocore` package into `.venv`.

- [ ] **Step 4: Run focused DE and checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py tests/unit/test_de_checkpointing.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS for all selected tests.

- [ ] **Step 5: Run full unit and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS for all unit and integration tests.

- [ ] **Step 6: Confirm Rust verification is unnecessary**

Run:

```powershell
git diff --name-only main...HEAD -- src Cargo.toml Cargo.lock evocore\_core.pyi
```

Expected: no output.

If the command prints any Rust, Cargo, or PyO3 stub file, run the Rust verification from `AGENTS.md` before opening the PR:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

- [ ] **Step 7: Push the branch**

Run:

```powershell
git status --short --branch
git push -u origin feature/de-stabilization
```

Expected: branch pushes successfully and tracks `origin/feature/de-stabilization`.

- [ ] **Step 8: Open a draft PR**

Use `.github/pull_request_template.md`. The PR body should include:

```markdown
## Summary

Stabilizes Differential Evolution as a first-class optimizer by adding v0.9.0 golden checkpoint fixtures, deterministic continuation tests, stricter checkpoint restore validation, stale batch lifecycle hardening, and docs/changelog coverage.

## Changes

- Added DE v0.9.0 checkpoint fixtures and manifest hash tests.
- Added DE fixture continuation, resume equivalence, and seeded reproducibility tests.
- Hardened DE checkpoint restore validation for required payload fields and trial target mappings.
- Rejected state-updating records for consumed DE batches.
- Expanded DE checkpointing, reproducibility, and limitation docs.

## User And Maintainer Impact

DE ask/tell checkpoints are now covered as a stable v0.9.0 fixture baseline. Invalid DE checkpoints and stale state-updating tells fail earlier and more explicitly.

## Risk And Compatibility

No public API rename or feature expansion. DE checkpoint schema v1 remains the intended stable surface; malformed or incomplete payloads that were previously accepted may now raise `CheckpointError`.

## Verification

- `.\.venv\Scripts\python.exe -m ruff format --check` passed.
- `.\.venv\Scripts\python.exe -m ruff check` passed.
- `.\.venv\Scripts\python.exe -m maturin develop --release` passed.
- `.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py tests/unit/test_de_checkpointing.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/integration/test_de_mixed_gene_space.py -v` passed.
- `.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v` passed.

## Checklist

- [x] Tests pass locally or in CI.
- [x] Documentation is updated for behavior, workflow, or public API changes.
- [x] `CHANGELOG.md` is updated for user-visible changes.
- [x] Type stubs are updated when public API or `evocore._core` exports change.
- [x] Release, packaging, seed, checkpoint, and serialization impact has been considered.
```

Run:

```powershell
$prBodyPath = Join-Path $env:TEMP "de-stabilization-pr-body.md"
@'
## Summary

Stabilizes Differential Evolution as a first-class optimizer by adding v0.9.0 golden checkpoint fixtures, deterministic continuation tests, stricter checkpoint restore validation, stale batch lifecycle hardening, and docs/changelog coverage.

## Changes

- Added DE v0.9.0 checkpoint fixtures and manifest hash tests.
- Added DE fixture continuation, resume equivalence, and seeded reproducibility tests.
- Hardened DE checkpoint restore validation for required payload fields and trial target mappings.
- Rejected state-updating records for consumed DE batches.
- Expanded DE checkpointing, reproducibility, and limitation docs.

## User And Maintainer Impact

DE ask/tell checkpoints are now covered as a stable v0.9.0 fixture baseline. Invalid DE checkpoints and stale state-updating tells fail earlier and more explicitly.

## Risk And Compatibility

No public API rename or feature expansion. DE checkpoint schema v1 remains the intended stable surface; malformed or incomplete payloads that were previously accepted may now raise `CheckpointError`.

## Verification

- `.\.venv\Scripts\python.exe -m ruff format --check` passed.
- `.\.venv\Scripts\python.exe -m ruff check` passed.
- `.\.venv\Scripts\python.exe -m maturin develop --release` passed.
- `.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py tests/unit/test_de_checkpointing.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/integration/test_de_mixed_gene_space.py -v` passed.
- `.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v` passed.

## Checklist

- [x] Tests pass locally or in CI.
- [x] Documentation is updated for behavior, workflow, or public API changes.
- [x] `CHANGELOG.md` is updated for user-visible changes.
- [x] Type stubs are updated when public API or `evocore._core` exports change.
- [x] Release, packaging, seed, checkpoint, and serialization impact has been considered.
'@ | Set-Content -Path $prBodyPath -Encoding UTF8
gh pr create --draft --title "test(de): stabilize checkpoint compatibility" --body-file $prBodyPath --base main --head feature/de-stabilization
```

Expected: GitHub returns the draft PR URL.
