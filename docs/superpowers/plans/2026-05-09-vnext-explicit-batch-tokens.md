# EvoCore vNext Explicit Batch Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit batch tokens and first-class asynchronous partial `tell()` semantics to vNext GA and CMA ask/tell engines.

**Architecture:** Add public batch identity to candidate/evaluation primitives and a small private batch ledger helper. Wire the ledger into GA and CMA so partial records can arrive safely, while keeping policy-driven `run()` strict enough to reject incomplete synchronous evaluator output. Tighten policy validation so exactly one `trusted_full` rung exists and it is final.

**Tech Stack:** Python 3.11+ dataclasses, pytest, ruff, Rust/PyO3 extension rebuild through maturin, Cargo checks.

---

## File Structure

- Modify `evocore/evaluation.py`: add `Candidate.batch_id`, `EvaluationRecord.batch_id`, and candidate/record batch mismatch validation.
- Create `evocore/batches.py`: private deterministic batch ID helper and `CandidateBatch` ledger.
- Modify `evocore/policies.py`: enforce exactly one final `trusted_full` rung.
- Modify `evocore/ga.py`: add vNext state reset, batch ledger registration, partial `tell()` validation, and strict evaluator-output validation in `run()`.
- Modify `evocore/cmaes.py`: add batch ledger registration and partial trusted-record accumulation before state updates.
- Modify `docs/site/ask-tell-engines.md`: document explicit batch tokens and partial tells.
- Modify `docs/site/api.md`: include `Candidate.batch_id` and `EvaluationRecord.batch_id` behavior in surrounding prose.
- Modify `CHANGELOG.md`: mention explicit batch-token ask/tell semantics.
- Modify `tests/unit/test_vnext_evaluation.py`: primitive batch ID tests.
- Modify `tests/unit/test_vnext_policy_scheduler.py`: policy final trusted rung tests.
- Modify `tests/unit/test_ga_ask_tell_vnext.py`: GA batch-token, partial tell, duplicate, mismatch, missing evaluator records, and repeat-run tests.
- Modify `tests/unit/test_cmaes_ask_tell_vnext.py`: CMA batch-token, partial completion, out-of-order completion, and duplicate consumed batch tests.

---

## Task 1: Add Batch Identity Primitives And Private Ledger

**Files:**
- Modify: `evocore/evaluation.py`
- Create: `evocore/batches.py`
- Modify: `tests/unit/test_vnext_evaluation.py`

- [ ] **Step 1: Add failing primitive tests**

Append these tests to `tests/unit/test_vnext_evaluation.py`:

```python
def test_candidate_and_record_expose_batch_ids() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-1",
        genes=[1.0],
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        batch_id="b-1",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    candidate.apply_record(record)

    assert candidate.batch_id == "b-1"
    assert record.batch_id == "b-1"
    assert candidate.status == "trusted"


def test_candidate_rejects_record_for_different_batch() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-left",
        genes=[1.0],
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        batch_id="b-right",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    with pytest.raises(FitnessError, match="batch_id"):
        candidate.apply_record(record)
```

- [ ] **Step 2: Run primitive tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py::test_candidate_and_record_expose_batch_ids tests/unit/test_vnext_evaluation.py::test_candidate_rejects_record_for_different_batch -v
```

Expected: FAIL with `TypeError` because `Candidate` and `EvaluationRecord` do not accept `batch_id`.

- [ ] **Step 3: Add batch fields and validation**

In `evocore/evaluation.py`, update `EvaluationRecord`:

```python
@dataclass(frozen=True)
class EvaluationRecord:
    """Record one evaluator result returned to an ask/tell engine."""

    candidate_id: str
    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None
```

In `evocore/evaluation.py`, update `Candidate`:

```python
@dataclass
class Candidate:
    """Represent a vNext optimizer candidate with lifecycle and lineage."""

    candidate_id: str
    genes: list[GeneValue]
    batch_id: str = ""
    params: dict[str, GeneValue] | None = None
    origin: CandidateOrigin = "random"
    parents: Sequence[str] = ()
    event_index: int = 0
    generation: int | None = None
    rung: str | None = None
    status: CandidateStatus = "proposed"
    confidence: EvaluationConfidence | None = None
    cost: float = 0.0
    scores: dict[str, CandidateScore] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

In `Candidate.apply_record()`, add this check immediately after the candidate ID check:

```python
        if record.batch_id is not None and self.batch_id and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
            )
```

- [ ] **Step 4: Create private batch ledger helper**

Create `evocore/batches.py`:

```python
"""Private vNext batch ledger helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from evocore import _core
from evocore.evaluation import EvaluationRecord
from evocore.exceptions import FitnessError


def batch_id_from_seed(master_seed: int, event_index: int) -> str:
    """Return a deterministic public batch ID for an ask event."""
    candidate_style_id = _core.candidate_id(int(master_seed), int(event_index), 0)
    return f"b-{candidate_style_id.removeprefix('c-')}"


@dataclass
class CandidateBatch:
    """Track records received for one ask() batch."""

    batch_id: str
    candidate_ids: tuple[str, ...]
    continuous_samples_by_id: dict[str, list[float]] = field(default_factory=dict)
    records_by_key: dict[tuple[str, str], EvaluationRecord] = field(default_factory=dict)
    consumed: bool = False
    _candidate_id_set: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._candidate_id_set = set(self.candidate_ids)

    def accept_record(self, record: EvaluationRecord, *, reject_consumed_trusted: bool = False) -> None:
        """Validate and store one record for this batch."""
        if reject_consumed_trusted and self.consumed and record.confidence == "trusted_full":
            raise FitnessError(f"batch {self.batch_id!r} has already been consumed.")
        if record.batch_id is not None and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
            )
        if record.candidate_id not in self._candidate_id_set:
            raise FitnessError(
                f"candidate_id {record.candidate_id!r} does not belong to batch {self.batch_id!r}."
            )
        if record.confidence == "trusted_full":
            for existing in self.records_by_key.values():
                if (
                    existing.candidate_id == record.candidate_id
                    and existing.confidence == "trusted_full"
                ):
                    raise FitnessError(
                        f"candidate_id {record.candidate_id!r} already has a trusted_full record "
                        f"for batch {self.batch_id!r}."
                    )
        key = (record.candidate_id, record.rung)
        if key in self.records_by_key:
            raise FitnessError(
                f"candidate_id {record.candidate_id!r} already has a record for rung "
                f"{record.rung!r} in batch {self.batch_id!r}."
            )
        self.records_by_key[key] = record

    def ordered_trusted_full_records(self) -> list[EvaluationRecord] | None:
        """Return trusted records in ask order once the batch is complete."""
        trusted_by_candidate: dict[str, EvaluationRecord] = {}
        for record in self.records_by_key.values():
            if record.confidence == "trusted_full":
                trusted_by_candidate[record.candidate_id] = record
        if any(candidate_id not in trusted_by_candidate for candidate_id in self.candidate_ids):
            return None
        return [trusted_by_candidate[candidate_id] for candidate_id in self.candidate_ids]
```

- [ ] **Step 5: Run primitive tests and format**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py -v
python -m ruff format evocore/evaluation.py evocore/batches.py tests/unit/test_vnext_evaluation.py
python -m ruff check evocore/evaluation.py evocore/batches.py tests/unit/test_vnext_evaluation.py
```

Expected: PASS for tests and Ruff.

- [ ] **Step 6: Commit primitives and ledger**

```powershell
git add evocore/evaluation.py evocore/batches.py tests/unit/test_vnext_evaluation.py
git commit -m "feat: add vnext batch identity primitives"
```

---

## Task 2: Enforce Final Trusted Full Policy Rung

**Files:**
- Modify: `evocore/policies.py`
- Modify: `tests/unit/test_vnext_policy_scheduler.py`

- [ ] **Step 1: Add failing policy tests**

Append these tests to `tests/unit/test_vnext_policy_scheduler.py`:

```python
def test_policy_requires_trusted_full_rung_to_be_final() -> None:
    with pytest.raises(ConfigurationError, match="final rung"):
        MultiFidelityPolicy(
            rungs=[
                Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
                Rung("audit", budget=1.0, promote_fraction=1.0, confidence="partial"),
            ],
            full_evaluation_budget=16,
        )


def test_policy_rejects_multiple_trusted_full_rungs() -> None:
    with pytest.raises(ConfigurationError, match="exactly one trusted_full"):
        MultiFidelityPolicy(
            rungs=[
                Rung("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
                Rung("full_a", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
                Rung("full_b", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
            ],
            full_evaluation_budget=16,
        )
```

- [ ] **Step 2: Run policy tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_vnext_policy_scheduler.py::test_policy_requires_trusted_full_rung_to_be_final tests/unit/test_vnext_policy_scheduler.py::test_policy_rejects_multiple_trusted_full_rungs -v
```

Expected: FAIL because the current policy accepts both shapes.

- [ ] **Step 3: Update policy validation**

In `evocore/policies.py`, replace the existing trusted-full check with:

```python
        trusted_full_rungs = [rung for rung in self.rungs if rung.confidence == "trusted_full"]
        if not trusted_full_rungs:
            raise ConfigurationError("MultiFidelityPolicy requires a trusted_full rung.")
        if len(trusted_full_rungs) != 1:
            raise ConfigurationError("MultiFidelityPolicy requires exactly one trusted_full rung.")
        if self.rungs[-1].confidence != "trusted_full":
            raise ConfigurationError("MultiFidelityPolicy final rung must be trusted_full.")
```

- [ ] **Step 4: Run policy tests and format**

Run:

```powershell
python -m pytest tests/unit/test_vnext_policy_scheduler.py -v
python -m ruff format evocore/policies.py tests/unit/test_vnext_policy_scheduler.py
python -m ruff check evocore/policies.py tests/unit/test_vnext_policy_scheduler.py
```

Expected: PASS.

- [ ] **Step 5: Commit policy validation**

```powershell
git add evocore/policies.py tests/unit/test_vnext_policy_scheduler.py
git commit -m "fix: require final trusted full policy rung"
```

---

## Task 3: Add GA Batch Tokens And Partial Tell Semantics

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`

- [ ] **Step 1: Add failing GA batch tests**

At the top of `tests/unit/test_ga_ask_tell_vnext.py`, add:

```python
import pytest
```

After the `from evocore import (...)` block, add:

```python
from evocore.exceptions import FitnessError
```

Append these tests to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
class DroppingEvaluator(Evaluator):
    def evaluate(self, candidates, rung):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=1.0,
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates[:-1]
        ]


def test_ga_ask_assigns_stable_batch_id_per_batch() -> None:
    engine = GAEngine(_space(), population_size=6, generations=5, seed=123)

    first = engine.ask(3)
    second = engine.ask(2)

    assert len({candidate.batch_id for candidate in first}) == 1
    assert len({candidate.batch_id for candidate in second}) == 1
    assert first[0].batch_id != second[0].batch_id
    assert first[0].batch_id.startswith("b-")


def test_ga_tell_accepts_partial_records_for_one_batch() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=float(index),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for index, candidate in enumerate(candidates)
    ]

    first = engine.tell(records[:2])
    second = engine.tell(records[2:])

    assert first.trusted_count == 2
    assert second.trusted_count == 2
    assert engine.vnext_telemetry.candidates_full_evaluated == 4
    assert engine.best_candidate.candidate_id == candidates[-1].candidate_id


def test_ga_tell_rejects_duplicate_candidate_rung_record() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidate = engine.ask(1)[0]
    record = EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    engine.tell([record])

    with pytest.raises(FitnessError, match="already has"):
        engine.tell([record])


def test_ga_tell_rejects_explicit_batch_mismatch() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidate = engine.ask(1)[0]

    with pytest.raises(FitnessError, match="batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-wrong",
                    score=1.0,
                    confidence="trusted_full",
                    rung="full",
                    cost=1.0,
                )
            ]
        )


def test_ga_run_rejects_evaluator_that_omits_assigned_records() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)

    with pytest.raises(FitnessError, match="missing evaluation records"):
        engine.run(DroppingEvaluator(), policy=MultiFidelityPolicy.single_full(budget=4, batch_size=4))


def test_ga_run_resets_vnext_state_for_repeated_runs() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    policy = MultiFidelityPolicy.single_full(budget=8, batch_size=4)

    first = engine.run(SphereEvaluator(), policy=policy)
    second = engine.run(SphereEvaluator(), policy=policy)

    assert first.n_evaluations == 8
    assert second.n_evaluations == 8
    assert len(second.final_population) == 8
    assert second.telemetry.candidates_full_evaluated == 8
```

- [ ] **Step 2: Run GA tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: FAIL because candidates do not have populated batch IDs, duplicate records are accepted, and `run()` does not reset state or validate evaluator record coverage.

- [ ] **Step 3: Add GA imports and state reset helper**

In `evocore/ga.py`, add this import:

```python
from evocore.batches import CandidateBatch, batch_id_from_seed
```

In `GAEngine.__init__`, replace the direct vNext field initialization:

```python
        self._event_index = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._trusted_population_vnext: list[Candidate] = []
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
```

with:

```python
        self._reset_vnext_state()
```

Add this method before `_warn_if_large_int_gene_without_sigma()`:

```python
    def _reset_vnext_state(self) -> None:
        self._event_index = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._trusted_population_vnext: list[Candidate] = []
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
```

- [ ] **Step 4: Add batch IDs to GA candidates and register batches**

Change `_candidate_from_genes()` to accept `batch_id`:

```python
    def _candidate_from_genes(
        self,
        genes: list[float | int | bool],
        *,
        batch_id: str,
        origin: str,
        event_index: int,
        candidate_index: int,
        parents: Sequence[str] = (),
    ) -> Candidate:
        candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
        params = self.gene_space.params_for(genes)
        return Candidate(
            candidate_id=candidate_id,
            batch_id=batch_id,
            genes=list(genes),
            params=params,
            origin=origin,
            parents=parents,
            event_index=event_index,
        )
```

In `ask()`, define `batch_id` immediately after `event_index`:

```python
        batch_id = batch_id_from_seed(self.seed, event_index)
```

Pass `batch_id=batch_id` into every `_candidate_from_genes()` call.

After storing candidates in `_candidates_by_id`, register the batch:

```python
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        )
```

- [ ] **Step 5: Validate and store records through the GA batch ledger**

In `tell()`, after the unknown candidate check and before `candidate.apply_record(record)`, add:

```python
            batch = self._batches_by_id.get(candidate.batch_id)
            if batch is None:
                raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
            batch.accept_record(record)
```

- [ ] **Step 6: Add strict evaluator-output validation for GA run**

At the top of `evocore/ga.py`, add this import:

```python
from collections import Counter
```

Add this method before `run()`:

```python
    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        assigned_by_id = {candidate.candidate_id: candidate for candidate in assigned}
        counts = Counter(record.candidate_id for record in records)
        duplicate_ids = sorted(candidate_id for candidate_id, count in counts.items() if count > 1)
        if duplicate_ids:
            raise FitnessError(
                f"evaluator returned duplicate evaluation records for candidate_ids: "
                f"{duplicate_ids!r}"
            )
        returned_id_set = set(counts)
        missing = set(assigned_by_id) - returned_id_set
        unknown = returned_id_set - set(assigned_by_id)
        if missing:
            raise FitnessError(f"evaluator returned missing evaluation records: {sorted(missing)!r}")
        if unknown:
            raise FitnessError(f"evaluator returned unknown candidate_ids: {sorted(unknown)!r}")
        for record in records:
            candidate = assigned_by_id[record.candidate_id]
            if record.batch_id is not None and record.batch_id != candidate.batch_id:
                raise FitnessError(
                    f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                    f"candidate batch {candidate.batch_id!r}."
                )
```

In `run()`, call the reset helper after validating evaluator type:

```python
        self._reset_vnext_state()
```

In `run()`, replace:

```python
                records = list(evaluator.evaluate(assigned, rung))
                self.tell(records)
```

with:

```python
                records = list(evaluator.evaluate(assigned, rung))
                self._validate_evaluator_records(assigned, records)
                self.tell(records)
```

- [ ] **Step 7: Run GA tests and format**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py tests/unit/test_rng_reproducibility.py -v
python -m ruff format evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py
python -m ruff check evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py
```

Expected: PASS.

- [ ] **Step 8: Commit GA batch semantics**

```powershell
git add evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py
git commit -m "fix: add ga explicit batch tell semantics"
```

---

## Task 4: Add CMA Batch Tokens And Partial Completion

**Files:**
- Modify: `evocore/cmaes.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`

- [ ] **Step 1: Add failing CMA batch tests**

At the top of `tests/unit/test_cmaes_ask_tell_vnext.py`, add:

```python
import pytest
```

After the `from evocore import ...` line, add:

```python
from evocore.exceptions import FitnessError
```

Append these helper and tests to `tests/unit/test_cmaes_ask_tell_vnext.py`:

```python
def _trusted_record(candidate, score: float) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )


def test_cma_ask_assigns_one_batch_id_per_population() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)

    first = engine.ask()
    second = engine.ask()

    assert len({candidate.batch_id for candidate in first}) == 1
    assert len({candidate.batch_id for candidate in second}) == 1
    assert first[0].batch_id != second[0].batch_id
    assert first[0].batch_id.startswith("b-")


def test_cma_partial_trusted_tells_advance_when_batch_is_complete() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    first = engine.tell([_trusted_record(candidate, 1.0) for candidate in candidates[:2]])
    assert first.trusted_count == 2
    assert engine.generation == 0

    second = engine.tell([_trusted_record(candidate, 2.0) for candidate in candidates[2:]])
    assert second.trusted_count == 2
    assert engine.generation == 1


def test_cma_trusted_records_update_in_original_ask_order() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [_trusted_record(candidate, float(index)) for index, candidate in enumerate(candidates)]

    engine.tell([records[2], records[0]])
    assert engine.generation == 0
    engine.tell([records[3], records[1]])

    assert engine.generation == 1


def test_cma_rejects_trusted_record_after_batch_consumption() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [_trusted_record(candidate, 1.0) for candidate in candidates]

    engine.tell(records)

    with pytest.raises(FitnessError, match="already been consumed"):
        engine.tell([_trusted_record(candidates[0], 1.0)])


def test_cma_tell_rejects_explicit_batch_mismatch() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidate = engine.ask()[0]

    with pytest.raises(FitnessError, match="batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-wrong",
                    score=1.0,
                    confidence="trusted_full",
                    rung="full",
                    cost=1.0,
                )
            ]
        )
```

- [ ] **Step 2: Run CMA tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: FAIL because CMA candidates do not have populated batch IDs and partial trusted tells do not accumulate.

- [ ] **Step 3: Add CMA imports and batch state**

In `evocore/cmaes.py`, add this import:

```python
from evocore.batches import CandidateBatch, batch_id_from_seed
```

In `CMAESEngine.__init__`, add this field after `_candidates_by_id`:

```python
        self._batches_by_id: dict[str, CandidateBatch] = {}
```

- [ ] **Step 4: Register CMA ask batches**

In `CMAESEngine.ask()`, define `batch_id` immediately after `event_index`:

```python
        batch_id = batch_id_from_seed(self.seed, event_index)
```

When creating each candidate, pass `batch_id=batch_id`:

```python
            candidate = Candidate(
                candidate_id=candidate_id,
                batch_id=batch_id,
                genes=list(individual.genes),
                params=individual.params,
                origin="cma_sample",
                event_index=event_index,
            )
```

After the candidate loop and before incrementing `_event_index`, register the batch:

```python
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            continuous_samples_by_id={
                candidate.candidate_id: list(samples_continuous[index])
                for index, candidate in enumerate(candidates)
            },
        )
```

Keep `_pending_samples_by_id` populated in this task so existing private references remain harmless.

- [ ] **Step 5: Accumulate CMA trusted records by batch**

Replace `CMAESEngine.tell()` with:

```python
    def tell(self, records: Sequence[EvaluationRecord]) -> EngineStateSummary:
        """Update CMA state from trusted evaluation records."""
        trusted = partial = surrogate = rejected = 0
        touched_batches: set[str] = set()
        for record in records:
            candidate = self._candidates_by_id.get(record.candidate_id)
            if candidate is None:
                raise FitnessError(
                    f"tell() received unknown candidate_id: {record.candidate_id!r}"
                )
            batch = self._batches_by_id.get(candidate.batch_id)
            if batch is None:
                raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
            batch.accept_record(record, reject_consumed_trusted=True)
            candidate.apply_record(record)
            touched_batches.add(batch.batch_id)
            if record.confidence == "trusted_full":
                trusted += 1
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence in ("partial", "cached"):
                partial += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, rung=record.rung)

        for batch_id in touched_batches:
            batch = self._batches_by_id[batch_id]
            ordered_records = batch.ordered_trusted_full_records()
            if ordered_records is None or batch.consumed:
                continue
            samples = [
                batch.continuous_samples_by_id[record.candidate_id]
                for record in ordered_records
            ]
            fitnesses = [
                float(record.score) for record in ordered_records if record.score is not None
            ]
            self._ensure_state().tell(samples, fitnesses)
            batch.consumed = True

        return EngineStateSummary(
            accepted_count=len(records),
            trusted_count=trusted,
            partial_count=partial,
            surrogate_count=surrogate,
            rejected_count=rejected,
        )
```

- [ ] **Step 6: Run CMA tests and format**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py -v
python -m ruff format evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py
python -m ruff check evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py
```

Expected: PASS.

- [ ] **Step 7: Commit CMA batch semantics**

```powershell
git add evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py
git commit -m "fix: add cma partial batch tell semantics"
```

---

## Task 5: Document Batch Tokens

**Files:**
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/api.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update ask/tell docs**

Append this section to `docs/site/ask-tell-engines.md`:

```markdown
## Batch Tokens

Every candidate returned by one `ask()` call carries the same `batch_id`. The token lets external
evaluators, queues, and cached backtests return partial results safely through `tell()`.

`EvaluationRecord.batch_id` is optional. If supplied, EvoCore verifies that it matches the stored
candidate batch. If omitted, the engine infers the batch from `candidate_id`.

```python
candidates = engine.ask(16)
batch_id = candidates[0].batch_id

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=batch_id,
        score=score_candidate(candidate),
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )
    for candidate in candidates[:4]
]
engine.tell(records)
```

GA updates trusted candidates as records arrive. CMA waits until all candidates in a CMA batch have
trusted full records, then updates the distribution exactly once in the original ask order.
```

- [ ] **Step 2: Update API docs prose**

In `docs/site/api.md`, under `## vNext Expensive Optimization`, insert this paragraph before the autodoc blocks:

```markdown
`Candidate.batch_id` groups all candidates produced by one `ask()` call. `EvaluationRecord.batch_id`
is optional; when present, engines validate it against the candidate ledger before accepting the
record.
```

- [ ] **Step 3: Update changelog**

In `CHANGELOG.md`, under the `0.7.0` Added section, add this bullet:

```markdown
- Explicit vNext batch tokens for asynchronous partial `tell()` workflows.
```

- [ ] **Step 4: Build docs**

Run:

```powershell
$siteDir = Join-Path $env:TEMP ('evocore-mkdocs-site-' + [guid]::NewGuid().ToString('N')); python -m mkdocs build --strict --site-dir $siteDir
```

Expected: PASS. MkDocs may print the existing Material advisory and may note the intentionally unnav'd `ga-benchmark-parity.md` page.

- [ ] **Step 5: Commit docs**

```powershell
git add docs/site/ask-tell-engines.md docs/site/api.md CHANGELOG.md
git commit -m "docs: document vnext batch tokens"
```

---

## Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run formatting and lint checks**

Run:

```powershell
python -m ruff format --check
python -m ruff check
cargo fmt --check
```

Expected: PASS.

- [ ] **Step 2: Run Rust checks**

Run:

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 3: Rebuild extension and run Python tests**

Run:

```powershell
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 4: Run focused benchmark and example smoke tests**

Run:

```powershell
python -m pytest tests/benchmarks/bench_vnext_multifidelity.py -v
python examples/vnext_budgeted_ga.py
```

Expected: pytest PASS, and the example prints `full_evals=32`.

- [ ] **Step 5: Run docs build**

Run:

```powershell
$siteDir = Join-Path $env:TEMP ('evocore-mkdocs-site-' + [guid]::NewGuid().ToString('N')); python -m mkdocs build --strict --site-dir $siteDir
```

Expected: PASS.

- [ ] **Step 6: Check diff hygiene**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors. `site/` may remain as an unrelated untracked generated directory and must not be staged.

- [ ] **Step 7: Commit verification fixes**

If any verification command required code or docs fixes, commit only those task-related files:

```powershell
git add evocore tests docs CHANGELOG.md
git commit -m "test: verify vnext batch tokens"
```

If no files changed during verification, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage: Tasks cover public `batch_id` fields, optional record batch IDs, private ledger, GA async partial tells, strict GA `run()` evaluator validation, repeated GA run reset, CMA partial-batch accumulation, once-only CMA updates, final trusted policy validation, docs, changelog, and final verification.
- Scope control: No distributed queue, persistence, retry, public `Batch` object, or CMA rejected-sample handling is included.
- Type consistency: Public names are `Candidate.batch_id`, `EvaluationRecord.batch_id`, `CandidateBatch`, and `batch_id_from_seed`.
- Test strategy: Every behavior change starts with failing tests, then minimal implementation, then focused verification.
