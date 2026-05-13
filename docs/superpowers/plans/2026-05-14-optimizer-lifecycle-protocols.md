# Optimizer Lifecycle Protocols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize EvoCore's clean-break, single-objective ask/tell lifecycle around structural `Optimizer` and `Evaluator` protocols.

**Architecture:** Put shared lifecycle records in `evocore.evaluation`, structural protocols in a new `evocore.protocols` module, and keep engine-specific ledgers private to `GAEngine` and `CMAESEngine`. `GAEngine` and `CMAESEngine` will both expose `direction`, return `TellResult` from `tell(...)`, and expose `state_summary()` without forcing users to subclass EvoCore base classes.

**Tech Stack:** Python dataclasses, `typing.Protocol`, PyO3-backed engine helpers, pytest, Ruff, Cargo, maturin.

---

## Scope Check

The approved spec covers one subsystem: the public ask/tell lifecycle contract. Keep result/history export redesign, problem/gene declaration redesign, constraints, multi-objective optimization, and the future `optimize(...)` wrapper out of this implementation.

## File Structure Map

- Create `evocore/protocols.py`: structural `Optimizer` and `Evaluator` protocols.
- Modify `evocore/evaluation.py`: lifecycle literals, `EvaluationContext`, `TellResult`, `EngineStateSummary`, direction helpers, metadata support.
- Modify `evocore/ga.py`: import shared result types, add direction handling, return `TellResult`, add `state_summary()`, pass `EvaluationContext` to evaluators.
- Modify `evocore/cmaes.py`: import shared result types, add direction handling, track best candidate for ask/tell, return `TellResult`, add `state_summary()`.
- Modify `evocore/__init__.py`: export protocols and lifecycle records from their new homes.
- Modify `tests/unit/test_vnext_evaluation.py`: record, context, tell result, and direction helper tests.
- Create `tests/unit/test_protocols.py`: runtime structural protocol tests.
- Modify `tests/unit/test_ga_ask_tell_vnext.py`: GA protocol, direction, no-op tell, state summary, and context tests.
- Modify `tests/unit/test_ga_engine.py`: existing GA run and multi-run evaluator fixtures using `EvaluationContext`.
- Modify `tests/unit/test_cmaes_ask_tell_vnext.py`: CMA protocol, direction, no-op tell, and state summary tests.
- Modify `tests/vnext_helpers.py`: structural evaluator helper using `EvaluationContext`.
- Modify `tests/benchmarks/bench_vnext_multifidelity.py`: benchmark evaluator fixture using `EvaluationContext`.
- Modify `README.md`, `examples/sphere_optimization.py`, `examples/mixed_gene_space.py`, `examples/onemax_binary.py`, `examples/vnext_budgeted_ga.py`, `docs/site/ga.md`, `docs/site/parallelism.md`, `docs/site/ask-tell-engines.md`, `docs/site/quickstart.md`, `docs/site/api.md`, and `CHANGELOG.md`: public docs and examples for the stabilized lifecycle.

## Preconditions

- [ ] **Step 1: Confirm branch and worktree**

Run:

```powershell
git status --short --branch
```

Expected: branch is `feature/general-optimizer-framework`; inspect any uncommitted files before editing and do not overwrite unrelated user work.

---

### Task 1: Add Lifecycle Contract Tests

**Files:**
- Modify: `tests/unit/test_vnext_evaluation.py`
- Create: `tests/unit/test_protocols.py`

- [ ] **Step 1: Extend evaluation imports in `tests/unit/test_vnext_evaluation.py`**

Change the existing import block to include the new lifecycle records and helper:

```python
from evocore.evaluation import (
    Candidate,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    OptimizationTelemetry,
    Rung,
    TellResult,
    score_for_direction,
)
```

- [ ] **Step 2: Add evaluation record and direction tests**

Append these tests to `tests/unit/test_vnext_evaluation.py`:

```python
def test_evaluation_record_preserves_metadata() -> None:
    record = EvaluationRecord(
        candidate_id="c-1",
        score=1.25,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
        metrics={"loss": 0.2},
        metadata={"source": "unit"},
        batch_id="b-1",
    )

    assert record.metadata["source"] == "unit"
    assert record.metrics["loss"] == pytest.approx(0.2)


def test_evaluation_context_carries_batch_rung_direction_and_budget() -> None:
    rung = Rung("cheap", budget=0.25, promote_fraction=0.5, confidence="partial")

    context = EvaluationContext(
        rung=rung,
        batch_id="b-1",
        event_index=3,
        direction="minimize",
        budget=0.25,
        metadata={"phase": "screen"},
    )

    assert context.rung is rung
    assert context.batch_id == "b-1"
    assert context.event_index == 3
    assert context.direction == "minimize"
    assert context.budget == pytest.approx(0.25)
    assert context.metadata["phase"] == "screen"


def test_tell_result_and_state_summary_have_stable_fields() -> None:
    telemetry = OptimizationTelemetry()
    tell_result = TellResult(
        accepted_count=3,
        trusted_count=1,
        partial_count=1,
        surrogate_count=0,
        cached_count=0,
        rejected_count=1,
        best_candidate_id="c-2",
        best_score=2.5,
        consumed_batch_ids=("b-1",),
        pending_batch_ids=("b-2",),
        telemetry=telemetry,
    )
    state = EngineStateSummary(
        best_candidate_id="c-2",
        best_score=2.5,
        event_index=4,
        pending_batch_ids=("b-2",),
        trusted_count=5,
        telemetry=telemetry,
    )

    assert tell_result.accepted_count == 3
    assert tell_result.cached_count == 0
    assert tell_result.consumed_batch_ids == ("b-1",)
    assert state.best_candidate_id == "c-2"
    assert state.pending_batch_ids == ("b-2",)


def test_candidate_best_observed_score_honors_direction() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[1.0], batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=10.0,
            confidence="partial",
            rung="cheap",
            cost=0.1,
        )
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=2.0,
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
    )

    assert candidate.best_observed_score("maximize") == pytest.approx(10.0)
    assert candidate.best_observed_score("minimize") == pytest.approx(2.0)
    assert candidate.comparison_score("maximize") == pytest.approx(10.0)
    assert candidate.comparison_score("minimize") == pytest.approx(-2.0)


def test_score_for_direction_rejects_invalid_direction() -> None:
    assert score_for_direction(3.0, "maximize") == pytest.approx(3.0)
    assert score_for_direction(3.0, "minimize") == pytest.approx(-3.0)

    with pytest.raises(ConfigurationError, match="direction"):
        score_for_direction(3.0, "lowest")  # type: ignore[arg-type]
```

- [ ] **Step 3: Create `tests/unit/test_protocols.py`**

Create the file with these contents:

```python
from evocore import (
    CMAESEngine,
    EvaluationContext,
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneDef,
    GeneSpace,
    Optimizer,
    Rung,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("period", "int", 2, 20),
        ]
    )


class StructuralSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


def test_ga_and_cma_satisfy_optimizer_protocol_at_runtime() -> None:
    assert isinstance(GAEngine(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(CMAESEngine(_space(), population_size=4, seed=1), Optimizer)


def test_structural_evaluator_satisfies_evaluator_protocol_at_runtime() -> None:
    evaluator = StructuralSphereEvaluator()
    rung = Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")
    context = EvaluationContext(
        rung=rung,
        batch_id="b-1",
        event_index=0,
        direction="minimize",
        budget=1.0,
    )

    assert isinstance(evaluator, Evaluator)
    assert evaluator.evaluate([], context) == []
```

- [ ] **Step 4: Run the new contract tests and verify they fail for missing API**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py -q
```

Expected: FAIL with import errors for `EngineStateSummary`, `EvaluationContext`, `TellResult`, `score_for_direction`, `Optimizer`, or `Evaluator`.

- [ ] **Step 5: Commit the failing tests**

Run:

```powershell
git add tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py
git commit -m "test: define optimizer lifecycle protocol contract"
```

Expected: commit succeeds with only the two test files staged.

---

### Task 2: Implement Shared Lifecycle Records And Protocols

**Files:**
- Create: `evocore/protocols.py`
- Modify: `evocore/evaluation.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_vnext_evaluation.py`
- Modify: `tests/unit/test_protocols.py`

- [ ] **Step 1: Add shared lifecycle types in `evocore/evaluation.py`**

In `evocore/evaluation.py`, add `Direction`, `score_for_direction`, and the new dataclasses near the existing literals:

```python
Direction = Literal["maximize", "minimize"]


def score_for_direction(score: float, direction: Direction) -> float:
    """Return a comparison score where larger is always better."""
    if direction == "maximize":
        return float(score)
    if direction == "minimize":
        return -float(score)
    raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
```

Add these dataclasses after `Rung`:

```python
@dataclass(frozen=True)
class EvaluationContext:
    """Describe the evaluator call context for one ask/tell batch."""

    rung: Rung | None
    batch_id: str
    event_index: int
    direction: Direction
    budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ConfigurationError("EvaluationContext batch_id must be non-empty.")
        if int(self.event_index) < 0:
            raise ConfigurationError("EvaluationContext event_index must be >= 0.")
        if self.direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        if self.budget is not None and (
            not math.isfinite(float(self.budget)) or float(self.budget) <= 0.0
        ):
            raise ConfigurationError("EvaluationContext budget must be finite and > 0.")


@dataclass(frozen=True)
class TellResult:
    """Summarize one optimizer tell() update."""

    accepted_count: int
    trusted_count: int
    partial_count: int
    surrogate_count: int
    cached_count: int
    rejected_count: int
    best_candidate_id: str | None = None
    best_score: float | None = None
    consumed_batch_ids: tuple[str, ...] = ()
    pending_batch_ids: tuple[str, ...] = ()
    telemetry: OptimizationTelemetry | None = None


@dataclass(frozen=True)
class EngineStateSummary:
    """Expose a stable read-only optimizer state summary."""

    best_candidate_id: str | None
    best_score: float | None
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    telemetry: OptimizationTelemetry
```

- [ ] **Step 2: Add metadata to score and record objects**

Update `CandidateScore` and `EvaluationRecord` in `evocore/evaluation.py`:

```python
@dataclass(frozen=True)
class CandidateScore:
    """Store one score observation for one candidate and rung."""

    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

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
    metadata: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None
```

- [ ] **Step 3: Update candidate record application and score helpers**

Replace `Candidate.apply_record`, `Candidate.best_observed_score`, and add `Candidate.comparison_score`:

```python
    def apply_record(self, record: EvaluationRecord) -> None:
        """Apply an evaluation record to this candidate."""
        if record.candidate_id != self.candidate_id:
            raise FitnessError(
                f"EvaluationRecord candidate_id {record.candidate_id!r} does not match "
                f"candidate {self.candidate_id!r}."
            )
        if record.batch_id is not None and self.batch_id and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
            )
        self.rung = record.rung
        self.confidence = record.confidence
        self.cost += record.cost
        self.scores[record.rung] = CandidateScore(
            score=record.score,
            confidence=record.confidence,
            rung=record.rung,
            cost=record.cost,
            metrics=dict(record.metrics),
            metadata=dict(record.metadata),
        )
        self.metadata["metrics"] = dict(record.metrics)
        self.metadata["record_metadata"] = dict(record.metadata)
        if record.confidence == "trusted_full":
            self.status = "trusted"
        elif record.confidence == "rejected":
            self.status = "eliminated"
        elif record.confidence in ("partial", "cached"):
            self.status = "racing"
        else:
            self.status = "screened"

    def best_observed_score(self, direction: Direction = "maximize") -> float:
        """Return the best raw finite score observed for this candidate."""
        values = [score.score for score in self.scores.values() if score.score is not None]
        if not values:
            return float("inf") if direction == "minimize" else float("-inf")
        if direction == "minimize":
            return min(float(value) for value in values)
        if direction == "maximize":
            return max(float(value) for value in values)
        raise ConfigurationError("direction must be 'maximize' or 'minimize'.")

    def comparison_score(self, direction: Direction = "maximize") -> float:
        """Return the best observed score normalized so larger is better."""
        best = self.best_observed_score(direction)
        if not math.isfinite(best):
            return best if direction == "maximize" else -best
        return score_for_direction(best, direction)
```

- [ ] **Step 4: Remove the old evaluator base class from `evocore/evaluation.py`**

Delete this class from the bottom of `evocore/evaluation.py`:

```python
class Evaluator:
    """Base class for vNext evaluators."""

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        rung: Rung,
    ) -> Sequence[EvaluationRecord]:
        """Evaluate candidates for a rung."""
        raise NotImplementedError("Evaluator.evaluate must be implemented by subclasses.")
```

- [ ] **Step 5: Create `evocore/protocols.py`**

Create the file with these contents:

```python
"""Structural protocols for EvoCore optimizer lifecycle APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from evocore.evaluation import (
    Candidate,
    Direction,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    TellResult,
)


@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol implemented by ask/tell optimizers."""

    direction: Direction

    def ask(self, n: int | None = None) -> Sequence[Candidate]:
        """Return candidates for external evaluation."""
        ...

    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        """Apply evaluation records and return a summary of accepted records."""
        ...

    def state_summary(self) -> EngineStateSummary:
        """Return a read-only optimizer state summary."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Structural protocol implemented by objective evaluators."""

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        context: EvaluationContext,
    ) -> Sequence[EvaluationRecord]:
        """Evaluate candidates in the supplied context."""
        ...
```

- [ ] **Step 6: Update top-level exports in `evocore/__init__.py`**

Change the evaluation import block:

```python
from evocore.evaluation import (
    Candidate,
    CandidateScore,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    OptimizationTelemetry,
    Rung,
    TellResult,
)
```

Add the protocol import:

```python
from evocore.protocols import Evaluator, Optimizer
```

Change the GA import block from:

```python
from evocore.ga import EngineStateSummary, GAEngine, MultiRunResult, RunResult
```

to:

```python
from evocore.ga import GAEngine, MultiRunResult, RunResult
```

Add these names to `__all__` in the CamelCase section:

```python
    "EngineStateSummary",
    "EvaluationContext",
    "Evaluator",
    "Optimizer",
    "TellResult",
```

- [ ] **Step 7: Run shared lifecycle tests**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py -q
```

Expected: tests in `test_vnext_evaluation.py` pass; protocol runtime tests may still fail until engines implement `direction` and `state_summary()`.

- [ ] **Step 8: Commit shared lifecycle records and protocols**

Run:

```powershell
git add evocore/evaluation.py evocore/protocols.py evocore/__init__.py tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py
git commit -m "feat: add optimizer lifecycle protocols"
```

Expected: commit succeeds with only shared lifecycle files and tests staged.

---

### Task 3: Make GAEngine Conform To The Lifecycle Protocol

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `tests/unit/test_protocols.py`

- [ ] **Step 1: Update imports and evaluator test shape in `tests/unit/test_ga_ask_tell_vnext.py`**

Change the import block to include `EvaluationContext` and remove `Evaluator` inheritance from local evaluator classes:

```python
from evocore import (
    EvaluationContext,
    EvaluationRecord,
    GAEngine,
    GeneDef,
    GeneSpace,
    MultiFidelityPolicy,
)
```

Replace the evaluator classes with context-based structural evaluators:

```python
class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


class DroppingEvaluator:
    def evaluate(self, candidates, context):
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=1.0,
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates[:-1]
        ]
```

- [ ] **Step 2: Add GA lifecycle behavior tests**

Append these tests to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
def test_ga_tell_empty_records_returns_noop_tell_result() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)

    result = engine.tell([])

    assert result.accepted_count == 0
    assert result.trusted_count == 0
    assert result.pending_batch_ids == ()


def test_ga_tell_rejects_unknown_explicit_batch_id() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidate = engine.ask(1)[0]

    with pytest.raises(FitnessError, match="unknown batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-missing",
                    score=1.0,
                    confidence="trusted_full",
                    rung="full",
                    cost=1.0,
                )
            ]
        )


def test_ga_state_summary_reports_best_and_pending_batches() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(2)

    before = engine.state_summary()

    assert before.best_candidate_id is None
    assert before.pending_batch_ids == (candidates[0].batch_id,)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=1.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            )
        ]
    )

    after = engine.state_summary()

    assert after.best_candidate_id == candidates[0].candidate_id
    assert after.best_score == pytest.approx(1.0)
    assert after.trusted_count == 1


def test_ga_minimize_direction_selects_lowest_trusted_score() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123, direction="minimize")
    candidates = engine.ask(2)

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=10.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            ),
            EvaluationRecord(
                candidate_id=candidates[1].candidate_id,
                batch_id=candidates[1].batch_id,
                score=2.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            ),
        ]
    )

    assert engine.best_candidate is not None
    assert engine.best_candidate.candidate_id == candidates[1].candidate_id
    assert result.best_score == pytest.approx(2.0)
```

- [ ] **Step 3: Run GA tests and verify they fail for missing behavior**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_protocols.py -q
```

Expected: FAIL for missing `direction`, `state_summary()`, `EvaluationContext` call shape, or `TellResult` fields.

- [ ] **Step 4: Update GA imports and remove local `EngineStateSummary`**

In `evocore/ga.py`, replace the evaluation import:

```python
from evocore.evaluation import (
    Candidate,
    Direction,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    OptimizationTelemetry,
    TellResult,
    score_for_direction,
)
```

Add protocol import:

```python
from evocore.protocols import Evaluator
```

Delete the local `EngineStateSummary` dataclass from `evocore/ga.py`.

- [ ] **Step 5: Add `direction` to `GAEngine.__init__`**

Add the constructor parameter after `seed`:

```python
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
```

Add validation after `self.seed = int(seed)`:

```python
        if direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        self.direction = direction
```

Keep `self.max_evaluations = max_evaluations` after this block.

- [ ] **Step 6: Propagate GA direction through copied engines**

In `_copy_with_seed`, add `direction=self.direction` after the copied seed:

```python
            seed=int(seed),
            direction=self.direction,
            max_evaluations=self.max_evaluations,
```

- [ ] **Step 7: Add GA summary helpers**

Add these private helpers after `_reset_vnext_state`:

```python
    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if len(batch.records_by_key) < len(batch.candidate_ids)
        )

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return (
            self.best_candidate.candidate_id,
            self.best_candidate.best_observed_score(self.direction),
        )

    def state_summary(self) -> EngineStateSummary:
        """Return a stable read-only vNext state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return EngineStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=len(self._trusted_population_vnext),
            telemetry=self.vnext_telemetry,
        )
```

- [ ] **Step 8: Add GA evaluator context construction**

Add this helper before `_validate_evaluator_records`:

```python
    def _evaluation_context(
        self,
        assigned: Sequence[Candidate],
        rung,
    ) -> EvaluationContext:
        batch_ids = {candidate.batch_id for candidate in assigned}
        if len(batch_ids) != 1:
            raise FitnessError(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            )
        batch_id = next(iter(batch_ids))
        event_index = assigned[0].event_index if assigned else self._event_index
        return EvaluationContext(
            rung=rung,
            batch_id=batch_id,
            event_index=event_index,
            direction=self.direction,
            budget=rung.budget,
        )
```

- [ ] **Step 9: Replace GA `tell(...)` implementation**

Replace `GAEngine.tell` with:

```python
    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        """Update GA state from vNext evaluation records."""
        trusted = partial = surrogate = cached = rejected = 0
        touched_batch_ids: set[str] = set()
        for record in records:
            candidate = self._candidates_by_id.get(record.candidate_id)
            if candidate is None:
                raise FitnessError(
                    f"tell() received unknown candidate_id: {record.candidate_id!r}"
                )
            if record.batch_id is not None and record.batch_id not in self._batches_by_id:
                raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
            batch = self._batches_by_id.get(candidate.batch_id)
            if batch is None:
                raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
            batch.accept_record(record)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            if record.confidence == "trusted_full":
                trusted += 1
                self._trusted_population_vnext.append(candidate)
                if (
                    self.best_candidate is None
                    or candidate.comparison_score(self.direction)
                    > self.best_candidate.comparison_score(self.direction)
                ):
                    self.best_candidate = candidate
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "cached":
                cached += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "partial":
                partial += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, rung=record.rung)

        self._trusted_population_vnext.sort(
            key=lambda candidate: candidate.comparison_score(self.direction),
            reverse=True,
        )
        self._trusted_population_vnext = self._trusted_population_vnext[: self.population_size]
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        consumed_batch_ids = tuple(
            batch_id
            for batch_id in touched_batch_ids
            if len(self._batches_by_id[batch_id].records_by_key)
            >= len(self._batches_by_id[batch_id].candidate_ids)
        )
        return TellResult(
            accepted_count=len(records),
            trusted_count=trusted,
            partial_count=partial,
            surrogate_count=surrogate,
            cached_count=cached,
            rejected_count=rejected,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=consumed_batch_ids,
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
        )
```

- [ ] **Step 10: Use direction-aware scores in GA ask and run**

In `GAEngine.ask`, replace the trusted individual construction fitness assignment:

```python
                    fitness=candidate.comparison_score(self.direction),
```

Replace the `fitnesses` line:

```python
            fitnesses = [individual.fitness or float("-inf") for individual in trusted_individuals]
```

Keep it unchanged after the assignment above; it now receives normalized comparison scores.

In `GAEngine.run`, replace best individual construction:

```python
        best = Individual(
            list(self.best_candidate.genes),
            fitness=self.best_candidate.best_observed_score(self.direction),
            fitness_valid=True,
            metadata={
                "params": self.best_candidate.params,
                "candidate_id": self.best_candidate.candidate_id,
            },
        )
```

Replace final population individual fitness:

```python
                    fitness=candidate.best_observed_score(self.direction),
```

- [ ] **Step 11: Pass `EvaluationContext` in `GAEngine.run`**

Replace the evaluator instance check:

```python
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "GAEngine.run requires an evaluator with evaluate(candidates, context)."
            )
```

Replace the evaluator call inside the rung loop:

```python
                context = self._evaluation_context(assigned, rung)
                records = list(evaluator.evaluate(assigned, context))
```

- [ ] **Step 12: Run GA lifecycle tests**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_protocols.py -q
```

Expected: PASS for GA tests and GA protocol checks. CMA protocol checks may still fail until Task 4.

- [ ] **Step 13: Commit GA lifecycle conformance**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_protocols.py
git commit -m "feat: align ga engine with lifecycle protocol"
```

Expected: commit succeeds with only GA implementation and tests staged.

---

### Task 4: Make CMAESEngine Conform To The Lifecycle Protocol

**Files:**
- Modify: `evocore/cmaes.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `tests/unit/test_protocols.py`

- [ ] **Step 1: Add CMA lifecycle behavior tests**

Append these tests to `tests/unit/test_cmaes_ask_tell_vnext.py`:

```python
def test_cma_tell_empty_records_returns_noop_tell_result() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)

    result = engine.tell([])

    assert result.accepted_count == 0
    assert result.trusted_count == 0
    assert result.pending_batch_ids == ()


def test_cma_tell_rejects_unknown_explicit_batch_id() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidate = engine.ask()[0]

    with pytest.raises(FitnessError, match="unknown batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-missing",
                    score=1.0,
                    confidence="trusted_full",
                    rung="full",
                    cost=1.0,
                )
            ]
        )


def test_cma_state_summary_reports_best_and_pending_batches() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    before = engine.state_summary()

    assert before.best_candidate_id is None
    assert before.pending_batch_ids == (candidates[0].batch_id,)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=3.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            )
        ]
    )

    after = engine.state_summary()

    assert after.best_candidate_id == candidates[0].candidate_id
    assert after.best_score == pytest.approx(3.0)
    assert after.trusted_count == 1


def test_cma_minimize_direction_tracks_lowest_trusted_score() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7, direction="minimize")
    candidates = engine.ask()

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=10.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            ),
            EvaluationRecord(
                candidate_id=candidates[1].candidate_id,
                batch_id=candidates[1].batch_id,
                score=2.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            ),
        ]
    )

    assert engine.best_candidate is not None
    assert engine.best_candidate.candidate_id == candidates[1].candidate_id
    assert result.best_score == pytest.approx(2.0)
```

- [ ] **Step 2: Run CMA tests and verify they fail for missing behavior**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_protocols.py -q
```

Expected: FAIL for missing `direction`, `state_summary()`, or `TellResult` fields on CMA.

- [ ] **Step 3: Update CMA imports**

In `evocore/cmaes.py`, replace the evaluation and GA imports:

```python
from evocore.evaluation import (
    Candidate,
    Direction,
    EngineStateSummary,
    EvaluationRecord,
    OptimizationTelemetry,
    TellResult,
    score_for_direction,
)
from evocore.ga import RunResult
```

- [ ] **Step 4: Add `direction` and best candidate state to `CMAESEngine.__init__`**

Add constructor parameter after `seed`:

```python
        seed: int = 0,
        direction: Direction = "maximize",
        track_diversity: bool = False,
```

Add validation after `self.seed = int(seed)`:

```python
        if direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        self.direction = direction
```

Add after `self.vnext_telemetry = OptimizationTelemetry()`:

```python
        self.best_candidate: Candidate | None = None
```

- [ ] **Step 5: Add CMA summary helpers**

Add these methods after `generation`:

```python
    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if not batch.consumed
            and len(batch.records_by_key) < len(batch.candidate_ids)
        )

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return (
            self.best_candidate.candidate_id,
            self.best_candidate.best_observed_score(self.direction),
        )

    def _trusted_count(self) -> int:
        return sum(
            1
            for candidate in self._candidates_by_id.values()
            if any(score.confidence == "trusted_full" for score in candidate.scores.values())
        )

    def state_summary(self) -> EngineStateSummary:
        """Return a stable read-only vNext state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return EngineStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            telemetry=self.vnext_telemetry,
        )
```

- [ ] **Step 6: Replace CMA `tell(...)` implementation**

Replace `CMAESEngine.tell` with:

```python
    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        """Update CMA state from trusted evaluation records."""
        trusted_records: list[EvaluationRecord] = []
        partial = surrogate = cached = rejected = 0
        touched_batch_ids: set[str] = set()
        consumed_batch_ids: set[str] = set()
        for record in records:
            candidate = self._candidates_by_id.get(record.candidate_id)
            if candidate is None:
                raise FitnessError(
                    f"tell() received unknown candidate_id: {record.candidate_id!r}"
                )
            if record.batch_id is not None and record.batch_id not in self._batches_by_id:
                raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
            batch = self._batches_by_id.get(candidate.batch_id)
            if batch is None:
                raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
            batch.accept_record(record, reject_consumed_trusted=True)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            if record.confidence == "trusted_full":
                trusted_records.append(record)
                if (
                    self.best_candidate is None
                    or candidate.comparison_score(self.direction)
                    > self.best_candidate.comparison_score(self.direction)
                ):
                    self.best_candidate = candidate
                self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "cached":
                cached += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "partial":
                partial += 1
                self.vnext_telemetry.record_partial(1, rung=record.rung, cost=record.cost)
            elif record.confidence == "surrogate":
                surrogate += 1
                self.vnext_telemetry.record_screened(1)
            else:
                rejected += 1
                self.vnext_telemetry.record_eliminated(1, rung=record.rung)

        for batch_id in touched_batch_ids:
            batch = self._batches_by_id[batch_id]
            ordered_records = batch.ordered_trusted_full_records()
            if ordered_records is None or batch.consumed:
                continue
            samples = []
            fitnesses = []
            for record in ordered_records:
                sample = batch.continuous_samples_by_id.get(record.candidate_id)
                if sample is None:
                    raise FitnessError(
                        f"missing continuous sample for candidate_id {record.candidate_id!r}."
                    )
                samples.append(sample)
                if record.score is None:
                    raise FitnessError(
                        f"trusted_full record for candidate_id {record.candidate_id!r} is missing score."
                    )
                fitnesses.append(score_for_direction(float(record.score), self.direction))
            self._ensure_state().tell(samples, fitnesses)
            batch.consumed = True
            consumed_batch_ids.add(batch.batch_id)

        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return TellResult(
            accepted_count=len(records),
            trusted_count=len(trusted_records),
            partial_count=partial,
            surrogate_count=surrogate,
            cached_count=cached,
            rejected_count=rejected,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=tuple(sorted(consumed_batch_ids)),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
        )
```

- [ ] **Step 7: Run CMA lifecycle tests**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_protocols.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit CMA lifecycle conformance**

Run:

```powershell
git add evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_protocols.py
git commit -m "feat: align cmaes engine with lifecycle protocol"
```

Expected: commit succeeds with only CMA implementation and tests staged.

---

### Task 5: Refresh Helpers, Docs, API Reference, And Changelog

**Files:**
- Modify: `tests/vnext_helpers.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/benchmarks/bench_vnext_multifidelity.py`
- Modify: `README.md`
- Modify: `examples/sphere_optimization.py`
- Modify: `examples/mixed_gene_space.py`
- Modify: `examples/onemax_binary.py`
- Modify: `examples/vnext_budgeted_ga.py`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/parallelism.md`
- Modify: `docs/site/quickstart.md`
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/api.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `tests/vnext_helpers.py` to use structural evaluators**

Replace the file contents with:

```python
from evocore import EvaluationContext, EvaluationRecord, MultiFidelityPolicy, Rung
from evocore.individual import Individual


class IndividualEvaluator:
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        records = []
        for candidate in candidates:
            individual = Individual(
                list(candidate.genes),
                metadata={
                    "params": candidate.params,
                    "candidate_id": candidate.candidate_id,
                },
            )
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(self.fn(individual)),
                    confidence=context.rung.confidence,
                    rung=context.rung.name,
                    cost=context.rung.budget,
                )
            )
        return records


def full_policy(budget: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        full_evaluation_budget=budget,
        batch_size=batch_size,
    )
```

- [ ] **Step 2: Update GA engine test evaluators**

In `tests/unit/test_ga_engine.py`, add `EvaluationContext` to the top-level `from evocore import (...)` block and remove `Evaluator` from that block.

Replace `CallableEvaluator` with:

```python
class CallableEvaluator:
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(self.fn(candidate.genes)),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]
```

Replace `ModuleSphereEvaluator` with:

```python
class ModuleSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(v) ** 2 for v in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]
```

- [ ] **Step 3: Update the multifidelity benchmark evaluator**

In `tests/benchmarks/bench_vnext_multifidelity.py`, replace the import block with:

```python
from evocore import (
    EvaluationContext,
    EvaluationRecord,
    GAEngine,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)
```

Replace `DeceptiveSphere` with:

```python
class DeceptiveSphere:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        records = []
        for candidate in candidates:
            true_score = -sum(float(value) ** 2 for value in candidate.genes)
            cheap_score = true_score + (0.1 if candidate.candidate_id.endswith("0") else 0.0)
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=cheap_score if context.rung.name == "cheap" else true_score,
                    confidence=context.rung.confidence,
                    rung=context.rung.name,
                    cost=context.rung.budget,
                )
            )
        return records
```

- [ ] **Step 4: Update README, examples, and GA docs evaluator signatures**

In `README.md`, `examples/sphere_optimization.py`, `examples/mixed_gene_space.py`, `examples/onemax_binary.py`, `examples/vnext_budgeted_ga.py`, and `docs/site/ga.md`, remove `Evaluator` from import lists and add `EvaluationContext`.

For sphere-style examples, use this evaluator shape:

```python
class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]
```

For mixed-gene examples, keep the existing score expression and use this record shape:

```python
EvaluationRecord(
    candidate_id=candidate.candidate_id,
    batch_id=candidate.batch_id,
    score=score,
    confidence=context.rung.confidence,
    rung=context.rung.name,
    cost=context.rung.budget,
)
```

For `examples/vnext_budgeted_ga.py`, keep the cheap/full branch and replace bare `rung` references with `context.rung`.

- [ ] **Step 5: Update parallelism docs**

In `docs/site/parallelism.md`, replace:

```markdown
`GAEngine.run()` calls your `Evaluator.evaluate(candidates, rung)` method for each scheduled
```

with:

```markdown
`GAEngine.run()` calls your `Evaluator.evaluate(candidates, context)` method for each scheduled
```

- [ ] **Step 6: Update the quickstart**

Replace `docs/site/quickstart.md` with:

~~~markdown
# Quickstart

```python
from evocore import EvaluationContext, EvaluationRecord, GAEngine, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


engine = GAEngine(
    GeneSpace.uniform(-5.0, 5.0, 10),
    population_size=100,
    generations=100,
    seed=42,
    direction="maximize",
)
result = engine.run(SphereEvaluator())

print(result.best_fitness)
print(result.best_individual.genes)
```
~~~

- [ ] **Step 7: Update ask/tell docs**

Replace `docs/site/ask-tell-engines.md` with:

~~~markdown
# Ask/Tell Engines

EvoCore optimizers expose a structural ask/tell lifecycle. An optimizer does not need
to inherit from a base class; it satisfies `Optimizer` when it exposes `direction`,
`ask(...)`, `tell(...)`, and `state_summary()`.

`ask()` returns `Candidate` objects with stable `candidate_id` values, decoded genes,
optional params, and one shared `batch_id` for the ask event. `tell()` accepts only
`EvaluationRecord` values and returns `TellResult`.

Evaluators satisfy the structural `Evaluator` protocol by implementing:

```python
def evaluate(candidates, context):
    return []
```

`EvaluationContext` carries the rung, batch ID, event index, direction, budget, and
metadata for the evaluator call.

`tell()` is asynchronous-friendly: callers may report any subset of a batch, in any
order, as long as each candidate/rung pair is reported at most once. `tell([])` is a
valid no-op for queue polling integrations.

Confidence values are explicit:

- `trusted_full` updates optimizer state by default.
- `partial` and `surrogate` can inform scheduling and telemetry.
- `cached` records are tracked separately so policies can decide whether to trust them.
- `rejected` records may omit score.

Raw user scores are preserved. Optimizers use `direction="maximize"` or
`direction="minimize"` to compare candidates without rewriting the score stored in
`EvaluationRecord`.

Invalid records raise `FitnessError`: unknown candidates, unknown explicit batch IDs,
batch mismatches, duplicate candidate/rung records, and non-finite non-rejected scores
are rejected.
~~~

- [ ] **Step 8: Update API reference**

In `docs/site/api.md`, replace the vNext section with:

```markdown
## Optimizer Lifecycle

`Optimizer` and `Evaluator` are structural protocols. Engines and evaluators conform by
shape, without subclassing.

::: evocore.protocols.Optimizer

::: evocore.protocols.Evaluator

::: evocore.evaluation.Candidate

::: evocore.evaluation.EvaluationRecord

::: evocore.evaluation.EvaluationContext

::: evocore.evaluation.TellResult

::: evocore.evaluation.EngineStateSummary

::: evocore.evaluation.Rung

::: evocore.evaluation.OptimizationTelemetry

::: evocore.policies.MultiFidelityPolicy

::: evocore.scheduler.EvaluationScheduler

::: evocore.advisors.InverseDistanceSurrogateAdvisor

::: evocore.mixed_cma.IntegerMargin

::: evocore.mixed_cma.CategoricalState
```

- [ ] **Step 9: Update changelog**

In `CHANGELOG.md`, under `[Unreleased]` `### Added`, add:

```markdown
- Structural `Optimizer` and `Evaluator` protocols for the clean-break ask/tell
  lifecycle.
- `EvaluationContext`, `TellResult`, and shared `EngineStateSummary` records for
  evaluator calls, `tell(...)` summaries, and stable state inspection.
```

Under `[Unreleased]` `### Changed`, add:

```markdown
- `GAEngine` and `CMAESEngine` now expose `direction` and preserve raw scores while
  using direction-aware comparisons for best-candidate tracking.
- Policy-driven evaluators now receive `EvaluationContext` instead of a bare rung.
```

- [ ] **Step 10: Verify no current docs, examples, or tests use the old evaluator signature**

Run:

```powershell
rg -n "Evaluator\)|def evaluate\(self, candidates, rung\)|evaluate\(candidates, rung\)" README.md docs/site examples tests evocore
```

Expected: no matches in current source, public docs, examples, or tests.

- [ ] **Step 11: Run documentation-adjacent tests**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_ga_engine.py -q
```

Expected: PASS.

- [ ] **Step 12: Commit docs and helper updates**

Run:

```powershell
git add tests/vnext_helpers.py tests/unit/test_ga_engine.py tests/benchmarks/bench_vnext_multifidelity.py README.md examples/sphere_optimization.py examples/mixed_gene_space.py examples/onemax_binary.py examples/vnext_budgeted_ga.py docs/site/ga.md docs/site/parallelism.md docs/site/quickstart.md docs/site/ask-tell-engines.md docs/site/api.md CHANGELOG.md
git commit -m "docs: document optimizer lifecycle protocols"
```

Expected: commit succeeds with only helper, docs, examples, benchmark, and changelog files staged.

---

### Task 6: Final Verification And Review

**Files:**
- Verify only; do not edit unless a command fails and the failure points to this lifecycle work.

- [ ] **Step 1: Run targeted lifecycle tests**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 2: Run Python formatting check**

Run:

```powershell
python -m ruff format --check
```

Expected: PASS.

- [ ] **Step 3: Run Python lint**

Run:

```powershell
python -m ruff check
```

Expected: PASS.

- [ ] **Step 4: Run Rust formatting check**

Run:

```powershell
cargo fmt --check
```

Expected: PASS.

- [ ] **Step 5: Run Rust lint**

Run:

```powershell
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 6: Run Rust tests**

Run:

```powershell
cargo test
```

Expected: PASS.

- [ ] **Step 7: Rebuild the Python extension**

Run:

```powershell
python -m maturin develop --release
```

Expected: PASS and installs the local EvoCore extension.

- [ ] **Step 8: Run Python unit and integration tests**

Run:

```powershell
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 9: Run property tests**

Run:

```powershell
python -m pytest tests/property/ -v
```

Expected: PASS.

- [ ] **Step 10: Inspect final diff**

Run:

```powershell
git status --short --branch
git log --oneline -n 6
```

Expected: working tree is clean after the task commits, and the latest commits correspond to this lifecycle protocol work.

- [ ] **Step 11: Stop if verification fails**

If any verification command fails, do not commit, push, or open a PR. Report the failing command, the relevant error summary, and the likely files involved.

- [ ] **Step 12: Prepare completion summary**

Summarize:

```markdown
Implemented clean-break optimizer lifecycle protocols:
- Added `Optimizer` and `Evaluator` structural protocols.
- Added `EvaluationContext`, `TellResult`, shared `EngineStateSummary`, and direction-aware score comparison.
- Aligned `GAEngine` and `CMAESEngine` ask/tell behavior with the lifecycle contract.
- Updated tests, helpers, docs, and changelog.

Verification:
- `python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_protocols.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_vnext_policy_scheduler.py -v`: result observed during Task 6 Step 1.
- `python -m ruff format --check`: result observed during Task 6 Step 2.
- `python -m ruff check`: result observed during Task 6 Step 3.
- `cargo fmt --check`: result observed during Task 6 Step 4.
- `cargo clippy --all-targets -- -D warnings`: result observed during Task 6 Step 5.
- `cargo test`: result observed during Task 6 Step 6.
- `python -m maturin develop --release`: result observed during Task 6 Step 7.
- `python -m pytest tests/unit/ tests/integration/ -v`: result observed during Task 6 Step 8.
- `python -m pytest tests/property/ -v`: result observed during Task 6 Step 9.
```
