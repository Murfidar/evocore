# EvoCore vNext Expensive Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EvoCore vNext as a full-override expensive black-box optimizer with ask/tell engines, multi-fidelity scheduling, racing, surrogate hooks, mixed-variable CMA foundations, and anti-overfit telemetry.

**Architecture:** Add vNext candidate/evaluation primitives first, then policy/scheduler, Rust ranking helpers, GA ask/tell override, advisor/racing support, CMA ask/tell plus mixed-variable foundations, and release/docs updates. DEAP parity is not an acceptance criterion; old parity-centered docs/tests should be demoted or rewritten around vNext behavior.

**Tech Stack:** Python 3.11+ dataclasses/protocol-style classes, Rust/PyO3 0.28.3, pytest, cargo test, maturin, MkDocs, ruff.

---

## Implementation Parts

Part 1 creates public vNext primitives.

Part 2 adds policies, scheduler, and telemetry aggregation.

Part 3 moves deterministic candidate IDs and hot ranking into Rust.

Part 4 fully overrides GA around ask/tell and policy-driven execution.

Part 5 adds racing and surrogate advisor foundations.

Part 6 overrides CMA around ask/tell and creates mixed-variable foundations.

Part 7 updates release versioning, changelog, MkDocs, README, examples, docstrings, and final verification.

## File Structure

- Create `evocore/evaluation.py`: candidate, rung, evaluation record, telemetry, and evaluator base classes.
- Create `evocore/policies.py`: `MultiFidelityPolicy` and validation for budgets, rungs, exploration, and audit settings.
- Create `evocore/scheduler.py`: successive-halving/racing scheduler and promotion decisions.
- Create `evocore/advisors.py`: advisor base class and a pure-Python baseline surrogate advisor.
- Modify `evocore/ga.py`: ask/tell state, policy-driven `run`, trusted-record population updates, vNext telemetry.
- Modify `evocore/cmaes.py`: ask/tell state, trusted-record CMA updates, fixed-gene reconstruction.
- Create `evocore/mixed_cma.py`: mixed-variable CMA utility classes for integer margins and categorical probability state.
- Modify `evocore/__init__.py`: export new public vNext APIs and bump fallback version.
- Modify `evocore/_core.pyi`: add new Rust helper stubs.
- Create `src/candidate.rs`: deterministic candidate ID and confidence-aware ranking helpers.
- Modify `src/lib.rs`: expose candidate helpers through PyO3.
- Modify `src/utils.rs`: add operation constants if needed for candidate IDs.
- Create `tests/unit/test_vnext_evaluation.py`: primitive validation tests.
- Create `tests/unit/test_vnext_policy_scheduler.py`: policy and scheduler tests.
- Create `tests/unit/test_vnext_core_helpers.py`: PyO3 helper tests.
- Create `tests/unit/test_ga_ask_tell_vnext.py`: GA vNext tests.
- Create `tests/unit/test_vnext_advisors.py`: advisor and audit tests.
- Create `tests/unit/test_cmaes_ask_tell_vnext.py`: CMA ask/tell tests.
- Create `tests/unit/test_mixed_cma_vnext.py`: mixed-variable CMA foundation tests.
- Create `tests/benchmarks/bench_vnext_multifidelity.py`: deterministic budget-savings benchmark.
- Modify `tests/unit/test_ga_engine.py`: remove or rewrite generation/parity assumptions that conflict with vNext.
- Modify `tests/unit/test_cmaes_engine.py`: replace single-loop assumptions with ask/tell semantics.
- Modify `pyproject.toml` and `Cargo.toml`: bump to `0.7.0`.
- Modify `CHANGELOG.md`: add breaking vNext section.
- Modify `README.md`: describe EvoCore as an expensive black-box optimizer.
- Modify `mkdocs.yml`: add vNext pages and demote parity page.
- Create `docs/site/budget-aware-optimization.md`.
- Create `docs/site/ask-tell-engines.md`.
- Create `docs/site/mixed-variable-search.md`.
- Create `docs/site/optimizer-telemetry.md`.
- Modify `docs/site/api.md`: include new public APIs.
- Create `examples/vnext_budgeted_ga.py`: runnable vNext multi-fidelity example.

---

## Part 1: Public vNext Primitives

### Task 1.1: Add Candidate, Rung, EvaluationRecord, And Telemetry Types

**Files:**
- Create: `evocore/evaluation.py`
- Create: `tests/unit/test_vnext_evaluation.py`

- [ ] **Step 1: Write failing primitive tests**

Create `tests/unit/test_vnext_evaluation.py` with this content:

```python
import pytest

from evocore.evaluation import (
    Candidate,
    EvaluationRecord,
    OptimizationTelemetry,
    Rung,
)
from evocore.exceptions import ConfigurationError, FitnessError


def test_rung_requires_valid_budget_and_promotion_fraction() -> None:
    Rung("cheap", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="rung name"):
        Rung("", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="budget"):
        Rung("bad_budget", budget=0.0, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="promote_fraction"):
        Rung("bad_fraction", budget=0.25, promote_fraction=1.5, confidence="partial")


def test_candidate_applies_trusted_record_and_tracks_score() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        genes=[1.0, 2],
        params={"x": 1.0, "mode": 2},
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        score=0.75,
        confidence="trusted_full",
        rung="full_snapshot",
        cost=1.0,
        metrics={"trade_count": 12},
    )

    candidate.apply_record(record)

    assert candidate.status == "trusted"
    assert candidate.confidence == "trusted_full"
    assert candidate.rung == "full_snapshot"
    assert candidate.cost == 1.0
    assert candidate.scores["full_snapshot"].score == 0.75
    assert candidate.metadata["metrics"]["trade_count"] == 12


def test_candidate_rejects_record_for_different_candidate() -> None:
    candidate = Candidate(candidate_id="left", genes=[1.0], origin="random", event_index=0)
    record = EvaluationRecord(
        candidate_id="right",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    with pytest.raises(FitnessError, match="does not match candidate"):
        candidate.apply_record(record)


def test_surrogate_score_does_not_mark_candidate_trusted() -> None:
    candidate = Candidate(candidate_id="c-2", genes=[0.0], origin="random", event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-2",
            score=0.2,
            confidence="surrogate",
            rung="surrogate",
            cost=0.0,
            metrics={"model": "baseline"},
        )
    )

    assert candidate.status == "screened"
    assert candidate.confidence == "surrogate"
    assert "surrogate" in candidate.scores


def test_rejected_record_can_omit_score() -> None:
    record = EvaluationRecord(
        candidate_id="bad",
        score=None,
        confidence="rejected",
        rung="cheap",
        cost=0.0,
        metrics={"reason": "no_signals"},
    )

    assert record.score is None


def test_non_rejected_record_requires_finite_score() -> None:
    with pytest.raises(FitnessError, match="finite score"):
        EvaluationRecord(
            candidate_id="nan",
            score=float("nan"),
            confidence="partial",
            rung="cheap",
            cost=0.1,
        )


def test_telemetry_records_counts_and_costs() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_proposed(5)
    telemetry.record_screened(2)
    telemetry.record_partial(3, rung="cheap", cost=0.6)
    telemetry.record_full(1, rung="full", cost=1.0)
    telemetry.record_promoted(2, rung="cheap")
    telemetry.record_eliminated(1, rung="cheap")

    assert telemetry.total_candidates_proposed == 5
    assert telemetry.candidates_screened == 2
    assert telemetry.candidates_partial_evaluated == 3
    assert telemetry.candidates_full_evaluated == 1
    assert telemetry.promoted_by_rung["cheap"] == 2
    assert telemetry.eliminated_by_rung["cheap"] == 1
    assert telemetry.cost_by_rung["cheap"] == pytest.approx(0.6)
    assert telemetry.cost_by_rung["full"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run primitive tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_evaluation.py -v
```

Expected: FAIL during import because `evocore.evaluation` does not exist.

- [ ] **Step 3: Implement primitives**

Create `evocore/evaluation.py` with this content:

```python
"""vNext candidate, evaluation, and telemetry primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.exceptions import ConfigurationError, FitnessError
from evocore.individual import GeneValue

CandidateOrigin = Literal[
    "random",
    "crossover",
    "mutation",
    "cma_sample",
    "surrogate_proposal",
    "memory_seed",
    "restart",
]
CandidateStatus = Literal[
    "proposed",
    "screened",
    "racing",
    "promoted",
    "trusted",
    "eliminated",
    "archived",
]
EvaluationConfidence = Literal["surrogate", "partial", "cached", "trusted_full", "rejected"]


@dataclass(frozen=True)
class Rung:
    """Describe one multi-fidelity evaluation rung."""

    name: str
    budget: float
    promote_fraction: float
    confidence: EvaluationConfidence

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("rung name must be a non-empty string.")
        if not math.isfinite(float(self.budget)) or self.budget <= 0.0:
            raise ConfigurationError("rung budget must be finite and > 0.")
        if not (0.0 < float(self.promote_fraction) <= 1.0):
            raise ConfigurationError("rung promote_fraction must be in (0, 1].")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise ConfigurationError("rung confidence is invalid.")


@dataclass(frozen=True)
class CandidateScore:
    """Store one score observation for one candidate and rung."""

    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Record one evaluator result returned to an ask/tell engine."""

    candidate_id: str
    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise FitnessError("EvaluationRecord candidate_id must be non-empty.")
        if not self.rung:
            raise FitnessError("EvaluationRecord rung must be non-empty.")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise FitnessError("EvaluationRecord confidence is invalid.")
        if self.confidence != "rejected":
            if self.score is None or not math.isfinite(float(self.score)):
                raise FitnessError("EvaluationRecord requires a finite score unless rejected.")
        if not math.isfinite(float(self.cost)) or self.cost < 0.0:
            raise FitnessError("EvaluationRecord cost must be finite and >= 0.")


@dataclass
class Candidate:
    """Represent a vNext optimizer candidate with lifecycle and lineage."""

    candidate_id: str
    genes: list[GeneValue]
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

    def apply_record(self, record: EvaluationRecord) -> None:
        """Apply an evaluation record to this candidate."""
        if record.candidate_id != self.candidate_id:
            raise FitnessError(
                f"EvaluationRecord candidate_id {record.candidate_id!r} does not match "
                f"candidate {self.candidate_id!r}."
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
        )
        self.metadata["metrics"] = dict(record.metrics)
        if record.confidence == "trusted_full":
            self.status = "trusted"
        elif record.confidence == "rejected":
            self.status = "eliminated"
        elif record.confidence in ("partial", "cached"):
            self.status = "racing"
        else:
            self.status = "screened"

    def best_observed_score(self) -> float:
        """Return the best finite score observed for this candidate."""
        values = [score.score for score in self.scores.values() if score.score is not None]
        return max(values) if values else float("-inf")


@dataclass
class OptimizationTelemetry:
    """Aggregate vNext optimizer budget and trial accounting."""

    total_candidates_proposed: int = 0
    unique_candidate_hashes: set[str] = field(default_factory=set)
    candidates_screened: int = 0
    candidates_partial_evaluated: int = 0
    candidates_full_evaluated: int = 0
    promoted_by_rung: dict[str, int] = field(default_factory=dict)
    eliminated_by_rung: dict[str, int] = field(default_factory=dict)
    cost_by_rung: dict[str, float] = field(default_factory=dict)

    def record_proposed(self, count: int) -> None:
        self.total_candidates_proposed += int(count)

    def record_screened(self, count: int) -> None:
        self.candidates_screened += int(count)

    def record_partial(self, count: int, *, rung: str, cost: float) -> None:
        self.candidates_partial_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_full(self, count: int, *, rung: str, cost: float) -> None:
        self.candidates_full_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_promoted(self, count: int, *, rung: str) -> None:
        self.promoted_by_rung[rung] = self.promoted_by_rung.get(rung, 0) + int(count)

    def record_eliminated(self, count: int, *, rung: str) -> None:
        self.eliminated_by_rung[rung] = self.eliminated_by_rung.get(rung, 0) + int(count)


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

- [ ] **Step 4: Run primitive tests and confirm pass**

Run:

```powershell
pytest tests/unit/test_vnext_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit primitives**

```powershell
git add evocore/evaluation.py tests/unit/test_vnext_evaluation.py
git commit -m "feat: add vnext evaluation primitives"
```

### Task 1.2: Export vNext Primitive APIs

**Files:**
- Modify: `evocore/__init__.py`
- Test: `tests/unit/test_package_init.py`

- [ ] **Step 1: Add failing export test**

Append this test to `tests/unit/test_package_init.py`:

```python
def test_vnext_public_exports_are_available() -> None:
    import evocore

    assert evocore.Candidate.__name__ == "Candidate"
    assert evocore.EvaluationRecord.__name__ == "EvaluationRecord"
    assert evocore.Rung.__name__ == "Rung"
    assert evocore.OptimizationTelemetry.__name__ == "OptimizationTelemetry"
```

- [ ] **Step 2: Run export test and confirm failure**

Run:

```powershell
pytest tests/unit/test_package_init.py::test_vnext_public_exports_are_available -v
```

Expected: FAIL because the names are not exported from `evocore`.

- [ ] **Step 3: Export primitive APIs**

In `evocore/__init__.py`, add this import block after the callback imports:

```python
from evocore.evaluation import (
    Candidate,
    CandidateScore,
    EvaluationRecord,
    Evaluator,
    OptimizationTelemetry,
    Rung,
)
```

Add these names to `__all__` in alphabetical public-API order:

```python
"Candidate",
"CandidateScore",
"EvaluationRecord",
"Evaluator",
"OptimizationTelemetry",
"Rung",
```

- [ ] **Step 4: Run export test and primitive tests**

Run:

```powershell
pytest tests/unit/test_package_init.py::test_vnext_public_exports_are_available tests/unit/test_vnext_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit exports**

```powershell
git add evocore/__init__.py tests/unit/test_package_init.py
git commit -m "feat: export vnext evaluation APIs"
```

---

## Part 2: Policy, Scheduler, And Telemetry

### Task 2.1: Add MultiFidelityPolicy Validation

**Files:**
- Create: `evocore/policies.py`
- Create: `tests/unit/test_vnext_policy_scheduler.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/unit/test_vnext_policy_scheduler.py` with this initial content:

```python
import pytest

from evocore.evaluation import Rung
from evocore.exceptions import ConfigurationError
from evocore.policies import MultiFidelityPolicy


def test_policy_requires_unique_rung_names_and_full_budget() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=32,
        batch_size=8,
        exploration_fraction=0.10,
        audit_fraction=0.05,
    )

    assert policy.rung_names == ("cheap", "full")
    assert policy.final_rung.name == "full"


def test_policy_rejects_duplicate_rung_names() -> None:
    with pytest.raises(ConfigurationError, match="duplicate rung"):
        MultiFidelityPolicy(
            rungs=[
                Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
                Rung("cheap", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
            ],
            full_evaluation_budget=16,
        )


def test_policy_rejects_missing_trusted_full_rung() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full"):
        MultiFidelityPolicy(
            rungs=[Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial")],
            full_evaluation_budget=16,
        )


def test_policy_rejects_invalid_budget_and_fractions() -> None:
    with pytest.raises(ConfigurationError, match="full_evaluation_budget"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=0,
        )

    with pytest.raises(ConfigurationError, match="exploration_fraction"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=1,
            exploration_fraction=1.5,
        )
```

- [ ] **Step 2: Run policy tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: FAIL because `evocore.policies` does not exist.

- [ ] **Step 3: Implement policy object**

Create `evocore/policies.py` with this content:

```python
"""vNext optimization policy objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evocore.evaluation import Rung
from evocore.exceptions import ConfigurationError


@dataclass(frozen=True)
class MultiFidelityPolicy:
    """Configure multi-fidelity scheduling for vNext engines."""

    rungs: list[Rung]
    full_evaluation_budget: int
    batch_size: int | None = None
    exploration_fraction: float = 0.10
    audit_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ConfigurationError("MultiFidelityPolicy requires at least one rung.")
        if int(self.full_evaluation_budget) <= 0:
            raise ConfigurationError("full_evaluation_budget must be positive.")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ConfigurationError("batch_size must be positive when provided.")
        if not (0.0 <= float(self.exploration_fraction) < 1.0):
            raise ConfigurationError("exploration_fraction must be in [0, 1).")
        if not (0.0 <= float(self.audit_fraction) < 1.0):
            raise ConfigurationError("audit_fraction must be in [0, 1).")

        names = [rung.name for rung in self.rungs]
        if len(names) != len(set(names)):
            raise ConfigurationError("MultiFidelityPolicy contains duplicate rung names.")
        if not any(rung.confidence == "trusted_full" for rung in self.rungs):
            raise ConfigurationError("MultiFidelityPolicy requires a trusted_full rung.")

    @property
    def rung_names(self) -> Sequence[str]:
        """Return rung names in execution order."""
        return tuple(rung.name for rung in self.rungs)

    @property
    def final_rung(self) -> Rung:
        """Return the last configured rung."""
        return self.rungs[-1]

    @classmethod
    def single_full(cls, *, budget: int, batch_size: int | None = None) -> MultiFidelityPolicy:
        """Create a one-rung full-evaluation vNext policy."""
        return cls(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=budget,
            batch_size=batch_size,
            exploration_fraction=0.0,
            audit_fraction=0.0,
        )
```

- [ ] **Step 4: Run policy tests and confirm pass**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 5: Export policy**

Modify `evocore/__init__.py`:

```python
from evocore.policies import MultiFidelityPolicy
```

Add `"MultiFidelityPolicy"` to `__all__`.

- [ ] **Step 6: Commit policy**

```powershell
git add evocore/__init__.py evocore/policies.py tests/unit/test_vnext_policy_scheduler.py
git commit -m "feat: add vnext multi-fidelity policy"
```

### Task 2.2: Add Successive-Halving Scheduler

**Files:**
- Create: `evocore/scheduler.py`
- Modify: `tests/unit/test_vnext_policy_scheduler.py`

- [ ] **Step 1: Add failing scheduler tests**

Append these tests to `tests/unit/test_vnext_policy_scheduler.py`:

```python
from evocore.evaluation import Candidate, EvaluationRecord
from evocore.scheduler import EvaluationScheduler


def _candidate(index: int) -> Candidate:
    return Candidate(candidate_id=f"c-{index}", genes=[float(index)], origin="random", event_index=0)


def test_scheduler_promotes_top_fraction_by_previous_rung_score() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.4, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(5)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")

    assert [candidate.candidate_id for candidate in promoted] == ["c-4", "c-3"]
    assert all(candidate.status == "promoted" for candidate in promoted)


def test_scheduler_assigns_first_rung_to_new_candidates() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(0), _candidate(1)]

    assigned = scheduler.assign_rung(candidates, rung_name="cheap")

    assert [candidate.rung for candidate in assigned] == ["cheap", "cheap"]
    assert [candidate.status for candidate in assigned] == ["racing", "racing"]


def test_scheduler_counts_eliminated_candidates() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(4)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")
    eliminated = [candidate for candidate in candidates if candidate.status == "eliminated"]

    assert len(promoted) == 2
    assert [candidate.candidate_id for candidate in eliminated] == ["c-0", "c-1"]
```

- [ ] **Step 2: Run scheduler tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: FAIL because `evocore.scheduler` does not exist.

- [ ] **Step 3: Implement scheduler**

Create `evocore/scheduler.py` with this content:

```python
"""vNext multi-fidelity schedulers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evocore.evaluation import Candidate
from evocore.exceptions import ConfigurationError
from evocore.policies import MultiFidelityPolicy


class EvaluationScheduler:
    """Schedule candidates across multi-fidelity rungs."""

    def __init__(self, policy: MultiFidelityPolicy) -> None:
        self.policy = policy

    def rung_after(self, completed_rung: str) -> str | None:
        """Return the next rung name after a completed rung."""
        names = self.policy.rung_names
        if completed_rung not in names:
            raise ConfigurationError(f"unknown rung: {completed_rung!r}")
        index = names.index(completed_rung)
        if index + 1 >= len(names):
            return None
        return names[index + 1]

    def assign_rung(self, candidates: Sequence[Candidate], *, rung_name: str) -> list[Candidate]:
        """Assign a rung to candidates selected for evaluation."""
        if rung_name not in self.policy.rung_names:
            raise ConfigurationError(f"unknown rung: {rung_name!r}")
        assigned = list(candidates)
        for candidate in assigned:
            candidate.rung = rung_name
            candidate.status = "racing"
        return assigned

    def promote(self, candidates: Sequence[Candidate], *, completed_rung: str) -> list[Candidate]:
        """Promote the top candidate fraction after a completed rung."""
        if completed_rung not in self.policy.rung_names:
            raise ConfigurationError(f"unknown rung: {completed_rung!r}")

        rung = self.policy.rungs[self.policy.rung_names.index(completed_rung)]
        ranked = sorted(candidates, key=lambda candidate: candidate.best_observed_score(), reverse=True)
        promote_count = max(1, int(math.ceil(len(ranked) * rung.promote_fraction)))
        promoted = ranked[:promote_count]
        promoted_ids = {candidate.candidate_id for candidate in promoted}
        for candidate in ranked:
            if candidate.candidate_id in promoted_ids:
                candidate.status = "promoted"
            else:
                candidate.status = "eliminated"
        return promoted
```

- [ ] **Step 4: Run scheduler tests**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 5: Export scheduler**

Modify `evocore/__init__.py`:

```python
from evocore.scheduler import EvaluationScheduler
```

Add `"EvaluationScheduler"` to `__all__`.

- [ ] **Step 6: Commit scheduler**

```powershell
git add evocore/__init__.py evocore/scheduler.py tests/unit/test_vnext_policy_scheduler.py
git commit -m "feat: add vnext evaluation scheduler"
```

---

## Part 3: Rust Candidate Helpers

### Task 3.1: Add Deterministic Candidate ID And Ranking Helpers

**Files:**
- Create: `src/candidate.rs`
- Modify: `src/lib.rs`
- Modify: `evocore/_core.pyi`
- Create: `tests/unit/test_vnext_core_helpers.py`

- [ ] **Step 1: Write failing PyO3 helper tests**

Create `tests/unit/test_vnext_core_helpers.py` with this content:

```python
from evocore import _core


def test_candidate_id_is_deterministic_and_distinguishes_index() -> None:
    left = _core.candidate_id(42, 3, 0)
    right = _core.candidate_id(42, 3, 0)
    other = _core.candidate_id(42, 3, 1)

    assert left == right
    assert left != other
    assert left.startswith("c-")


def test_rank_top_k_prefers_trusted_then_score() -> None:
    indices = _core.rank_top_k(
        scores=[0.9, 10.0, 0.7, 0.5],
        trusted_mask=[True, False, True, True],
        k=2,
    )

    assert indices == [0, 2]


def test_rank_top_k_uses_score_when_trust_matches() -> None:
    indices = _core.rank_top_k(
        scores=[1.0, 3.0, 2.0],
        trusted_mask=[True, True, True],
        k=3,
    )

    assert indices == [1, 2, 0]
```

- [ ] **Step 2: Run PyO3 tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_core_helpers.py -v
```

Expected: FAIL because `_core.candidate_id` and `_core.rank_top_k` are missing.

- [ ] **Step 3: Add Rust module**

Create `src/candidate.rs` with this content:

```rust
use std::cmp::Ordering;

use crate::selection::safe_fitness;
use crate::utils::{derive_seed, OP_SELECTION};

pub fn candidate_id(master_seed: u64, event_index: u64, candidate_index: u64) -> String {
    let left = derive_seed(master_seed, event_index, candidate_index, OP_SELECTION);
    let right = derive_seed(master_seed ^ 0xA5A5_A5A5_A5A5_A5A5, candidate_index, event_index, OP_SELECTION);
    format!("c-{left:016x}{right:016x}")
}

pub fn rank_top_k(scores: &[f64], trusted_mask: &[bool], k: usize) -> Vec<usize> {
    assert_eq!(scores.len(), trusted_mask.len(), "scores and trusted_mask length mismatch");
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&left, &right| {
        match trusted_mask[right].cmp(&trusted_mask[left]) {
            Ordering::Equal => safe_fitness(scores[right])
                .partial_cmp(&safe_fitness(scores[left]))
                .unwrap_or(Ordering::Equal),
            ordering => ordering,
        }
    });
    indices.truncate(k.min(indices.len()));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_id_deterministic() {
        assert_eq!(candidate_id(42, 1, 2), candidate_id(42, 1, 2));
        assert_ne!(candidate_id(42, 1, 2), candidate_id(42, 1, 3));
    }

    #[test]
    fn test_rank_top_k_prefers_trusted() {
        let ranked = rank_top_k(&[0.9, 10.0, 0.7], &[true, false, true], 2);
        assert_eq!(ranked, vec![0, 2]);
    }
}
```

- [ ] **Step 4: Expose Rust helpers through PyO3**

In `src/lib.rs`, add this module near the other module declarations:

```rust
pub mod candidate;
```

Add these imports near the other `use` declarations:

```rust
use candidate::{candidate_id as candidate_id_impl, rank_top_k as rank_top_k_impl};
```

Add these PyO3 functions before the `_core` module initializer:

```rust
#[pyfunction]
fn candidate_id(master_seed: u64, event_index: u64, candidate_index: u64) -> String {
    candidate_id_impl(master_seed, event_index, candidate_index)
}

#[pyfunction]
fn rank_top_k(scores: Vec<f64>, trusted_mask: Vec<bool>, k: usize) -> Vec<usize> {
    rank_top_k_impl(&scores, &trusted_mask, k)
}
```

Register the functions inside the `_core` module initializer:

```rust
m.add_function(wrap_pyfunction!(candidate_id, m)?)?;
m.add_function(wrap_pyfunction!(rank_top_k, m)?)?;
```

- [ ] **Step 5: Update type stubs**

Append these stubs to `evocore/_core.pyi` after `py_derive_seed`:

```python
def candidate_id(master_seed: int, event_index: int, candidate_index: int) -> str: pass
def rank_top_k(scores: Sequence[float], trusted_mask: Sequence[bool], k: int) -> list[int]: pass
```

- [ ] **Step 6: Run Rust and PyO3 tests**

Run:

```powershell
cargo test candidate
maturin develop --release
pytest tests/unit/test_vnext_core_helpers.py -v
```

Expected: all commands PASS.

- [ ] **Step 7: Commit Rust helpers**

```powershell
git add src/candidate.rs src/lib.rs evocore/_core.pyi tests/unit/test_vnext_core_helpers.py
git commit -m "feat: add vnext candidate rust helpers"
```

---

## Part 4: GA Ask/Tell Full Override

### Task 4.1: Add GA Ask/Tell State And Tests

**Files:**
- Modify: `evocore/ga.py`
- Create: `tests/unit/test_ga_ask_tell_vnext.py`

- [ ] **Step 1: Write failing GA ask/tell tests**

Create `tests/unit/test_ga_ask_tell_vnext.py` with this content:

```python
from evocore import EvaluationRecord, GAEngine, GeneDef, GeneSpace


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("mode", "int", 0, 3),
        ]
    )


def test_ga_ask_returns_candidates_with_params_and_ids() -> None:
    engine = GAEngine(_space(), population_size=6, generations=5, seed=123)

    candidates = engine.ask(4)

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert all(candidate.params is not None for candidate in candidates)
    assert all(candidate.origin == "random" for candidate in candidates)


def test_ga_tell_trusted_records_builds_trusted_population() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=float(index),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for index, candidate in enumerate(candidates)
    ]

    summary = engine.tell(records)

    assert summary.trusted_count == 4
    assert engine.vnext_telemetry.candidates_full_evaluated == 4
    assert engine.best_candidate.candidate_id == candidates[-1].candidate_id


def test_ga_tell_surrogate_records_do_not_build_trusted_population() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=100.0,
                confidence="surrogate",
                rung="surrogate",
                cost=0.0,
            )
            for candidate in candidates
        ]
    )

    assert engine.vnext_telemetry.candidates_screened == 4
    assert engine.best_candidate is None
```

- [ ] **Step 2: Run GA ask/tell tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: FAIL because `GAEngine.ask` and `GAEngine.tell` are missing.

- [ ] **Step 3: Add summary type and state fields**

In `evocore/ga.py`, add these imports:

```python
from dataclasses import dataclass
from evocore.evaluation import Candidate, EvaluationRecord, OptimizationTelemetry
```

Add this dataclass near `RunResult` and `MultiRunResult`:

```python
@dataclass(frozen=True)
class EngineStateSummary:
    """Summarize one vNext tell() state update."""

    accepted_count: int
    trusted_count: int
    partial_count: int
    surrogate_count: int
    rejected_count: int
```

Inside `GAEngine.__init__`, add these fields after `_fitness_warning_emitted`:

```python
self._event_index = 0
self._candidates_by_id: dict[str, Candidate] = {}
self._trusted_population_vnext: list[Candidate] = []
self.vnext_telemetry = OptimizationTelemetry()
self.best_candidate: Candidate | None = None
```

- [ ] **Step 4: Add ask() helper methods**

In `GAEngine`, add these methods before `run`:

```python
def _candidate_from_genes(
    self,
    genes: list[float | int | bool],
    *,
    origin: str,
    event_index: int,
    candidate_index: int,
    parents: Sequence[str] = (),
) -> Candidate:
    candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
    params = self.gene_space.params_for(genes)
    return Candidate(
        candidate_id=candidate_id,
        genes=list(genes),
        params=params,
        origin=origin,
        parents=parents,
        event_index=event_index,
    )

def ask(self, n: int | None = None) -> list[Candidate]:
    """Return vNext candidates for external evaluation."""
    count = int(n or self.population_size)
    if count <= 0:
        raise ConfigurationError("ask(n) requires n > 0.")

    event_index = self._event_index
    if not self._trusted_population_vnext:
        encoded = _core.init_population(
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            count,
            int(_core.py_derive_seed(self.seed, event_index, 0, _core.OP_INIT)),
        )
        individuals = self.operators.decode_population(encoded)
        candidates = [
            self._candidate_from_genes(
                individual.genes,
                origin="random",
                event_index=event_index,
                candidate_index=index,
            )
            for index, individual in enumerate(individuals)
        ]
    else:
        trusted_individuals = [
            Individual(
                list(candidate.genes),
                fitness=candidate.best_observed_score(),
                fitness_valid=True,
                metadata={"params": candidate.params} if candidate.params else {},
            )
            for candidate in self._trusted_population_vnext
        ]
        fitnesses = [individual.fitness or float("-inf") for individual in trusted_individuals]
        offspring = self._make_offspring(
            trusted_individuals,
            fitnesses,
            gen=event_index,
            offspring_count=count,
        )
        candidates = [
            self._candidate_from_genes(
                individual.genes,
                origin="mutation",
                event_index=event_index,
                candidate_index=index,
            )
            for index, individual in enumerate(offspring)
        ]

    for candidate in candidates:
        self._candidates_by_id[candidate.candidate_id] = candidate
    self._event_index += 1
    self.vnext_telemetry.record_proposed(len(candidates))
    return candidates
```

- [ ] **Step 5: Add tell()**

Add this method after `ask`:

```python
def tell(self, records: Sequence[EvaluationRecord]) -> EngineStateSummary:
    """Update GA state from vNext evaluation records."""
    trusted = partial = surrogate = rejected = 0
    for record in records:
        candidate = self._candidates_by_id.get(record.candidate_id)
        if candidate is None:
            raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
        candidate.apply_record(record)
        if record.confidence == "trusted_full":
            trusted += 1
            self._trusted_population_vnext.append(candidate)
            if self.best_candidate is None or candidate.best_observed_score() > self.best_candidate.best_observed_score():
                self.best_candidate = candidate
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

    self._trusted_population_vnext.sort(key=lambda candidate: candidate.best_observed_score(), reverse=True)
    self._trusted_population_vnext = self._trusted_population_vnext[: self.population_size]
    return EngineStateSummary(
        accepted_count=len(records),
        trusted_count=trusted,
        partial_count=partial,
        surrogate_count=surrogate,
        rejected_count=rejected,
    )
```

- [ ] **Step 6: Run GA ask/tell tests**

Run:

```powershell
pytest tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: PASS.

- [ ] **Step 7: Export summary type**

Modify `evocore/__init__.py`:

```python
from evocore.ga import EngineStateSummary, GAEngine, MultiRunResult, RunResult
```

Add `"EngineStateSummary"` to `__all__`.

- [ ] **Step 8: Commit ask/tell GA state**

```powershell
git add evocore/ga.py evocore/__init__.py tests/unit/test_ga_ask_tell_vnext.py
git commit -m "feat: add ga ask tell vnext state"
```

### Task 4.2: Replace GA run() With Policy-Driven vNext Execution

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing policy-driven run test**

Append this test to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
from evocore import Evaluator, MultiFidelityPolicy, Rung


class SphereEvaluator(Evaluator):
    def evaluate(self, candidates, rung):
        confidence = rung.confidence
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates
        ]


def test_ga_run_uses_policy_and_returns_vnext_telemetry() -> None:
    engine = GAEngine(_space(), population_size=6, generations=20, seed=123)
    policy = MultiFidelityPolicy.single_full(budget=12, batch_size=4)

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.n_evaluations == 12
    assert result.best_individual.fitness_valid
    assert result.telemetry.candidates_full_evaluated == 12
    assert result.stop_reason == "max_evaluations"
```

- [ ] **Step 2: Run vNext run test and confirm failure**

Run:

```powershell
pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_uses_policy_and_returns_vnext_telemetry -v
```

Expected: FAIL because `run()` still expects a fitness function and `RunResult` has no `telemetry`.

- [ ] **Step 3: Extend RunResult**

In the `RunResult` dataclass in `evocore/ga.py`, add this field at the end:

```python
telemetry: OptimizationTelemetry = field(default_factory=OptimizationTelemetry)
```

Ensure `field` is imported from `dataclasses`:

```python
from dataclasses import dataclass, field
```

- [ ] **Step 4: Replace GA run() body with vNext policy orchestration**

Replace the existing `run` method in `GAEngine` with this version:

```python
def run(self, evaluator: Evaluator, policy: MultiFidelityPolicy | None = None) -> RunResult:
    """Run vNext policy-driven GA optimization."""
    if not isinstance(evaluator, Evaluator):
        raise ConfigurationError(
            "GAEngine.run now requires an Evaluator instance with evaluate(candidates, rung)."
        )

    resolved_policy = policy or MultiFidelityPolicy.single_full(
        budget=max(1, self.population_size * max(1, self.generations)),
        batch_size=self.population_size,
    )
    scheduler = EvaluationScheduler(resolved_policy)
    start = time.perf_counter()
    n_evaluations = 0
    final_candidates: list[Candidate] = []

    while self.vnext_telemetry.candidates_full_evaluated < resolved_policy.full_evaluation_budget:
        remaining = resolved_policy.full_evaluation_budget - self.vnext_telemetry.candidates_full_evaluated
        batch_size = min(resolved_policy.batch_size or self.population_size, remaining)
        candidates = self.ask(batch_size)
        active_candidates = candidates

        for rung in resolved_policy.rungs:
            assigned = scheduler.assign_rung(active_candidates, rung_name=rung.name)
            records = list(evaluator.evaluate(assigned, rung))
            self.tell(records)
            if rung.confidence == "trusted_full":
                n_evaluations += len(records)
                final_candidates.extend(assigned)
                break
            promoted = scheduler.promote(assigned, completed_rung=rung.name)
            next_rung = scheduler.rung_after(rung.name)
            if next_rung is None:
                active_candidates = promoted
            else:
                active_candidates = promoted

    if self.best_candidate is None:
        raise FitnessError("GA run completed without a trusted_full candidate.")

    best = Individual(
        list(self.best_candidate.genes),
        fitness=self.best_candidate.best_observed_score(),
        fitness_valid=True,
        metadata={"params": self.best_candidate.params, "candidate_id": self.best_candidate.candidate_id},
    )
    final_population = Population(
        [
            Individual(
                list(candidate.genes),
                fitness=candidate.best_observed_score(),
                fitness_valid=candidate.confidence == "trusted_full",
                metadata={"params": candidate.params, "candidate_id": candidate.candidate_id},
            )
            for candidate in final_candidates
        ]
    )
    return RunResult(
        best_individual=best,
        best_fitness=float(best.fitness),
        final_population=final_population,
        logbook=Logbook(),
        wall_time_seconds=time.perf_counter() - start,
        n_evaluations=n_evaluations,
        elite_history=[],
        diversity_history=[],
        seed=self.seed,
        stopped_early=True,
        max_evaluations=resolved_policy.full_evaluation_budget,
        stop_reason="max_evaluations",
        budget_reached=True,
        telemetry=self.vnext_telemetry,
    )
```

Add these imports:

```python
from evocore.evaluation import Evaluator
from evocore.policies import MultiFidelityPolicy
from evocore.scheduler import EvaluationScheduler
```

- [ ] **Step 5: Update old GA run tests to vNext evaluator style**

In `tests/unit/test_ga_engine.py`, add this helper near the top:

```python
from evocore import EvaluationRecord, Evaluator, MultiFidelityPolicy, Rung


class CallableEvaluator(Evaluator):
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, rung):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(self.fn(candidate.genes)),
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates
        ]


def full_policy(budget: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        full_evaluation_budget=budget,
        batch_size=batch_size,
    )
```

For each remaining old-style callable assertion, replace the callable with an explicit
vNext evaluator call such as
`engine.run(CallableEvaluator(lambda genes: -sum(float(value) ** 2 for value in genes)), policy=full_policy(8))`.
Remove assertions that explicitly require DEAP-parity generation semantics, outer mutation
gates, or tournament-with-replacement behavior as a product contract.

- [ ] **Step 6: Run GA tests**

Run:

```powershell
pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit GA vNext run override**

```powershell
git add evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py
git commit -m "feat: make ga run policy driven"
```

---

## Part 5: Racing And Advisor Foundations

### Task 5.1: Add Audit-Aware Racing Behavior

**Files:**
- Modify: `evocore/scheduler.py`
- Modify: `tests/unit/test_vnext_policy_scheduler.py`

- [ ] **Step 1: Add failing audit quota test**

Append this test to `tests/unit/test_vnext_policy_scheduler.py`:

```python
def test_scheduler_audit_fraction_promotes_one_low_ranked_candidate() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.25, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
        audit_fraction=0.25,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(8)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")
    promoted_ids = {candidate.candidate_id for candidate in promoted}

    assert {"c-7", "c-6"}.issubset(promoted_ids)
    assert len(promoted) == 3
    assert any(candidate_id in promoted_ids for candidate_id in {"c-0", "c-1", "c-2", "c-3", "c-4", "c-5"})
```

- [ ] **Step 2: Run audit test and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py::test_scheduler_audit_fraction_promotes_one_low_ranked_candidate -v
```

Expected: FAIL because audit quota is not implemented.

- [ ] **Step 3: Implement deterministic audit promotion**

Replace `promote()` in `evocore/scheduler.py` with this implementation:

```python
def promote(self, candidates: Sequence[Candidate], *, completed_rung: str) -> list[Candidate]:
    """Promote top candidates plus deterministic audit samples."""
    if completed_rung not in self.policy.rung_names:
        raise ConfigurationError(f"unknown rung: {completed_rung!r}")

    rung = self.policy.rungs[self.policy.rung_names.index(completed_rung)]
    ranked = sorted(candidates, key=lambda candidate: candidate.best_observed_score(), reverse=True)
    promote_count = max(1, int(math.ceil(len(ranked) * rung.promote_fraction)))
    audit_count = int(math.floor(len(ranked) * self.policy.audit_fraction))
    promoted = list(ranked[:promote_count])

    if audit_count > 0 and len(ranked) > promote_count:
        audit_pool = ranked[promote_count:]
        promoted.extend(audit_pool[:audit_count])

    promoted_ids = {candidate.candidate_id for candidate in promoted}
    for candidate in ranked:
        if candidate.candidate_id in promoted_ids:
            candidate.status = "promoted"
        else:
            candidate.status = "eliminated"
    return promoted
```

- [ ] **Step 4: Run scheduler tests**

Run:

```powershell
pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit audit quota**

```powershell
git add evocore/scheduler.py tests/unit/test_vnext_policy_scheduler.py
git commit -m "feat: add scheduler audit quota"
```

### Task 5.2: Add Baseline Surrogate Advisor

**Files:**
- Create: `evocore/advisors.py`
- Create: `tests/unit/test_vnext_advisors.py`

- [ ] **Step 1: Write failing advisor tests**

Create `tests/unit/test_vnext_advisors.py` with this content:

```python
from evocore.advisors import InverseDistanceSurrogateAdvisor
from evocore.evaluation import Candidate, EvaluationRecord


def _candidate(candidate_id: str, x: float) -> Candidate:
    return Candidate(candidate_id=candidate_id, genes=[x], origin="random", event_index=0)


def test_surrogate_advisor_scores_near_known_good_candidate_higher() -> None:
    advisor = InverseDistanceSurrogateAdvisor()
    good = _candidate("good", 1.0)
    bad = _candidate("bad", 5.0)
    advisor.observe(
        [
            EvaluationRecord("good", score=10.0, confidence="trusted_full", rung="full", cost=1.0),
            EvaluationRecord("bad", score=-10.0, confidence="trusted_full", rung="full", cost=1.0),
        ],
        candidates={"good": good, "bad": bad},
    )

    near_good = _candidate("near_good", 1.1)
    near_bad = _candidate("near_bad", 4.9)
    rankings = advisor.rank([near_bad, near_good])

    assert rankings[0].candidate_id == "near_good"
    assert rankings[0].confidence == "surrogate"


def test_surrogate_advisor_returns_zero_scores_before_observations() -> None:
    advisor = InverseDistanceSurrogateAdvisor()
    rankings = advisor.rank([_candidate("x", 1.0)])

    assert rankings[0].score == 0.0
    assert rankings[0].reason == "no_training_data"
```

- [ ] **Step 2: Run advisor tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_vnext_advisors.py -v
```

Expected: FAIL because `evocore.advisors` does not exist.

- [ ] **Step 3: Implement advisor module**

Create `evocore/advisors.py` with this content:

```python
"""vNext optimizer advisors."""

from __future__ import annotations

import math
from dataclasses import dataclass

from evocore.evaluation import Candidate, EvaluationConfidence, EvaluationRecord


@dataclass(frozen=True)
class AdvisorScore:
    """Rank one candidate using an advisor."""

    candidate_id: str
    score: float
    confidence: EvaluationConfidence
    reason: str


class InverseDistanceSurrogateAdvisor:
    """Pure-Python inverse-distance baseline surrogate advisor."""

    def __init__(self) -> None:
        self._observations: list[tuple[list[float], float]] = []

    def observe(
        self,
        records: list[EvaluationRecord],
        *,
        candidates: dict[str, Candidate],
    ) -> None:
        """Observe trusted records for surrogate ranking."""
        for record in records:
            if record.confidence != "trusted_full" or record.score is None:
                continue
            candidate = candidates[record.candidate_id]
            self._observations.append(([float(value) for value in candidate.genes], float(record.score)))

    def rank(self, candidates: list[Candidate]) -> list[AdvisorScore]:
        """Rank candidates by inverse-distance weighted known scores."""
        rankings: list[AdvisorScore] = []
        for candidate in candidates:
            if not self._observations:
                rankings.append(
                    AdvisorScore(
                        candidate_id=candidate.candidate_id,
                        score=0.0,
                        confidence="surrogate",
                        reason="no_training_data",
                    )
                )
                continue
            genes = [float(value) for value in candidate.genes]
            weighted_sum = 0.0
            weight_total = 0.0
            for observed_genes, observed_score in self._observations:
                distance = math.sqrt(
                    sum((left - right) ** 2 for left, right in zip(genes, observed_genes, strict=True))
                )
                weight = 1.0 / max(distance, 1e-9)
                weighted_sum += observed_score * weight
                weight_total += weight
            rankings.append(
                AdvisorScore(
                    candidate_id=candidate.candidate_id,
                    score=weighted_sum / weight_total,
                    confidence="surrogate",
                    reason="inverse_distance",
                )
            )
        return sorted(rankings, key=lambda item: item.score, reverse=True)
```

- [ ] **Step 4: Export advisor**

Modify `evocore/__init__.py`:

```python
from evocore.advisors import AdvisorScore, InverseDistanceSurrogateAdvisor
```

Add `"AdvisorScore"` and `"InverseDistanceSurrogateAdvisor"` to `__all__`.

- [ ] **Step 5: Run advisor tests**

Run:

```powershell
pytest tests/unit/test_vnext_advisors.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit advisor**

```powershell
git add evocore/advisors.py evocore/__init__.py tests/unit/test_vnext_advisors.py
git commit -m "feat: add baseline surrogate advisor"
```

---

## Part 6: CMA Ask/Tell And Mixed-Variable Foundations

### Task 6.1: Add CMA Ask/Tell Trusted Update Seam

**Files:**
- Modify: `evocore/cmaes.py`
- Create: `tests/unit/test_cmaes_ask_tell_vnext.py`

- [ ] **Step 1: Write failing CMA ask/tell tests**

Create `tests/unit/test_cmaes_ask_tell_vnext.py` with this content:

```python
from evocore import CMAESEngine, EvaluationRecord, GeneDef, GeneSpace


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("period", "int", 2, 20),
        ]
    )


def test_cma_ask_returns_candidate_batch() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)

    candidates = engine.ask()

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert all(candidate.params is not None for candidate in candidates)


def test_cma_tell_ignores_partial_records_for_state_update() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    generation_before = engine.generation

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=1.0,
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
            for candidate in candidates
        ]
    )

    assert summary.partial_count == 4
    assert engine.generation == generation_before


def test_cma_tell_trusted_records_updates_state() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            )
            for candidate in candidates
        ]
    )

    assert summary.trusted_count == 4
    assert engine.generation == 1
```

- [ ] **Step 2: Run CMA tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: FAIL because `CMAESEngine.ask`, `CMAESEngine.tell`, and `generation` are missing.

- [ ] **Step 3: Add CMA vNext state fields**

In `evocore/cmaes.py`, add imports:

```python
from evocore.evaluation import Candidate, EvaluationRecord, OptimizationTelemetry
from evocore.ga import EngineStateSummary
```

Inside `CMAESEngine.__init__`, add:

```python
self._state: _core.PyCMAESState | None = None
self._event_index = 0
self._pending_samples_by_id: dict[str, list[float]] = {}
self._candidates_by_id: dict[str, Candidate] = {}
self.vnext_telemetry = OptimizationTelemetry()
```

Add this property:

```python
@property
def generation(self) -> int:
    """Return the current CMA generation."""
    return 0 if self._state is None else int(self._state.generation)
```

- [ ] **Step 4: Add ask() and tell() to CMAESEngine**

Add these methods before `run` in `evocore/cmaes.py`:

```python
def _ensure_state(self) -> _core.PyCMAESState:
    if self._state is None:
        self._state = _core.PyCMAESState(
            self._initial_mean_encoded(),
            self._sigma_abs(),
            self.population_size,
            self._bounds_list,
        )
    return self._state

def ask(self, n: int | None = None) -> list[Candidate]:
    """Return a CMA candidate batch."""
    if n is not None and int(n) != self.population_size:
        raise ConfigurationError("CMAESEngine.ask currently requires n to equal population_size.")
    state = self._ensure_state()
    event_index = self._event_index
    samples_continuous = state.ask(self.seed, event_index)
    samples_discrete = [self._apply_bounds_and_round(sample) for sample in samples_continuous]
    candidates: list[Candidate] = []
    for index, sample in enumerate(samples_discrete):
        individual = self._decode_individual(sample)
        candidate_id = _core.candidate_id(self.seed, event_index, index)
        candidate = Candidate(
            candidate_id=candidate_id,
            genes=list(individual.genes),
            params=individual.params,
            origin="cma_sample",
            event_index=event_index,
        )
        self._pending_samples_by_id[candidate_id] = list(samples_continuous[index])
        self._candidates_by_id[candidate_id] = candidate
        candidates.append(candidate)
    self._event_index += 1
    self.vnext_telemetry.record_proposed(len(candidates))
    return candidates

def tell(self, records: Sequence[EvaluationRecord]) -> EngineStateSummary:
    """Update CMA state from trusted evaluation records."""
    trusted_records: list[EvaluationRecord] = []
    partial = surrogate = rejected = 0
    for record in records:
        candidate = self._candidates_by_id.get(record.candidate_id)
        if candidate is None:
            raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
        candidate.apply_record(record)
        if record.confidence == "trusted_full":
            trusted_records.append(record)
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

    if len(trusted_records) == self.population_size:
        samples = [self._pending_samples_by_id[record.candidate_id] for record in trusted_records]
        fitnesses = [float(record.score) for record in trusted_records if record.score is not None]
        self._ensure_state().tell(samples, fitnesses)

    return EngineStateSummary(
        accepted_count=len(records),
        trusted_count=len(trusted_records),
        partial_count=partial,
        surrogate_count=surrogate,
        rejected_count=rejected,
    )
```

- [ ] **Step 5: Run CMA ask/tell tests**

Run:

```powershell
pytest tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit CMA ask/tell seam**

```powershell
git add evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py
git commit -m "feat: add cma ask tell vnext seam"
```

### Task 6.2: Add Mixed-Variable CMA Foundation Types

**Files:**
- Create: `evocore/mixed_cma.py`
- Create: `tests/unit/test_mixed_cma_vnext.py`
- Modify: `evocore/__init__.py`

- [ ] **Step 1: Write failing mixed-variable foundation tests**

Create `tests/unit/test_mixed_cma_vnext.py` with this content:

```python
import pytest

from evocore.mixed_cma import CategoricalState, IntegerMargin


def test_integer_margin_keeps_probability_mass_inside_bounds() -> None:
    margin = IntegerMargin(low=0, high=3, min_probability=0.10)

    probabilities = margin.probabilities(mean=1.4, sigma=0.2)

    assert set(probabilities) == {0, 1, 2, 3}
    assert all(value >= 0.10 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_categorical_state_updates_toward_better_category() -> None:
    state = CategoricalState(categories=(0, 1, 2), learning_rate=0.5)

    state.update(weighted_observations=[(2, 1.0), (1, 0.0)])

    assert state.probabilities[2] > state.probabilities[0]
    assert state.probabilities[2] > state.probabilities[1]
    assert sum(state.probabilities.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run mixed-variable tests and confirm failure**

Run:

```powershell
pytest tests/unit/test_mixed_cma_vnext.py -v
```

Expected: FAIL because `evocore.mixed_cma` does not exist.

- [ ] **Step 3: Implement mixed-variable foundation module**

Create `evocore/mixed_cma.py` with this content:

```python
"""Mixed-variable CMA foundations for vNext."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from evocore.exceptions import ConfigurationError


@dataclass(frozen=True)
class IntegerMargin:
    """Convert continuous integer samples into margin-protected probabilities."""

    low: int
    high: int
    min_probability: float = 0.02

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ConfigurationError("IntegerMargin requires low <= high.")
        if not (0.0 < self.min_probability < 1.0):
            raise ConfigurationError("IntegerMargin min_probability must be in (0, 1).")

    def probabilities(self, *, mean: float, sigma: float) -> dict[int, float]:
        """Return normalized integer probabilities with a minimum margin."""
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ConfigurationError("IntegerMargin sigma must be finite and > 0.")
        raw: dict[int, float] = {}
        for value in range(self.low, self.high + 1):
            z = (float(value) - float(mean)) / sigma
            raw[value] = math.exp(-0.5 * z * z)
        total = sum(raw.values())
        probabilities = {key: value / total for key, value in raw.items()}
        categories = len(probabilities)
        floor_total = self.min_probability * categories
        if floor_total >= 1.0:
            raise ConfigurationError("IntegerMargin min_probability is too large for range.")
        adjusted = {
            key: self.min_probability + (1.0 - floor_total) * value
            for key, value in probabilities.items()
        }
        adjusted_total = sum(adjusted.values())
        return {key: value / adjusted_total for key, value in adjusted.items()}


@dataclass
class CategoricalState:
    """Maintain a categorical distribution for mixed-variable CMA."""

    categories: Sequence[int]
    learning_rate: float = 0.20
    probabilities: dict[int, float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.categories:
            raise ConfigurationError("CategoricalState requires at least one category.")
        if len(set(self.categories)) != len(self.categories):
            raise ConfigurationError("CategoricalState categories must be unique.")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ConfigurationError("CategoricalState learning_rate must be in (0, 1].")
        probability = 1.0 / len(self.categories)
        self.probabilities = {category: probability for category in self.categories}

    def update(self, *, weighted_observations: list[tuple[int, float]]) -> None:
        """Move probability mass toward weighted observed categories."""
        target = {category: 0.0 for category in self.categories}
        for category, weight in weighted_observations:
            if category not in target:
                raise ConfigurationError(f"unknown category: {category!r}")
            target[category] += max(float(weight), 0.0)
        total = sum(target.values())
        if total <= 0.0:
            return
        target = {category: value / total for category, value in target.items()}
        for category in self.categories:
            self.probabilities[category] = (
                (1.0 - self.learning_rate) * self.probabilities[category]
                + self.learning_rate * target[category]
            )
        normalizer = sum(self.probabilities.values())
        self.probabilities = {
            category: value / normalizer for category, value in self.probabilities.items()
        }
```

- [ ] **Step 4: Export mixed-variable APIs**

Modify `evocore/__init__.py`:

```python
from evocore.mixed_cma import CategoricalState, IntegerMargin
```

Add `"CategoricalState"` and `"IntegerMargin"` to `__all__`.

- [ ] **Step 5: Run mixed-variable tests**

Run:

```powershell
pytest tests/unit/test_mixed_cma_vnext.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit mixed-variable foundations**

```powershell
git add evocore/mixed_cma.py evocore/__init__.py tests/unit/test_mixed_cma_vnext.py
git commit -m "feat: add mixed variable cma foundations"
```

---

## Part 7: Release, Docs, Examples, And Verification

### Task 7.1: Bump Version And Changelog For vNext

**Files:**
- Modify: `pyproject.toml`
- Modify: `Cargo.toml`
- Modify: `evocore/__init__.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update Python package version**

In `pyproject.toml`, replace:

```toml
version = "0.6.1"
```

with:

```toml
version = "0.7.0"
```

- [ ] **Step 2: Update Cargo version**

In `Cargo.toml`, replace:

```toml
version = "0.6.1"
```

with:

```toml
version = "0.7.0"
```

- [ ] **Step 3: Update development fallback version**

In `evocore/__init__.py`, replace:

```python
__version__ = "0.6.1"
```

with:

```python
__version__ = "0.7.0"
```

- [ ] **Step 4: Add changelog section**

In `CHANGELOG.md`, under `## [Unreleased]`, add:

```markdown
## [0.7.0] - 2026-05-07

### Breaking

- Reoriented EvoCore around vNext expensive black-box optimization rather than DEAP parity.
- Replaced GA execution with ask/tell and policy-driven multi-fidelity semantics.
- Added vNext CMA ask/tell semantics for trusted-record distribution updates.

### Added

- Candidate, rung, evaluation record, and optimizer telemetry APIs.
- Multi-fidelity policy and scheduler primitives.
- Deterministic Rust candidate ID and confidence-aware ranking helpers.
- Baseline surrogate advisor and audit-aware promotion support.
- Mixed-variable CMA foundation types for integer margins and categorical state.
- vNext docs and examples for budget-aware optimization.
```

- [ ] **Step 5: Run version smoke**

Run:

```powershell
python -c "import evocore; print(evocore.__version__)"
```

Expected: prints `0.7.0` in editable/development context.

- [ ] **Step 6: Commit release metadata**

```powershell
git add pyproject.toml Cargo.toml evocore/__init__.py CHANGELOG.md
git commit -m "chore: prepare vnext version metadata"
```

### Task 7.2: Update Public Docs And MkDocs Navigation

**Files:**
- Modify: `mkdocs.yml`
- Modify: `docs/site/api.md`
- Create: `docs/site/budget-aware-optimization.md`
- Create: `docs/site/ask-tell-engines.md`
- Create: `docs/site/mixed-variable-search.md`
- Create: `docs/site/optimizer-telemetry.md`

- [ ] **Step 1: Create budget-aware optimization page**

Create `docs/site/budget-aware-optimization.md`:

```markdown
# Budget-Aware Optimization

EvoCore vNext treats full fitness calls as scarce resources.

Use `MultiFidelityPolicy` and `Rung` to describe cheap, partial, and full evaluation
levels. Engines ask for candidates, schedulers assign rungs, evaluators return
`EvaluationRecord` objects, and engines update state through `tell()`.

```python
from evocore import MultiFidelityPolicy, Rung

policy = MultiFidelityPolicy(
    rungs=[
        Rung("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        Rung("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    full_evaluation_budget=64,
    batch_size=16,
)
```
```

- [ ] **Step 2: Create ask/tell page**

Create `docs/site/ask-tell-engines.md`:

```markdown
# Ask/Tell Engines

`GAEngine` and `CMAESEngine` expose ask/tell state.

`ask()` returns candidates with stable IDs and decoded params. `tell()` accepts
`EvaluationRecord` values and updates engine state only according to each record's
confidence level.

Trusted full records update optimizer state by default. Surrogate and partial records are
used for scheduling, screening, and telemetry unless a policy explicitly allows aggressive
state updates.
```

- [ ] **Step 3: Create mixed-variable page**

Create `docs/site/mixed-variable-search.md`:

```markdown
# Mixed-Variable Search

EvoCore vNext starts moving beyond continuous-only CMA by separating continuous, integer,
categorical, and fixed-gene behavior.

`IntegerMargin` protects integer probability mass so integer genes do not collapse too
quickly. `CategoricalState` tracks categorical-by-integer probability updates for future
mixed CMA engines.
```

- [ ] **Step 4: Create telemetry page**

Create `docs/site/optimizer-telemetry.md`:

```markdown
# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Telemetry includes proposed candidates, screened candidates, partial evaluations, full
evaluations, promoted and eliminated counts by rung, and cost by rung. Trading systems can
use this evidence for anti-overfitting workflows such as Deflated Sharpe Ratio, White's
Reality Check, Hansen SPA, and Model Confidence Set.
```

- [ ] **Step 5: Update MkDocs navigation**

In `mkdocs.yml`, replace the GA parity nav entry:

```yaml
  - GA Benchmark Parity: ga-benchmark-parity.md
```

with these vNext entries:

```yaml
  - Budget-Aware Optimization: budget-aware-optimization.md
  - Ask/Tell Engines: ask-tell-engines.md
  - Mixed-Variable Search: mixed-variable-search.md
  - Optimizer Telemetry: optimizer-telemetry.md
```

- [ ] **Step 6: Update API page**

Append this to `docs/site/api.md`:

```markdown
## vNext Expensive Optimization

::: evocore.evaluation.Candidate

::: evocore.evaluation.EvaluationRecord

::: evocore.evaluation.Rung

::: evocore.evaluation.OptimizationTelemetry

::: evocore.policies.MultiFidelityPolicy

::: evocore.scheduler.EvaluationScheduler

::: evocore.advisors.InverseDistanceSurrogateAdvisor

::: evocore.mixed_cma.IntegerMargin

::: evocore.mixed_cma.CategoricalState
```

- [ ] **Step 7: Build docs**

Run:

```powershell
python -m mkdocs build
```

Expected: PASS and generated site without missing nav pages.

- [ ] **Step 8: Commit docs**

```powershell
git add mkdocs.yml docs/site/api.md docs/site/budget-aware-optimization.md docs/site/ask-tell-engines.md docs/site/mixed-variable-search.md docs/site/optimizer-telemetry.md
git commit -m "docs: add vnext optimizer documentation"
```

### Task 7.3: Add vNext Example And Benchmark Smoke

**Files:**
- Create: `examples/vnext_budgeted_ga.py`
- Create: `tests/benchmarks/bench_vnext_multifidelity.py`

- [ ] **Step 1: Create runnable example**

Create `examples/vnext_budgeted_ga.py`:

```python
"""Budget-aware EvoCore vNext GA example."""

from __future__ import annotations

from evocore import (
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneDef,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)


class TwoRungSphere(Evaluator):
    def evaluate(self, candidates, rung):
        scale = 0.5 if rung.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
                metrics={"rung": rung.name},
            )
            for candidate in candidates
        ]


def main() -> None:
    space = GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("y", "float", -5.0, 5.0),
        ]
    )
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            Rung("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        full_evaluation_budget=32,
        batch_size=8,
        audit_fraction=0.10,
    )
    result = GAEngine(space, population_size=8, generations=20, seed=42).run(
        TwoRungSphere(),
        policy=policy,
    )
    print(f"best={result.best_fitness:.6f}")
    print(f"full_evals={result.telemetry.candidates_full_evaluated}")
    print(f"partial_evals={result.telemetry.candidates_partial_evaluated}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add benchmark smoke**

Create `tests/benchmarks/bench_vnext_multifidelity.py`:

```python
from evocore import (
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)


class DeceptiveSphere(Evaluator):
    def evaluate(self, candidates, rung):
        records = []
        for candidate in candidates:
            true_score = -sum(float(value) ** 2 for value in candidate.genes)
            cheap_score = true_score + (0.1 if candidate.candidate_id.endswith("0") else 0.0)
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    score=cheap_score if rung.name == "cheap" else true_score,
                    confidence=rung.confidence,
                    rung=rung.name,
                    cost=rung.budget,
                )
            )
        return records


def test_vnext_multifidelity_benchmark_smoke() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=16,
        batch_size=8,
        audit_fraction=0.25,
    )
    result = GAEngine(GeneSpace.uniform(-5.0, 5.0, 3), population_size=8, seed=11).run(
        DeceptiveSphere(),
        policy=policy,
    )

    assert result.telemetry.candidates_full_evaluated == 16
    assert result.telemetry.candidates_partial_evaluated >= 16
    assert result.best_individual.fitness_valid
```

- [ ] **Step 3: Run example and benchmark smoke**

Run:

```powershell
python examples/vnext_budgeted_ga.py
pytest tests/benchmarks/bench_vnext_multifidelity.py -v
```

Expected: example prints `best=`, `full_evals=32`, and `partial_evals=`; benchmark smoke PASS.

- [ ] **Step 4: Commit example and benchmark**

```powershell
git add examples/vnext_budgeted_ga.py tests/benchmarks/bench_vnext_multifidelity.py
git commit -m "docs: add vnext budgeted ga example"
```

### Task 7.4: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run formatting checks**

Run:

```powershell
ruff format --check
ruff check --select ALL
cargo fmt --check
```

Expected: all commands PASS.

- [ ] **Step 2: Run Rust checks**

Run:

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
```

Expected: all commands PASS.

- [ ] **Step 3: Rebuild extension and run Python tests**

Run:

```powershell
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

Expected: all tests PASS. If old DEAP-parity assertions fail, rewrite them to vNext semantics or remove them when they only assert historical parity behavior.

- [ ] **Step 4: Run docs and example smoke**

Run:

```powershell
python -m mkdocs build
python examples/vnext_budgeted_ga.py
```

Expected: docs build PASS; example prints best fitness and telemetry counts.

- [ ] **Step 5: Check diff hygiene**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional uncommitted changes before the final commit.

- [ ] **Step 6: Commit verification fixes**

If verification required fixes, commit them:

```powershell
git add .
git commit -m "test: verify vnext optimizer rollout"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage: Parts 1-7 cover candidate state, evaluation records, ask/tell GA, ask/tell CMA, scheduler, advisor, mixed-variable foundations, anti-overfit telemetry, Trading-Algo reference shape, version bump, changelog, MkDocs, examples, docstrings, and final verification.
- Scope control: Mixed-variable CMA is delivered as foundation types and ask/tell CMA seam in this plan, not a full CatCMAwM implementation in one branch.
- Breaking stance: The plan rewrites GA execution and demotes DEAP parity as a product constraint.
- Test strategy: Every part starts with failing tests, implements the smallest useful code, verifies passing tests, and commits.
- Release hygiene: Version, changelog, MkDocs, API docs, README direction, examples, and type stubs are included.
