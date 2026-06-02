# Differential Evolution Feature Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Differential Evolution feature parity for deterministic multi-run execution, policy-driven `run(evaluator, policy=None)`, and supporting docs/examples while leaving new DE strategies for future work.

**Architecture:** Keep DE implementation in `evocore/optimizers/de/`. Add a focused `multi_run.py` mixin for child-seed execution and aggregation, and update `engine.py` to resolve `BudgetPolicy`, validate evaluator output, run policy stages, close screened-out candidates with rejected records, and build policy-aware results.

**Tech Stack:** Python 3, EvoCore lifecycle primitives (`BudgetPolicy`, `BudgetScheduler`, `EvaluationStage`, `EvaluationRecord`), PyO3-backed `_core` seed derivation, pytest, ruff, maturin.

---

## File Map

- Create `evocore/optimizers/de/multi_run.py`: DE-specific `_copy_with_seed(...)`, `run_child_optimizer(...)`, and `run_multiple(...)`.
- Modify `evocore/optimizers/de/engine.py`: inherit the multi-run mixin, resolve policies, validate evaluator records, evaluate staged policy batches, close screened-out candidates, and rebuild `run(...)` around policy orchestration.
- Modify `evocore/optimizers/de/__init__.py`: no export change expected; keep exporting `DifferentialEvolutionOptimizer`.
- Modify `tests/unit/test_de_engine.py`: add policy-run tests and expand existing `max_evaluations` expectations.
- Create `tests/unit/test_de_multi_run.py`: focused tests for DE multi-run behavior.
- Modify `tests/integration/test_de_mixed_gene_space.py`: add a budget-aware mixed-space DE integration test.
- Create `examples/budgeted_de.py`: two-stage budget-aware DE example.
- Modify `docs/site/de.md`: document `run_multiple(...)`, policy-driven run, callbacks, and remaining strategy limitations.
- Modify `docs/site/budget-aware-optimization.md`: list DE as policy-driven, keep CMA-ES wording accurate.
- Modify `docs/site/callbacks-checkpointing.md`: clarify manual DE checkpoints and unsupported policy-run mid-loop resume.
- Modify `CHANGELOG.md`: add a user-visible DE feature-parity entry.

## Branch And Setup

Use the repository-local virtual environment for Python commands.

- [ ] **Step 1: Start from an implementation branch**

Run:

```powershell
git status --short --branch
```

Expected: either a clean task branch or `main`.

If on `main`, run:

```powershell
git switch -c feature/de-feature-parity
```

Expected: `Switched to a new branch 'feature/de-feature-parity'`.

- [ ] **Step 2: Confirm the design spec is available**

Run:

```powershell
Test-Path docs\superpowers\specs\2026-06-02-differential-evolution-feature-parity-design.md
```

Expected: `True`.

---

## Task 1: Add Failing Multi-Run Tests

**Files:**
- Create: `tests/unit/test_de_multi_run.py`

- [ ] **Step 1: Create DE multi-run tests**

Add this file:

```python
import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    GeneSpace,
    OptimizationBatchResult,
    _core,
)
from evocore.core.errors import ConfigurationError


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


def _optimizer(seed: int = 42) -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=seed,
    )


def test_de_run_multiple_returns_sorted_batch_result() -> None:
    result = _optimizer().run_multiple(SphereEvaluator(), n_runs=4)

    assert isinstance(result, OptimizationBatchResult)
    assert result.n_runs == 4
    assert result.best is result.all_runs[0]
    assert result.direction == "maximize"
    assert [run.best_score for run in result.all_runs] == sorted(
        [run.best_score for run in result.all_runs],
        reverse=True,
    )


def test_de_run_multiple_uses_deterministic_child_seeds() -> None:
    first = _optimizer(seed=7).run_multiple(SphereEvaluator(), n_runs=3)
    second = _optimizer(seed=7).run_multiple(SphereEvaluator(), n_runs=3)
    expected_seeds = {
        int(_core.py_derive_seed(7, 0, run_idx, _core.OP_MULTI_RUN))
        for run_idx in range(3)
    }

    assert {run.seed for run in first.all_runs} == expected_seeds
    assert [run.seed for run in first.all_runs] == [run.seed for run in second.all_runs]
    assert [run.best_score for run in first.all_runs] == pytest.approx(
        [run.best_score for run in second.all_runs]
    )


def test_de_run_multiple_rejects_invalid_arguments() -> None:
    engine = _optimizer()

    with pytest.raises(ConfigurationError, match="n_runs must be positive"):
        engine.run_multiple(SphereEvaluator(), n_runs=0)
    with pytest.raises(ConfigurationError, match="aggregate must be 'best' or 'all'"):
        engine.run_multiple(SphereEvaluator(), aggregate="median")


def test_de_run_multiple_parallel_requires_picklable_evaluator() -> None:
    class NestedEvaluator:
        def evaluate(self, candidates, context):
            return SphereEvaluator().evaluate(candidates, context)

    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        _optimizer().run_multiple(NestedEvaluator(), n_runs=2, run_parallel=True)
```

- [ ] **Step 2: Run the failing multi-run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_multi_run.py -v
```

Expected before implementation: failures mentioning `DifferentialEvolutionOptimizer` has no attribute `run_multiple`.

---

## Task 2: Implement DE Multi-Run Mixin

**Files:**
- Create: `evocore/optimizers/de/multi_run.py`
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_multi_run.py`

- [ ] **Step 1: Add the multi-run module**

Create `evocore/optimizers/de/multi_run.py`:

```python
from __future__ import annotations

import copy
import logging
import os
import time
from typing import TYPE_CHECKING, Protocol

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.core.parallel import ensure_picklable
from evocore.lifecycle import Evaluator, score_for_direction
from evocore.results import OptimizationBatchResult, OptimizationResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer


class _ChildOptimizer(Protocol):
    def _copy_with_seed(self, seed: int) -> _ChildOptimizer:
        raise NotImplementedError

    def run(self, evaluator: Evaluator) -> OptimizationResult:
        raise NotImplementedError


def run_child_optimizer(
    engine: _ChildOptimizer,
    seed: int,
    evaluator: Evaluator,
) -> OptimizationResult:
    """Run one child optimizer with a derived seed for process-pool execution."""
    return engine._copy_with_seed(seed).run(evaluator)


class DifferentialEvolutionMultiRunMixin:
    """Seed derivation, optimizer copying, and multi-run execution for DE."""

    def _copy_with_seed(self, seed: int) -> DifferentialEvolutionOptimizer:
        from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer

        return DifferentialEvolutionOptimizer(
            gene_space=self.gene_space,
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
            strategy=self.strategy,
            parallel=self.parallel,
            n_workers=self.n_workers,
            process_initializer=self.process_initializer,
            process_initargs=self.process_initargs,
            seed=int(seed),
            direction=self.direction,
            max_evaluations=self.max_evaluations,
            track_diversity=self.track_diversity,
            callbacks=copy.deepcopy(self.callbacks),
        )

    def run_multiple(
        self,
        evaluator: Evaluator,
        n_runs: int = 10,
        aggregate: str = "best",
        run_parallel: bool = False,
    ) -> OptimizationBatchResult:
        """Run multiple deterministic child DE runs from derived seeds."""
        if n_runs <= 0:
            raise ConfigurationError("n_runs must be positive.")
        if aggregate not in ("best", "all"):
            raise ConfigurationError("aggregate must be 'best' or 'all'.")

        child_seeds = [
            int(_core.py_derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
            for run_idx in range(n_runs)
        ]
        logger.debug("DE run_multiple n_runs=%s child_seeds=%s", n_runs, child_seeds)

        started = time.perf_counter()
        if run_parallel:
            ensure_picklable(evaluator, context="run_multiple(run_parallel=True)")
            ensure_picklable(self, context="run_multiple(run_parallel=True) engine")

            import concurrent.futures
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_runs, self.n_workers or os.cpu_count() or 1),
                mp_context=ctx,
            )
            try:
                futures = [
                    pool.submit(run_child_optimizer, self, seed, evaluator)
                    for seed in child_seeds
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            finally:
                pool.shutdown(cancel_futures=True, wait=False)
        else:
            results = [self._copy_with_seed(seed).run(evaluator) for seed in child_seeds]

        results.sort(
            key=lambda run: score_for_direction(run.best_score, self.direction),
            reverse=True,
        )
        return OptimizationBatchResult(
            best=results[0],
            all_runs=results,
            n_runs=n_runs,
            wall_time_seconds=time.perf_counter() - started,
            direction=self.direction,
        )


__all__ = ["DifferentialEvolutionMultiRunMixin", "run_child_optimizer"]
```

- [ ] **Step 2: Wire the mixin into the optimizer class**

In `evocore/optimizers/de/engine.py`, add the import:

```python
from evocore.optimizers.de.multi_run import DifferentialEvolutionMultiRunMixin
```

Change the class inheritance to:

```python
class DifferentialEvolutionOptimizer(
    DifferentialEvolutionCheckpointingMixin,
    DifferentialEvolutionAskTellMixin,
    DifferentialEvolutionMultiRunMixin,
):
    """Run Differential Evolution over a flat EvoCore GeneSpace."""
```

- [ ] **Step 3: Run multi-run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_multi_run.py -v
```

Expected: all tests in `tests/unit/test_de_multi_run.py` pass.

- [ ] **Step 4: Commit multi-run support**

Run:

```powershell
git add evocore/optimizers/de/engine.py evocore/optimizers/de/multi_run.py tests/unit/test_de_multi_run.py
git commit -m "feat(de): add deterministic multi-run execution"
```

Expected: commit succeeds with only the listed files staged.

---

## Task 3: Add Failing Policy-Run Tests

**Files:**
- Modify: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Extend imports in `tests/unit/test_de_engine.py`**

Change the top import block to include policy primitives and `FitnessError`:

```python
from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    Gene,
    GeneSpace,
)
from evocore.callbacks import Callback
from evocore.core.errors import ConfigurationError, FitnessError
```

- [ ] **Step 2: Add policy helper classes and functions**

Add these helpers after `SphereEvaluator`:

```python
def _two_stage_policy(max_evaluations: int = 12, batch_size: int = 6) -> BudgetPolicy:
    return BudgetPolicy(
        stages=[
            EvaluationStage(
                "cheap",
                budget=0.10,
                promote_fraction=0.50,
                confidence="partial",
            ),
            EvaluationStage(
                "full",
                budget=1.00,
                promote_fraction=1.00,
                confidence="trusted_full",
            ),
        ],
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )


class TwoStageSphereEvaluator:
    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, int]] = []

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        stage_name = context.stage.name
        self.stage_calls.append((stage_name, len(candidates)))
        scale = 0.25 if stage_name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale
                * sum(
                    float(value) ** 2
                    for value in candidate.genes
                    if type(value) is not bool
                ),
                confidence=context.stage.confidence,
                stage=stage_name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CachedFinalEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        confidence = "cached" if context.stage.name == "full" else context.stage.confidence
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(
                    float(value) ** 2
                    for value in candidate.genes
                    if type(value) is not bool
                ),
                confidence=confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]
```

- [ ] **Step 3: Add policy-run tests**

Append these tests before `test_de_public_checkpoint_example_smoke`:

```python
def test_de_run_accepts_explicit_single_full_policy() -> None:
    policy = BudgetPolicy.single_full(max_evaluations=8, batch_size=4)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        seed=42,
    )

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations == 8
    assert result.max_evaluations == 8
    assert result.telemetry.candidates_full_evaluated == 8


def test_de_run_prefers_explicit_policy_over_constructor_max_evaluations() -> None:
    policy = BudgetPolicy.single_full(max_evaluations=9, batch_size=3)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        max_evaluations=4,
        seed=42,
    )

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.max_evaluations == 9
    assert result.n_evaluations == 9


def test_de_run_rejects_non_policy_argument() -> None:
    engine = DifferentialEvolutionOptimizer(GeneSpace.uniform(-2.0, 2.0, 3), seed=42)

    with pytest.raises(ConfigurationError, match="policy must be a BudgetPolicy"):
        engine.run(SphereEvaluator(), policy=object())


def test_de_run_two_stage_policy_screens_and_closes_batches() -> None:
    evaluator = TwoStageSphereEvaluator()
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(evaluator, policy=_two_stage_policy(max_evaluations=9, batch_size=6))

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations == 9
    assert result.telemetry.candidates_partial_evaluated > 0
    assert result.telemetry.candidates_full_evaluated == 9
    assert engine.state_summary().pending_batch_ids == ()
    assert any(stage_name == "cheap" for stage_name, _ in evaluator.stage_calls)
    assert any(stage_name == "full" for stage_name, _ in evaluator.stage_calls)


def test_de_run_cached_final_records_update_state_without_spending_fresh_budget() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(
        CachedFinalEvaluator(),
        policy=_two_stage_policy(max_evaluations=6, batch_size=6),
    )

    assert result.n_evaluations == 0
    assert result.telemetry.candidates_cached > 0
    assert result.best_candidate_id is not None
    assert len(result.final_solutions) == 6


class MissingRecordEvaluator:
    def evaluate(self, candidates, context):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=1.0,
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in list(candidates)[:-1]
        ]


def test_de_run_rejects_missing_evaluator_records() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    with pytest.raises(FitnessError, match="missing evaluation records"):
        engine.run(MissingRecordEvaluator(), policy=BudgetPolicy.single_full(max_evaluations=6))
```

- [ ] **Step 4: Run the failing policy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py -k "policy or cached or missing_evaluator or honors_max_evaluations" -v
```

Expected before implementation: failures from unsupported policy handling and missing evaluator validation.

---

## Task 4: Add Policy Helper Methods To DE Engine

**Files:**
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Update imports in `engine.py`**

At the top of `evocore/optimizers/de/engine.py`, add `Counter` and `Solution` imports and lifecycle policy imports:

```python
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any
```

Update the lifecycle import list to include:

```python
    BudgetPolicy,
    BudgetScheduler,
    is_state_update_confidence,
```

Update the search-space import:

```python
from evocore.search_space import GeneSpace, Solution, SolutionSet
```

- [ ] **Step 2: Add `_resolve_policy(...)`**

Add this method before `_evaluate_candidates(...)`:

```python
    def _resolve_policy(self, policy: BudgetPolicy | None) -> BudgetPolicy:
        """Resolve explicit or constructor shorthand budget settings."""
        if policy is not None and not isinstance(policy, BudgetPolicy):
            raise ConfigurationError("policy must be a BudgetPolicy when provided.")
        if policy is not None:
            return policy
        max_evaluations = self.max_evaluations
        if max_evaluations is None:
            max_evaluations = max(1, self.population_size * (self.max_generations + 1))
        return BudgetPolicy.single_full(
            max_evaluations=max_evaluations,
            batch_size=self.population_size,
        )
```

- [ ] **Step 3: Add evaluator record validation**

Add this method after `_evaluation_context(...)`:

```python
    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        """Reject incomplete or mismatched synchronous evaluator results."""
        expected_ids = [candidate.candidate_id for candidate in assigned]
        returned_ids = [record.candidate_id for record in records]
        expected_counts = Counter(expected_ids)
        returned_counts = Counter(returned_ids)

        missing_ids = [
            candidate_id for candidate_id in expected_ids if returned_counts[candidate_id] == 0
        ]
        unexpected_ids = [
            candidate_id for candidate_id in returned_counts if candidate_id not in expected_counts
        ]
        duplicate_ids = [
            candidate_id
            for candidate_id, count in returned_counts.items()
            if count > expected_counts[candidate_id]
        ]

        if missing_ids:
            raise FitnessError(
                "Evaluator returned missing evaluation records for candidate_ids: "
                f"{sorted(set(missing_ids))!r}."
            )
        if unexpected_ids:
            raise FitnessError(
                "Evaluator returned unknown evaluation records for candidate_ids: "
                f"{sorted(unexpected_ids)!r}."
            )
        if duplicate_ids:
            raise FitnessError(
                "Evaluator returned duplicate evaluation records for candidate_ids: "
                f"{sorted(duplicate_ids)!r}."
            )

        batch_ids = {candidate.batch_id for candidate in assigned}
        if len(batch_ids) != 1:
            raise FitnessError("DE run candidates must belong to exactly one batch.")
        expected_batch_id = next(iter(batch_ids))
        for record in records:
            if record.batch_id is not None and record.batch_id != expected_batch_id:
                raise FitnessError(
                    f"Evaluator returned record batch_id {record.batch_id!r} for batch "
                    f"{expected_batch_id!r}."
                )
```

- [ ] **Step 4: Add screened-out record helpers**

Add these methods after `_validate_evaluator_records(...)`:

```python
    def _candidate_has_terminal_record(self, candidate: Candidate) -> bool:
        batch = self._batches_by_id[candidate.batch_id]
        return any(
            record.candidate_id == candidate.candidate_id
            and (is_state_update_confidence(record.confidence) or record.confidence == "rejected")
            for record in batch.records_by_key.values()
        )

    def _screened_out_records(
        self,
        candidates: Sequence[Candidate],
        *,
        completed_stage: str,
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        synthetic_stage = f"{completed_stage}__de_screened_out"
        for candidate in candidates:
            if self._candidate_has_terminal_record(candidate):
                continue
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=None,
                    confidence="rejected",
                    stage=synthetic_stage,
                    cost=0.0,
                    metadata={
                        "reason": "not_promoted",
                        "completed_stage": completed_stage,
                        "target_candidate_id": candidate.metadata.get("target_candidate_id"),
                        "target_slot": candidate.metadata.get("target_slot"),
                    },
                )
            )
        return records

    def _reject_screened_out(
        self,
        candidates: Sequence[Candidate],
        *,
        completed_stage: str,
    ) -> None:
        records = self._screened_out_records(candidates, completed_stage=completed_stage)
        if records:
            self.tell(records)
```

- [ ] **Step 5: Run focused helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_run_rejects_non_policy_argument tests/unit/test_de_engine.py::test_de_run_rejects_missing_evaluator_records -v
```

Expected after implementation in this task: non-policy argument passes, missing evaluator record may still fail for the unsupported policy path until `run(...)` is rewritten in the next task.

Do not commit this task separately if the missing-record test still fails because `run(...)` has not been rewired.

---

## Task 5: Rewrite DE `run(...)` Around BudgetPolicy

**Files:**
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add a policy stage evaluator helper**

Add this method before `run(...)`:

```python
    def _evaluate_policy_stages(
        self,
        candidates: Sequence[Candidate],
        evaluator: Evaluator,
        scheduler: BudgetScheduler,
        policy: BudgetPolicy,
    ) -> tuple[int, list[Candidate], list[EvaluationRecord]]:
        active_candidates = list(candidates)
        for stage in policy.stages:
            if not active_candidates:
                return 0, [], []
            assigned = scheduler.assign_stage(active_candidates, stage_name=stage.name)
            context = self._evaluation_context(assigned, stage)
            records = self._evaluate_candidates(assigned, evaluator, context)
            self._validate_evaluator_records(assigned, records)
            self.tell(records)

            if stage.confidence == "trusted_full":
                state_eligible_count = sum(
                    1 for record in records if is_state_update_confidence(record.confidence)
                )
                fresh_full_count = sum(
                    1 for record in records if record.confidence == "trusted_full"
                )
                if state_eligible_count == 0:
                    raise FitnessError(
                        "Evaluator returned no state-eligible records for the final stage; "
                        "trusted_full or cached records are required."
                    )
                return fresh_full_count, list(assigned), list(records)

            promoted = scheduler.promote(assigned, completed_stage=stage.name)
            promoted_ids = {candidate.candidate_id for candidate in promoted}
            screened_out = [
                candidate for candidate in assigned if candidate.candidate_id not in promoted_ids
            ]
            self._reject_screened_out(screened_out, completed_stage=stage.name)
            active_candidates = promoted

        return 0, [], []
```

- [ ] **Step 2: Add a result builder helper**

Add this method before `run(...)`:

```python
    def _build_run_result(
        self,
        *,
        started: float,
        generation_history: GenerationHistory,
        diversity_history: list[list[float]],
        elite_history: list[Solution],
        n_evaluations: int,
        stop_reason: StopReason,
        max_evaluations: int,
    ) -> OptimizationResult:
        final_solutions = self._target_solutions()
        if len(final_solutions):
            best_solution = final_solutions.best(1)[0].clone()
            best_score = float(best_solution.score)
        else:
            best_solution = Solution([], score=float("-inf"), score_valid=False)
            final_solutions = SolutionSet([best_solution])
            best_score = float("-inf")

        result = OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - started,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_history,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=max_evaluations,
            telemetry=self.vnext_telemetry,
            direction=self.direction,
            optimizer_type="DifferentialEvolutionOptimizer",
            best_candidate_id=self.best_candidate.candidate_id if self.best_candidate else None,
            events=self.events,
            reproducibility=self._reproducibility_metadata(),
        )
        append_run_stop_event(
            result.events,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result
```

- [ ] **Step 3: Replace `run(...)` with policy orchestration**

Replace the existing `run(...)` body with:

```python
    def run(
        self,
        evaluator: Evaluator,
        policy: BudgetPolicy | None = None,
    ) -> OptimizationResult:  # noqa: PLR0912, PLR0915
        """Run one synchronous policy-driven DE optimization."""
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run requires an evaluator with evaluate(candidates, context)."
            )
        resolved_policy = self._resolve_policy(policy)
        scheduler = BudgetScheduler(resolved_policy)
        self._reset_vnext_state()
        self._bind_callbacks()

        started = time.perf_counter()
        generation_history = GenerationHistory()
        diversity_history: list[list[float]] = []
        elite_history: list[Solution] = []
        n_evaluations = 0
        stop_reason: StopReason = "max_generations"

        while (
            len(self._target_candidate_ids) < self.population_size
            and self.vnext_telemetry.candidates_full_evaluated < resolved_policy.max_evaluations
        ):
            remaining = (
                resolved_policy.max_evaluations
                - self.vnext_telemetry.candidates_full_evaluated
            )
            batch_size = min(
                resolved_policy.batch_size or self.population_size,
                self.population_size - len(self._target_candidate_ids),
                remaining,
            )
            if batch_size <= 0:
                stop_reason = "max_evaluations"
                break
            candidates = self.ask(batch_size)
            fresh_count, _, _ = self._evaluate_policy_stages(
                candidates,
                evaluator,
                scheduler,
                resolved_policy,
            )
            n_evaluations += fresh_count

        if self.vnext_telemetry.candidates_full_evaluated >= resolved_policy.max_evaluations:
            stop_reason = "max_evaluations"

        for gen in range(self.max_generations):
            if stop_reason == "max_evaluations":
                break
            gen_start = time.perf_counter()
            current_solutions = self._target_solutions()
            for callback in self.callbacks:
                callback.on_generation_start(gen, current_solutions)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break

            remaining = (
                resolved_policy.max_evaluations
                - self.vnext_telemetry.candidates_full_evaluated
            )
            trial_count = min(
                self.population_size,
                resolved_policy.batch_size or self.population_size,
                remaining,
            )
            if trial_count <= 0:
                stop_reason = "max_evaluations"
                break

            trials = self.ask(trial_count)
            fresh_count, final_candidates, _ = self._evaluate_policy_stages(
                trials,
                evaluator,
                scheduler,
                resolved_policy,
            )
            n_evaluations += fresh_count
            self._append_generation_record(
                generation_history,
                gen=gen,
                gen_start=gen_start,
                n_evaluations=fresh_count,
            )

            solutions = self._target_solutions()
            diversity = solutions.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            if len(solutions):
                elite_history.append(solutions.best(1)[0].clone())
            info = GenerationInfo(gen, 0, 0)
            for callback in self.callbacks:
                callback.on_generation_end(gen, solutions, info)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break
            if (
                fresh_count > 0
                and self.vnext_telemetry.candidates_full_evaluated
                >= resolved_policy.max_evaluations
            ):
                stop_reason = "max_evaluations"
                break
            if not final_candidates:
                stop_reason = "max_evaluations"
                break

        return self._build_run_result(
            started=started,
            generation_history=generation_history,
            diversity_history=diversity_history,
            elite_history=elite_history,
            n_evaluations=n_evaluations,
            stop_reason=stop_reason,
            max_evaluations=resolved_policy.max_evaluations,
        )
```

- [ ] **Step 4: Run DE engine tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py -v
```

Expected: `tests/unit/test_de_engine.py` passes. If `test_de_run_returns_optimization_result_with_events_and_generations` reports `stop_reason` differences, keep result shape stable and update only assertions that explicitly cover the new policy-owned budget semantics.

- [ ] **Step 5: Run DE ask/tell tests for regression coverage**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py -v
```

Expected: all existing ask/tell tests pass.

- [ ] **Step 6: Commit policy-run engine support**

Run:

```powershell
git add evocore/optimizers/de/engine.py tests/unit/test_de_engine.py
git commit -m "feat(de): add policy-driven run"
```

Expected: commit succeeds with only the engine and DE engine tests staged.

---

## Task 6: Add Focused Screened-Trial Regression Test

**Files:**
- Modify: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add evaluator that rejects half the trial candidates before final stage**

Add this class near the other evaluator helpers:

```python
class HalfPromotionEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        if context.stage.name == "cheap":
            return [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(index),
                    confidence="partial",
                    stage="cheap",
                    cost=context.stage.budget,
                )
                for index, candidate in enumerate(candidates)
            ]
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=100.0,
                confidence="trusted_full",
                stage="full",
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]
```

- [ ] **Step 2: Add the screened-trial test**

Add this test:

```python
def test_de_policy_screened_out_trials_leave_targets_unchanged() -> None:
    policy = _two_stage_policy(max_evaluations=9, batch_size=6)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(HalfPromotionEvaluator(), policy=policy)
    rejected_trial_events = [
        event
        for event in result.events
        if event.event_type == "tell"
        and event.origin == "mutation"
        and event.confidence == "rejected"
        and event.metadata.get("reason") == "not_promoted"
    ]
    final_candidate_ids = {
        solution.metadata["candidate_id"] for solution in result.final_solutions
    }

    assert rejected_trial_events
    assert engine.state_summary().pending_batch_ids == ()
    for event in rejected_trial_events:
        assert event.metadata["target_candidate_id"] in final_candidate_ids
```

- [ ] **Step 3: Run the screened-trial test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_policy_screened_out_trials_leave_targets_unchanged -v
```

Expected: the test passes.

- [ ] **Step 4: Commit screened-trial regression coverage**

Run:

```powershell
git add tests/unit/test_de_engine.py
git commit -m "test(de): cover screened trial policy behavior"
```

Expected: commit succeeds with only `tests/unit/test_de_engine.py` staged.

---

## Task 7: Add Mixed-Space Integration Coverage

**Files:**
- Modify: `tests/integration/test_de_mixed_gene_space.py`

- [ ] **Step 1: Add imports for policy primitives**

At the top of `tests/integration/test_de_mixed_gene_space.py`, include:

```python
from evocore import BudgetPolicy, EvaluationStage
```

Keep existing imports for `DifferentialEvolutionOptimizer`, `EvaluationRecord`, `Gene`, and `GeneSpace`.

- [ ] **Step 2: Add a two-stage mixed evaluator**

Add this class near the existing evaluator helpers:

```python
class TwoStageMixedEvaluator:
    def evaluate(self, candidates, context):
        assert context.stage is not None
        scale = 0.5 if context.stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale
                * (
                    float(candidate.params["x"]) ** 2
                    + float(candidate.params["period"] - 8) ** 2
                    + (0.0 if candidate.params["enabled"] else 1.0)
                ),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]
```

- [ ] **Step 3: Add the integration test**

Add this test:

```python
def test_de_budgeted_run_supports_mixed_gene_space() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        max_evaluations=18,
        batch_size=6,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        max_generations=3,
        seed=7,
    )

    result = optimizer.run(TwoStageMixedEvaluator(), policy=policy)

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.n_evaluations == 18
    assert result.telemetry.candidates_partial_evaluated > 0
    assert result.telemetry.candidates_full_evaluated == 18
    assert len(result.final_solutions) == 6
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)
```

- [ ] **Step 4: Run the integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_de_mixed_gene_space.py -v
```

Expected: all tests in `tests/integration/test_de_mixed_gene_space.py` pass.

- [ ] **Step 5: Commit integration coverage**

Run:

```powershell
git add tests/integration/test_de_mixed_gene_space.py
git commit -m "test(de): cover budgeted mixed-space run"
```

Expected: commit succeeds with only the integration test file staged.

---

## Task 8: Add Budgeted DE Example

**Files:**
- Create: `examples/budgeted_de.py`

- [ ] **Step 1: Create the example**

Create `examples/budgeted_de.py`:

```python
"""Budget-aware Differential Evolution example."""

from __future__ import annotations

from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
    Gene,
    GeneSpace,
)


class TwoStageSphere:
    def evaluate(self, candidates, context):
        stage = context.stage
        if stage is None:
            raise ValueError("TwoStageSphere requires a scheduled stage.")
        scale = 0.5 if stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
                metrics={"stage": stage.name},
            )
            for candidate in candidates
        ]


def main() -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
            EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
        ],
        max_evaluations=32,
        batch_size=8,
        audit_fraction=0.10,
    )
    result = DifferentialEvolutionOptimizer(
        space,
        population_size=8,
        max_generations=20,
        seed=42,
    ).run(
        TwoStageSphere(),
        policy=policy,
    )
    print(f"best={result.best_score:.6f}")
    print(f"full_evals={result.telemetry.candidates_full_evaluated}")
    print(f"partial_evals={result.telemetry.candidates_partial_evaluated}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run:

```powershell
.\.venv\Scripts\python.exe examples\budgeted_de.py
```

Expected: output includes `best=`, `full_evals=32`, and `partial_evals=` with a positive integer.

- [ ] **Step 3: Commit the example**

Run:

```powershell
git add examples/budgeted_de.py
git commit -m "docs(de): add budgeted example"
```

Expected: commit succeeds with only `examples/budgeted_de.py` staged.

---

## Task 9: Update DE Documentation And Changelog

**Files:**
- Modify: `docs/site/de.md`
- Modify: `docs/site/budget-aware-optimization.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `docs/site/de.md`**

Add a `Budgeted Evaluation` section after the first synchronous run example:

```markdown
## Budgeted Evaluation

`DifferentialEvolutionOptimizer.run(evaluator, policy=...)` uses the same
`BudgetPolicy` and `EvaluationStage` vocabulary as GA. Non-final stages can
screen candidates, while DE target slots are initialized or replaced only after
final state-eligible `trusted_full` or `cached` records.

```python
from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
    GeneSpace,
)


class TwoStageSphere:
    def evaluate(self, candidates, context):
        stage = context.stage
        scale = 0.5 if stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


policy = BudgetPolicy(
    stages=[
        EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    max_evaluations=32,
    batch_size=8,
)

result = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 2),
    population_size=8,
    seed=42,
).run(TwoStageSphere(), policy=policy)
```
```

Add a `Multi-Run Execution` section before `Reproducibility`:

```markdown
## Multi-Run Execution

Use `run_multiple(...)` when you want deterministic child runs from one DE
configuration:

```python
batch = optimizer.run_multiple(TwoStageSphere(), n_runs=5)
best = batch.best
scores = [run.best_score for run in batch.all_runs]
```

Child seeds are derived from the optimizer seed and results are sorted
best-first using the optimizer direction.
```

Replace the `Current Limitations` paragraph with:

```markdown
DE does not yet expose custom strategy plugins or a Rust-backed variation
kernel. Those remain future feature and performance tracks. Policy-driven
mid-loop checkpoint resume is also outside checkpoint v1; use manual ask/tell
checkpoints when evaluation work must survive process restarts.
```

- [ ] **Step 2: Update `docs/site/budget-aware-optimization.md`**

Add this paragraph after the opening policy example:

```markdown
`GeneticAlgorithmOptimizer` and `DifferentialEvolutionOptimizer` can drive this
policy directly through `run(evaluator, policy=policy)`. For DE, non-final
stages screen candidates and final state-eligible records initialize or replace
target slots. CMA-ES remains manual ask/tell for policy-shaped external
evaluation in this release line.
```

- [ ] **Step 3: Update `docs/site/callbacks-checkpointing.md`**

In the unsupported checkpoint surfaces section, include DE policy-run wording:

```markdown
Policy-driven `run(evaluator, policy=...)` mid-loop resume is not part of
checkpoint v1 for GA or DE. `EventHistory` remains audit data and is not
replayed to rebuild optimizer state.
```

In the DE checkpoint section, add:

```markdown
For synchronous DE `run(...)`, callbacks can observe generation start, generation
end, and run end. Manual ask/tell checkpoints remain the stable DE resume path;
`CheckpointCallback` is not advertised as a DE policy-run resume mechanism.
```

- [ ] **Step 4: Update `CHANGELOG.md`**

Add this entry under the current unreleased or next-version section:

```markdown
- Added Differential Evolution feature parity planning for deterministic
  `run_multiple(...)`, policy-driven `run(evaluator, policy=...)`, budget-aware
  examples, and clearer DE checkpoint guidance.
```

If implementation changes are committed in the same PR, use this stronger entry:

```markdown
- Added `DifferentialEvolutionOptimizer.run_multiple(...)` and policy-driven
  `DifferentialEvolutionOptimizer.run(evaluator, policy=...)` with delayed
  target-slot replacement until final state-eligible budget stages.
```

- [ ] **Step 5: Run docs/example smoke checks**

Run:

```powershell
.\.venv\Scripts\python.exe examples\budgeted_de.py
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py tests/unit/test_de_multi_run.py -v
```

Expected: the example prints budget counts and the listed tests pass.

- [ ] **Step 6: Commit docs and changelog**

Run:

```powershell
git add docs/site/de.md docs/site/budget-aware-optimization.md docs/site/callbacks-checkpointing.md CHANGELOG.md
git commit -m "docs(de): document feature parity support"
```

Expected: commit succeeds with only docs and changelog staged.

---

## Task 10: Final Verification

**Files:**
- No source edits in this task.

- [ ] **Step 1: Run formatting check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: exit code 0.

- [ ] **Step 2: Run lint check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: exit code 0.

- [ ] **Step 3: Rebuild the extension**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: exit code 0 and the package is installed into `.venv`.

- [ ] **Step 4: Run unit and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: exit code 0 with all selected tests passing.

- [ ] **Step 5: Inspect final status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on the feature branch.

- [ ] **Step 6: Push and open draft PR**

Run:

```powershell
git push -u origin feature/de-feature-parity
gh pr create --draft --base main --head feature/de-feature-parity --title "feat(de): add feature parity" --body-file .github\pull_request_template.md
```

Expected: branch push succeeds and GitHub creates a draft pull request. Replace the generated PR body with a filled template that lists the verification commands and results.

## Self-Review Checklist

- Spec coverage: Tasks cover multi-run, policy resolution, multi-stage screening, delayed replacement, screened-out batch closure, callbacks, manual checkpoint docs, mixed-space integration, examples, changelog, and verification.
- Type consistency: The plan uses `BudgetPolicy`, `BudgetScheduler`, `EvaluationStage`, `EvaluationRecord`, `Evaluator`, `OptimizationBatchResult`, `Solution`, and `SolutionSet` with names already present in the codebase.
- Scope control: The plan keeps `strategy="rand1bin"` unchanged and does not touch Rust, PyO3 stubs, or CMA-ES.
