# Objective Evaluation Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the approved objective/evaluation semantics contract: finite raw scores for valid records, `score=None` for rejected records, cached observations that update state without spending fresh full-evaluation budget, and consistent docs/tests.

**Architecture:** Keep validation centralized in `evocore.evaluation.EvaluationRecord` and `OptimizationTelemetry`. Engines consume shared confidence semantics instead of duplicating score rules. GA policy-driven runs count fresh `trusted_full` records only; remaining legacy callable helpers become strict about non-finite values until the callable APIs are removed in a separate clean-break cleanup.

**Tech Stack:** Python 3.11+, dataclasses, pytest, Ruff, maturin/PyO3 extension build, MkDocs.

---

## Scope Check

The design spec is one contract layer, but it touches validation, telemetry, engine loops,
tests, docs, and changelog. This plan keeps those changes together because they must land
atomically for public semantics to be coherent.

Full deletion of legacy `fitness_fn` entrypoints is intentionally not part of this plan.
That cleanup crosses checkpoint/resume behavior and CMA convenience APIs. This plan removes
their non-finite sanitization behavior so any remaining callable path is strict instead of
contradicting the objective contract.

---

## File Structure

- Modify `evocore/evaluation.py`: require `rejected` records to use `score=None`; add cached telemetry accounting and deterministic export.
- Modify `evocore/ga.py`: record cached observations separately from full evaluations; count only fresh `trusted_full` records in policy-driven `run(...)`; make legacy non-finite callable helper behavior raise `FitnessError`.
- Modify `evocore/cmaes.py`: record cached observations separately from full evaluations; make legacy non-finite callable helper behavior raise `FitnessError`.
- Modify `tests/unit/test_vnext_evaluation.py`: cover strict rejected-record validation and cached telemetry export.
- Modify `tests/unit/test_ga_ask_tell_vnext.py`: cover GA cached state eligibility without full-budget consumption and mixed cached/fresh policy runs.
- Modify `tests/unit/test_cmaes_ask_tell_vnext.py`: cover CMA cached state eligibility without full-budget consumption.
- Modify `tests/unit/test_ga_engine.py`: replace non-finite sanitization expectations with strict `FitnessError` expectations.
- Modify `tests/unit/test_runtime_observability.py`: replace non-finite warning/log expectations with strict exception behavior.
- Modify `docs/site/ask-tell-engines.md`: document strict rejected records and cached budget semantics.
- Modify `docs/site/optimizer-telemetry.md`: document the new cached telemetry field and changed budget semantics.
- Modify `docs/site/ga.md`: document that GA budget accounting counts fresh `trusted_full` records only.
- Modify `docs/site/cmaes.md`: document cached state updates without fresh budget accounting.
- Modify `CHANGELOG.md`: record the public behavior changes.

---

### Task 0: Confirm Branch And Worktree

**Files:**
- Read-only: git worktree metadata

- [ ] **Step 1: Check branch and uncommitted files**

Run:

```powershell
git status --short --branch
```

Expected: on `feature/general-optimizer-framework` or another task branch, not `main`.
If unrelated uncommitted files exist, leave them untouched and stage only files listed in
this plan.

---

### Task 1: Add Evaluation Validation And Telemetry Tests

**Files:**
- Modify: `tests/unit/test_vnext_evaluation.py`

- [ ] **Step 1: Add failing tests for strict rejected records and cached telemetry**

Append these tests near the existing `EvaluationRecord` and telemetry tests in
`tests/unit/test_vnext_evaluation.py`:

```python
def test_rejected_record_rejects_score() -> None:
    with pytest.raises(FitnessError, match="rejected"):
        EvaluationRecord(
            candidate_id="bad",
            score=0.0,
            confidence="rejected",
            rung="full",
            cost=0.0,
            metadata={"reason": "constraint_violation"},
        )


def test_telemetry_records_cached_without_full_evaluation_count() -> None:
    telemetry = OptimizationTelemetry()

    telemetry.record_cached(2, rung="full", cost=0.0)

    assert telemetry.candidates_cached == 2
    assert telemetry.candidates_full_evaluated == 0
    assert telemetry.to_dict()["candidates_cached"] == 2
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py::test_rejected_record_rejects_score tests/unit/test_vnext_evaluation.py::test_telemetry_records_cached_without_full_evaluation_count -v
```

Expected: both tests fail. The rejected-record test currently allows scored rejected
records, and `OptimizationTelemetry.record_cached(...)` does not exist yet.

---

### Task 2: Implement Strict Records And Cached Telemetry

**Files:**
- Modify: `evocore/evaluation.py`
- Modify: `tests/unit/test_vnext_evaluation.py`

- [ ] **Step 1: Update `EvaluationRecord.__post_init__`**

In `evocore/evaluation.py`, replace the existing score validation block in
`EvaluationRecord.__post_init__` with this code:

```python
        if self.confidence == "rejected":
            if self.score is not None:
                raise FitnessError("EvaluationRecord with confidence='rejected' requires score=None.")
        elif self.score is None or not math.isfinite(float(self.score)):
            raise FitnessError("EvaluationRecord requires a finite score unless rejected.")
```

Keep the existing candidate ID, rung, confidence, and cost validation around this block.

- [ ] **Step 2: Add cached telemetry state and export**

In `evocore/evaluation.py`, update `OptimizationTelemetry` to include the cached count and
helper:

```python
@dataclass
class OptimizationTelemetry:
    """Aggregate vNext optimizer budget and trial accounting."""

    total_candidates_proposed: int = 0
    unique_candidate_hashes: set[str] = field(default_factory=set)
    candidates_screened: int = 0
    candidates_partial_evaluated: int = 0
    candidates_full_evaluated: int = 0
    candidates_cached: int = 0
    promoted_by_rung: dict[str, int] = field(default_factory=dict)
    eliminated_by_rung: dict[str, int] = field(default_factory=dict)
    cost_by_rung: dict[str, float] = field(default_factory=dict)

    def record_cached(self, count: int, *, rung: str, cost: float) -> None:
        """Record cached trusted observations without spending fresh full-evaluation budget."""
        self.candidates_cached += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)
```

Then add `"candidates_cached": self.candidates_cached,` to `OptimizationTelemetry.to_dict()`
immediately after `"candidates_full_evaluated": self.candidates_full_evaluated,`.

- [ ] **Step 3: Update the existing telemetry export assertion**

In `tests/unit/test_vnext_evaluation.py`, update
`test_telemetry_to_dict_exports_sorted_hashes_and_unique_count` so it sets the new cached
field and expects it in the exported payload:

```python
    telemetry.candidates_cached = 1
```

Expected dictionary addition:

```python
        "candidates_cached": 1,
```

Place it after `"candidates_full_evaluated": 3,`.

- [ ] **Step 4: Run the focused validation and telemetry tests**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py -v
```

Expected: all tests in `test_vnext_evaluation.py` pass.

- [ ] **Step 5: Commit Task 1-2**

Run:

```powershell
git add evocore/evaluation.py tests/unit/test_vnext_evaluation.py
git commit -m "fix: enforce strict evaluation record semantics"
```

Expected: commit succeeds with only those two files staged.

---

### Task 3: Update GA Cached Budget Semantics

**Files:**
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `evocore/ga.py`

- [ ] **Step 1: Strengthen the GA cached ask/tell test**

In `tests/unit/test_ga_ask_tell_vnext.py`, extend
`test_ga_cached_records_are_eligible_for_best_state` with these assertions after the
existing `result.cached_count == 1` assertion:

```python
    assert result.trusted_count == 1
    assert engine.vnext_telemetry.candidates_cached == 1
    assert engine.vnext_telemetry.candidates_full_evaluated == 1
```

- [ ] **Step 2: Add a GA policy-run test with one cached final-rung record**

Append this evaluator and test to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
class OneCachedThenFreshEvaluator:
    def __init__(self) -> None:
        self._returned_cached = False

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        records = []
        for candidate in candidates:
            if not self._returned_cached:
                self._returned_cached = True
                records.append(
                    EvaluationRecord(
                        candidate_id=candidate.candidate_id,
                        batch_id=candidate.batch_id,
                        score=100.0,
                        confidence="cached",
                        rung=context.rung.name,
                        cost=0.0,
                    )
                )
            else:
                records.append(
                    EvaluationRecord(
                        candidate_id=candidate.candidate_id,
                        batch_id=candidate.batch_id,
                        score=-sum(float(value) ** 2 for value in candidate.genes),
                        confidence="trusted_full",
                        rung=context.rung.name,
                        cost=context.rung.budget,
                    )
                )
        return records


def test_ga_run_cached_records_do_not_consume_full_evaluation_budget() -> None:
    engine = GAEngine(_space(), population_size=4, generations=20, seed=123)
    policy = MultiFidelityPolicy.single_full(budget=4, batch_size=4)

    result = engine.run(OneCachedThenFreshEvaluator(), policy=policy)

    assert result.n_evaluations == 4
    assert result.telemetry.candidates_full_evaluated == 4
    assert result.telemetry.candidates_cached == 1
    assert result.best_score == pytest.approx(100.0)
```

- [ ] **Step 3: Run the new GA tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_cached_records_are_eligible_for_best_state tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_cached_records_do_not_consume_full_evaluation_budget -v
```

Expected: failures show cached records still increment full-evaluation telemetry and/or
policy-run `n_evaluations`.

- [ ] **Step 4: Update `GAEngine.tell(...)` cached handling**

In `evocore/ga.py`, replace the cached branch inside `GAEngine.tell(...)` with:

```python
            elif record.confidence == "cached":
                cached += 1
                self.vnext_telemetry.record_cached(1, rung=record.rung, cost=record.cost)
```

Do not call `record_full(...)` for cached records.

- [ ] **Step 5: Update `GAEngine.run(...)` fresh evaluation counting**

In `evocore/ga.py`, inside `GAEngine.run(...)`, replace this final-rung block:

```python
                if rung.confidence == "trusted_full":
                    n_evaluations += len(records)
                    final_candidates.extend(assigned)
                    break
```

with:

```python
                if rung.confidence == "trusted_full":
                    fresh_count = sum(1 for record in records if record.confidence == "trusted_full")
                    n_evaluations += fresh_count
                    final_candidates.extend(assigned)
                    if fresh_count == 0:
                        raise FitnessError(
                            "Evaluator returned no fresh trusted_full records for the final rung; "
                            "cached records do not consume full-evaluation budget."
                        )
                    break
```

This prevents a cache-only final batch from looping forever while still allowing mixed
cached/fresh batches to continue until the fresh full-evaluation budget is reached.

- [ ] **Step 6: Run GA ask/tell tests**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: all GA ask/tell vNext tests pass.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py
git commit -m "fix: exclude cached records from GA evaluation budget"
```

Expected: commit succeeds with only those two files staged.

---

### Task 4: Update CMA Cached Budget Semantics

**Files:**
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `evocore/cmaes.py`

- [ ] **Step 1: Strengthen the CMA cached ask/tell test**

In `tests/unit/test_cmaes_ask_tell_vnext.py`, extend
`test_cma_cached_records_are_eligible_for_best_state_and_batch_update` with:

```python
    assert engine.vnext_telemetry.candidates_cached == 4
    assert engine.vnext_telemetry.candidates_full_evaluated == 0
```

- [ ] **Step 2: Run the focused CMA cached test and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py::test_cma_cached_records_are_eligible_for_best_state_and_batch_update -v
```

Expected: failure shows cached records are still counted as full evaluations or the cached
telemetry field is missing.

- [ ] **Step 3: Update `CMAESEngine._apply_record_confidence(...)` cached handling**

In `evocore/cmaes.py`, replace this cached branch:

```python
        if record.confidence == "cached":
            self.vnext_telemetry.record_full(1, rung=record.rung, cost=record.cost)
            return "cached"
```

with:

```python
        if record.confidence == "cached":
            self.vnext_telemetry.record_cached(1, rung=record.rung, cost=record.cost)
            return "cached"
```

- [ ] **Step 4: Run CMA ask/tell tests**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: all CMA ask/tell vNext tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py
git commit -m "fix: exclude cached records from CMA evaluation budget"
```

Expected: commit succeeds with only those two files staged.

---

### Task 5: Make Remaining Callable Fitness Helpers Strict

**Files:**
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_runtime_observability.py`
- Modify: `evocore/ga.py`
- Modify: `evocore/cmaes.py`

- [ ] **Step 1: Replace the GA non-finite sanitization test**

In `tests/unit/test_ga_engine.py`, replace
`test_nan_fitness_warns_once_and_sanitizes` with:

```python
def test_non_finite_fitness_raises() -> None:
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    ind = Individual([0.0, 0.0])

    with pytest.raises(FitnessError, match="finite"):
        engine._evaluate_all([ind], lambda _ind: float("nan"), gen=0)

    assert ind.fitness is None
    assert ind.fitness_valid is False
```

Remove `FitnessWarning` from the import list in this file if it becomes unused. Remove the
top-level `import warnings` if no other test in the file uses it.

- [ ] **Step 2: Replace the runtime observability non-finite warning test**

In `tests/unit/test_runtime_observability.py`, replace
`test_ga_logs_non_finite_fitness_warning` with:

```python
def test_ga_non_finite_fitness_raises_without_warning_log(caplog) -> None:
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1, seed=7)

    with caplog.at_level(logging.WARNING, logger="evocore"):
        with pytest.raises(FitnessError, match="finite"):
            engine._run_from_population(
                engine._initial_population(), non_finite_once, start_generation=0
            )

    messages = [record.getMessage() for record in caplog.records if record.name == "evocore.ga"]
    assert not any("assigned fitness=-inf" in message for message in messages)
```

Update the imports at the top of the file to:

```python
from evocore import CMAESEngine, FitnessError, GAEngine, GeneSpace
```

- [ ] **Step 3: Add a CMA callable strictness test**

Append this test to `tests/unit/test_cmaes_engine.py`:

```python
def test_cmaes_non_finite_fitness_raises() -> None:
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), population_size=6, generations=1)

    with pytest.raises(FitnessError, match="finite"):
        engine.run(lambda _ind: float("nan"))
```

Update the import in `tests/unit/test_cmaes_engine.py` to include `FitnessError`:

```python
from evocore import ConfigurationError, FitnessError, GeneDef, GeneSpace
```

- [ ] **Step 4: Run focused strictness tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py::test_non_finite_fitness_raises tests/unit/test_runtime_observability.py::test_ga_non_finite_fitness_raises_without_warning_log tests/unit/test_cmaes_engine.py::test_cmaes_non_finite_fitness_raises -v
```

Expected: tests fail because callable helper paths still sanitize non-finite values.

- [ ] **Step 5: Update `GAEngine._normalise_fitness_result(...)`**

In `evocore/ga.py`, replace the non-finite block in `_normalise_fitness_result(...)`:

```python
        if not math.isfinite(fitness):
            ind.metadata["raw_fitness"] = fitness
            ind.fitness = float("-inf")
            ind.fitness_valid = True
            return float("-inf"), 1
```

with:

```python
        if not math.isfinite(fitness):
            raise FitnessError(
                f"fitness_fn must return a finite float at generation {gen}, index {idx}; "
                f"got {fitness!r}."
            )
```

Then remove the warning-emission block from `_evaluate_all(...)`:

```python
        if nan_count and not self._fitness_warning_emitted:
            logger.warning(
                "GA generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
                gen,
                nan_count,
            )
            warnings.warn(
                f"{nan_count} individuals in generation {gen} returned NaN or Inf fitness. "
                "They have been assigned fitness=-inf for selection.",
                FitnessWarning,
                stacklevel=2,
            )
            self._fitness_warning_emitted = True
```

Remove `import warnings` and `FitnessWarning` from `evocore/ga.py` if unused after this
edit. Keep the return type as `tuple[list[float], int]`; `nan_count` remains `0` for
successful evaluations until legacy logbook fields are removed in a separate cleanup.

- [ ] **Step 6: Update `CMAESEngine._normalise_fitness_result(...)`**

In `evocore/cmaes.py`, replace the non-finite block in `_normalise_fitness_result(...)`:

```python
        if not math.isfinite(fitness):
            ind.metadata["raw_fitness"] = fitness
            ind.fitness = float("-inf")
            ind.fitness_valid = True
            return float("-inf"), 1
```

with:

```python
        if not math.isfinite(fitness):
            raise FitnessError(
                f"fitness_fn must return a finite float at generation {gen}, index {idx}; "
                f"got {fitness!r}."
            )
```

Then remove the warning-emission block from `_evaluate_all(...)`:

```python
        if nan_count and not self._fitness_warning_emitted:
            logger.warning(
                "CMA-ES generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
                gen,
                nan_count,
            )
            warnings.warn(
                f"{nan_count} individuals in generation {gen} returned NaN or Inf fitness. "
                "They have been assigned fitness=-inf for selection.",
                FitnessWarning,
                stacklevel=2,
            )
            self._fitness_warning_emitted = True
```

Remove `import warnings` and `FitnessWarning` from `evocore/cmaes.py` if unused after this
edit. Keep `nan_count` as `0` for successful callable runs.

- [ ] **Step 7: Run focused strictness tests**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py::test_non_finite_fitness_raises tests/unit/test_runtime_observability.py::test_ga_non_finite_fitness_raises_without_warning_log tests/unit/test_cmaes_engine.py::test_cmaes_non_finite_fitness_raises -v
```

Expected: all focused strictness tests pass.

- [ ] **Step 8: Run related legacy-helper tests**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py::test_tuple_fitness_stores_metrics tests/unit/test_ga_engine.py::test_fitness_exception_wrapped tests/unit/test_cmaes_engine.py tests/unit/test_runtime_observability.py -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 5**

Run:

```powershell
git add evocore/ga.py evocore/cmaes.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py tests/unit/test_runtime_observability.py
git commit -m "fix: reject non-finite callable fitness values"
```

Expected: commit succeeds with only the listed files staged.

---

### Task 6: Update Public Docs And Changelog

**Files:**
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update ask/tell confidence semantics**

In `docs/site/ask-tell-engines.md`, replace the confidence bullets with:

```markdown
Confidence values are explicit:

- `trusted_full` records carry finite raw scores from fresh full objective work. They
  update optimizer state and consume full-evaluation budget.
- `cached` records carry finite raw scores from trusted previous full evaluations. They
  update optimizer state but do not consume fresh full-evaluation budget.
- `partial` and `surrogate` records carry finite scores for scheduling, telemetry, and
  history, but they cannot become optimizer best state.
- `rejected` records represent recoverable candidate-level failures. They must use
  `score=None` and carry diagnostics in `metrics` or `metadata`.
```

Then replace the invalid-record sentence with:

```markdown
Invalid records raise `FitnessError`: unknown candidates, unknown explicit batch IDs,
batch mismatches, duplicate candidate/rung records, non-finite non-rejected scores, and
scored `rejected` records are rejected.
```

- [ ] **Step 2: Update telemetry docs**

In `docs/site/optimizer-telemetry.md`, add `candidates_cached` to the stable export field
list immediately after `candidates_full_evaluated`, and replace the cached paragraph with:

```markdown
Cached evaluation records are state-eligible but do not count as fresh full evaluations.
They are visible through `OptimizationTelemetry.candidates_cached`,
`TellResult.cached_count`, and event history rows with `confidence="cached"`.
```

- [ ] **Step 3: Update GA docs**

In `docs/site/ga.md`, after the opening paragraph, add:

```markdown
Full-evaluation budget accounting counts fresh `trusted_full` records only. Cached records
can update best-candidate state, but they do not spend fresh objective budget.
```

- [ ] **Step 4: Update CMA docs**

In `docs/site/cmaes.md`, replace the sentence:

```markdown
`direction="maximize"` and `direction="minimize"` preserve raw user fitness values in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state.
```

with:

```markdown
`direction="maximize"` and `direction="minimize"` preserve raw user fitness values in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state. Cached records
reuse trusted previous full observations and do not spend fresh full-evaluation budget.
```

- [ ] **Step 5: Update changelog**

In `CHANGELOG.md`, under `[Unreleased]` `### Changed`, replace:

```markdown
- Cached evaluation records are now eligible for optimizer state updates and full-budget
  accounting while remaining separately counted in `TellResult.cached_count`.
```

with:

```markdown
- Cached evaluation records remain eligible for optimizer state updates but no longer
  consume fresh full-evaluation budget; they are counted through
  `OptimizationTelemetry.candidates_cached` and `TellResult.cached_count`.
- Objective records now reject non-finite scores uniformly, and `rejected` records must
  use `score=None` with diagnostics in metrics or metadata.
```

- [ ] **Step 6: Run docs checks**

Run:

```powershell
python -m mkdocs build --strict
```

Expected: documentation builds successfully without warnings.

- [ ] **Step 7: Commit Task 6**

Run:

```powershell
git add docs/site/ask-tell-engines.md docs/site/optimizer-telemetry.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "docs: update objective evaluation semantics"
```

Expected: commit succeeds with only docs and changelog staged.

---

### Task 7: Final Verification

**Files:**
- Read-only: full touched surface

- [ ] **Step 1: Confirm branch state**

Run:

```powershell
git status --short --branch
```

Expected: on the task branch. If the worktree is dirty, only files intentionally changed
by this plan are present.

- [ ] **Step 2: Run formatting check**

Run:

```powershell
python -m ruff format --check
```

Expected: command exits `0`.

- [ ] **Step 3: Run lint**

Run:

```powershell
python -m ruff check
```

Expected: command exits `0`.

- [ ] **Step 4: Rebuild the Python extension**

Run:

```powershell
python -m maturin develop --release
```

Expected: extension builds and installs successfully.

- [ ] **Step 5: Run unit and integration tests**

Run:

```powershell
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Run docs build**

Run:

```powershell
python -m mkdocs build --strict
```

Expected: documentation builds successfully.

- [ ] **Step 7: Commit any verification-only fixes**

If verification required formatting or small fixes, stage the touched files from this plan:

```powershell
git add evocore/evaluation.py evocore/ga.py evocore/cmaes.py tests/unit/test_vnext_evaluation.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py tests/unit/test_runtime_observability.py docs/site/ask-tell-engines.md docs/site/optimizer-telemetry.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "fix: finalize objective evaluation semantics"
```

Expected: no commit is needed if Task 1-6 commits already pass verification.

- [ ] **Step 8: Push the branch**

Run:

```powershell
git push origin feature/general-optimizer-framework
```

Expected: branch pushes successfully to the existing draft PR.
