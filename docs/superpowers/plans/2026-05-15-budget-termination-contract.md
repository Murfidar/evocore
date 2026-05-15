# Budget Termination Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved budget and termination contract: hard rename `generations` to `max_generations`, unify policy budget naming on `max_evaluations`, remove legacy result stop booleans, and append final `run_stop` history events.

**Architecture:** Keep shared result and history vocabulary in `evocore.stats` and `evocore.ga.RunResult`, then have GA and CMA emit the same final stop contract. Rename policy fields at the source in `evocore.policies`, and update all test, docs, benchmark, and reproducibility surfaces to use the new names. Preserve the approved objective semantics: `n_evaluations` and `max_evaluations` count fresh `trusted_full` records only.

**Tech Stack:** Python 3.11+, dataclasses, typing `Literal`, pytest, Ruff, maturin/PyO3 extension build, MkDocs.

---

## Scope Check

This plan is one stabilization slice because the public names, result export shape, docs,
and tests must change atomically. It touches Python source, tests, benchmark helpers,
docs, and changelog, but it does not touch Rust/PyO3 code.

Out of scope:

- Adding `target_score`, patience, or wall-clock controls.
- Adding island-model budget composition.
- Adding multi-objective termination.
- Migrating old checkpoint or result payloads.

---

## File Structure

- Modify `evocore/policies.py`: rename `full_evaluation_budget` to `max_evaluations`; rename `single_full(budget=...)` to `single_full(max_evaluations=...)`; reject legacy `budget` in `single_full(...)` with `ConfigurationError`.
- Modify `evocore/stats.py`: add shared `StopReason`, allow `EventRecord.event_type="run_stop"`, and add an `append_run_stop_event(...)` helper.
- Modify `evocore/ga.py`: import shared stop helpers; remove `stopped_early` and `budget_reached`; add `max_generations`; rename internal `self.generations`; update result export, default policy creation, policy budget references, run stop reasoning, reproducibility metadata, and final `run_stop` event emission.
- Modify `evocore/cmaes.py`: rename constructor/internal `generations` to `max_generations`; update callback binding, loops, reproducibility metadata, stop reason, result construction, and final `run_stop` event emission.
- Modify `evocore/callbacks.py`: bind progress bars from `max_generations` instead of `generations`.
- Modify tests under `tests/unit/`, `tests/integration/`, `tests/benchmarks/`, and `tests/vnext_helpers.py`: update API names, result expectations, policy budget names, and final history events.
- Modify docs under `docs/site/`: update examples and public contract text.
- Modify `CHANGELOG.md`: document the breaking budget/termination cleanup.

---

### Task 0: Confirm Branch And Worktree

**Files:**
- Read-only: git worktree metadata

- [ ] **Step 1: Check branch and uncommitted files**

Run:

```powershell
git status --short --branch
```

Expected: branch is `feature/general-optimizer-framework` or another task branch, not
`main`. The plan/spec commits may be ahead of origin. If unrelated uncommitted files
exist, leave them untouched and stage only files listed in the current task.

---

### Task 1: Add Policy Rename Tests

**Files:**
- Modify: `tests/unit/test_vnext_policy_scheduler.py`
- Modify: `tests/vnext_helpers.py`

- [ ] **Step 1: Update policy tests to use `max_evaluations` and reject `budget`**

In `tests/unit/test_vnext_policy_scheduler.py`, replace every
`full_evaluation_budget=` constructor argument with `max_evaluations=`.

Replace `test_policy_requires_unique_rung_names_and_full_budget` with:

```python
def test_policy_requires_unique_rung_names_and_max_evaluations() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=32,
        batch_size=8,
        exploration_fraction=0.10,
        audit_fraction=0.05,
    )

    assert policy.max_evaluations == 32
    assert policy.rung_names == ("cheap", "full")
    assert policy.final_rung.name == "full"
```

Replace the invalid-budget assertion inside
`test_policy_rejects_invalid_budget_and_fractions` with:

```python
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            max_evaluations=0,
        )
```

Append these tests near the policy validation tests:

```python
def test_policy_rejects_legacy_full_evaluation_budget_name() -> None:
    with pytest.raises(TypeError, match="full_evaluation_budget"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=16,
        )


def test_single_full_uses_max_evaluations_and_rejects_budget() -> None:
    policy = MultiFidelityPolicy.single_full(max_evaluations=12, batch_size=4)

    assert policy.max_evaluations == 12
    assert policy.batch_size == 4
    assert policy.rung_names == ("full",)

    with pytest.raises(ConfigurationError, match="max_evaluations"):
        MultiFidelityPolicy.single_full(budget=12, batch_size=4)
```

In `tests/vnext_helpers.py`, replace the helper with:

```python
def full_policy(max_evaluations: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        max_evaluations=max_evaluations,
        batch_size=batch_size,
    )
```

- [ ] **Step 2: Run focused policy tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: failures mention unexpected `max_evaluations`, missing
`policy.max_evaluations`, or old `single_full(budget=...)` behavior.

---

### Task 2: Implement Policy Rename

**Files:**
- Modify: `evocore/policies.py`
- Modify: `tests/unit/test_vnext_policy_scheduler.py`
- Modify: `tests/vnext_helpers.py`

- [ ] **Step 1: Rename the policy dataclass field and validation**

In `evocore/policies.py`, replace the `MultiFidelityPolicy` dataclass with:

```python
@dataclass(frozen=True)
class MultiFidelityPolicy:
    """Configure multi-fidelity scheduling for vNext engines."""

    rungs: list[Rung]
    max_evaluations: int
    batch_size: int | None = None
    exploration_fraction: float = 0.10
    audit_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ConfigurationError("MultiFidelityPolicy requires at least one rung.")
        if int(self.max_evaluations) <= 0:
            raise ConfigurationError("max_evaluations must be positive.")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ConfigurationError("batch_size must be positive when provided.")
        if not (0.0 <= float(self.exploration_fraction) < 1.0):
            raise ConfigurationError("exploration_fraction must be in [0, 1).")
        if not (0.0 <= float(self.audit_fraction) < 1.0):
            raise ConfigurationError("audit_fraction must be in [0, 1).")

        names = [rung.name for rung in self.rungs]
        if len(names) != len(set(names)):
            raise ConfigurationError("MultiFidelityPolicy contains duplicate rung names.")
        trusted_full_rungs = [rung for rung in self.rungs if rung.confidence == "trusted_full"]
        if not trusted_full_rungs:
            raise ConfigurationError("MultiFidelityPolicy requires a trusted_full rung.")
        if len(trusted_full_rungs) != 1:
            raise ConfigurationError("MultiFidelityPolicy requires exactly one trusted_full rung.")
        if self.rungs[-1].confidence != "trusted_full":
            raise ConfigurationError("MultiFidelityPolicy final rung must be trusted_full.")

    @property
    def rung_names(self) -> Sequence[str]:
        """Return rung names in execution order."""
        return tuple(rung.name for rung in self.rungs)

    @property
    def final_rung(self) -> Rung:
        """Return the last configured rung."""
        return self.rungs[-1]

    @classmethod
    def single_full(
        cls,
        *,
        max_evaluations: int | None = None,
        batch_size: int | None = None,
        **legacy_kwargs: object,
    ) -> MultiFidelityPolicy:
        """Create a one-rung full-evaluation vNext policy."""
        if "budget" in legacy_kwargs:
            raise ConfigurationError(
                "MultiFidelityPolicy.single_full() uses max_evaluations=..., not budget=...."
            )
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(
                f"MultiFidelityPolicy.single_full() got unexpected argument(s): {unknown}."
            )
        if max_evaluations is None:
            raise ConfigurationError("single_full() requires max_evaluations.")
        return cls(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            exploration_fraction=0.0,
            audit_fraction=0.0,
        )
```

- [ ] **Step 2: Run policy tests**

Run:

```powershell
python -m pytest tests/unit/test_vnext_policy_scheduler.py -v
```

Expected: all policy scheduler tests pass.

- [ ] **Step 3: Commit policy rename**

Run:

```powershell
git add evocore/policies.py tests/unit/test_vnext_policy_scheduler.py tests/vnext_helpers.py
git commit -m "refactor: rename policy budget to max evaluations"
```

Expected: commit succeeds with only these files staged.

---

### Task 3: Add Result And Run-Stop History Tests

**Files:**
- Modify: `tests/unit/test_stats.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add event-history support tests for `run_stop`**

In `tests/unit/test_stats.py`, update the import block to include
`append_run_stop_event`:

```python
from evocore.stats import (
    EventHistory,
    EventRecord,
    Logbook,
    LogEntry,
    ReproducibilityMetadata,
    append_run_stop_event,
    gene_space_hash,
    gene_space_signature,
)
```

Append this test after `test_event_history_to_rows_preserves_append_order`:

```python
def test_append_run_stop_event_records_terminal_metadata() -> None:
    history = EventHistory()

    append_run_stop_event(
        history,
        stop_reason="max_evaluations",
        max_evaluations=12,
        max_generations=3,
        n_evaluations=12,
    )

    assert len(history) == 1
    row = history.to_rows()[0]
    assert row["event_type"] == "run_stop"
    assert row["metadata"] == {
        "max_evaluations": 12,
        "max_generations": 3,
        "n_evaluations": 12,
        "stop_reason": "max_evaluations",
    }
```

- [ ] **Step 2: Update `RunResult` construction tests for hard cleanup**

In `tests/unit/test_ga_engine.py`, replace `make_result(...)` with:

```python
def make_result(seed: int, fitness: float) -> RunResult:
    ind = Individual([fitness], fitness=fitness, fitness_valid=True)
    return RunResult(
        best_individual=ind,
        best_fitness=fitness,
        final_population=Population([ind]),
        logbook=Logbook(),
        wall_time_seconds=0.01,
        n_evaluations=1,
        elite_history=[ind],
        diversity_history=[],
        seed=seed,
        stop_reason="max_generations",
        max_generations=5,
    )
```

Replace `test_run_result_preserves_existing_positional_construction` with:

```python
def test_run_result_uses_stop_reason_without_legacy_booleans():
    result = make_result(7, 1.25)

    assert result.best_fitness == pytest.approx(1.25)
    assert result.stop_reason == "max_generations"
    assert result.max_generations == 5
    assert not hasattr(result, "stopped_early")
    assert not hasattr(result, "budget_reached")
    assert result.direction == "maximize"
    assert result.engine_type == ""
    assert result.best_candidate_id is None
    assert result.best_score is None
    assert len(result.history) == 0
    assert result.metadata == {}
```

In `test_run_result_to_dict_excludes_runtime_by_default`, add these assertions after
the existing `payload["n_evaluations"]` assertion:

```python
    assert payload["stop"] == {"reason": "max_generations"}
    assert payload["budget"] == {
        "max_evaluations": None,
        "max_generations": 5,
        "n_evaluations": 1,
    }
    assert "stopped_early" not in payload["stop"]
    assert "budget_reached" not in payload["budget"]
```

- [ ] **Step 3: Run focused result/history tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_append_run_stop_event_records_terminal_metadata tests/unit/test_ga_engine.py::test_run_result_uses_stop_reason_without_legacy_booleans tests/unit/test_ga_engine.py::test_run_result_to_dict_excludes_runtime_by_default -v
```

Expected: failures show `append_run_stop_event` is missing, `RunResult` still requires
`stopped_early`, or old export booleans remain.

---

### Task 4: Implement Shared Stop Vocabulary, RunResult Cleanup, And Run-Stop Helper

**Files:**
- Modify: `evocore/stats.py`
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_stats.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add shared stop reason vocabulary and `run_stop` event helper**

In `evocore/stats.py`, replace the `EventRecord.event_type` annotation with:

```python
    event_type: Literal["ask", "tell", "generation", "run_stop"]
```

Add this type alias near the other public type aliases/imports in `evocore/stats.py`:

```python
StopReason = Literal[
    "max_evaluations",
    "max_generations",
    "callback",
    "manual",
    "optimizer_converged",
    "target_score",
    "patience",
    "wall_time",
]
```

Add this helper immediately after the `EventHistory` class:

```python
def append_run_stop_event(
    history: EventHistory,
    *,
    stop_reason: StopReason,
    max_evaluations: int | None,
    max_generations: int | None,
    n_evaluations: int,
) -> None:
    """Append one terminal run-level stop event."""
    history.append(
        EventRecord(
            event_index=len(history),
            event_type="run_stop",
            metadata={
                "stop_reason": stop_reason,
                "max_evaluations": max_evaluations,
                "max_generations": max_generations,
                "n_evaluations": n_evaluations,
            },
        )
    )
```

- [ ] **Step 2: Update `RunResult` dataclass and export shape**

In `evocore/ga.py`, remove the local `StopReason = ...` line.

Update the `evocore.stats` import to include:

```python
    StopReason,
    append_run_stop_event,
```

Replace the `RunResult` dataclass fields from `seed` through `metadata` with:

```python
    seed: int
    stop_reason: StopReason = "max_generations"
    max_generations: int | None = None
    max_evaluations: int | None = None
    telemetry: OptimizationTelemetry = field(default_factory=OptimizationTelemetry)
    direction: Direction = "maximize"
    engine_type: str = ""
    best_candidate_id: str | None = None
    best_score: float | None = None
    history: EventHistory = field(default_factory=EventHistory)
    reproducibility: ReproducibilityMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Replace the `stop`, `budget`, and top-level `n_evaluations` portion in
`RunResult.to_dict(...)` with:

```python
            "stop": {
                "reason": self.stop_reason,
            },
            "budget": {
                "max_evaluations": self.max_evaluations,
                "max_generations": self.max_generations,
                "n_evaluations": self.n_evaluations,
            },
            "n_evaluations": self.n_evaluations,
```

- [ ] **Step 3: Run focused result/history tests**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_append_run_stop_event_records_terminal_metadata tests/unit/test_ga_engine.py::test_run_result_uses_stop_reason_without_legacy_booleans tests/unit/test_ga_engine.py::test_run_result_to_dict_excludes_runtime_by_default -v
```

Expected: focused tests pass.

- [ ] **Step 4: Commit shared result cleanup**

Run:

```powershell
git add evocore/stats.py evocore/ga.py tests/unit/test_stats.py tests/unit/test_ga_engine.py
git commit -m "refactor: simplify run stop result contract"
```

Expected: commit succeeds with only these files staged.

---

### Task 5: Add GA Max-Generation And Stop-Reason Tests

**Files:**
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`

- [ ] **Step 1: Update GA test helpers to use new policy names**

In `tests/unit/test_ga_engine.py`, replace the helper `full_policy(...)` with:

```python
def full_policy(max_evaluations: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        max_evaluations=max_evaluations,
        batch_size=batch_size,
    )
```

In `tests/unit/test_ga_ask_tell_vnext.py`, replace all
`MultiFidelityPolicy.single_full(budget=` calls with
`MultiFidelityPolicy.single_full(max_evaluations=`.

- [ ] **Step 2: Add GA constructor and generation-loop stop tests**

In `tests/unit/test_ga_engine.py`, add these tests near the existing configuration and
run diagnostics tests:

```python
def test_ga_uses_max_generations_and_rejects_generations_keyword():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=3,
        seed=42,
    )

    assert engine.max_generations == 3
    assert not hasattr(engine, "generations")

    with pytest.raises(ConfigurationError, match="max_generations"):
        GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), generations=3)


def test_ga_generation_loop_reports_max_generations_and_run_stop_event():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine._run_from_population(
        engine._initial_population(),
        lambda ind: -sum(float(v) ** 2 for v in ind.genes),
        start_generation=0,
    )

    assert result.max_generations == 2
    assert result.stop_reason == "max_generations"
    assert [event.event_type for event in result.history][-1] == "run_stop"
    assert result.history.to_rows()[-1]["metadata"] == {
        "max_evaluations": None,
        "max_generations": 2,
        "n_evaluations": result.n_evaluations,
        "stop_reason": "max_generations",
    }


def test_ga_callback_stop_precedes_max_evaluations():
    class StopAfterFirstGeneration(Callback):
        def on_generation_end(self, gen, pop, info):
            self.should_stop = True

    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=5,
        max_evaluations=4,
        callbacks=[StopAfterFirstGeneration()],
        seed=42,
    )

    result = engine._run_from_population(
        engine._initial_population(),
        lambda ind: -sum(float(v) ** 2 for v in ind.genes),
        start_generation=0,
    )

    assert result.stop_reason == "callback"
    assert result.history.to_rows()[-1]["metadata"]["stop_reason"] == "callback"
```

- [ ] **Step 3: Update existing GA result expectations**

In `tests/unit/test_ga_engine.py`:

- Replace all `generations=` constructor arguments with `max_generations=`.
- Replace assertions for `result.budget_reached` with assertions that the attribute is
  absent:

```python
    assert not hasattr(result, "budget_reached")
```

- Replace expected natural generation event lists such as:

```python
    assert [event.event_type for event in result.history] == ["generation", "generation"]
```

with:

```python
    assert [event.event_type for event in result.history] == [
        "generation",
        "generation",
        "run_stop",
    ]
```

- Replace reproducibility assertions for `"generations"` with:

```python
    assert result.reproducibility.optimizer_config["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config
```

In `tests/unit/test_ga_ask_tell_vnext.py`, replace every `generations=` constructor
argument with `max_generations=`.

- [ ] **Step 4: Run focused GA tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py::test_ga_uses_max_generations_and_rejects_generations_keyword tests/unit/test_ga_engine.py::test_ga_generation_loop_reports_max_generations_and_run_stop_event tests/unit/test_ga_engine.py::test_ga_callback_stop_precedes_max_evaluations tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_uses_policy_and_returns_vnext_telemetry tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_cached_records_do_not_consume_full_evaluation_budget -v
```

Expected: failures mention the old `generations` attribute, old policy field names,
missing `run_stop`, or removed result constructor fields.

---

### Task 6: Implement GA Budget And Termination Cleanup

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`

- [ ] **Step 1: Rename the GA constructor argument and internal field**

In `evocore/ga.py`, update the constructor docstring argument line to:

```python
        max_generations: Maximum number of generations to run.
```

Replace the `GAEngine.__init__` signature prefix and suffix with this shape:

```python
    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 100,
        max_generations: int = 100,
        crossover: str = "sbx",
        crossover_prob: float = 0.9,
        crossover_eta: float = 2.0,
        crossover_alpha: float = 0.5,
        mutation: str = "gaussian",
        mutation_prob: float = 0.1,
        mutation_individual_prob: float = 1.0,
        mutation_sigma: float = 0.2,
        mutation_sigma_schedule: str = "constant",
        mutation_sigma_end: float = 0.02,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 1,
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: Callable[..., object] | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
        **legacy_kwargs: object,
    ) -> None:
```

Immediately after the `gene_space is None` check, add:

```python
        if "generations" in legacy_kwargs:
            raise ConfigurationError("GAEngine uses max_generations=..., not generations=....")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(f"GAEngine got unexpected argument(s): {unknown}.")
```

Replace generation validation and assignment with:

```python
        if max_generations < 0:
            raise ConfigurationError("max_generations must be >= 0.")
```

and:

```python
        self.max_generations = max_generations
```

Remove `self.generations = generations`.

- [ ] **Step 2: Replace internal `self.generations` references**

In `evocore/ga.py`, replace these snippets:

```python
        if self.generations <= 1 or self.mutation_sigma_schedule == "constant":
```

with:

```python
        if self.max_generations <= 1 or self.mutation_sigma_schedule == "constant":
```

Replace:

```python
        t = gen / max(1, self.generations - 1)
```

with:

```python
        t = gen / max(1, self.max_generations - 1)
```

Replace callback binding:

```python
            callback.bind_context(seed=self.seed, generations=self.generations)
```

with:

```python
            callback.bind_context(seed=self.seed, max_generations=self.max_generations)
```

Replace generation loops:

```python
        for gen in range(start_generation, self.generations):
```

with:

```python
        for gen in range(start_generation, self.max_generations):
```

Replace `_copy_with_seed(...)` constructor argument:

```python
            generations=self.generations,
```

with:

```python
            max_generations=self.max_generations,
```

- [ ] **Step 3: Update GA stop reasons and result construction**

In `_run_generation(...)`, replace the final return:

```python
        return next_population, fitnesses, n_evaluations, False, "generations"
```

with:

```python
        return next_population, fitnesses, n_evaluations, False, "max_generations"
```

In `_run_from_population(...)`, replace:

```python
        stopped_early = False
        stop_reason: StopReason = "generations"
```

with:

```python
        stop_reason: StopReason = "max_generations"
```

Remove every `stopped_early = True` assignment.

In the `RunResult(...)` call in `_run_from_population(...)`, remove
`stopped_early=...` and `budget_reached=...`, and add `max_generations`:

```python
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=self.max_evaluations,
```

After constructing the result and before invoking callbacks, append the terminal event:

```python
        append_run_stop_event(
            result.history,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
```

- [ ] **Step 4: Update GA reproducibility and default policy**

In `_optimizer_config(...)`, replace:

```python
                "generations": self.generations,
```

with:

```python
                "max_generations": self.max_generations,
```

In `run(...)`, replace default policy creation with:

```python
        resolved_policy = policy or MultiFidelityPolicy.single_full(
            max_evaluations=max(1, self.population_size * max(1, self.max_generations)),
            batch_size=self.population_size,
        )
```

Replace every `resolved_policy.full_evaluation_budget` reference with
`resolved_policy.max_evaluations`.

In both policy-driven `RunResult(...)` calls, remove `stopped_early=...` and
`budget_reached=...`, and use:

```python
                stop_reason="max_evaluations",
                max_generations=self.max_generations,
                max_evaluations=resolved_policy.max_evaluations,
```

and:

```python
            stop_reason="max_evaluations",
            max_generations=self.max_generations,
            max_evaluations=resolved_policy.max_evaluations,
```

Before returning each policy-driven `RunResult`, assign it to `result`, append the
terminal event, then return it:

```python
        append_run_stop_event(
            result.history,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
        return result
```

- [ ] **Step 5: Run GA tests**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: all selected GA tests pass.

- [ ] **Step 6: Commit GA cleanup**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py
git commit -m "refactor: align GA budget termination contract"
```

Expected: commit succeeds with only these files staged.

---

### Task 7: Add CMA Max-Generation And Stop-Reason Tests

**Files:**
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`

- [ ] **Step 1: Add CMA constructor and run-stop tests**

In `tests/unit/test_cmaes_engine.py`, add these tests near the existing constructor tests:

```python
def test_cmaes_uses_max_generations_and_rejects_generations_keyword():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    assert engine.max_generations == 2
    assert not hasattr(engine, "generations")

    with pytest.raises(ConfigurationError, match="max_generations"):
        CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), generations=2)


def test_cmaes_run_reports_max_generations_and_run_stop_event():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine.run(sphere)

    assert result.max_generations == 2
    assert result.max_evaluations is None
    assert result.stop_reason == "max_generations"
    assert [event.event_type for event in result.history][-1] == "run_stop"
    assert result.history.to_rows()[-1]["metadata"] == {
        "max_evaluations": None,
        "max_generations": 2,
        "n_evaluations": result.n_evaluations,
        "stop_reason": "max_generations",
    }
```

- [ ] **Step 2: Update existing CMA tests**

In `tests/unit/test_cmaes_engine.py`, replace all `generations=` constructor arguments
with `max_generations=`.

Replace expected history event lists such as:

```python
    assert [event.event_type for event in result.history] == ["generation", "generation"]
```

with:

```python
    assert [event.event_type for event in result.history] == [
        "generation",
        "generation",
        "run_stop",
    ]
```

Add reproducibility assertions to `test_cma_generation_loop_result_attaches_history_and_reproducibility`:

```python
    assert result.reproducibility.optimizer_config["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config
```

In `tests/unit/test_cmaes_ask_tell_vnext.py`, no constructor currently needs
`generations=` changes except any added by future local edits; keep default
`max_generations` where omitted.

- [ ] **Step 3: Run focused CMA tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_engine.py::test_cmaes_uses_max_generations_and_rejects_generations_keyword tests/unit/test_cmaes_engine.py::test_cmaes_run_reports_max_generations_and_run_stop_event tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: failures mention old `generations`, missing `run_stop`, or old
`RunResult` constructor fields.

---

### Task 8: Implement CMA Cleanup And Callback Context Rename

**Files:**
- Modify: `evocore/cmaes.py`
- Modify: `evocore/callbacks.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `tests/unit/test_callbacks.py`

- [ ] **Step 1: Rename CMA constructor argument and internal field**

In `evocore/cmaes.py`, update the constructor docstring line to:

```python
        max_generations: Maximum number of generations to run.
```

Replace the constructor parameter:

```python
        generations: int = 300,
```

with:

```python
        max_generations: int = 300,
```

Add `**legacy_kwargs: object` at the end of the constructor signature:

```python
        track_diversity: bool = False,
        **legacy_kwargs: object,
    ) -> None:
```

Immediately after the `gene_space is None` check, add:

```python
        if "generations" in legacy_kwargs:
            raise ConfigurationError("CMAESEngine uses max_generations=..., not generations=....")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(f"CMAESEngine got unexpected argument(s): {unknown}.")
```

Replace validation and assignment:

```python
        if generations < 0:
            raise ConfigurationError("generations must be >= 0.")
```

with:

```python
        if max_generations < 0:
            raise ConfigurationError("max_generations must be >= 0.")
```

Replace:

```python
        self.generations = generations
```

with:

```python
        self.max_generations = max_generations
```

- [ ] **Step 2: Replace CMA internal generation references**

In `evocore/cmaes.py`, replace callback binding:

```python
            callback.bind_context(seed=self.seed, generations=self.generations)
```

with:

```python
            callback.bind_context(seed=self.seed, max_generations=self.max_generations)
```

Replace `_optimizer_config(...)` field:

```python
                "generations": self.generations,
```

with:

```python
                "max_generations": self.max_generations,
```

Replace the run loop:

```python
        for gen in range(self.generations):
```

with:

```python
        for gen in range(self.max_generations):
```

Remove `stopped_early = False` and all `stopped_early = True` assignments from
`run(...)`. Add before the loop:

```python
        stop_reason: StopReason = "max_generations"
```

When callbacks stop before or after a generation, set:

```python
                stop_reason = "callback"
```

Update the `RunResult(...)` call by removing `stopped_early=...` and adding:

```python
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=None,
```

After creating `result`, append the stop event:

```python
        append_run_stop_event(
            result.history,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
```

Update the import from `evocore.stats` in `evocore/cmaes.py` to include:

```python
    StopReason,
    append_run_stop_event,
```

- [ ] **Step 3: Update callback progress binding**

In `evocore/callbacks.py`, replace:

```python
        self._total = kwargs.get("generations")
```

with:

```python
        self._total = kwargs.get("max_generations")
```

Append this test to `tests/unit/test_callbacks.py`:

```python
def test_progress_bar_binds_max_generations():
    cb = ProgressBar()

    cb.bind_context(seed=42, max_generations=7)

    assert cb._total == 7
```

Update the import at the top of `tests/unit/test_callbacks.py` to include
`ProgressBar`:

```python
from evocore.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    GenerationInfo,
    MetricsLogger,
    ProgressBar,
)
```

- [ ] **Step 4: Run CMA and callback tests**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_callbacks.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit CMA cleanup**

Run:

```powershell
git add evocore/cmaes.py evocore/callbacks.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_callbacks.py
git commit -m "refactor: align CMA termination contract"
```

Expected: commit succeeds with only these files staged.

---

### Task 9: Rename Remaining Test, Benchmark, And Helper References

**Files:**
- Modify: `tests/unit/test_rng_reproducibility.py`
- Modify: `tests/unit/test_runtime_observability.py`
- Modify: `tests/integration/test_binary_onemax.py`
- Modify: `tests/integration/test_cmaes_rosenbrock.py`
- Modify: `tests/integration/test_mixed_gene_space.py`
- Modify: `tests/integration/test_rastrigin.py`
- Modify: `tests/integration/test_sphere_function.py`
- Modify: `tests/benchmarks/bench_ga_vs_deap.py`
- Modify: `tests/benchmarks/bench_parallel_scaling.py`
- Modify: `tests/benchmarks/bench_vnext_multifidelity.py`

- [ ] **Step 1: Replace public constructor and policy names in remaining tests**

Run:

```powershell
rg -n "generations=|full_evaluation_budget=|single_full\(budget=|budget_reached|stopped_early" tests
```

For every constructor call in tests and benchmarks, replace `generations=` with
`max_generations=`.

For every policy constructor, replace `full_evaluation_budget=` with
`max_evaluations=`.

For every `single_full(...)` call, replace `budget=` with `max_evaluations=`.

Delete expectations for `budget_reached` and `stopped_early`; replace them with
`stop_reason` assertions or absence checks:

```python
assert result.stop_reason == "max_evaluations"
assert not hasattr(result, "budget_reached")
assert not hasattr(result, "stopped_early")
```

- [ ] **Step 2: Run remaining Python tests that were touched**

Run:

```powershell
python -m pytest tests/unit/test_rng_reproducibility.py tests/unit/test_runtime_observability.py tests/integration/ -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Run benchmark import smoke checks**

Run:

```powershell
python -m py_compile tests/benchmarks/bench_ga_vs_deap.py tests/benchmarks/bench_parallel_scaling.py tests/benchmarks/bench_vnext_multifidelity.py
```

Expected: command exits `0`.

- [ ] **Step 4: Commit remaining test updates**

Run:

```powershell
git add tests/unit/test_rng_reproducibility.py tests/unit/test_runtime_observability.py tests/integration/test_binary_onemax.py tests/integration/test_cmaes_rosenbrock.py tests/integration/test_mixed_gene_space.py tests/integration/test_rastrigin.py tests/integration/test_sphere_function.py tests/benchmarks/bench_ga_vs_deap.py tests/benchmarks/bench_parallel_scaling.py tests/benchmarks/bench_vnext_multifidelity.py
git commit -m "test: update budget termination API names"
```

Expected: commit succeeds. If `rg` found and changed additional test files, include only
those task-related files in `git add`.

---

### Task 10: Update Public Docs And Changelog

**Files:**
- Modify: `docs/site/quickstart.md`
- Modify: `docs/site/budget-aware-optimization.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update docs examples and budget terminology**

Run:

```powershell
rg -n "generations=|full_evaluation_budget=|single_full\(budget=|budget_reached|stopped_early|stop_reason" docs/site CHANGELOG.md
```

Make these concrete replacements:

- `generations=` -> `max_generations=`
- `full_evaluation_budget=` -> `max_evaluations=`
- `single_full(budget=` -> `single_full(max_evaluations=`
- Remove public docs text that recommends `budget_reached` or `stopped_early`.
- Describe `stop_reason` as the only final stop status.
- Describe `n_evaluations` as fresh `trusted_full` observations consumed.

In `docs/site/budget-aware-optimization.md`, replace the policy example with:

```markdown
policy = MultiFidelityPolicy(
    rungs=[
        Rung("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        Rung("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    max_evaluations=64,
    batch_size=16,
)
```

Then add:

```markdown
`max_evaluations` counts fresh `trusted_full` observations only. Cached records can
update optimizer state, but they do not spend fresh full-evaluation budget.
```

In `CHANGELOG.md`, under `[Unreleased]` `### Changed`, add:

```markdown
- Budget and termination vocabulary now uses `max_generations` and `max_evaluations`
  consistently. The legacy `generations`, `full_evaluation_budget`,
  `single_full(budget=...)`, `RunResult.stopped_early`, and `RunResult.budget_reached`
  public surfaces were removed in favor of `RunResult.stop_reason`.
```

- [ ] **Step 2: Run docs build and confirm failures are fixed**

Run:

```powershell
python -m mkdocs build --strict
```

Expected: documentation builds successfully without warnings.

- [ ] **Step 3: Commit docs updates**

Run:

```powershell
git add docs/site/quickstart.md docs/site/budget-aware-optimization.md docs/site/ga.md docs/site/cmaes.md docs/site/ask-tell-engines.md docs/site/optimizer-telemetry.md CHANGELOG.md
git commit -m "docs: update budget termination vocabulary"
```

Expected: commit succeeds. If `rg` found and changed additional docs files, include only
those task-related docs in `git add`.

---

### Task 11: Final Repository Sweep And Verification

**Files:**
- Read-only unless verification requires formatting-only fixes

- [ ] **Step 1: Confirm no current public old-name references remain**

Run:

```powershell
rg -n "generations=|full_evaluation_budget=|single_full\(budget=|budget_reached|stopped_early|\"generations\"|self\.generations|\.generations" evocore tests docs/site CHANGELOG.md
```

Expected: no matches, except old historical design/plan documents under
`docs/superpowers/` if you intentionally include that directory in a separate search.

- [ ] **Step 2: Confirm intentional historical references are limited to superpowers docs**

Run:

```powershell
rg -n "generations=|full_evaluation_budget=|single_full\(budget=|budget_reached|stopped_early" docs/superpowers
```

Expected: matches may exist in older specs/plans and in the
`2026-05-15-budget-termination-contract-design.md` migration notes. Do not rewrite old
historical plans unless the user asks.

- [ ] **Step 3: Run formatting check**

Run:

```powershell
python -m ruff format --check
```

Expected: command exits `0`.

- [ ] **Step 4: Run lint**

Run:

```powershell
python -m ruff check
```

Expected: command exits `0`.

- [ ] **Step 5: Rebuild the Python extension**

Run:

```powershell
python -m maturin develop --release
```

Expected: extension builds and installs successfully.

- [ ] **Step 6: Run unit and integration tests**

Run:

```powershell
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Run property tests**

Run:

```powershell
python -m pytest tests/property/ -v
```

Expected: all property tests pass.

- [ ] **Step 8: Run docs build**

Run:

```powershell
python -m mkdocs build --strict
```

Expected: documentation builds successfully without warnings.

- [ ] **Step 9: Commit verification-only fixes if needed**

If verification required formatting or small fixes, stage only task-related files:

```powershell
git add evocore/policies.py evocore/stats.py evocore/ga.py evocore/cmaes.py evocore/callbacks.py tests/unit/test_vnext_policy_scheduler.py tests/vnext_helpers.py tests/unit/test_stats.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_callbacks.py tests/unit/test_rng_reproducibility.py tests/unit/test_runtime_observability.py tests/integration/test_binary_onemax.py tests/integration/test_cmaes_rosenbrock.py tests/integration/test_mixed_gene_space.py tests/integration/test_rastrigin.py tests/integration/test_sphere_function.py tests/benchmarks/bench_ga_vs_deap.py tests/benchmarks/bench_parallel_scaling.py tests/benchmarks/bench_vnext_multifidelity.py docs/site/quickstart.md docs/site/budget-aware-optimization.md docs/site/ga.md docs/site/cmaes.md docs/site/ask-tell-engines.md docs/site/optimizer-telemetry.md CHANGELOG.md
git commit -m "fix: finalize budget termination contract"
```

Expected: no commit is needed if earlier commits already pass verification.

- [ ] **Step 10: Final branch status**

Run:

```powershell
git status --short --branch
```

Expected: working tree is clean on `feature/general-optimizer-framework` or the active
task branch. If dirty files remain, report them and do not push until they are handled.

---

## Implementation Notes

- Do not rename `Rung.budget`; it is rung/fidelity intensity, not run-level evaluation
  budget.
- Keep `RunResult.n_evaluations` as the stable consumed fresh full-evaluation count.
- Keep top-level `RunResult.to_dict()["n_evaluations"]` while also adding
  `RunResult.to_dict()["budget"]["n_evaluations"]`; the top-level field is already part
  of the result envelope.
- Do not add `target_score`, patience, or wall-clock controls in this implementation.
- Do not change old historical plans/specs under `docs/superpowers/` except the new
  implementation plan itself.
