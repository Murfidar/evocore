# Differential Evolution Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `DifferentialEvolutionOptimizer` with mixed bool support, stable ask/tell checkpointing, and a shared optimizer state-acceptance contract.

**Architecture:** Build DE as a Python-only optimizer package under `evocore/optimizers/de/`, following the GA/CMA-ES domain layout. First add the shared acceptance result type, then add DE config, ask/tell state, replacement logic, checkpointing, synchronous run support, and docs. Keep `UpdateResult.accepted_count` as record-ledger acceptance and use `AcceptanceDecision.accepted_for_state` for optimizer-state acceptance.

**Tech Stack:** Python dataclasses and protocols, existing EvoCore lifecycle/checkpoint/result helpers, PyO3 `_core` seed and population helpers, pytest, ruff, maturin.

---

## Reference Documents

- Spec: `docs/superpowers/specs/2026-05-21-differential-evolution-optimizer-design.md`
- Root instructions: `AGENTS.md`

## Branch And Environment Rules

- Start by running `git status --short --branch`.
- If on `main`, run `git switch -c feature/differential-evolution-optimizer`.
- If already on a task branch, continue there unless the user explicitly asks for a new branch.
- Use the repository virtual environment for Python commands:
  `.\.venv\Scripts\python.exe -m ...`
- If `.\.venv\Scripts\python.exe` is missing, stop and report that before using another interpreter.

## File Structure

Create:

- `evocore/optimizers/de/__init__.py`: public DE package exports.
- `evocore/optimizers/de/engine.py`: public optimizer constructor, state summary, config/reproducibility, run helpers.
- `evocore/optimizers/de/ask_tell.py`: candidate proposal, trial generation, tell validation, replacement decisions, events.
- `evocore/optimizers/de/checkpointing.py`: stable ask/tell checkpoint and resume.
- `evocore/optimizers/de/config.py`: config signatures, compatibility validation, reproducibility hooks.
- `tests/unit/test_de_engine.py`: constructor, config, imports, protocol, run behavior.
- `tests/unit/test_de_ask_tell.py`: DE ask/tell initialization, trials, replacement, mixed bool behavior.
- `tests/unit/test_de_checkpointing.py`: stable ask/tell checkpoint/resume coverage.
- `tests/integration/test_de_mixed_gene_space.py`: numeric and mixed-space integration smoke tests.
- `docs/site/de.md`: DE user guide.

Modify:

- `evocore/lifecycle/telemetry.py`: add `AcceptanceDecision`; extend `UpdateResult`.
- `evocore/lifecycle/__init__.py`: export `AcceptanceDecision`.
- `evocore/optimizers/ga/ask_tell.py`: populate acceptance decisions.
- `evocore/optimizers/cmaes/ask_tell.py`: populate acceptance decisions.
- `evocore/optimizers/__init__.py`: lazy export `DifferentialEvolutionOptimizer`.
- `evocore/__init__.py`: top-level export `DifferentialEvolutionOptimizer` and `AcceptanceDecision`.
- `docs/site/api.md`: API entries for acceptance decisions and DE.
- `docs/site/ask-tell-engines.md`: explain ledger acceptance vs state acceptance.
- `docs/site/callbacks-checkpointing.md`: DE checkpoint example.
- `docs/site/gene-space.md`: note DE mixed bool support.
- `mkdocs.yml`: add DE guide to nav.
- `CHANGELOG.md`: public API and behavior entry.
- Existing import/protocol tests: update `tests/unit/test_domain_imports.py`, `tests/unit/test_package_init.py`, `tests/unit/test_protocols.py`.

Do not modify Rust files for this v1 implementation.

---

### Task 1: Add The Shared Acceptance Contract

**Files:**
- Modify: `evocore/lifecycle/telemetry.py`
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`
- Test: `tests/unit/test_vnext_evaluation.py`
- Test: `tests/unit/test_domain_imports.py`
- Test: `tests/unit/test_package_init.py`

- [ ] **Step 1: Write failing lifecycle tests**

Append these tests to `tests/unit/test_vnext_evaluation.py`:

```python
from evocore.lifecycle import AcceptanceDecision


def test_acceptance_decision_exposes_state_acceptance_bool() -> None:
    decision = AcceptanceDecision(
        candidate_id="c-trial",
        batch_id="b-1",
        accepted_for_state=True,
        reason="trial_replaced_target",
        target_candidate_id="c-target",
        target_slot=2,
    )

    assert decision.candidate_id == "c-trial"
    assert decision.batch_id == "b-1"
    assert decision.accepted_for_state is True
    assert decision.reason == "trial_replaced_target"
    assert decision.target_candidate_id == "c-target"
    assert decision.target_slot == 2


def test_update_result_defaults_acceptance_decisions_for_existing_callers() -> None:
    result = UpdateResult(
        accepted_count=0,
        trusted_count=0,
        partial_count=0,
        surrogate_count=0,
        cached_count=0,
        rejected_count=0,
    )

    assert result.acceptance_decisions == ()
    assert result.state_accepted_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vnext_evaluation.py::test_acceptance_decision_exposes_state_acceptance_bool tests/unit/test_vnext_evaluation.py::test_update_result_defaults_acceptance_decisions_for_existing_callers -v
```

Expected: FAIL with `ImportError` or `AttributeError` for `AcceptanceDecision`.

- [ ] **Step 3: Add the dataclass and UpdateResult fields**

In `evocore/lifecycle/telemetry.py`, add the dataclass above `UpdateResult`:

```python
@dataclass(frozen=True)
class AcceptanceDecision:
    """Describe whether one accepted record changed optimizer state."""

    candidate_id: str
    batch_id: str
    accepted_for_state: bool
    reason: str
    target_candidate_id: str | None = None
    target_slot: int | None = None
```

Extend `UpdateResult` with defaulted fields at the end of the dataclass:

```python
    acceptance_decisions: tuple[AcceptanceDecision, ...] = ()
    state_accepted_count: int = 0
```

Update the module export:

```python
__all__ = [
    "AcceptanceDecision",
    "OptimizationTelemetry",
    "OptimizerStateSummary",
    "UpdateResult",
]
```

- [ ] **Step 4: Export AcceptanceDecision from lifecycle and top-level package**

In `evocore/lifecycle/__init__.py`, update the telemetry import:

```python
from evocore.lifecycle.telemetry import (
    AcceptanceDecision,
    OptimizationTelemetry,
    OptimizerStateSummary,
    UpdateResult,
)
```

Add `"AcceptanceDecision"` to `__all__`.

In `evocore/__init__.py`, add `AcceptanceDecision` to the lifecycle import block and to top-level `__all__`.

- [ ] **Step 5: Update import-surface tests**

In `tests/unit/test_domain_imports.py`, update `test_domain_packages_export_symbols_owned_by_focused_modules`:

```python
    from evocore.lifecycle import (
        AcceptanceDecision,
        OptimizationTelemetry,
        OptimizerStateSummary,
        UpdateResult,
        candidate_to_solution,
        solution_to_candidate,
    )
```

Add this assertion near the other lifecycle assertions:

```python
    assert AcceptanceDecision.__module__ == "evocore.lifecycle.telemetry"
```

In `tests/unit/test_package_init.py`, add an assertion in the top-level lifecycle export test:

```python
    assert evocore.AcceptanceDecision.__name__ == "AcceptanceDecision"
```

- [ ] **Step 6: Run the focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add evocore/lifecycle/telemetry.py evocore/lifecycle/__init__.py evocore/__init__.py tests/unit/test_vnext_evaluation.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py
git commit -m "feat(lifecycle): add optimizer acceptance decisions"
```

---

### Task 2: Populate Acceptance Decisions In GA And CMA-ES

**Files:**
- Modify: `evocore/optimizers/ga/ask_tell.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
- Test: `tests/unit/test_ga_ask_tell_vnext.py`
- Test: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Write failing GA acceptance tests**

Append this test to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
def test_ga_tell_reports_state_acceptance_decisions_for_trusted_records() -> None:
    engine = GeneticAlgorithmOptimizer(_space(), population_size=4, max_generations=5, seed=123)
    candidates = engine.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=float(index),
            confidence="trusted_full",
            stage="full",
        )
        for index, candidate in enumerate(candidates)
    ]

    result = engine.tell(records)

    assert result.accepted_count == 4
    assert result.state_accepted_count == 4
    assert [decision.accepted_for_state for decision in result.acceptance_decisions] == [
        True,
        True,
        True,
        True,
    ]
    assert {decision.reason for decision in result.acceptance_decisions} == {
        "state_record_accepted"
    }
    assert {decision.target_candidate_id for decision in result.acceptance_decisions} == {None}
```

Append this no-op test near the existing `tell([])` test:

```python
def test_ga_tell_empty_has_no_acceptance_decisions() -> None:
    engine = GeneticAlgorithmOptimizer(_space(), population_size=4, max_generations=5, seed=123)

    result = engine.tell([])

    assert result.accepted_count == 0
    assert result.state_accepted_count == 0
    assert result.acceptance_decisions == ()
```

- [ ] **Step 2: Write failing CMA-ES acceptance test**

Append this test to `tests/unit/test_cmaes_engine.py`:

```python
def test_cmaes_tell_reports_state_acceptance_decisions_for_complete_batch() -> None:
    engine = CMAESOptimizer(GeneSpace.uniform(-2.0, 2.0, 3), population_size=6, seed=42)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-float(index),
            confidence="trusted_full",
            stage="full",
        )
        for index, candidate in enumerate(candidates)
    ]

    result = engine.tell(records)

    assert result.accepted_count == 6
    assert result.state_accepted_count == 6
    assert len(result.acceptance_decisions) == 6
    assert all(decision.accepted_for_state for decision in result.acceptance_decisions)
    assert {decision.reason for decision in result.acceptance_decisions} == {
        "state_record_accepted"
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_tell_reports_state_acceptance_decisions_for_trusted_records tests/unit/test_ga_ask_tell_vnext.py::test_ga_tell_empty_has_no_acceptance_decisions tests/unit/test_cmaes_engine.py::test_cmaes_tell_reports_state_acceptance_decisions_for_complete_batch -v
```

Expected: FAIL because optimizers do not populate the new fields.

- [ ] **Step 4: Populate GA decisions**

In `evocore/optimizers/ga/ask_tell.py`, import `AcceptanceDecision` from `evocore.lifecycle`.

In `tell`, initialize a list before the loop:

```python
        acceptance_decisions: list[AcceptanceDecision] = []
```

Inside the loop, after `candidate.apply_record(record)`, add:

```python
            if is_state_update_confidence(record.confidence):
                acceptance_decisions.append(
                    AcceptanceDecision(
                        candidate_id=record.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="state_record_accepted",
                    )
                )
```

In the returned `UpdateResult`, add:

```python
            acceptance_decisions=tuple(acceptance_decisions),
            state_accepted_count=len(acceptance_decisions),
```

- [ ] **Step 5: Populate CMA-ES decisions**

In `evocore/optimizers/cmaes/ask_tell.py`, import `AcceptanceDecision` from `evocore.lifecycle`.

In `tell`, initialize a list before the loop:

```python
        acceptance_decisions: list[AcceptanceDecision] = []
```

Inside the loop, after `_apply_record_confidence`, add:

```python
            if is_state_update_confidence(record.confidence):
                acceptance_decisions.append(
                    AcceptanceDecision(
                        candidate_id=record.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="state_record_accepted",
                    )
                )
```

In the returned `UpdateResult`, add:

```python
            acceptance_decisions=tuple(acceptance_decisions),
            state_accepted_count=len(acceptance_decisions),
```

- [ ] **Step 6: Run focused optimizer tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_tell_reports_state_acceptance_decisions_for_trusted_records tests/unit/test_ga_ask_tell_vnext.py::test_ga_tell_empty_has_no_acceptance_decisions tests/unit/test_cmaes_engine.py::test_cmaes_tell_reports_state_acceptance_decisions_for_complete_batch -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add evocore/optimizers/ga/ask_tell.py evocore/optimizers/cmaes/ask_tell.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py
git commit -m "feat(optimizers): report state acceptance decisions"
```

---

### Task 3: Add DE Config, Constructor, And Public Imports

**Files:**
- Create: `evocore/optimizers/de/__init__.py`
- Create: `evocore/optimizers/de/config.py`
- Create: `evocore/optimizers/de/engine.py`
- Modify: `evocore/optimizers/__init__.py`
- Modify: `evocore/__init__.py`
- Test: `tests/unit/test_de_engine.py`
- Test: `tests/unit/test_domain_imports.py`
- Test: `tests/unit/test_package_init.py`

- [ ] **Step 1: Write failing constructor and config tests**

Create `tests/unit/test_de_engine.py`:

```python
import pytest

from evocore import DifferentialEvolutionOptimizer, Gene, GeneSpace
from evocore.core.errors import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def test_de_constructor_sets_public_configuration() -> None:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=8,
        max_generations=12,
        mutation_factor=0.7,
        crossover_rate=0.6,
        seed=123,
        direction="minimize",
    )

    assert engine.population_size == 8
    assert engine.max_generations == 12
    assert engine.mutation_factor == pytest.approx(0.7)
    assert engine.crossover_rate == pytest.approx(0.6)
    assert engine.strategy == "rand1bin"
    assert engine.seed == 123
    assert engine.direction == "minimize"
    assert engine.state_summary().trusted_count == 0


def test_de_config_signature_is_stable_and_hash_changes_with_parameters() -> None:
    left = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.5)
    right = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.9)

    assert left.config_signature()["optimizer_type"] == "DifferentialEvolutionOptimizer"
    assert left.config_hash() != right.config_hash()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gene_space": None}, "gene_space required"),
        ({"population_size": 3}, "population_size"),
        ({"mutation_factor": -0.1}, "mutation_factor"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"parallel": "gpu"}, "parallel"),
        ({"direction": "lowest"}, "direction"),
    ],
)
def test_de_rejects_invalid_configuration(kwargs, message) -> None:
    params = {"gene_space": _space(), **kwargs}
    with pytest.raises(ConfigurationError, match=message):
        DifferentialEvolutionOptimizer(**params)
```

- [ ] **Step 2: Update failing import tests**

In `tests/unit/test_domain_imports.py`, add `"evocore.optimizers.de"` to `modules`.

In `test_new_domain_symbols_are_importable`, import and assert:

```python
    from evocore.optimizers.de import DifferentialEvolutionOptimizer

    assert DifferentialEvolutionOptimizer is not None
```

In `test_domain_packages_export_symbols_owned_by_focused_modules`, import:

```python
    from evocore.optimizers.de import DifferentialEvolutionOptimizer
```

Add:

```python
    assert DifferentialEvolutionOptimizer.__module__ == "evocore.optimizers.de.engine"
```

In `tests/unit/test_package_init.py`, add `DifferentialEvolutionOptimizer` to the top-level optimizer import assertion.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: FAIL because `DifferentialEvolutionOptimizer` does not exist.

- [ ] **Step 4: Create DE config helpers**

Create `evocore/optimizers/de/config.py`:

```python
"""Differential Evolution optimizer configuration helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
)
from evocore.search_space import GeneSpace


class _DEOptimizerLike(Protocol):
    population_size: int
    max_generations: int
    mutation_factor: float
    crossover_rate: float
    strategy: str
    seed: int
    direction: str
    parallel: str
    n_workers: int | None
    track_diversity: bool
    callbacks: Sequence[object]
    max_evaluations: int | None
    gene_space: GeneSpace | None


def build_de_config(optimizer: _DEOptimizerLike) -> OptimizerConfig:
    """Build the canonical Differential Evolution optimizer config."""
    return OptimizerConfig(
        optimizer_type="DifferentialEvolutionOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "max_generations": optimizer.max_generations,
            "mutation_factor": optimizer.mutation_factor,
            "crossover_rate": optimizer.crossover_rate,
            "strategy": optimizer.strategy,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
            "max_evaluations": optimizer.max_evaluations,
            "track_diversity": optimizer.track_diversity,
        },
        components={
            "strategy": {
                "type": optimizer.strategy,
                "parameters": {
                    "mutation_factor": optimizer.mutation_factor,
                    "crossover_rate": optimizer.crossover_rate,
                },
            }
        },
    )


def de_runtime_hooks(optimizer: _DEOptimizerLike) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a DE optimizer."""
    return callback_hook_signatures(optimizer.callbacks)


def de_reproducibility_status(
    optimizer: _DEOptimizerLike,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a DE optimizer."""
    return reproducibility_from_hooks(de_runtime_hooks(optimizer))


def validate_de_compatibility(optimizer: _DEOptimizerLike) -> None:
    """Validate DE optimizer and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for DifferentialEvolutionOptimizer. "
            "Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    if optimizer.strategy != "rand1bin":
        raise ConfigurationError("DifferentialEvolutionOptimizer strategy must be 'rand1bin'.")
    if optimizer.population_size < 4:
        raise ConfigurationError("population_size must be at least 4 for strategy='rand1bin'.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if not math.isfinite(float(optimizer.mutation_factor)) or optimizer.mutation_factor < 0.0:
        raise ConfigurationError("mutation_factor must be finite and >= 0.")
    if not 0.0 <= float(optimizer.crossover_rate) <= 1.0:
        raise ConfigurationError("crossover_rate must be in [0, 1].")
    if optimizer.max_evaluations is not None and optimizer.max_evaluations <= 0:
        raise ConfigurationError("max_evaluations must be positive when provided.")
    if optimizer.parallel not in ("none", "thread", "process"):
        raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
    if optimizer.direction not in ("maximize", "minimize"):
        raise ConfigurationError("direction must be 'maximize' or 'minimize'.")


__all__ = [
    "build_de_config",
    "de_reproducibility_status",
    "de_runtime_hooks",
    "validate_de_compatibility",
]
```

- [ ] **Step 5: Create DE engine skeleton**

Create `evocore/optimizers/de/engine.py`:

```python
"""Differential Evolution optimizer engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evocore.callbacks import Callback
from evocore.core.serialization import package_version
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    Direction,
    OptimizationTelemetry,
    OptimizerStateSummary,
)
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.de.config import (
    build_de_config,
    de_reproducibility_status,
    de_runtime_hooks,
    validate_de_compatibility,
)
from evocore.results import EventHistory, ReproducibilityMetadata
from evocore.search_space import GeneSpace


class DifferentialEvolutionOptimizer:
    """Run Differential Evolution over a flat EvoCore GeneSpace."""

    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        max_generations: int = 300,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.9,
        strategy: str = "rand1bin",
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: object | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            from evocore.core.errors import ConfigurationError

            raise ConfigurationError(
                f"DifferentialEvolutionOptimizer got unexpected argument(s): {unknown}."
            )
        self.gene_space = gene_space
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.strategy = str(strategy)
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.track_diversity = bool(track_diversity)
        self.callbacks = list(callbacks or [])
        validate_de_compatibility(self)
        self._reset_vnext_state()

    def _reset_vnext_state(self) -> None:
        """Reset state used by DE ask/tell and run APIs."""
        self._event_index = 0
        self.generation = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._target_candidate_ids: list[str] = []
        self._trial_target_slots: dict[str, int] = {}
        self._trial_target_candidate_ids: dict[str, str] = {}
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
        self.events = EventHistory()

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(batch_id for batch_id, batch in self._batches_by_id.items() if not batch.consumed)

    def _trusted_count(self) -> int:
        return len(self._target_candidate_ids)

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return self.best_candidate.candidate_id, self.best_candidate.best_state_score(self.direction)

    def state_summary(self) -> OptimizerStateSummary:
        """Return a stable read-only DE state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return OptimizerStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            telemetry=self.vnext_telemetry,
        )

    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_de_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer and gene-space compatibility."""
        validate_de_compatibility(self)

    def _optimizer_config(self) -> dict[str, Any]:
        return self.config_signature()

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        status, notes = de_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="DifferentialEvolutionOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=de_runtime_hooks(self),
        )
```

- [ ] **Step 6: Add DE package exports**

Create `evocore/optimizers/de/__init__.py`:

```python
"""Differential Evolution optimizer implementation."""

from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer

__all__ = ["DifferentialEvolutionOptimizer"]
```

In `evocore/optimizers/__init__.py`, add a lazy branch:

```python
    if name == "DifferentialEvolutionOptimizer":
        from evocore.optimizers.de import DifferentialEvolutionOptimizer

        return DifferentialEvolutionOptimizer
```

Add `"DifferentialEvolutionOptimizer"` to `__all__`.

In `evocore/__init__.py`, import from `evocore.optimizers.de` and add to `__all__`:

```python
from evocore.optimizers.de import DifferentialEvolutionOptimizer
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add evocore/optimizers/de evocore/optimizers/__init__.py evocore/__init__.py tests/unit/test_de_engine.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py
git commit -m "feat(de): add optimizer configuration surface"
```

---

### Task 4: Implement DE Initialization Ask/Tell

**Files:**
- Create: `evocore/optimizers/de/ask_tell.py`
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_ask_tell.py`
- Test: `tests/unit/test_protocols.py`

- [ ] **Step 1: Write failing initialization tests**

Create `tests/unit/test_de_ask_tell.py`:

```python
import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import ConfigurationError, FitnessError


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _records(candidates, scores, confidence="trusted_full"):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence=confidence,
            stage="full",
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def test_de_initial_ask_returns_valid_decoded_candidates() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    candidates = engine.ask()

    assert len(candidates) == 6
    assert {candidate.batch_id for candidate in candidates} == {candidates[0].batch_id}
    assert [candidate.origin for candidate in candidates] == ["random"] * 6
    for candidate in candidates:
        assert isinstance(candidate.genes[0], float)
        assert isinstance(candidate.genes[1], int)
        assert type(candidate.genes[2]) is bool
        assert candidate.genes[3] == pytest.approx(1.5)
        _mixed_space().validate_genes(candidate.genes)


def test_de_initial_tell_fills_target_population_and_best_candidate() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()

    result = engine.tell(_records(candidates, [0.0, 1.0, 5.0, 2.0, 3.0, 4.0]))

    assert result.accepted_count == 6
    assert result.state_accepted_count == 6
    assert len(result.acceptance_decisions) == 6
    assert all(decision.accepted_for_state for decision in result.acceptance_decisions)
    assert result.best_candidate_id == candidates[2].candidate_id
    assert result.best_score == pytest.approx(5.0)
    assert engine.state_summary().trusted_count == 6
    assert engine.state_summary().pending_batch_ids == ()


def test_de_ask_rejects_non_positive_count() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    with pytest.raises(ConfigurationError, match="ask\\(n\\) requires n > 0"):
        engine.ask(0)


def test_de_tell_rejects_unknown_candidate() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    with pytest.raises(FitnessError, match="unknown candidate_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id="missing",
                    batch_id="b-missing",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                )
            ]
        )
```

In `tests/unit/test_protocols.py`, update imports and the runtime optimizer test:

```python
from evocore import DifferentialEvolutionOptimizer


def test_ga_cma_and_de_satisfy_optimizer_protocol_at_runtime() -> None:
    assert isinstance(GeneticAlgorithmOptimizer(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(CMAESOptimizer(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(DifferentialEvolutionOptimizer(_space(), population_size=4, seed=1), Optimizer)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_initial_ask_returns_valid_decoded_candidates tests/unit/test_de_ask_tell.py::test_de_initial_tell_fills_target_population_and_best_candidate tests/unit/test_de_ask_tell.py::test_de_ask_rejects_non_positive_count tests/unit/test_de_ask_tell.py::test_de_tell_rejects_unknown_candidate tests/unit/test_protocols.py::test_ga_cma_and_de_satisfy_optimizer_protocol_at_runtime -v
```

Expected: FAIL because `ask()` and `tell()` are temporary methods.

- [ ] **Step 3: Create ask/tell mixin with initialization helpers**

Create `evocore/optimizers/de/ask_tell.py` with these imports and helpers:

```python
from __future__ import annotations

import math
from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.lifecycle import (
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
    is_state_update_confidence,
    score_for_direction,
    solution_to_candidate,
)
from evocore.results import EventRecord
from evocore.search_space import Solution


def _decode_de_values(gene_space, encoded: Sequence[float]) -> list[float | int | bool]:
    if len(encoded) != gene_space.length:
        raise ConfigurationError(
            f"Expected {gene_space.length} encoded genes, got {len(encoded)}."
        )
    decoded: list[float | int | bool] = []
    for value, gene in zip(encoded, gene_space.genes, strict=False):
        if gene.kind == "bool":
            decoded.append(bool(float(value) >= 0.5))
        elif gene.kind == "int":
            low = float(gene.low)
            high = float(gene.high)
            decoded.append(int(round(min(max(float(value), low), high))))
        else:
            low = float(gene.low)
            high = float(gene.high)
            decoded.append(float(min(max(float(value), low), high)))
    gene_space.validate_genes(decoded)
    return decoded


class DifferentialEvolutionAskTellMixin:
    """Ask/tell lifecycle helpers for Differential Evolution."""

    def _candidate_from_genes(
        self,
        genes: Sequence[float | int | bool],
        *,
        batch_id: str,
        origin: str,
        event_index: int,
        candidate_index: int,
        metadata: dict | None = None,
    ) -> Candidate:
        candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
        candidate = solution_to_candidate(
            Solution(list(genes)),
            gene_space=self.gene_space,
            candidate_id=candidate_id,
            batch_id=batch_id,
            origin=origin,
            event_index=event_index,
        )
        candidate.generation = self.generation
        candidate.metadata.update(dict(metadata or {}))
        return candidate

    def _initial_candidates(self, count: int, event_index: int, batch_id: str) -> list[Candidate]:
        encoded_population = _core.init_population(
            self.gene_space.rust_bounds,
            self.gene_space.kinds,
            count,
            int(_core.py_derive_seed(self.seed, event_index, 0, _core.OP_INIT)),
        )
        return [
            self._candidate_from_genes(
                _decode_de_values(self.gene_space, encoded),
                batch_id=batch_id,
                origin="random",
                event_index=event_index,
                candidate_index=index,
            )
            for index, encoded in enumerate(encoded_population)
        ]
```

- [ ] **Step 4: Add ask, event, and pending helpers**

In the same mixin, add:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        for candidate in candidates:
            self.events.append(
                EventRecord(
                    event_index=len(self.events),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(self.gene_space),
                    generation=candidate.generation,
                    origin=candidate.origin,
                    parents=tuple(candidate.parents),
                    genes=tuple(candidate.genes),
                    params=dict(candidate.params) if candidate.params is not None else None,
                    metadata=dict(candidate.metadata),
                )
            )

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id for batch_id, batch in self._batches_by_id.items() if not batch.consumed
        )

    def ask(self, n: int | None = None) -> list[Candidate]:
        count = int(n or self.population_size)
        if count <= 0:
            raise ConfigurationError("ask(n) requires n > 0.")
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        if len(self._target_candidate_ids) < self.population_size:
            needed = self.population_size - len(self._target_candidate_ids)
            candidates = self._initial_candidates(min(count, needed), event_index, batch_id)
        else:
            candidates = self._trial_candidates(count, event_index, batch_id)
        for candidate in candidates:
            self._candidates_by_id[candidate.candidate_id] = candidate
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        )
        self._event_index += 1
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
        self._append_ask_events(candidates)
        return candidates
```

- [ ] **Step 5: Add initialization tell support**

In the same mixin, add:

```python
    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        candidate = self._candidates_by_id.get(record.candidate_id)
        if candidate is None:
            raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
        if record.batch_id is not None and record.batch_id not in self._batches_by_id:
            raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
        batch = self._batches_by_id.get(candidate.batch_id)
        if batch is None:
            raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
        return candidate, batch

    def _record_best_candidate(self, candidate: Candidate) -> None:
        if self.best_candidate is None or candidate.state_comparison_score(
            self.direction
        ) > self.best_candidate.state_comparison_score(self.direction):
            self.best_candidate = candidate

    def _apply_telemetry_for_record(self, record: EvaluationRecord) -> str:
        if record.confidence == "trusted_full":
            self.vnext_telemetry.record_full(1, stage=record.stage, cost=record.cost)
            return "trusted"
        if record.confidence == "cached":
            self.vnext_telemetry.record_cached(1, stage=record.stage, cost=record.cost)
            return "cached"
        if record.confidence == "partial":
            self.vnext_telemetry.record_partial(1, stage=record.stage, cost=record.cost)
            return "partial"
        if record.confidence == "surrogate":
            self.vnext_telemetry.record_screened(1)
            return "surrogate"
        self.vnext_telemetry.record_eliminated(1, stage=record.stage)
        return "rejected"

    def _append_tell_event(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        *,
        metadata: dict | None = None,
    ) -> None:
        raw_score = float(record.score) if record.score is not None else None
        comparison_score = (
            score_for_direction(raw_score, self.direction)
            if raw_score is not None and math.isfinite(raw_score)
            else None
        )
        event_metadata = dict(record.metadata)
        event_metadata.update(dict(metadata or {}))
        self.events.append(
            EventRecord(
                event_index=len(self.events),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(self.gene_space),
                generation=candidate.generation,
                stage=record.stage,
                confidence=record.confidence,
                raw_score=raw_score,
                comparison_score=comparison_score,
                cost=record.cost,
                status=candidate.status,
                origin=candidate.origin,
                parents=tuple(candidate.parents),
                genes=tuple(candidate.genes),
                params=dict(candidate.params) if candidate.params is not None else None,
                metrics=dict(record.metrics),
                metadata=event_metadata,
            )
        )

    def _batch_complete_for_de(self, batch: CandidateBatch) -> bool:
        terminal_candidate_ids = {
            record.candidate_id
            for record in batch.records_by_key.values()
            if is_state_update_confidence(record.confidence) or record.confidence == "rejected"
        }
        return all(candidate_id in terminal_candidate_ids for candidate_id in batch.candidate_ids)
```

Add the first `tell` implementation:

```python
    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
        trusted = partial = surrogate = cached = rejected = 0
        touched_batch_ids: set[str] = set()
        consumed_batch_ids: set[str] = set()
        acceptance_decisions: list[AcceptanceDecision] = []
        for record in records:
            candidate, batch = self._candidate_and_batch_for_record(record)
            batch.accept_record(record)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            confidence = self._apply_telemetry_for_record(record)
            if confidence == "trusted":
                trusted += 1
            elif confidence == "cached":
                cached += 1
            elif confidence == "partial":
                partial += 1
            elif confidence == "surrogate":
                surrogate += 1
            else:
                rejected += 1
            if is_state_update_confidence(record.confidence):
                if candidate.candidate_id not in self._trial_target_slots:
                    self._target_candidate_ids.append(candidate.candidate_id)
                    self._record_best_candidate(candidate)
                    decision = AcceptanceDecision(
                        candidate_id=candidate.candidate_id,
                        batch_id=batch.batch_id,
                        accepted_for_state=True,
                        reason="initial_target_accepted",
                        target_slot=len(self._target_candidate_ids) - 1,
                    )
                    acceptance_decisions.append(decision)
                    self._append_tell_event(
                        candidate,
                        record,
                        metadata={
                            "accepted_for_state": True,
                            "acceptance_reason": decision.reason,
                            "target_slot": decision.target_slot,
                        },
                    )
                else:
                    decision = self._apply_trial_replacement(candidate, record, batch)
                    acceptance_decisions.append(decision)
            else:
                self._append_tell_event(
                    candidate,
                    record,
                    metadata={
                        "accepted_for_state": False,
                        "acceptance_reason": "record_not_state_eligible",
                    },
                )
            if self._batch_complete_for_de(batch):
                batch.consumed = True
                consumed_batch_ids.add(batch.batch_id)
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=len(records),
            trusted_count=trusted,
            partial_count=partial,
            surrogate_count=surrogate,
            cached_count=cached,
            rejected_count=rejected,
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=tuple(sorted(consumed_batch_ids)),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
            acceptance_decisions=tuple(acceptance_decisions),
            state_accepted_count=sum(
                1 for decision in acceptance_decisions if decision.accepted_for_state
            ),
        )
```

- [ ] **Step 6: Wire mixin into engine**

In `evocore/optimizers/de/engine.py`, import the mixin:

```python
from evocore.optimizers.de.ask_tell import DifferentialEvolutionAskTellMixin
```

Change the class definition:

```python
class DifferentialEvolutionOptimizer(DifferentialEvolutionAskTellMixin):
```

Remove the temporary `ask` and `tell` methods from Task 3.

Keep `engine._pending_batch_ids` only if the ask/tell mixin does not define it; otherwise remove the engine copy to avoid ambiguity.

- [ ] **Step 7: Run focused initialization tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_initial_ask_returns_valid_decoded_candidates tests/unit/test_de_ask_tell.py::test_de_initial_tell_fills_target_population_and_best_candidate tests/unit/test_de_ask_tell.py::test_de_ask_rejects_non_positive_count tests/unit/test_de_ask_tell.py::test_de_tell_rejects_unknown_candidate tests/unit/test_protocols.py::test_ga_cma_and_de_satisfy_optimizer_protocol_at_runtime -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add evocore/optimizers/de/ask_tell.py evocore/optimizers/de/engine.py tests/unit/test_de_ask_tell.py tests/unit/test_protocols.py
git commit -m "feat(de): add initialization ask tell lifecycle"
```

---

### Task 5: Implement Hybrid DE Trial Generation

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Test: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Add failing trial generation tests**

Append these tests to `tests/unit/test_de_ask_tell.py`:

```python
def _trusted_engine() -> tuple[DifferentialEvolutionOptimizer, list]:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0.0, 1.0, 5.0, 2.0, 3.0, 4.0]))
    return engine, candidates


def test_de_trial_ask_returns_one_trial_per_target_with_mapping_metadata() -> None:
    engine, targets = _trusted_engine()

    trials = engine.ask()

    assert len(trials) == 6
    assert {trial.origin for trial in trials} == {"mutation"}
    assert set(engine._trial_target_slots) == {trial.candidate_id for trial in trials}
    assert set(engine._trial_target_candidate_ids) == {trial.candidate_id for trial in trials}
    assert {trial.metadata["target_candidate_id"] for trial in trials} == {
        target.candidate_id for target in targets
    }
    assert {trial.metadata["target_slot"] for trial in trials} == set(range(6))


def test_de_trial_generation_is_deterministic_for_same_seed_and_state() -> None:
    left, _ = _trusted_engine()
    right, _ = _trusted_engine()

    left_trials = left.ask()
    right_trials = right.ask()

    assert [trial.genes for trial in left_trials] == [trial.genes for trial in right_trials]
    assert [trial.metadata["target_slot"] for trial in left_trials] == [
        trial.metadata["target_slot"] for trial in right_trials
    ]


def test_de_trial_generation_preserves_gene_types_and_fixed_values() -> None:
    engine, _ = _trusted_engine()

    trials = engine.ask()

    for trial in trials:
        assert isinstance(trial.genes[0], float)
        assert isinstance(trial.genes[1], int)
        assert type(trial.genes[2]) is bool
        assert trial.genes[3] == pytest.approx(1.5)
        _mixed_space().validate_genes(trial.genes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_trial_ask_returns_one_trial_per_target_with_mapping_metadata tests/unit/test_de_ask_tell.py::test_de_trial_generation_is_deterministic_for_same_seed_and_state tests/unit/test_de_ask_tell.py::test_de_trial_generation_preserves_gene_types_and_fixed_values -v
```

Expected: FAIL because trial generation has not been added.

- [ ] **Step 3: Add deterministic trial helpers**

In `evocore/optimizers/de/ask_tell.py`, import `random`:

```python
import random
```

Add these methods to `DifferentialEvolutionAskTellMixin` above `_trial_candidates`:

```python
    def _rng_for_trial(self, target_slot: int, op: int) -> random.Random:
        seed = int(_core.py_derive_seed(self.seed, self.generation, target_slot, op))
        return random.Random(seed)

    def _donor_slots(self, target_slot: int) -> tuple[int, int, int]:
        choices = [slot for slot in range(len(self._target_candidate_ids)) if slot != target_slot]
        rng = self._rng_for_trial(target_slot, _core.OP_SELECTION)
        selected = rng.sample(choices, 3)
        return int(selected[0]), int(selected[1]), int(selected[2])

    def _target_candidate(self, slot: int) -> Candidate:
        return self._candidates_by_id[self._target_candidate_ids[slot]]

    def _repair_gene_value(self, value: float, gene) -> float | int | bool:
        if gene.kind == "bool":
            return bool(value >= 0.5)
        low = float(gene.low)
        high = float(gene.high)
        clamped = min(max(float(value), low), high)
        if gene.kind == "int":
            return int(round(clamped))
        return float(clamped)
```

- [ ] **Step 4: Implement per-gene trial value logic**

Add these methods to `DifferentialEvolutionAskTellMixin`:

```python
    def _trial_values_for_slot(self, target_slot: int) -> list[float | int | bool]:
        target = self._target_candidate(target_slot)
        a_slot, b_slot, c_slot = self._donor_slots(target_slot)
        donor_a = self._target_candidate(a_slot)
        donor_b = self._target_candidate(b_slot)
        donor_c = self._target_candidate(c_slot)
        mask_rng = self._rng_for_trial(target_slot, _core.OP_CROSSOVER)
        bool_rng = self._rng_for_trial(target_slot, _core.OP_MUTATION)
        variable_indices = self.gene_space.variable_indices
        forced_index = variable_indices[
            mask_rng.randrange(len(variable_indices))
        ] if variable_indices else 0
        values: list[float | int | bool] = []
        for index, gene in enumerate(self.gene_space.genes):
            if gene.is_fixed:
                values.append(self._repair_gene_value(float(gene.low), gene))
                continue
            selected = index == forced_index or mask_rng.random() < self.crossover_rate
            if not selected:
                values.append(target.genes[index])
                continue
            if gene.kind == "bool":
                trial_bool = bool(donor_a.genes[index])
                if bool(donor_b.genes[index]) != bool(donor_c.genes[index]):
                    if bool_rng.random() < min(1.0, self.mutation_factor):
                        trial_bool = not trial_bool
                values.append(trial_bool)
                continue
            mutant = (
                float(donor_a.genes[index])
                + self.mutation_factor
                * (float(donor_b.genes[index]) - float(donor_c.genes[index]))
            )
            values.append(self._repair_gene_value(mutant, gene))
        self.gene_space.validate_genes(values)
        return values

    def _trial_candidates(self, count: int, event_index: int, batch_id: str) -> list[Candidate]:
        target_count = len(self._target_candidate_ids)
        trial_count = min(count, target_count)
        candidates: list[Candidate] = []
        for target_slot in range(trial_count):
            target = self._target_candidate(target_slot)
            genes = self._trial_values_for_slot(target_slot)
            candidate = self._candidate_from_genes(
                genes,
                batch_id=batch_id,
                origin="mutation",
                event_index=event_index,
                candidate_index=target_slot,
                metadata={
                    "target_slot": target_slot,
                    "target_candidate_id": target.candidate_id,
                },
            )
            self._trial_target_slots[candidate.candidate_id] = target_slot
            self._trial_target_candidate_ids[candidate.candidate_id] = target.candidate_id
            candidates.append(candidate)
        return candidates
```

- [ ] **Step 5: Run focused trial generation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_trial_ask_returns_one_trial_per_target_with_mapping_metadata tests/unit/test_de_ask_tell.py::test_de_trial_generation_is_deterministic_for_same_seed_and_state tests/unit/test_de_ask_tell.py::test_de_trial_generation_preserves_gene_types_and_fixed_values -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add evocore/optimizers/de/ask_tell.py tests/unit/test_de_ask_tell.py
git commit -m "feat(de): generate mixed-space trial candidates"
```

---

### Task 6: Implement DE Replacement Semantics

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Test: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Add failing replacement tests**

Append these tests to `tests/unit/test_de_ask_tell.py`:

```python
def test_de_tell_replaces_target_when_trial_is_better() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()
    trial = trials[0]
    target_id = trial.metadata["target_candidate_id"]
    target_slot = trial.metadata["target_slot"]

    result = engine.tell(_records([trial], [100.0]))

    assert result.state_accepted_count == 1
    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].reason == "trial_replaced_target"
    assert result.acceptance_decisions[0].target_candidate_id == target_id
    assert result.acceptance_decisions[0].target_slot == target_slot
    assert engine._target_candidate_ids[target_slot] == trial.candidate_id
    assert targets[0].candidate_id not in engine._target_candidate_ids


def test_de_tell_keeps_target_when_trial_is_worse() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()
    trial = trials[2]
    target_slot = trial.metadata["target_slot"]
    target_id = trial.metadata["target_candidate_id"]

    result = engine.tell(_records([trial], [-100.0]))

    assert result.state_accepted_count == 0
    assert result.acceptance_decisions[0].accepted_for_state is False
    assert result.acceptance_decisions[0].reason == "trial_kept_target"
    assert result.acceptance_decisions[0].target_candidate_id == target_id
    assert result.acceptance_decisions[0].target_slot == target_slot
    assert engine._target_candidate_ids[target_slot] == targets[target_slot].candidate_id


def test_de_tell_replaces_on_equal_score() -> None:
    engine, _ = _trusted_engine()
    trials = engine.ask()
    trial = trials[1]

    result = engine.tell(_records([trial], [1.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].reason == "trial_replaced_target"


def test_de_minimize_replaces_when_trial_score_is_lower() -> None:
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction="minimize",
    )
    targets = engine.ask()
    engine.tell(_records(targets, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]))
    trial = engine.ask()[0]

    result = engine.tell(_records([trial], [1.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.best_candidate_id == trial.candidate_id
    assert result.best_score == pytest.approx(1.0)


def test_de_rejected_trial_does_not_replace_target_and_can_complete_batch() -> None:
    engine, targets = _trusted_engine()
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

    result = engine.tell(rejected)

    assert result.rejected_count == len(trials)
    assert result.state_accepted_count == 0
    assert result.consumed_batch_ids == (trials[0].batch_id,)
    assert engine._target_candidate_ids == [target.candidate_id for target in targets]
    assert engine.generation == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_tell_replaces_target_when_trial_is_better tests/unit/test_de_ask_tell.py::test_de_tell_keeps_target_when_trial_is_worse tests/unit/test_de_ask_tell.py::test_de_tell_replaces_on_equal_score tests/unit/test_de_ask_tell.py::test_de_minimize_replaces_when_trial_score_is_lower tests/unit/test_de_ask_tell.py::test_de_rejected_trial_does_not_replace_target_and_can_complete_batch -v
```

Expected: FAIL because trial replacement has not been added.

- [ ] **Step 3: Implement replacement helper**

Add `_apply_trial_replacement` in `evocore/optimizers/de/ask_tell.py`:

```python
    def _apply_trial_replacement(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        batch: CandidateBatch,
    ) -> AcceptanceDecision:
        target_slot = self._trial_target_slots[candidate.candidate_id]
        target_candidate_id = self._trial_target_candidate_ids[candidate.candidate_id]
        target = self._candidates_by_id[target_candidate_id]
        accepted = candidate.state_comparison_score(self.direction) >= target.state_comparison_score(
            self.direction
        )
        if accepted:
            self._target_candidate_ids[target_slot] = candidate.candidate_id
            self._record_best_candidate(candidate)
            reason = "trial_replaced_target"
        else:
            self._record_best_candidate(target)
            reason = "trial_kept_target"
        decision = AcceptanceDecision(
            candidate_id=candidate.candidate_id,
            batch_id=batch.batch_id,
            accepted_for_state=accepted,
            reason=reason,
            target_candidate_id=target_candidate_id,
            target_slot=target_slot,
        )
        self._append_tell_event(
            candidate,
            record,
            metadata={
                "accepted_for_state": accepted,
                "acceptance_reason": reason,
                "target_candidate_id": target_candidate_id,
                "target_slot": target_slot,
            },
        )
        return decision
```

- [ ] **Step 4: Increment generation after completed trial sweeps**

In `tell`, after marking a batch consumed, add:

```python
                if all(candidate_id in self._trial_target_slots for candidate_id in batch.candidate_ids):
                    self.generation += 1
```

Keep this inside the `if self._batch_complete_for_de(batch):` block so partial batches do not advance generation.

- [ ] **Step 5: Remove completed trial mappings after batch completion**

Inside the same completed-batch block, after the generation increment:

```python
                for candidate_id in batch.candidate_ids:
                    self._trial_target_slots.pop(candidate_id, None)
                    self._trial_target_candidate_ids.pop(candidate_id, None)
```

- [ ] **Step 6: Run full DE ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add evocore/optimizers/de/ask_tell.py tests/unit/test_de_ask_tell.py
git commit -m "feat(de): apply target replacement decisions"
```

---

### Task 7: Add Stable DE Ask/Tell Checkpointing

**Files:**
- Create: `evocore/optimizers/de/checkpointing.py`
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_checkpointing.py`

- [ ] **Step 1: Write failing checkpoint tests**

Create `tests/unit/test_de_checkpointing.py`:

```python
import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import CheckpointError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def _records(candidates, scores):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence="trusted_full",
            stage="full",
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def test_de_checkpoint_restores_after_initial_ask() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    snapshot = engine.ask_tell_checkpoint(metadata={"phase": "submitted"})

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    summary = restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert restored.tell(_records(candidates, [0, 1, 2, 3, 4, 5])).trusted_count == 6


def test_de_checkpoint_restores_after_partial_initial_tell() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates[:3], [0, 1, 2]))
    snapshot = engine.ask_tell_checkpoint()

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())
    result = restored.tell(_records(candidates[3:], [3, 4, 5]))

    assert result.best_score == pytest.approx(5.0)
    assert restored.state_summary().trusted_count == 6


def test_de_checkpoint_restores_pending_trial_mapping() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    targets = engine.ask()
    engine.tell(_records(targets, [0, 1, 2, 3, 4, 5]))
    trials = engine.ask()
    snapshot = engine.ask_tell_checkpoint()

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())
    result = restored.tell(_records([trials[0]], [100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].target_slot == 0


def test_de_checkpoint_rejects_wrong_optimizer_identity() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    snapshot = engine.ask_tell_checkpoint().to_dict()
    snapshot["optimizer"]["optimizer_type"] = "GeneticAlgorithmOptimizer"
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="optimizer_type"):
        restored.resume_ask_tell_checkpoint(snapshot)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -v
```

Expected: FAIL because checkpoint methods do not exist.

- [ ] **Step 3: Create checkpointing mixin**

Create `evocore/optimizers/de/checkpointing.py`:

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.lifecycle import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
from evocore.results import CheckpointSnapshot, validate_checkpoint_identity
from evocore.results import load_checkpoint as load_checkpoint_payload
from evocore.results import save_checkpoint as save_checkpoint_payload

DE_ASK_TELL_STATE_KIND = "de_ask_tell"
DE_CHECKPOINT_STATE_SCHEMA_VERSION = 1


class DifferentialEvolutionCheckpointingMixin:
    """Stable checkpoint helpers for DE ask/tell workflows."""

    @staticmethod
    def load_checkpoint(checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
        return load_checkpoint_payload(checkpoint)

    @staticmethod
    def save_checkpoint(
        checkpoint: str | os.PathLike[str],
        snapshot: CheckpointSnapshot | Mapping[str, Any],
    ) -> None:
        save_checkpoint_payload(checkpoint, snapshot)

    def _validate_stable_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        validate_checkpoint_identity(
            payload,
            optimizer_type="DifferentialEvolutionOptimizer",
            gene_space_hash=self.gene_space.hash(),
            optimizer_config_hash=self.config_hash(),
            seed=self.seed,
            direction=self.direction,
        )
```

- [ ] **Step 4: Add checkpoint creation**

In the same mixin, add:

```python
    def ask_tell_checkpoint(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        best_candidate_id = None if self.best_candidate is None else self.best_candidate.candidate_id
        state_payload = {
            "state_kind": DE_ASK_TELL_STATE_KIND,
            "event_index": self._event_index,
            "generation": self.generation,
            "candidates_by_id": {
                candidate_id: candidate_to_checkpoint(candidate)
                for candidate_id, candidate in sorted(self._candidates_by_id.items())
            },
            "batches_by_id": {
                batch_id: batch_to_checkpoint(batch)
                for batch_id, batch in sorted(self._batches_by_id.items())
            },
            "target_candidate_ids": list(self._target_candidate_ids),
            "trial_target_slots": dict(sorted(self._trial_target_slots.items())),
            "trial_target_candidate_ids": dict(sorted(self._trial_target_candidate_ids.items())),
            "best_candidate_id": best_candidate_id,
            "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            "events": event_history_to_checkpoint(self.events),
        }
        return CheckpointSnapshot(
            optimizer_type="DifferentialEvolutionOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "mode": "ask_tell",
                "event_index": self._event_index,
                "generation": self.generation,
                "pending_batch_ids": list(self._pending_batch_ids()),
                "best_candidate_id": best_candidate_id,
            },
            state={
                "optimizer_type": "DifferentialEvolutionOptimizer",
                "schema_version": DE_CHECKPOINT_STATE_SCHEMA_VERSION,
                "payload": state_payload,
            },
            audit={
                "events": event_history_to_checkpoint(self.events),
                "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            },
            metadata=dict(metadata or {}),
        )
```

- [ ] **Step 5: Add checkpoint restore**

In the same mixin, add:

```python
    def _ask_tell_payload_from_checkpoint(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        state = payload["state"]
        if state.get("schema_version") != DE_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise CheckpointError("checkpoint state.schema_version must be 1.")
        state_payload = state["payload"]
        if state_payload.get("state_kind") != DE_ASK_TELL_STATE_KIND:
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by DE ask/tell resume."
            )
        return state_payload

    def _restore_ask_tell_state(self, state_payload: Mapping[str, Any]) -> None:
        raw_candidates = state_payload.get("candidates_by_id")
        if not isinstance(raw_candidates, Mapping):
            raise CheckpointError("checkpoint state.payload.candidates_by_id must be an object.")
        candidates = {
            str(candidate_id): candidate_from_checkpoint(candidate_payload)
            for candidate_id, candidate_payload in raw_candidates.items()
        }
        for candidate_id, candidate in candidates.items():
            if candidate.candidate_id != candidate_id:
                raise CheckpointError(
                    f"checkpoint candidate key {candidate_id!r} does not match "
                    f"candidate_id {candidate.candidate_id!r}."
                )

        raw_batches = state_payload.get("batches_by_id")
        if not isinstance(raw_batches, Mapping):
            raise CheckpointError("checkpoint state.payload.batches_by_id must be an object.")
        batches = {
            str(batch_id): batch_from_checkpoint(batch_payload)
            for batch_id, batch_payload in raw_batches.items()
        }
        for batch_id, batch in batches.items():
            if batch.batch_id != batch_id:
                raise CheckpointError(
                    f"checkpoint batch key {batch_id!r} does not match batch_id {batch.batch_id!r}."
                )
            for candidate_id in batch.candidate_ids:
                if candidate_id not in candidates:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} references unknown candidate_id {candidate_id!r}."
                    )

        target_candidate_ids = [str(value) for value in state_payload.get("target_candidate_ids") or []]
        for candidate_id in target_candidate_ids:
            if candidate_id not in candidates:
                raise CheckpointError(
                    f"checkpoint target_candidate_id {candidate_id!r} is unknown."
                )
        trial_target_slots = {
            str(candidate_id): int(slot)
            for candidate_id, slot in (state_payload.get("trial_target_slots") or {}).items()
        }
        trial_target_candidate_ids = {
            str(candidate_id): str(target_id)
            for candidate_id, target_id in (
                state_payload.get("trial_target_candidate_ids") or {}
            ).items()
        }
        for candidate_id in set(trial_target_slots) | set(trial_target_candidate_ids):
            if candidate_id not in candidates:
                raise CheckpointError(f"checkpoint trial candidate_id {candidate_id!r} is unknown.")
            if candidate_id not in trial_target_slots or candidate_id not in trial_target_candidate_ids:
                raise CheckpointError(
                    f"checkpoint trial candidate_id {candidate_id!r} must have slot and target mappings."
                )
            target_id = trial_target_candidate_ids[candidate_id]
            if target_id not in candidates:
                raise CheckpointError(
                    f"checkpoint trial target_candidate_id {target_id!r} is unknown."
                )
        best_candidate_id = state_payload.get("best_candidate_id")
        if best_candidate_id is not None and best_candidate_id not in candidates:
            raise CheckpointError(f"checkpoint best_candidate_id {best_candidate_id!r} is unknown.")

        self._candidates_by_id = candidates
        self._batches_by_id = batches
        self._target_candidate_ids = target_candidate_ids
        self._trial_target_slots = trial_target_slots
        self._trial_target_candidate_ids = trial_target_candidate_ids
        self.best_candidate = None if best_candidate_id is None else candidates[best_candidate_id]
        self.vnext_telemetry = telemetry_from_checkpoint(state_payload.get("telemetry") or {})
        self.events = event_history_from_checkpoint(state_payload.get("events") or [])
        self._event_index = int(state_payload.get("event_index", 0))
        self.generation = int(state_payload.get("generation", 0))

    def resume_ask_tell_checkpoint(self, checkpoint: str | os.PathLike[str] | Mapping[str, Any]):
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        state_payload = self._ask_tell_payload_from_checkpoint(payload)
        self._restore_ask_tell_state(state_payload)
        return self.state_summary()
```

Add module exports:

```python
__all__ = [
    "DE_ASK_TELL_STATE_KIND",
    "DE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "DifferentialEvolutionCheckpointingMixin",
]
```

- [ ] **Step 6: Wire checkpointing into engine**

In `evocore/optimizers/de/engine.py`, import:

```python
from evocore.optimizers.de.checkpointing import DifferentialEvolutionCheckpointingMixin
```

Change the class definition:

```python
class DifferentialEvolutionOptimizer(
    DifferentialEvolutionCheckpointingMixin,
    DifferentialEvolutionAskTellMixin,
):
```

- [ ] **Step 7: Run checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add evocore/optimizers/de/checkpointing.py evocore/optimizers/de/engine.py tests/unit/test_de_checkpointing.py
git commit -m "feat(de): add ask tell checkpoints"
```

---

### Task 8: Add Synchronous Run, Generation History, Callbacks, And Parallel Evaluation

**Files:**
- Modify: `evocore/optimizers/de/engine.py`
- Test: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add failing run tests**

Append these tests to `tests/unit/test_de_engine.py`:

```python
from evocore import EvaluationContext, EvaluationRecord, EvaluationStage
from evocore.callbacks import Callback


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes if type(value) is not bool),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CountingCallback(Callback):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.completed = False

    def on_generation_start(self, generation, population) -> None:
        self.starts += 1

    def on_generation_end(self, generation, population, info) -> None:
        self.ends += 1

    def on_run_end(self, result) -> None:
        self.completed = True


def test_de_run_returns_optimization_result_with_events_and_generations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.best_candidate_id is not None
    assert result.n_evaluations >= 6
    assert len(result.generations) == 2
    assert len(result.events) > 0
    assert result.reproducibility is not None
    assert result.reproducibility.optimizer_type == "DifferentialEvolutionOptimizer"


def test_de_run_invokes_callbacks() -> None:
    callback = CountingCallback()
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
        callbacks=[callback],
    )

    engine.run(SphereEvaluator())

    assert callback.starts == 2
    assert callback.ends == 2
    assert callback.completed is True


def test_de_run_honors_max_evaluations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        max_evaluations=8,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations <= 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_run_returns_optimization_result_with_events_and_generations tests/unit/test_de_engine.py::test_de_run_invokes_callbacks tests/unit/test_de_engine.py::test_de_run_honors_max_evaluations -v
```

Expected: FAIL because the engine has no `run` method.

- [ ] **Step 3: Add run imports**

In `evocore/optimizers/de/engine.py`, add imports:

```python
import time
from collections.abc import Callable

from evocore.callbacks import GenerationInfo
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.lifecycle import EvaluationContext, EvaluationStage, Evaluator
from evocore.results import (
    GenerationHistory,
    GenerationRecord,
    OptimizationResult,
    StopReason,
    append_run_stop_event,
)
from evocore.search_space import SolutionSet, candidate_to_solution
```

- [ ] **Step 4: Add callback and evaluation helpers**

Inside `DifferentialEvolutionOptimizer`, add:

```python
    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(seed=self.seed, max_generations=self.max_generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _evaluate_candidates(self, candidates, evaluator: Evaluator, context: EvaluationContext):
        if self.parallel == "process":
            ensure_picklable(evaluator, context="DifferentialEvolutionOptimizer.run parallel='process'")
            with ProcessParallel(
                self.n_workers,
                initializer=self.process_initializer,
                initargs=self.process_initargs,
            ) as parallel:
                return parallel.evaluate(candidates, lambda candidate: evaluator.evaluate([candidate], context)[0])
        if self.parallel == "thread":
            return ThreadParallel(self.n_workers).evaluate(
                candidates,
                lambda candidate: evaluator.evaluate([candidate], context)[0],
            )
        return list(evaluator.evaluate(candidates, context))

    def _evaluation_context(self, candidates, stage: EvaluationStage) -> EvaluationContext:
        batch_ids = {candidate.batch_id for candidate in candidates}
        if len(batch_ids) != 1:
            raise FitnessError("DE run candidates must belong to exactly one batch.")
        return EvaluationContext(
            stage=stage,
            batch_id=next(iter(batch_ids)),
            event_index=candidates[0].event_index if candidates else self._event_index,
            direction=self.direction,
            budget=stage.budget,
        )
```

If the process-mode lambda fails pickling in tests or ruff, replace process support with a top-level helper in `evocore/optimizers/de/engine.py`:

```python
def evaluate_one_candidate(args):
    evaluator, candidate, context = args
    return evaluator.evaluate([candidate], context)[0]
```

Then call `parallel.evaluate([(evaluator, candidate, context) for candidate in candidates], evaluate_one_candidate)`.

- [ ] **Step 5: Add result conversion helpers**

Inside the class, add:

```python
    def _target_solutions(self) -> SolutionSet:
        return SolutionSet(
            [
                candidate_to_solution(
                    self._candidates_by_id[candidate_id],
                    direction=self.direction,
                    gene_space=self.gene_space,
                )
                for candidate_id in self._target_candidate_ids
            ]
        )

    def _append_generation_record(
        self,
        history: GenerationHistory,
        *,
        gen: int,
        gen_start: float,
        n_evaluations: int,
    ) -> None:
        solutions = self._target_solutions()
        if not len(solutions):
            return
        best = solutions.best(1)[0]
        history.append(
            GenerationRecord(
                gen=gen,
                best_score=float(best.score),
                mean_score=solutions.mean_score(),
                std_score=solutions.std_score(),
                wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
                n_evaluations=n_evaluations,
                nan_score_count=0,
                cached_count=0,
                diversity=solutions.diversity() if self.track_diversity else [],
                custom=dict(best.metadata.get("metrics", {})),
            )
        )
```

- [ ] **Step 6: Implement run**

Inside the class, add:

```python
    def run(self, evaluator: Evaluator, policy=None) -> OptimizationResult:
        if policy is not None:
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run does not support policy yet."
            )
        if not isinstance(evaluator, Evaluator):
            raise ConfigurationError(
                "DifferentialEvolutionOptimizer.run requires an evaluator with evaluate(candidates, context)."
            )
        self._reset_vnext_state()
        self._bind_callbacks()
        stage = EvaluationStage(
            name="full",
            budget=1.0,
            promote_fraction=1.0,
            confidence="trusted_full",
        )
        started = time.perf_counter()
        generation_history = GenerationHistory()
        diversity_history: list[list[float]] = []
        elite_history = []
        n_evaluations = 0
        stop_reason: StopReason = "max_generations"

        initial = self.ask(self.population_size)
        initial_context = self._evaluation_context(initial, stage)
        initial_records = self._evaluate_candidates(initial, evaluator, initial_context)
        self.tell(initial_records)
        n_evaluations += len(initial_records)

        for gen in range(self.max_generations):
            gen_start = time.perf_counter()
            current_solutions = self._target_solutions()
            for callback in self.callbacks:
                callback.on_generation_start(gen, current_solutions)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break
            if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
                stop_reason = "max_evaluations"
                break
            remaining = (
                self.population_size
                if self.max_evaluations is None
                else max(self.max_evaluations - n_evaluations, 0)
            )
            trial_count = min(self.population_size, remaining)
            if trial_count <= 0:
                stop_reason = "max_evaluations"
                break
            trials = self.ask(trial_count)
            context = self._evaluation_context(trials, stage)
            records = self._evaluate_candidates(trials, evaluator, context)
            self.tell(records)
            n_evaluations += len(records)
            self._append_generation_record(
                generation_history,
                gen=gen,
                gen_start=gen_start,
                n_evaluations=len(records),
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

        final_solutions = self._target_solutions()
        if not len(final_solutions):
            raise FitnessError("DE run produced no evaluated target candidates.")
        best_solution = final_solutions.best(1)[0].clone()
        result = OptimizationResult(
            best_solution=best_solution,
            best_score=float(best_solution.score),
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - started,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_history,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=self.max_evaluations,
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

- [ ] **Step 7: Run focused run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_run_returns_optimization_result_with_events_and_generations tests/unit/test_de_engine.py::test_de_run_invokes_callbacks tests/unit/test_de_engine.py::test_de_run_honors_max_evaluations -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add evocore/optimizers/de/engine.py tests/unit/test_de_engine.py
git commit -m "feat(de): add synchronous evaluator run"
```

---

### Task 9: Add Integration Coverage For Numeric And Mixed Spaces

**Files:**
- Create: `tests/integration/test_de_mixed_gene_space.py`
- Test: `tests/integration/test_de_mixed_gene_space.py`

- [ ] **Step 1: Write integration tests**

Create `tests/integration/test_de_mixed_gene_space.py`:

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationContext, EvaluationRecord, Gene, GeneSpace


class NumericSphereEvaluator:
    def evaluate(self, candidates, context: EvaluationContext):
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


class MixedSwitchEvaluator:
    def evaluate(self, candidates, context: EvaluationContext):
        records = []
        for candidate in candidates:
            x, period, enabled = candidate.genes
            score = -abs(float(x) - 0.25) - abs(int(period) - 7)
            if enabled:
                score += 2.0
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence=context.stage.confidence,
                    stage=context.stage.name,
                    cost=context.stage.budget,
                )
            )
        return records


def test_de_improves_numeric_sphere_smoke() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 4),
        population_size=12,
        max_generations=5,
        seed=42,
    )

    result = optimizer.run(NumericSphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.best_score > -100.0
    assert result.n_evaluations > 0


def test_de_runs_mixed_bool_numeric_space_smoke() -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -2.0, 2.0),
            Gene("period", "int", 2, 12),
            Gene("enabled", "bool"),
        ]
    )
    optimizer = DifferentialEvolutionOptimizer(space, population_size=10, max_generations=4, seed=7)

    result = optimizer.run(MixedSwitchEvaluator())

    assert type(result.best_solution.values[2]) is bool
    assert result.best_solution.metadata["params"]["enabled"] in (True, False)
    assert result.best_score > -20.0
```

- [ ] **Step 2: Run integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

Run:

```powershell
git add tests/integration/test_de_mixed_gene_space.py
git commit -m "test(de): cover numeric and mixed optimization"
```

---

### Task 10: Add Docs, API References, And Changelog

**Files:**
- Create: `docs/site/de.md`
- Modify: `docs/site/api.md`
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `docs/site/gene-space.md`
- Modify: `mkdocs.yml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Create the DE guide**

Create `docs/site/de.md`:

```markdown
# Differential Evolution

`DifferentialEvolutionOptimizer` proposes one trial candidate per target slot and
keeps the trial when it is at least as good as the incumbent target for the
configured direction.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
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


optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=12,
    max_generations=20,
    seed=42,
)
result = optimizer.run(SphereEvaluator())
```

## Mixed Bool And Numeric Spaces

DE supports flat spaces containing `float`, `int`, and `bool` genes. Float genes
use arithmetic DE variation, integer genes are rounded and clamped, and bool
genes use a deterministic binary rule inspired by GA mixed-space mutation.

```python
from evocore import DifferentialEvolutionOptimizer, Gene, GeneSpace

space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("period", "int", 2, 50),
        Gene("enabled", "bool"),
    ]
)

optimizer = DifferentialEvolutionOptimizer(space, population_size=10, seed=7)
```

## Ask/Tell Acceptance Decisions

`tell()` returns `UpdateResult.acceptance_decisions`. For DE,
`accepted_for_state=True` means the trial replaced its target slot.
`accepted_count` still counts records accepted by the ask/tell ledger.
```

- [ ] **Step 2: Add docs nav and API entries**

In `mkdocs.yml`, add DE after Genetic Algorithms:

```yaml
  - Genetic Algorithms: ga.md
  - Differential Evolution: de.md
  - Budget-Aware Optimization: budget-aware-optimization.md
```

In `docs/site/api.md`, add:

```markdown
::: evocore.lifecycle.AcceptanceDecision

::: evocore.optimizers.de.DifferentialEvolutionOptimizer
```

Place `AcceptanceDecision` near `UpdateResult` and DE near the optimizer references.

- [ ] **Step 3: Update ask/tell docs**

In `docs/site/ask-tell-engines.md`, add this paragraph after the `tell()` introduction:

```markdown
`UpdateResult.accepted_count` counts records accepted by the ask/tell ledger.
Optimizers that make a separate state decision also return
`acceptance_decisions`. `AcceptanceDecision.accepted_for_state` is the per-record
boolean for whether optimizer state changed. For Differential Evolution, this
means a trial replaced its target slot.
```

- [ ] **Step 4: Update checkpoint docs**

In `docs/site/callbacks-checkpointing.md`, add a DE ask/tell section after the CMA-ES ask/tell section:

```markdown
## Differential Evolution Ask/Tell Checkpoints

Differential Evolution ask/tell checkpoints store target slots and pending trial
mappings in addition to candidates, batches, telemetry, and events. This lets a
restored optimizer compare returned trial records against the same target
candidate after resume.
```

- [ ] **Step 5: Update gene-space docs**

In `docs/site/gene-space.md`, update the optimizer support paragraph:

```markdown
`GeneticAlgorithmOptimizer` and `DifferentialEvolutionOptimizer` support flat
spaces that mix `float`, `int`, and `bool` genes.
```

- [ ] **Step 6: Update changelog**

Add an unreleased entry to `CHANGELOG.md`:

```markdown
- Added `DifferentialEvolutionOptimizer` with mixed bool/numeric gene support,
  ask/tell replacement decisions, stable ask/tell checkpoints, and synchronous
  evaluator-driven runs.
- Added `AcceptanceDecision` and `UpdateResult.state_accepted_count` to
  distinguish ask/tell record acceptance from optimizer state acceptance.
```

- [ ] **Step 7: Run docs-related smoke checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add docs/site/de.md docs/site/api.md docs/site/ask-tell-engines.md docs/site/callbacks-checkpointing.md docs/site/gene-space.md mkdocs.yml CHANGELOG.md
git commit -m "docs(de): document differential evolution optimizer"
```

---

### Task 11: Run Focused Verification And Fix Regressions

**Files:**
- Modify only files directly involved in failures from the commands in this task.

- [ ] **Step 1: Run ruff format check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: PASS. If it fails, run:

```powershell
.\.venv\Scripts\python.exe -m ruff format
```

Then rerun the format check and commit formatting-only changes:

```powershell
git add evocore tests docs
git commit -m "style: format differential evolution changes"
```

- [ ] **Step 2: Run ruff lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS. If it fails, fix the named files and rerun this command until it passes. Commit lint fixes:

```powershell
git add evocore tests docs
git commit -m "fix: address differential evolution lint issues"
```

- [ ] **Step 3: Build the extension into the local venv**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: PASS.

- [ ] **Step 4: Run unit and integration tests touched by this plan**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_de_engine.py tests/unit/test_de_ask_tell.py tests/unit/test_de_checkpointing.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py tests/unit/test_protocols.py tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit verification fixes if any**

If files changed during verification, run:

```powershell
git status --short
git add evocore/lifecycle/telemetry.py evocore/lifecycle/__init__.py evocore/__init__.py evocore/optimizers/__init__.py evocore/optimizers/ga/ask_tell.py evocore/optimizers/cmaes/ask_tell.py evocore/optimizers/de tests/unit/test_vnext_evaluation.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_de_engine.py tests/unit/test_de_ask_tell.py tests/unit/test_de_checkpointing.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py tests/unit/test_protocols.py tests/integration/test_de_mixed_gene_space.py docs/site/de.md docs/site/api.md docs/site/ask-tell-engines.md docs/site/callbacks-checkpointing.md docs/site/gene-space.md mkdocs.yml CHANGELOG.md
git commit -m "fix(de): resolve verification issues"
```

---

### Task 12: Final Full Relevant Verification

**Files:**
- No planned source edits.

- [ ] **Step 1: Run full relevant Python verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: all commands PASS.

- [ ] **Step 2: Check Rust status**

Run:

```powershell
git diff --name-only HEAD~12..HEAD -- src Cargo.toml Cargo.lock
```

Expected: no output. If this command shows Rust files, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all Rust commands PASS.

- [ ] **Step 3: Record final status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on the implementation branch.

---

## Self-Review Notes

- Spec coverage: the plan covers the shared acceptance contract, GA/CMA consistency, DE package structure, mixed bool trial generation, replacement semantics, ask/tell checkpointing, synchronous run support, public exports, docs, changelog, and verification.
- Deferred scope from the spec remains deferred: no legacy pickle checkpointing, no `run_multiple(...)`, no custom strategy plugins, and no Rust DE kernel.
- Type consistency: the plan consistently uses `AcceptanceDecision`, `accepted_for_state`, `state_accepted_count`, `DifferentialEvolutionOptimizer`, `DE_ASK_TELL_STATE_KIND`, and `strategy="rand1bin"`.
