# DE Strategy Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `DifferentialEvolutionOptimizer` so the existing `rand1bin` trial generation flows through an internal strategy contract without changing public behavior.

**Architecture:** Add `evocore/optimizers/de/strategies.py` for strategy specs, deterministic trial proposals, shared mixed-gene repair, and `rand1bin`. Keep `ask_tell.py` responsible for candidates, batches, target slots, telemetry, events, and replacement. Keep `config.py` responsible for validation and stable config export.

**Tech Stack:** Python 3.14, EvoCore Python package, Rust-backed `_core` seed helpers, pytest, ruff, maturin-built extension already available in `.venv`.

---

## Dependencies

This plan implements Gate 1 from:

```text
docs/superpowers/specs/2026-06-03-differential-evolution-strategy-adaptation-design.md
```

Create and work on this branch:

```powershell
git switch main
git pull --ff-only
git switch -c feature/de-strategy-contract
```

Do not start the built-in strategy or jDE plans until this branch passes verification.

## File Structure

- Create `evocore/optimizers/de/strategies.py`: internal strategy specs, shared gene repair, deterministic RNG helpers, `TrialContext`, `TrialProposal`, and `rand1bin` proposal generation.
- Modify `evocore/optimizers/de/ask_tell.py`: delegate trial value construction to `strategies.py`; keep candidate and batch lifecycle unchanged.
- Modify `evocore/optimizers/de/config.py`: resolve strategy validation through `strategies.py` while still only accepting `rand1bin`.
- Modify `tests/unit/test_de_ask_tell.py`: lock current deterministic `rand1bin` trial proposals with concrete expected values.
- Create `tests/unit/test_de_strategies.py`: unit-test the new strategy contract.

## Task 1: Lock Current `rand1bin` Behavior

**Files:**
- Modify: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Add a deterministic golden test for current trial genes**

Append this test after `test_de_trial_generation_is_deterministic_for_same_seed_and_state`:

```python
def test_de_rand1bin_trial_generation_matches_locked_fixture() -> None:
    engine, _ = _trusted_engine()

    trials = engine.ask()

    assert [trial.candidate_id for trial in trials] == [
        "c-2ecb9a886a1c5508197db432ebefc5b1",
        "c-f61cb32190a1eb220d5de69d29a9ecd7",
        "c-f53cc27736ff8d1a0f22af8298c8e374",
        "c-19f55af7a3d037b865a01a90e8d00068",
        "c-3df85dda8060132d4d1c1ffd070710eb",
        "c-1520ec64a9be278b82b86b1858f7ee28",
    ]
    assert [trial.genes for trial in trials] == [
        [pytest.approx(1.5811814881513984), 2, True, pytest.approx(1.5)],
        [pytest.approx(3.425316280812999), 2, False, pytest.approx(1.5)],
        [pytest.approx(3.675373942863314), 8, False, pytest.approx(1.5)],
        [pytest.approx(-3.555239267967455), 15, False, pytest.approx(1.5)],
        [pytest.approx(-3.555239267967455), 9, False, pytest.approx(1.5)],
        [pytest.approx(-2.116060077343211), 20, True, pytest.approx(1.5)],
    ]
    assert [trial.metadata["target_slot"] for trial in trials] == [0, 1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run the golden test before refactoring**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_rand1bin_trial_generation_matches_locked_fixture -v
```

Expected: PASS. This is a characterization test, so it should pass before the refactor.

- [ ] **Step 3: Commit the characterization test**

Run:

```powershell
git add tests/unit/test_de_ask_tell.py
git commit -m "test(de): lock rand1bin trial generation"
```

## Task 2: Add Strategy Contract Tests

**Files:**
- Create: `tests/unit/test_de_strategies.py`

- [ ] **Step 1: Create failing strategy-contract tests**

Create `tests/unit/test_de_strategies.py` with this content:

```python
import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.optimizers.de.strategies import (
    TrialContext,
    repair_de_gene_value,
    strategy_spec_for,
    trial_proposal_for_strategy,
)


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
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


def _trusted_population():
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    return engine


def test_strategy_spec_for_returns_rand1bin_contract() -> None:
    spec = strategy_spec_for("rand1bin")

    assert spec.name == "rand1bin"
    assert spec.min_population_size == 4
    assert spec.is_adaptive is False
    assert spec.checkpoint_state_schema is None


def test_strategy_spec_for_rejects_unknown_strategy() -> None:
    with pytest.raises(ConfigurationError, match="strategy must be one of 'rand1bin'"):
        strategy_spec_for("best1bin")


def test_repair_de_gene_value_preserves_mixed_gene_contract() -> None:
    space = _mixed_space()

    assert repair_de_gene_value(-7.0, space.genes[0]) == pytest.approx(-5.0)
    assert repair_de_gene_value(99.0, space.genes[1]) == 20
    assert repair_de_gene_value(0.49, space.genes[2]) is False
    assert repair_de_gene_value(0.50, space.genes[2]) is True
    assert repair_de_gene_value(2.0, space.genes[3]) == pytest.approx(1.5)


def test_rand1bin_strategy_proposal_matches_optimizer_fixture() -> None:
    engine = _trusted_population()
    population = [engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name="rand1bin",
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    assert proposal.genes == [
        pytest.approx(1.5811814881513984),
        2,
        True,
        pytest.approx(1.5),
    ]
    assert proposal.metadata["strategy"] == "rand1bin"
    assert proposal.metadata["target_slot"] == 0
    assert proposal.metadata["donor_slots"] == (4, 2, 1)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evocore.optimizers.de.strategies'`.

## Task 3: Implement `evocore/optimizers/de/strategies.py`

**Files:**
- Create: `evocore/optimizers/de/strategies.py`

- [ ] **Step 1: Create the strategy module**

Create `evocore/optimizers/de/strategies.py` with this content:

```python
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.search_space import GeneSpace


@dataclass(frozen=True)
class DEStrategySpec:
    """Internal Differential Evolution strategy metadata."""

    name: str
    min_population_size: int
    is_adaptive: bool = False
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_state_schema: int | None = None


@dataclass(frozen=True)
class TrialContext:
    """Inputs needed to build one DE trial vector."""

    strategy_name: str
    gene_space: GeneSpace
    population: Sequence[Candidate]
    target_slot: int
    generation: int
    seed: int
    mutation_factor: float
    crossover_rate: float
    strategy_state: object | None = None


@dataclass(frozen=True)
class TrialProposal:
    """Strategy output before ask/tell wraps it as a Candidate."""

    genes: list[float | int | bool]
    metadata: dict[str, object]


SUPPORTED_DE_STRATEGIES: dict[str, DEStrategySpec] = {
    "rand1bin": DEStrategySpec(name="rand1bin", min_population_size=4),
}


def supported_strategy_names():
    """Return strategy names in a stable display order."""

    return tuple(SUPPORTED_DE_STRATEGIES)


def strategy_spec_for(strategy: str) -> DEStrategySpec:
    """Return the internal strategy spec or raise a user-facing config error."""

    try:
        return SUPPORTED_DE_STRATEGIES[str(strategy)]
    except KeyError as exc:
        accepted = "', '".join(supported_strategy_names())
        raise ConfigurationError(
            f"DifferentialEvolutionOptimizer strategy must be one of '{accepted}'."
        ) from exc


def validate_strategy_population_size(strategy: str, population_size: int) -> None:
    """Validate population size against the selected strategy."""

    spec = strategy_spec_for(strategy)
    if int(population_size) < spec.min_population_size:
        raise ConfigurationError(
            "population_size must be at least "
            f"{spec.min_population_size} for strategy={spec.name!r}."
        )


def rng_for_de_trial(seed: int, generation: int, target_slot: int, op: int) -> random.Random:
    """Return deterministic per-trial RNG matching the original DE implementation."""

    derived = int(_core.py_derive_seed(int(seed), int(generation), int(target_slot), int(op)))
    return random.Random(derived)  # noqa: S311 - deterministic optimizer sampling.


def repair_de_gene_value(value: float, gene) -> float | int | bool:
    """Repair one DE gene value according to the mixed GeneSpace contract."""

    if gene.kind == "bool":
        return bool(float(value) >= 0.5)
    low = float(gene.low)
    high = float(gene.high)
    clamped = min(max(float(value), low), high)
    if gene.kind == "int":
        return int(round(clamped))
    return float(clamped)


def _target_candidate(context: TrialContext) -> Candidate:
    return context.population[context.target_slot]


def _rand1bin_donor_slots(context: TrialContext) -> tuple[int, int, int]:
    choices = [slot for slot in range(len(context.population)) if slot != context.target_slot]
    rng = rng_for_de_trial(context.seed, context.generation, context.target_slot, _core.OP_SELECTION)
    selected = rng.sample(choices, 3)
    return int(selected[0]), int(selected[1]), int(selected[2])


def _forced_variable_index(context: TrialContext, rng: random.Random) -> int:
    variable_indices = context.gene_space.variable_indices
    return variable_indices[rng.randrange(len(variable_indices))] if variable_indices else 0


def _rand1bin_trial(context: TrialContext) -> TrialProposal:
    if len(context.population) < strategy_spec_for("rand1bin").min_population_size:
        validate_strategy_population_size("rand1bin", len(context.population))

    target = _target_candidate(context)
    a_slot, b_slot, c_slot = _rand1bin_donor_slots(context)
    donor_a = context.population[a_slot]
    donor_b = context.population[b_slot]
    donor_c = context.population[c_slot]
    mask_rng = rng_for_de_trial(
        context.seed, context.generation, context.target_slot, _core.OP_CROSSOVER
    )
    bool_rng = rng_for_de_trial(
        context.seed, context.generation, context.target_slot, _core.OP_MUTATION
    )
    forced_index = _forced_variable_index(context, mask_rng)
    values: list[float | int | bool] = []

    for index, gene in enumerate(context.gene_space.genes):
        if gene.is_fixed:
            values.append(repair_de_gene_value(float(gene.low), gene))
            continue
        selected = index == forced_index or mask_rng.random() < context.crossover_rate
        if not selected:
            values.append(target.genes[index])
            continue
        if gene.kind == "bool":
            trial_bool = bool(donor_a.genes[index])
            if bool(donor_b.genes[index]) != bool(
                donor_c.genes[index]
            ) and bool_rng.random() < min(1.0, context.mutation_factor):
                trial_bool = not trial_bool
            values.append(trial_bool)
            continue
        mutant = float(donor_a.genes[index]) + context.mutation_factor * (
            float(donor_b.genes[index]) - float(donor_c.genes[index])
        )
        values.append(repair_de_gene_value(mutant, gene))

    context.gene_space.validate_genes(values)
    return TrialProposal(
        genes=values,
        metadata={
            "strategy": "rand1bin",
            "target_slot": context.target_slot,
            "donor_slots": (a_slot, b_slot, c_slot),
        },
    )


def trial_proposal_for_strategy(context: TrialContext) -> TrialProposal:
    """Build a trial proposal for the selected internal strategy."""

    spec = strategy_spec_for(context.strategy_name)
    validate_strategy_population_size(spec.name, len(context.population))
    if spec.name == "rand1bin":
        return _rand1bin_trial(context)
    raise ConfigurationError(f"Unsupported DE strategy implementation: {spec.name!r}.")


__all__ = [
    "DEStrategySpec",
    "SUPPORTED_DE_STRATEGIES",
    "TrialContext",
    "TrialProposal",
    "repair_de_gene_value",
    "rng_for_de_trial",
    "strategy_spec_for",
    "supported_strategy_names",
    "trial_proposal_for_strategy",
    "validate_strategy_population_size",
]
```

- [ ] **Step 2: Run the strategy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py -v
```

Expected: PASS.

## Task 4: Delegate Ask/Tell Trial Generation To The Strategy Module

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`

- [ ] **Step 1: Replace strategy-related imports**

In `evocore/optimizers/de/ask_tell.py`, remove:

```python
import random
```

Add this import after the lifecycle imports:

```python
from evocore.optimizers.de.strategies import (
    TrialContext,
    repair_de_gene_value,
    rng_for_de_trial,
    trial_proposal_for_strategy,
)
```

- [ ] **Step 2: Replace local RNG and repair helpers with delegates**

Replace the existing `_rng_for_trial` and `_repair_gene_value` methods with:

```python
    def _rng_for_trial(self, target_slot: int, op: int):
        return rng_for_de_trial(self.seed, self.generation, target_slot, op)

    def _repair_gene_value(self, value: float, gene) -> float | int | bool:
        return repair_de_gene_value(value, gene)
```

Keep these delegates temporarily so existing internal tests and future diffs stay small.

- [ ] **Step 3: Replace `_donor_slots` and `_trial_values_for_slot`**

Remove the existing `_donor_slots` method.

Replace `_trial_values_for_slot` with these two methods:

```python
    def _target_population(self) -> list[Candidate]:
        return [
            self._candidates_by_id[candidate_id]
            for candidate_id in self._target_candidate_ids
        ]

    def _trial_proposal_for_slot(self, target_slot: int):
        return trial_proposal_for_strategy(
            TrialContext(
                strategy_name=self.strategy,
                gene_space=self.gene_space,
                population=self._target_population(),
                target_slot=target_slot,
                generation=self.generation,
                seed=self.seed,
                mutation_factor=self.mutation_factor,
                crossover_rate=self.crossover_rate,
            )
        )
```

- [ ] **Step 4: Update `_trial_candidates` to use proposals**

Inside `_trial_candidates`, replace:

```python
            genes = self._trial_values_for_slot(target_slot)
```

with:

```python
            proposal = self._trial_proposal_for_slot(target_slot)
            genes = proposal.genes
            metadata = dict(proposal.metadata)
            metadata["target_candidate_id"] = target.candidate_id
```

Then replace the candidate metadata block:

```python
                metadata={
                    "target_slot": target_slot,
                    "target_candidate_id": target.candidate_id,
                },
```

with:

```python
                metadata=metadata,
```

- [ ] **Step 5: Run ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_strategies.py -v
```

Expected: PASS.

## Task 5: Route Config Validation Through Strategy Specs

**Files:**
- Modify: `evocore/optimizers/de/config.py`
- Test: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add a config validation test for the new error wording**

In `tests/unit/test_de_engine.py`, add this test after `test_de_rejects_invalid_configuration`:

```python
def test_de_rejects_unknown_strategy_with_supported_names() -> None:
    with pytest.raises(ConfigurationError, match="strategy must be one of 'rand1bin'"):
        DifferentialEvolutionOptimizer(_space(), population_size=8, strategy="best1bin")
```

- [ ] **Step 2: Run the new test before config changes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_rejects_unknown_strategy_with_supported_names -v
```

Expected: FAIL because the current message says `strategy must be 'rand1bin'`.

- [ ] **Step 3: Modify `config.py` imports**

Add this import below the existing optimizer config imports:

```python
from evocore.optimizers.de.strategies import (
    strategy_spec_for,
    validate_strategy_population_size,
)
```

- [ ] **Step 4: Replace strategy validation in `validate_de_compatibility`**

Replace:

```python
    if optimizer.strategy != "rand1bin":
        raise ConfigurationError("DifferentialEvolutionOptimizer strategy must be 'rand1bin'.")
    if optimizer.population_size < 4:
        raise ConfigurationError("population_size must be at least 4 for strategy='rand1bin'.")
```

with:

```python
    strategy_spec_for(optimizer.strategy)
    validate_strategy_population_size(optimizer.strategy, optimizer.population_size)
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_rejects_unknown_strategy_with_supported_names tests/unit/test_de_engine.py::test_de_rejects_invalid_configuration tests/unit/test_de_engine.py::test_de_config_signature_is_stable_and_hash_changes_with_parameters -v
```

Expected: PASS.

## Task 6: Verify Checkpoint Compatibility

**Files:**
- Test: `tests/unit/test_de_checkpointing.py`
- Test: `tests/unit/test_checkpoint_golden_fixtures.py`

- [ ] **Step 1: Run DE checkpoint unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -v
```

Expected: PASS. Existing `rand1bin` checkpoints must restore after the refactor.

- [ ] **Step 2: Run golden checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit the strategy contract refactor**

Run:

```powershell
git add evocore/optimizers/de/strategies.py evocore/optimizers/de/ask_tell.py evocore/optimizers/de/config.py tests/unit/test_de_strategies.py tests/unit/test_de_engine.py
git commit -m "refactor(de): route rand1bin through strategy contract"
```

## Task 7: Full Verification For The Contract Slice

**Files:**
- Verify: Python source, tests, and docs unchanged by this slice.

- [ ] **Step 1: Run formatting check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: PASS.

- [ ] **Step 2: Run lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS.

- [ ] **Step 3: Run DE-focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/unit/test_de_checkpointing.py tests/unit/test_de_multi_run.py -v
```

Expected: PASS.

- [ ] **Step 4: Rebuild the extension if local imports fail**

Run this only if pytest reports that `evocore._core` cannot be imported:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: command exits successfully and subsequent pytest imports `_core`.

- [ ] **Step 5: Check branch status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `feature/de-strategy-contract`.

## Self-Review Checklist

- [ ] `rand1bin` trial genes match the locked fixture.
- [ ] `ask_tell.py` still owns `Candidate` creation and replacement decisions.
- [ ] `strategies.py` does not import `DifferentialEvolutionOptimizer`.
- [ ] `config_signature()` for `rand1bin` keeps the same public shape.
- [ ] Existing DE checkpoints restore without adding `strategy_state`.
- [ ] The built-in strategy and jDE plans can depend on `TrialContext`, `TrialProposal`, and `trial_proposal_for_strategy`.
