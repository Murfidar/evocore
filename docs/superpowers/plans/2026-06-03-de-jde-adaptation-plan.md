# DE jDE Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `strategy="jde-rand1bin"` as the first stateful adaptive Differential Evolution strategy with deterministic per-slot `F` and `CR` state.

**Architecture:** Build on the strategy contract from `feature/de-strategy-contract`. Add `adaptive.py` for jDE state and checkpoint helpers, extend `strategies.py` so jDE reuses `rand1bin` with per-slot parameters, notify adaptive state from `ask_tell.py` after final replacement decisions, and serialize adaptive state in DE ask/tell checkpoints.

**Tech Stack:** Python 3.14, EvoCore Python package, pytest, ruff, existing `_core.py_derive_seed` deterministic seed helper, stable JSON checkpoint envelope.

---

## Dependencies

This plan implements Gate 3 from:

```text
docs/superpowers/specs/2026-06-03-differential-evolution-strategy-adaptation-design.md
```

Start from the completed contract branch. If the built-in strategies branch has already merged, branch from that branch instead and keep the same implementation steps.

```powershell
git switch feature/de-strategy-contract
git pull --ff-only
git switch -c feature/de-jde-adaptation
```

This plan assumes the contract plan already added:

```text
evocore/optimizers/de/strategies.py
TrialContext
TrialProposal
trial_proposal_for_strategy
```

This plan does not implement SHADE-style memory adaptation.

## File Structure

- Create `evocore/optimizers/de/adaptive.py`: jDE state, deterministic parameter proposal, pending trial registration, acceptance handling, checkpoint serialization, and checkpoint restore validation.
- Modify `evocore/optimizers/de/strategies.py`: add `jde-rand1bin` strategy spec and dispatch that reuses `rand1bin` with per-slot jDE parameters.
- Modify `evocore/optimizers/de/engine.py`: initialize adaptive strategy state on reset.
- Modify `evocore/optimizers/de/ask_tell.py`: register pending jDE trial parameters after candidate creation and notify jDE state after final acceptance/rejection.
- Modify `evocore/optimizers/de/checkpointing.py`: save and restore `strategy_state` for jDE.
- Create `tests/unit/test_de_jde.py`: jDE state, metadata, ask/tell adaptation, policy-stage behavior, and checkpoint tests.
- Modify `tests/unit/test_de_engine.py`: constructor/config tests for `jde-rand1bin`.
- Modify `tests/integration/test_de_mixed_gene_space.py`: mixed-space jDE smoke test.
- Modify `docs/site/de.md`: document jDE as the first adaptive strategy and keep SHADE future-facing.
- Modify `CHANGELOG.md`: add jDE entry.

## Task 1: Add Failing jDE Tests

**Files:**
- Create: `tests/unit/test_de_jde.py`
- Modify: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add jDE constructor/config tests**

In `tests/unit/test_de_engine.py`, append:

```python
def test_de_accepts_jde_rand1bin_strategy() -> None:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )

    assert engine.strategy == "jde-rand1bin"
    assert engine.config_signature()["parameters"]["strategy"] == "jde-rand1bin"
    assert engine.config_signature()["components"]["strategy"]["type"] == "jde-rand1bin"


def test_de_jde_strategy_requires_rand1bin_population_size() -> None:
    with pytest.raises(ConfigurationError, match="at least 4"):
        DifferentialEvolutionOptimizer(
            _space(),
            population_size=3,
            strategy="jde-rand1bin",
            seed=42,
        )
```

- [ ] **Step 2: Create `tests/unit/test_de_jde.py`**

Create `tests/unit/test_de_jde.py` with this content:

```python
import copy

import pytest

from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
    Gene,
    GeneSpace,
)
from evocore.core.errors import CheckpointError
from evocore.optimizers.de.adaptive import JDETrialParameters


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def _records(candidates, scores, confidence="trusted_full", stage="full"):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence=confidence,
            stage=stage,
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def _trusted_jde_engine() -> DifferentialEvolutionOptimizer:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )
    targets = engine.ask()
    engine.tell(_records(targets, [0, 1, 2, 3, 4, 5]))
    return engine


def _state_tuple(engine: DifferentialEvolutionOptimizer):
    state = engine._de_strategy_state
    return (
        tuple(engine._target_candidate_ids),
        tuple(round(value, 12) for value in state.f_by_slot),
        tuple(round(value, 12) for value in state.cr_by_slot),
        tuple(sorted(state.pending_trial_params)),
        engine.state_summary().best_candidate_id,
        engine.state_summary().trusted_count,
    )


def test_jde_trial_metadata_is_deterministic_for_same_seed() -> None:
    left = _trusted_jde_engine()
    right = _trusted_jde_engine()

    left_trials = left.ask()
    right_trials = right.ask()

    left_metadata = [
        (
            trial.metadata["strategy"],
            trial.metadata["target_slot"],
            trial.metadata["adaptive_slot"],
            trial.metadata["mutation_factor"],
            trial.metadata["crossover_rate"],
            trial.genes,
        )
        for trial in left_trials
    ]
    right_metadata = [
        (
            trial.metadata["strategy"],
            trial.metadata["target_slot"],
            trial.metadata["adaptive_slot"],
            trial.metadata["mutation_factor"],
            trial.metadata["crossover_rate"],
            trial.genes,
        )
        for trial in right_trials
    ]

    assert left_metadata == right_metadata
    assert {trial.metadata["strategy"] for trial in left_trials} == {"jde-rand1bin"}


def test_jde_acceptance_commits_trial_parameters() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    engine._de_strategy_state.pending_trial_params[trial.candidate_id] = JDETrialParameters(
        target_slot=slot,
        mutation_factor=0.37,
        crossover_rate=0.41,
    )

    result = engine.tell(_records([trial], [100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(0.37)
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(0.41)
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_rejection_preserves_previous_parameters() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )
    engine._de_strategy_state.pending_trial_params[trial.candidate_id] = JDETrialParameters(
        target_slot=slot,
        mutation_factor=0.22,
        crossover_rate=0.33,
    )

    result = engine.tell(_records([trial], [-100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is False
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_partial_records_do_not_adapt_or_clear_pending_trial() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )

    result = engine.tell(_records([trial], [10.0], confidence="partial", stage="cheap"))

    assert result.partial_count == 1
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id in engine._de_strategy_state.pending_trial_params


def test_jde_rejected_record_preserves_previous_parameters_and_clears_pending() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )

    result = engine.tell(_records([trial], [None], confidence="rejected", stage="cheap__de_screened_out"))

    assert result.rejected_count == 1
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_checkpoint_resume_matches_uninterrupted_pending_trial() -> None:
    original = _trusted_jde_engine()
    trials = original.ask()
    original.tell(_records(trials[:2], [100.0, -100.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(trials[2:], [-101.0, 99.0, -102.0, 98.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count


def test_jde_checkpoint_rejects_missing_strategy_state() -> None:
    engine = _trusted_jde_engine()
    engine.ask()
    payload = engine.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"]["strategy_state"]
    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )

    with pytest.raises(CheckpointError, match="strategy_state"):
        restored.resume_ask_tell_checkpoint(payload)
```

- [ ] **Step 3: Run jDE tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_accepts_jde_rand1bin_strategy tests/unit/test_de_jde.py -v
```

Expected: FAIL because `jde-rand1bin` and `evocore.optimizers.de.adaptive` do not exist yet.

## Task 2: Add jDE Adaptive State Module

**Files:**
- Create: `evocore/optimizers/de/adaptive.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Create `adaptive.py`**

Create `evocore/optimizers/de/adaptive.py` with this content:

```python
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from evocore import _core
from evocore.core.errors import CheckpointError

JDE_STATE_SCHEMA_VERSION = 1
JDE_F_REFRESH_PROBABILITY = 0.1
JDE_CR_REFRESH_PROBABILITY = 0.1
JDE_F_LOW = 0.1
JDE_F_HIGH = 1.0


@dataclass(frozen=True)
class JDETrialParameters:
    """Per-trial jDE parameters attached to one pending trial candidate."""

    target_slot: int
    mutation_factor: float
    crossover_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "target_slot": self.target_slot,
            "mutation_factor": self.mutation_factor,
            "crossover_rate": self.crossover_rate,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> JDETrialParameters:
        try:
            return cls(
                target_slot=int(payload["target_slot"]),
                mutation_factor=float(payload["mutation_factor"]),
                crossover_rate=float(payload["crossover_rate"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint state.payload.strategy_state pending trial params are invalid.") from exc


@dataclass
class JDEAdaptiveState:
    """Checkpointable jDE per-slot parameter state."""

    f_by_slot: list[float]
    cr_by_slot: list[float]
    pending_trial_params: dict[str, JDETrialParameters] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        *,
        population_size: int,
        mutation_factor: float,
        crossover_rate: float,
    ) -> JDEAdaptiveState:
        return cls(
            f_by_slot=[float(mutation_factor)] * int(population_size),
            cr_by_slot=[float(crossover_rate)] * int(population_size),
        )

    def propose_parameters(
        self,
        *,
        seed: int,
        generation: int,
        target_slot: int,
    ) -> JDETrialParameters:
        f_value = self.f_by_slot[target_slot]
        cr_value = self.cr_by_slot[target_slot]
        f_rng = _jde_rng(seed, generation, target_slot, offset=1)
        cr_rng = _jde_rng(seed, generation, target_slot, offset=2)
        if f_rng.random() < JDE_F_REFRESH_PROBABILITY:
            f_value = JDE_F_LOW + f_rng.random() * (JDE_F_HIGH - JDE_F_LOW)
        if cr_rng.random() < JDE_CR_REFRESH_PROBABILITY:
            cr_value = cr_rng.random()
        return JDETrialParameters(
            target_slot=target_slot,
            mutation_factor=f_value,
            crossover_rate=cr_value,
        )

    def register_pending(self, candidate_id: str, params: JDETrialParameters) -> None:
        self.pending_trial_params[str(candidate_id)] = params

    def complete_pending(self, candidate_id: str, *, accepted: bool) -> None:
        params = self.pending_trial_params.pop(str(candidate_id), None)
        if params is None:
            return
        if accepted:
            self.f_by_slot[params.target_slot] = params.mutation_factor
            self.cr_by_slot[params.target_slot] = params.crossover_rate

    def discard_pending(self, candidate_id: str) -> None:
        self.pending_trial_params.pop(str(candidate_id), None)

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "strategy": "jde-rand1bin",
            "strategy_state_schema_version": JDE_STATE_SCHEMA_VERSION,
            "f_by_slot": list(self.f_by_slot),
            "cr_by_slot": list(self.cr_by_slot),
            "pending_trial_params": {
                candidate_id: params.to_dict()
                for candidate_id, params in sorted(self.pending_trial_params.items())
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        population_size: int,
    ) -> JDEAdaptiveState:
        if payload.get("strategy") != "jde-rand1bin":
            raise CheckpointError("checkpoint state.payload.strategy_state strategy must be 'jde-rand1bin'.")
        if payload.get("strategy_state_schema_version") != JDE_STATE_SCHEMA_VERSION:
            raise CheckpointError("strategy_state_schema_version 1 is required for strategy='jde-rand1bin'.")
        f_by_slot = _float_list(payload, "f_by_slot")
        cr_by_slot = _float_list(payload, "cr_by_slot")
        if len(f_by_slot) != population_size or len(cr_by_slot) != population_size:
            raise CheckpointError("checkpoint state.payload.strategy_state slot arrays must match population_size.")
        raw_pending = payload.get("pending_trial_params")
        if not isinstance(raw_pending, Mapping):
            raise CheckpointError("checkpoint state.payload.strategy_state.pending_trial_params must be an object.")
        return cls(
            f_by_slot=f_by_slot,
            cr_by_slot=cr_by_slot,
            pending_trial_params={
                str(candidate_id): JDETrialParameters.from_mapping(params)
                for candidate_id, params in raw_pending.items()
            },
        )


def _jde_rng(seed: int, generation: int, target_slot: int, *, offset: int) -> random.Random:
    derived = int(
        _core.py_derive_seed(
            int(seed),
            int(generation),
            int(target_slot) * 10 + int(offset),
            _core.OP_MUTATION,
        )
    )
    return random.Random(derived)  # noqa: S311 - deterministic optimizer sampling.


def _float_list(payload: Mapping[str, Any], key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise CheckpointError(f"checkpoint state.payload.strategy_state.{key} must be an array.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint state.payload.strategy_state.{key} must contain floats.") from exc


def initial_strategy_state(
    *,
    strategy: str,
    population_size: int,
    mutation_factor: float,
    crossover_rate: float,
) -> JDEAdaptiveState | None:
    if strategy == "jde-rand1bin":
        return JDEAdaptiveState.initial(
            population_size=population_size,
            mutation_factor=mutation_factor,
            crossover_rate=crossover_rate,
        )
    return None


def strategy_state_to_checkpoint(state: object | None) -> dict[str, object] | None:
    if isinstance(state, JDEAdaptiveState):
        return state.to_checkpoint()
    return None


def strategy_state_from_checkpoint(
    *,
    strategy: str,
    payload: object,
    population_size: int,
) -> JDEAdaptiveState | None:
    if strategy != "jde-rand1bin":
        return None
    if not isinstance(payload, Mapping):
        raise CheckpointError("strategy_state is required for strategy='jde-rand1bin' checkpoints.")
    return JDEAdaptiveState.from_checkpoint(payload, population_size=population_size)


__all__ = [
    "JDEAdaptiveState",
    "JDETrialParameters",
    "initial_strategy_state",
    "strategy_state_from_checkpoint",
    "strategy_state_to_checkpoint",
]
```

- [ ] **Step 2: Run import-focused jDE tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py::test_jde_trial_metadata_is_deterministic_for_same_seed -v
```

Expected: FAIL because `jde-rand1bin` is still not a supported strategy.

## Task 3: Add `jde-rand1bin` Strategy Dispatch

**Files:**
- Modify: `evocore/optimizers/de/strategies.py`
- Test: `tests/unit/test_de_engine.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Import adaptive state**

Add this import to `strategies.py`:

```python
from evocore.optimizers.de.adaptive import JDEAdaptiveState
```

- [ ] **Step 2: Add the jDE strategy spec**

Add this item to `SUPPORTED_DE_STRATEGIES`:

```python
    "jde-rand1bin": DEStrategySpec(
        name="jde-rand1bin",
        min_population_size=4,
        is_adaptive=True,
        checkpoint_state_schema=1,
    ),
```

- [ ] **Step 3: Add the jDE trial function**

Add this function below `_rand1bin_trial`:

```python
def _jde_rand1bin_trial(context: TrialContext) -> TrialProposal:
    if not isinstance(context.strategy_state, JDEAdaptiveState):
        raise ConfigurationError("strategy_state is required for strategy='jde-rand1bin'.")
    params = context.strategy_state.propose_parameters(
        seed=context.seed,
        generation=context.generation,
        target_slot=context.target_slot,
    )
    proposal = _rand1bin_trial(
        TrialContext(
            strategy_name="rand1bin",
            gene_space=context.gene_space,
            population=context.population,
            target_slot=context.target_slot,
            generation=context.generation,
            seed=context.seed,
            mutation_factor=params.mutation_factor,
            crossover_rate=params.crossover_rate,
            strategy_state=None,
        )
    )
    metadata = dict(proposal.metadata)
    metadata.update(
        {
            "strategy": "jde-rand1bin",
            "adaptive_slot": context.target_slot,
            "mutation_factor": params.mutation_factor,
            "crossover_rate": params.crossover_rate,
        }
    )
    return TrialProposal(genes=proposal.genes, metadata=metadata)
```

- [ ] **Step 4: Extend `trial_proposal_for_strategy`**

Add this branch before the unsupported implementation error:

```python
    if spec.name == "jde-rand1bin":
        return _jde_rand1bin_trial(context)
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_accepts_jde_rand1bin_strategy tests/unit/test_de_engine.py::test_de_jde_strategy_requires_rand1bin_population_size -v
```

Expected: FAIL because the optimizer has not initialized `_de_strategy_state` yet.

## Task 4: Initialize And Register jDE State In The Optimizer

**Files:**
- Modify: `evocore/optimizers/de/engine.py`
- Modify: `evocore/optimizers/de/ask_tell.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Initialize strategy state in `_reset_vnext_state`**

In `evocore/optimizers/de/engine.py`, add this import:

```python
from evocore.optimizers.de.adaptive import initial_strategy_state
```

At the end of `_reset_vnext_state`, add:

```python
        self._de_strategy_state = initial_strategy_state(
            strategy=self.strategy,
            population_size=self.population_size,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
        )
```

- [ ] **Step 2: Pass strategy state to `TrialContext`**

In `evocore/optimizers/de/ask_tell.py`, add this argument when constructing `TrialContext`:

```python
                strategy_state=self._de_strategy_state,
```

- [ ] **Step 3: Import jDE state types into `ask_tell.py`**

Add:

```python
from evocore.optimizers.de.adaptive import JDEAdaptiveState, JDETrialParameters
```

- [ ] **Step 4: Add pending registration helper**

Add this method to `DifferentialEvolutionAskTellMixin` near `_trial_proposal_for_slot`:

```python
    def _record_pending_strategy_trial(self, candidate: Candidate) -> None:
        if not isinstance(self._de_strategy_state, JDEAdaptiveState):
            return
        self._de_strategy_state.register_pending(
            candidate.candidate_id,
            JDETrialParameters(
                target_slot=int(candidate.metadata["adaptive_slot"]),
                mutation_factor=float(candidate.metadata["mutation_factor"]),
                crossover_rate=float(candidate.metadata["crossover_rate"]),
            ),
        )
```

- [ ] **Step 5: Register pending jDE parameters after candidate creation**

In `_trial_candidates`, immediately after creating `candidate`, add:

```python
            self._record_pending_strategy_trial(candidate)
```

- [ ] **Step 6: Run deterministic metadata test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py::test_jde_trial_metadata_is_deterministic_for_same_seed -v
```

Expected: PASS.

- [ ] **Step 7: Commit jDE proposal generation**

Run:

```powershell
git add evocore/optimizers/de/adaptive.py evocore/optimizers/de/strategies.py evocore/optimizers/de/engine.py evocore/optimizers/de/ask_tell.py tests/unit/test_de_engine.py tests/unit/test_de_jde.py
git commit -m "feat(de): add jde trial parameter proposals"
```

## Task 5: Notify jDE State From Tell Decisions

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Add completion helpers**

Add these methods to `DifferentialEvolutionAskTellMixin` near `_record_pending_strategy_trial`:

```python
    def _complete_pending_strategy_trial(self, candidate_id: str, *, accepted: bool) -> None:
        if isinstance(self._de_strategy_state, JDEAdaptiveState):
            self._de_strategy_state.complete_pending(candidate_id, accepted=accepted)

    def _discard_pending_strategy_trial(self, candidate_id: str) -> None:
        if isinstance(self._de_strategy_state, JDEAdaptiveState):
            self._de_strategy_state.discard_pending(candidate_id)
```

- [ ] **Step 2: Notify jDE after trial replacement**

In `_apply_trial_replacement`, after the `decision = AcceptanceDecision` block and before the `_append_tell_event` call, add:

```python
        self._complete_pending_strategy_trial(
            candidate.candidate_id,
            accepted=accepted,
        )
```

- [ ] **Step 3: Discard jDE pending params for rejected terminal records**

In the non-state-eligible branch of `tell`, before the `_append_tell_event` call, add:

```python
                if record.confidence == "rejected":
                    self._discard_pending_strategy_trial(candidate.candidate_id)
```

- [ ] **Step 4: Run jDE adaptation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py::test_jde_acceptance_commits_trial_parameters tests/unit/test_de_jde.py::test_jde_rejection_preserves_previous_parameters tests/unit/test_de_jde.py::test_jde_partial_records_do_not_adapt_or_clear_pending_trial tests/unit/test_de_jde.py::test_jde_rejected_record_preserves_previous_parameters_and_clears_pending -v
```

Expected: PASS.

- [ ] **Step 5: Run DE ask/tell regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_jde.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit jDE tell adaptation**

Run:

```powershell
git add evocore/optimizers/de/ask_tell.py tests/unit/test_de_jde.py
git commit -m "feat(de): update jde parameters from trial acceptance"
```

## Task 6: Add jDE Checkpoint Serialization

**Files:**
- Modify: `evocore/optimizers/de/checkpointing.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Import adaptive checkpoint helpers**

Add this import to `checkpointing.py`:

```python
from evocore.optimizers.de.adaptive import (
    strategy_state_from_checkpoint,
    strategy_state_to_checkpoint,
)
```

- [ ] **Step 2: Save strategy state in `ask_tell_checkpoint`**

In `state_payload`, add:

```python
            "strategy_state": strategy_state_to_checkpoint(self._de_strategy_state),
```

- [ ] **Step 3: Restore strategy state in `_restore_ask_tell_state`**

At the end of `_restore_ask_tell_state`, after assigning `self.generation`, add:

```python
        self._de_strategy_state = strategy_state_from_checkpoint(
            strategy=self.strategy,
            payload=state_payload.get("strategy_state"),
            population_size=self.population_size,
        )
```

- [ ] **Step 4: Update required-field tests for stateless checkpoints**

Do not add `strategy_state` to the required field list in `tests/unit/test_de_checkpointing.py`. Stateless `rand1bin` checkpoints must continue to restore with `strategy_state` missing.

- [ ] **Step 5: Run checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py tests/unit/test_de_jde.py::test_jde_checkpoint_resume_matches_uninterrupted_pending_trial tests/unit/test_de_jde.py::test_jde_checkpoint_rejects_missing_strategy_state -v
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint support**

Run:

```powershell
git add evocore/optimizers/de/checkpointing.py tests/unit/test_de_jde.py
git commit -m "feat(de): checkpoint jde adaptive state"
```

## Task 7: Add Policy And Integration Coverage

**Files:**
- Modify: `tests/unit/test_de_jde.py`
- Modify: `tests/integration/test_de_mixed_gene_space.py`

- [ ] **Step 1: Add policy-driven jDE test**

Append this test to `tests/unit/test_de_jde.py`:

```python
class TwoStageSphere:
    def evaluate(self, candidates, context):
        assert context.stage is not None
        scale = 0.5 if context.stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


def test_jde_policy_run_keeps_strategy_state_consistent() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=18,
        batch_size=6,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        strategy="jde-rand1bin",
        seed=42,
    )

    result = optimizer.run(TwoStageSphere(), policy=policy)

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.reproducibility.optimizer_config["parameters"]["strategy"] == "jde-rand1bin"
    assert len(optimizer._de_strategy_state.f_by_slot) == 6
    assert len(optimizer._de_strategy_state.cr_by_slot) == 6
    assert not optimizer._de_strategy_state.pending_trial_params
```

- [ ] **Step 2: Add mixed-space integration smoke test**

Append this test to `tests/integration/test_de_mixed_gene_space.py`:

```python
def test_de_jde_runs_mixed_bool_numeric_space_smoke() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=8,
        max_generations=3,
        strategy="jde-rand1bin",
        seed=123,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.reproducibility.optimizer_config["parameters"]["strategy"] == "jde-rand1bin"
    assert result.final_solutions
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)
```

- [ ] **Step 3: Run policy and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py::test_jde_policy_run_keeps_strategy_state_consistent tests/integration/test_de_mixed_gene_space.py::test_de_jde_runs_mixed_bool_numeric_space_smoke -v
```

Expected: PASS.

- [ ] **Step 4: Commit policy and integration coverage**

Run:

```powershell
git add tests/unit/test_de_jde.py tests/integration/test_de_mixed_gene_space.py
git commit -m "test(de): cover jde policy and mixed-space runs"
```

## Task 8: Update Docs And Changelog

**Files:**
- Modify: `docs/site/de.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add jDE to the strategy docs**

In `docs/site/de.md`, add this section after the stateless Strategies section:

````markdown
## jDE Adaptation

`strategy="jde-rand1bin"` uses the `rand1bin` donor and crossover shape, but each
target slot carries its own adaptive mutation factor and crossover rate. Trial
parameters are proposed deterministically from the optimizer seed and are kept
only when the trial replaces its target.

```python
adaptive = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=12,
    strategy="jde-rand1bin",
    mutation_factor=0.5,
    crossover_rate=0.9,
    seed=42,
)
```

jDE checkpoints include committed per-slot parameters and pending trial
parameters, so ask/tell checkpoint resume remains deterministic after a trial
batch has been proposed.
````

- [ ] **Step 2: Update the Current Limitations section**

Replace the limitation sentence that says adaptive strategies are not exposed with:

```markdown
DE does not yet expose custom strategy plugins, SHADE-style memory adaptation, or
a Rust-backed variation kernel.
```

- [ ] **Step 3: Add changelog entry**

Under `## [Unreleased]` / `### Added`, add:

```markdown
- Added `strategy="jde-rand1bin"` for simple jDE-style Differential Evolution
  adaptation with checkpointed per-slot mutation and crossover parameters.
```

- [ ] **Step 4: Run docs grep smoke check**

Run:

```powershell
rg -n "jde-rand1bin|jDE|SHADE-style" docs/site/de.md CHANGELOG.md
```

Expected: output includes the new jDE docs, limitation wording, and changelog entry.

- [ ] **Step 5: Commit docs**

Run:

```powershell
git add docs/site/de.md CHANGELOG.md
git commit -m "docs(de): document jde adaptation"
```

## Task 9: Full Verification For jDE

**Files:**
- Verify: Python source, tests, docs, and changelog changes.

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

- [ ] **Step 3: Run DE unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py tests/unit/test_de_strategies.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/unit/test_de_checkpointing.py tests/unit/test_de_multi_run.py -v
```

Expected: PASS.

- [ ] **Step 4: Run DE integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS.

- [ ] **Step 5: Check branch status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `feature/de-jde-adaptation`.

## Self-Review Checklist

- [ ] `jde-rand1bin` uses the `rand1bin` donor/crossover path with per-slot `F` and `CR`.
- [ ] Partial and surrogate records do not update adaptive state.
- [ ] Rejected terminal records clear pending trial params without committing them.
- [ ] Accepted final state records commit pending trial params to the target slot.
- [ ] Existing stateless `rand1bin` checkpoints still restore when `strategy_state` is missing.
- [ ] jDE checkpoints fail clearly when `strategy_state` is missing or malformed.
- [ ] Docs name SHADE-style adaptation as future work.
