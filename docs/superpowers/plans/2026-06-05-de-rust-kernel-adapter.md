# DE Rust Kernel Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Differential Evolution Rust-kernel marshalling out of ask/tell lifecycle code and into a focused Python adapter.

**Architecture:** Add a private `evocore/optimizers/de/kernel.py` adapter that consumes shared search-space codec helpers, calls `_core.de_generate_trials(...)`, validates Rust proposal payloads, and returns `TrialProposal` objects. DE ask/tell remains responsible for candidates, pending maps, events, telemetry, jDE commit/discard, replacement, and checkpoints.

**Tech Stack:** Python 3.11+, PyO3 extension, pytest, ruff.

---

## Prerequisite

Complete `docs/superpowers/plans/2026-06-05-search-space-codec-contract.md` first. This plan assumes `encode_gene_values(...)` and `decode_gene_values(...)` exist in `evocore.search_space`.

---

## File Structure

- Create: `evocore/optimizers/de/kernel.py`
  - Private adapter for `_core.de_generate_trials(...)`.
- Modify: `evocore/optimizers/de/ask_tell.py`
  - Replaces `_rust_trial_proposals(...)` raw marshalling with the adapter.
- Create: `tests/unit/test_de_kernel_adapter.py`
  - Adapter unit tests with `_core.de_generate_trials` monkeypatched.
- Modify: `tests/unit/test_de_ask_tell.py`
  - Keeps lifecycle behavior covered after adapter extraction.

---

### Task 1: Add Failing Adapter Tests

**Files:**
- Create: `tests/unit/test_de_kernel_adapter.py`

- [ ] **Step 1: Write tests for adapter marshalling and decoding**

Create `tests/unit/test_de_kernel_adapter.py` with:

```python
from __future__ import annotations

import pytest

from evocore import EvaluationRecord, _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.optimizers.de.kernel import DERustKernelAdapter
from evocore.search_space import Gene, GeneSpace


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _candidate(candidate_id: str, genes) -> Candidate:
    candidate = Candidate(candidate_id=candidate_id, genes=list(genes), batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id=candidate_id,
            batch_id="b-1",
            score=float(candidate_id.rsplit("-", 1)[-1]),
            confidence="trusted_full",
            stage="full",
        )
    )
    return candidate


def test_adapter_passes_encoded_population_and_kernel_arguments(monkeypatch) -> None:
    calls = []

    def fake_de_generate_trials(
        population,
        scores,
        gene_bounds,
        gene_kinds,
        strategy,
        mutation_factor,
        crossover_rate,
        seed,
        generation,
        target_slots,
        direction,
        jde_state=None,
    ):
        calls.append(
            {
                "population": population,
                "scores": scores,
                "gene_bounds": gene_bounds,
                "gene_kinds": gene_kinds,
                "strategy": strategy,
                "mutation_factor": mutation_factor,
                "crossover_rate": crossover_rate,
                "seed": seed,
                "generation": generation,
                "target_slots": target_slots,
                "direction": direction,
                "jde_state": jde_state,
            }
        )
        return [
            {
                "target_slot": 0,
                "genes": [99.0, 20.8, 0.2, -9.0],
                "metadata": {
                    "strategy": strategy,
                    "target_slot": 0,
                    "base_slot": 1,
                    "donor_slots": [1, 2, 3],
                    "difference_pairs": [[2, 3]],
                },
            }
        ]

    monkeypatch.setattr(_core, "de_generate_trials", fake_de_generate_trials)

    proposals = DERustKernelAdapter().generate_trials(
        target_population=[
            _candidate("candidate-0", [0.25, 7, True, 1.5]),
            _candidate("candidate-1", [0.5, 8, False, 1.5]),
            _candidate("candidate-2", [0.75, 9, True, 1.5]),
            _candidate("candidate-3", [1.0, 10, False, 1.5]),
        ],
        scores=[0.0, 1.0, 2.0, 3.0],
        gene_space=_space(),
        strategy="rand1bin",
        mutation_factor=0.7,
        crossover_rate=0.9,
        seed=42,
        generation=3,
        target_slots=[0],
        direction="maximize",
        jde_state={"f_by_slot": [0.5] * 4, "cr_by_slot": [0.9] * 4},
    )

    assert calls == [
        {
            "population": [
                [0.25, 7.0, 1.0, 1.5],
                [0.5, 8.0, 0.0, 1.5],
                [0.75, 9.0, 1.0, 1.5],
                [1.0, 10.0, 0.0, 1.5],
            ],
            "scores": [0.0, 1.0, 2.0, 3.0],
            "gene_bounds": [(-5.0, 5.0), (2.0, 20.0), (0.0, 1.0), (1.5, 1.5)],
            "gene_kinds": ["float", "int", "bool", "float"],
            "strategy": "rand1bin",
            "mutation_factor": 0.7,
            "crossover_rate": 0.9,
            "seed": 42,
            "generation": 3,
            "target_slots": [0],
            "direction": "maximize",
            "jde_state": {"f_by_slot": [0.5] * 4, "cr_by_slot": [0.9] * 4},
        }
    ]
    assert len(proposals) == 1
    assert proposals[0].genes == [5.0, 20, False, pytest.approx(1.5)]
    assert proposals[0].metadata["strategy"] == "rand1bin"


def test_adapter_rejects_score_population_length_mismatch() -> None:
    with pytest.raises(ConfigurationError, match="scores length"):
        DERustKernelAdapter().generate_trials(
            target_population=[_candidate("candidate-0", [0.25, 7, True, 1.5])],
            scores=[],
            gene_space=_space(),
            strategy="rand1bin",
            mutation_factor=0.7,
            crossover_rate=0.9,
            seed=42,
            generation=3,
            target_slots=[0],
            direction="maximize",
            jde_state=None,
        )


def test_adapter_rejects_malformed_rust_payload(monkeypatch) -> None:
    monkeypatch.setattr(_core, "de_generate_trials", lambda *args, **kwargs: [{"genes": [0.0]}])

    with pytest.raises(ConfigurationError, match="metadata"):
        DERustKernelAdapter().generate_trials(
            target_population=[
                _candidate("candidate-0", [0.25, 7, True, 1.5]),
                _candidate("candidate-1", [0.5, 8, False, 1.5]),
                _candidate("candidate-2", [0.75, 9, True, 1.5]),
                _candidate("candidate-3", [1.0, 10, False, 1.5]),
            ],
            scores=[0.0, 1.0, 2.0, 3.0],
            gene_space=_space(),
            strategy="rand1bin",
            mutation_factor=0.7,
            crossover_rate=0.9,
            seed=42,
            generation=3,
            target_slots=[0],
            direction="maximize",
            jde_state=None,
        )
```

- [ ] **Step 2: Run adapter tests and verify they fail because the module is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_kernel_adapter.py -v
```

Expected: FAIL during import with missing `evocore.optimizers.de.kernel`.

---

### Task 2: Implement The Adapter

**Files:**
- Create: `evocore/optimizers/de/kernel.py`

- [ ] **Step 1: Create the private adapter module**

Create `evocore/optimizers/de/kernel.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.optimizers.de.strategies import TrialProposal
from evocore.search_space import GeneSpace, decode_gene_values, encode_gene_values


def _require_mapping(raw: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            f"DE Rust kernel returned {type(raw).__name__}, expected mapping."
        )
    return raw


def _metadata_from_raw(raw: Mapping[str, Any]) -> dict[str, object]:
    if "metadata" not in raw:
        raise ConfigurationError("DE Rust kernel proposal is missing metadata.")
    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("DE Rust kernel proposal metadata must be a mapping.")
    return dict(metadata)


def _genes_from_raw(gene_space: GeneSpace, raw: Mapping[str, Any]) -> list[float | int | bool]:
    if "genes" not in raw:
        raise ConfigurationError("DE Rust kernel proposal is missing genes.")
    genes = raw["genes"]
    if not isinstance(genes, Sequence) or isinstance(genes, str | bytes):
        raise ConfigurationError("DE Rust kernel proposal genes must be a sequence.")
    return decode_gene_values(gene_space, genes)


class DERustKernelAdapter:
    """Convert Python DE state to and from the Rust proposal kernel."""

    def generate_trials(
        self,
        *,
        target_population: Sequence[Candidate],
        scores: Sequence[float],
        gene_space: GeneSpace,
        strategy: str,
        mutation_factor: float,
        crossover_rate: float,
        seed: int,
        generation: int,
        target_slots: Sequence[int],
        direction: str,
        jde_state: Mapping[str, Sequence[float]] | None,
    ) -> list[TrialProposal]:
        if len(scores) != len(target_population):
            raise ConfigurationError(
                "DE Rust kernel scores length must match target population length."
            )

        population_encoded = [
            encode_gene_values(gene_space, candidate.genes) for candidate in target_population
        ]
        raw_proposals = _core.de_generate_trials(
            population_encoded,
            [float(score) for score in scores],
            gene_space.rust_bounds,
            gene_space.kinds,
            strategy,
            mutation_factor,
            crossover_rate,
            seed,
            generation,
            list(target_slots),
            direction,
            jde_state,
        )

        proposals: list[TrialProposal] = []
        for raw_item in raw_proposals:
            raw = _require_mapping(raw_item)
            proposals.append(
                TrialProposal(
                    genes=_genes_from_raw(gene_space, raw),
                    metadata=_metadata_from_raw(raw),
                )
            )
        return proposals


__all__ = ["DERustKernelAdapter"]
```

- [ ] **Step 2: Run adapter tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_kernel_adapter.py -v
```

Expected: PASS.

---

### Task 3: Use Adapter From DE Ask/Tell

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Update imports**

In `evocore/optimizers/de/ask_tell.py`, remove `from evocore import _core` only if `_candidate_from_genes(...)` no longer uses `_core.candidate_id`; otherwise keep it.

Add:

```python
from evocore.optimizers.de.kernel import DERustKernelAdapter
```

- [ ] **Step 2: Replace `_rust_trial_proposals(...)` marshalling**

Replace `_rust_trial_proposals(...)` with:

```python
    def _rust_trial_proposals(self, count: int) -> list[TrialProposal]:
        target_population = self._target_population()
        trial_count = min(int(count), len(target_population))
        target_slots = list(range(trial_count))
        scores = [candidate.best_state_score(self.direction) for candidate in target_population]
        jde_state = None
        to_rust_committed_state = getattr(
            self._de_strategy_state,
            "to_rust_committed_state",
            None,
        )
        if callable(to_rust_committed_state):
            jde_state = to_rust_committed_state()

        return DERustKernelAdapter().generate_trials(
            target_population=target_population,
            scores=scores,
            gene_space=self.gene_space,
            strategy=self.strategy,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
            seed=self.seed,
            generation=self.generation,
            target_slots=target_slots,
            direction=self.direction,
            jde_state=jde_state,
        )
```

- [ ] **Step 3: Keep lifecycle test focused on lifecycle output**

In `tests/unit/test_de_ask_tell.py`, keep `test_de_trial_ask_uses_rust_kernel_output` but do not assert adapter internals beyond the `_core.de_generate_trials` call already captured. The adapter test owns detailed marshalling assertions.

- [ ] **Step 4: Run DE adapter and ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_kernel_adapter.py tests/unit/test_de_ask_tell.py -v
```

Expected: PASS.

---

### Task 4: Verify jDE And Checkpoint Behavior Still Belongs To Python

**Files:**
- Existing tests only unless failures expose a behavior gap.

- [ ] **Step 1: Run jDE and checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 2: If a test fails, keep ownership in Python**

If a failure involves pending jDE parameters, update only DE ask/tell or adaptive-state Python code so:

```python
self._record_pending_strategy_trial(candidate)
self._complete_pending_strategy_trial(candidate.candidate_id, accepted=accepted)
self._discard_pending_strategy_trial(candidate.candidate_id)
```

remain lifecycle-owned Python calls. Do not move pending/commit/discard semantics into `DERustKernelAdapter`.

---

### Task 5: Final Verification And Commit

**Files:**
- All files touched in Tasks 1-4.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_kernel_adapter.py tests/unit/test_de_ask_tell.py tests/unit/test_de_jde.py tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 2: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both PASS.

- [ ] **Step 3: Commit task-related files only**

Run:

```powershell
git status --short
git add evocore/optimizers/de/kernel.py evocore/optimizers/de/ask_tell.py tests/unit/test_de_kernel_adapter.py tests/unit/test_de_ask_tell.py
git commit -m "refactor(de): isolate rust kernel adapter"
```

Expected: commit succeeds with only DE adapter files staged.
