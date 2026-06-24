# EvoCore Phase 3C Active Mixed CMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add projected CMA warm starts, opt-in integer-margin CMA behavior, exact checkpoint resume, and lifecycle-managed CMA restart helpers.

**Architecture:** CMA continues to optimize a flat numeric `GeneSpace`. Projection builds active spaces and historical means; CMA owns integer sampling strategy, continuous latent sample bookkeeping, config identity, checkpoint identity, and fresh-run restart construction.

**Tech Stack:** Python CMA optimizer mixins, Rust/PyO3 state if coordinate statistics are needed, EvoCore checkpoints, deterministic seed helpers, pytest, hypothesis, ruff, maturin.

---

## Dependency

- Complete Phase 3A before projected warm starts.
- Complete Phase 3B before invalid projected candidates are converted into penalty records.
- Source design: `docs/superpowers/specs/2026-06-22-evocore-phase-3-projection-cma-design.md`

## File Structure

- Create: `evocore/optimizers/cmaes/projection.py`
  - `ProjectedWarmStartResult`, `build_projected_cma_mean`.
- Modify: `evocore/optimizers/cmaes/mixed.py`
  - Add integer strategy helpers around existing `IntegerMarginDistribution`.
- Modify: `evocore/optimizers/cmaes/engine.py`
  - Add `integer_strategy`, `integer_min_probability`, validation, and decode path.
- Modify: `evocore/optimizers/cmaes/config.py`
  - Include integer strategy in config signatures.
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
  - Apply margin sampling on ask while preserving continuous samples for tell.
- Modify: `evocore/optimizers/cmaes/checkpointing.py`
  - Persist any margin state required for exact resume.
- Create: `evocore/optimizers/cmaes/restarts.py`
  - Restart policies, decisions, and fresh optimizer factory.
- Modify: `evocore/optimizers/cmaes/__init__.py`, `evocore/__init__.py`
  - Re-export public CMA helper names.
- Modify if Rust state access is required: `src/`, `evocore/_core.pyi`
  - Expose minimal coordinate statistics only.
- Create: `tests/unit/test_cmaes_projection.py`
- Modify: `tests/unit/test_mixed_cma_vnext.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_checkpointing.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Create: `tests/property/test_integer_margin_properties.py`
- Create: `tests/unit/test_cmaes_restarts.py`
- Modify: `tests/unit/test_package_init.py`

## Public API

Export these names from `evocore.optimizers.cmaes` and top-level `evocore`:

```python
CMAESRestartDecision
CMAESRestartPolicy
FixedCMAESRestartPolicy
IPOPCMAESRestartPolicy
ProjectedWarmStartResult
build_projected_cma_mean
create_cmaes_restart
```

Add `CMAESOptimizer` constructor options:

```python
integer_strategy: Literal["round", "margin"] = "round"
integer_min_probability: float = 0.02
```

## Task 1: Projected CMA Warm Starts

**Files:**
- Create: `tests/unit/test_cmaes_projection.py`
- Create: `evocore/optimizers/cmaes/projection.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`
- Modify: `evocore/__init__.py`

- [ ] **Step 1: Write failing projected warm-start tests**

Create `tests/unit/test_cmaes_projection.py`:

```python
import pytest

from evocore import Gene, GeneSpace, WarmStartRecord
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import build_projected_cma_mean
from evocore.search_space import ActiveGeneProjection, BinaryThresholdTransform


def _projection() -> ActiveGeneProjection:
    return ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": BinaryThresholdTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )


def test_build_projected_cma_mean_from_best_matching_record() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[
            WarmStartRecord(
                params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True},
                score=12.0,
            ),
            WarmStartRecord(
                params={"template": 1, "fast": 8.0, "slow": 50.0, "flag": False},
                score=10.0,
            ),
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean == [5.0, 40.0, 1.0]
    assert result.accepted_count == 2


def test_projected_mean_rejects_template_mismatch() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[
            WarmStartRecord(
                params={"template": 2, "fast": 5.0, "slow": 40.0, "flag": True},
                score=12.0,
            )
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean is None
    assert result.rejected[0]["reason"] == "projection_mismatch"


def test_projected_mean_reports_non_invertible_transform() -> None:
    class DecodeOnlyTransform:
        checkpointable = True

        def decode(self, value):
            return bool(value)

        def encode(self, value):
            raise ConfigurationError("DecodeOnlyTransform cannot encode historical values.")

        def signature(self):
            return {"type": "decode_only", "version": 1}

    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": DecodeOnlyTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )

    with pytest.raises(ConfigurationError, match="encode"):
        build_projected_cma_mean(
            projection=projection,
            records=[
                WarmStartRecord(
                    params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True},
                    score=12.0,
                )
            ],
            direction="maximize",
            strategy="best",
        )
```

- [ ] **Step 2: Run test and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_projection.py -v
```

Expected: fails with import error for `build_projected_cma_mean`.

- [ ] **Step 3: Implement projected mean helper**

Create `evocore/optimizers/cmaes/projection.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Direction, WarmStartRecord, score_for_direction
from evocore.search_space import ParameterProjection


@dataclass(frozen=True)
class ProjectedWarmStartResult:
    initial_mean: list[float] | None
    accepted_count: int
    rejected: tuple[Mapping[str, object], ...]
    source_candidate_hashes: tuple[str, ...]
    metadata: Mapping[str, object]


def build_projected_cma_mean(
    *,
    projection: ParameterProjection,
    records: Sequence[WarmStartRecord],
    direction: Direction,
    strategy: Literal["best", "top_k_centroid"] = "best",
    top_k: int | None = None,
) -> ProjectedWarmStartResult:
    if strategy not in ("best", "top_k_centroid"):
        raise ConfigurationError("strategy must be 'best' or 'top_k_centroid'.")
    if top_k is not None and int(top_k) <= 0:
        raise ConfigurationError("top_k must be positive when provided.")

    ranked = []
    rejected: list[Mapping[str, object]] = []
    for index, record in enumerate(records):
        try:
            projected = projection.project(dict(record.params or {}))
        except ConfigurationError as exc:
            rejected.append(
                {"record_index": index, "reason": "projection_mismatch", "message": str(exc)}
            )
            continue
        ranked.append((score_for_direction(record.score, direction), projected))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ProjectedWarmStartResult(None, 0, tuple(rejected), (), {"strategy": strategy})

    selected = ranked[: (len(ranked) if top_k is None else int(top_k))]
    if strategy == "best":
        mean = [float(value) for value in selected[0][1].optimizer_values]
    else:
        mean = [
            sum(float(item[1].optimizer_values[index]) for item in selected) / len(selected)
            for index in range(projection.optimizer_space.length)
        ]
    projection.optimizer_space.validate_genes(mean)
    return ProjectedWarmStartResult(
        mean,
        len(ranked),
        tuple(rejected),
        tuple(projected.projection_hash for _, projected in selected),
        {"strategy": strategy, "top_k": top_k},
    )
```

- [ ] **Step 4: Export and commit projected warm starts**

Update `evocore/optimizers/cmaes/__init__.py` and `evocore/__init__.py`, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_projection.py tests/unit/test_cmaes_external_state.py tests/unit/test_package_init.py -v
git add evocore/optimizers/cmaes/projection.py evocore/optimizers/cmaes/__init__.py evocore/__init__.py tests/unit/test_cmaes_projection.py tests/unit/test_package_init.py
git commit -m "feat(cmaes): add projected warm-start helpers"
```

Expected: selected tests pass and commit succeeds.

## Task 2: Integer Strategy and Margin CMA

**Files:**
- Modify: `evocore/optimizers/cmaes/mixed.py`
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `evocore/optimizers/cmaes/config.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
- Modify: `evocore/optimizers/cmaes/checkpointing.py`
- Modify if Rust support is required: `src/`, `evocore/_core.pyi`
- Modify: `tests/unit/test_mixed_cma_vnext.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_checkpointing.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Create: `tests/property/test_integer_margin_properties.py`

- [ ] **Step 1: Write failing integer strategy tests**

Add these tests to `tests/unit/test_cmaes_engine.py` and `tests/unit/test_optimizer_config.py`:

```python
def test_cmaes_default_integer_strategy_is_round() -> None:
    optimizer = CMAESOptimizer(
        GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)]),
        population_size=4,
        seed=1,
    )

    assert optimizer.integer_strategy == "round"


def test_cmaes_rejects_invalid_integer_strategy() -> None:
    with pytest.raises(ConfigurationError, match="integer_strategy"):
        CMAESOptimizer(GeneSpace([Gene("x", "int", 0, 3)]), integer_strategy="bad")


def test_margin_strategy_changes_config_hash() -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])

    assert CMAESOptimizer(space, integer_strategy="round").config_hash() != CMAESOptimizer(
        space,
        integer_strategy="margin",
    ).config_hash()
```

- [ ] **Step 2: Write failing margin resume test**

Add to `tests/unit/test_cmaes_ask_tell_checkpointing.py`:

```python
def test_margin_cma_resume_next_ask_matches_uninterrupted(tmp_path) -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])
    uninterrupted = CMAESOptimizer(
        space,
        population_size=4,
        seed=12,
        integer_strategy="margin",
    )
    restored = CMAESOptimizer(
        space,
        population_size=4,
        seed=12,
        integer_strategy="margin",
    )

    batch = uninterrupted.ask()
    uninterrupted.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(batch)
        ]
    )
    checkpoint = uninterrupted.ask_tell_checkpoint()
    restored.resume_ask_tell_checkpoint(checkpoint.to_dict())

    assert [candidate.genes for candidate in restored.ask()] == [
        candidate.genes for candidate in uninterrupted.ask()
    ]
```

- [ ] **Step 3: Add integer-margin property tests**

Create `tests/property/test_integer_margin_properties.py`:

```python
from hypothesis import given, strategies as st

from evocore.optimizers.cmaes import IntegerMarginDistribution


@given(
    low=st.integers(-5, 0),
    high=st.integers(1, 8),
    mean=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    sigma=st.floats(0.05, 10, allow_nan=False, allow_infinity=False),
)
def test_integer_margin_probabilities_are_bounded_and_normalized(
    low: int,
    high: int,
    mean: float,
    sigma: float,
) -> None:
    margin = IntegerMarginDistribution(low=low, high=high, min_probability=0.01)
    probabilities = margin.probabilities(mean=mean, sigma=sigma)

    assert set(probabilities) == set(range(low, high + 1))
    assert abs(sum(probabilities.values()) - 1.0) < 1.0e-12
    assert all(value >= 0.01 for value in probabilities.values())
```

- [ ] **Step 4: Implement config and ask/tell behavior**

Update `CMAESOptimizer.__init__`:

```python
integer_strategy: Literal["round", "margin"] = "round",
integer_min_probability: float = 0.02,
```

Validation rules:

- `integer_strategy` must be `"round"` or `"margin"`.
- `integer_min_probability` must be in `(0, 1)`.
- `integer_min_probability * integer_range_size < 1.0` for every active integer gene using margin.
- Default `"round"` must preserve existing candidate sequences and config/checkpoint shape where practical.

Implementation rules for `integer_strategy="margin"`:

- Keep Rust CMA continuous samples in `CandidateBatch.continuous_samples_by_id`.
- For each integer gene, use `IntegerMarginDistribution(low, high, integer_min_probability)`.
- Derive deterministic coordinate randomness from seed, event index, candidate index, gene index, and continuous latent sample value.
- Sample user-facing integer genes from margin probabilities.
- Tell Rust with original continuous samples and direction-adjusted fitnesses.
- Include strategy and min probability in `build_cmaes_config()`.
- Persist any additional deterministic state needed by `ask_tell_checkpoint()`.
- If Rust coordinate statistics are required for correct sigma, expose only the minimal PyO3 accessors and update `evocore/_core.pyi`.

- [ ] **Step 5: Run and commit integer-margin CMA**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_mixed_cma_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_optimizer_config.py tests/property/test_integer_margin_properties.py -v
```

Expected: selected tests pass.

Commit:

```powershell
git add evocore/optimizers/cmaes evocore/_core.pyi src tests/unit/test_mixed_cma_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_optimizer_config.py tests/property/test_integer_margin_properties.py
git commit -m "feat(cmaes): integrate integer margin strategy"
```

## Task 3: CMA Restart Helpers

**Files:**
- Create: `evocore/optimizers/cmaes/restarts.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`
- Modify: `evocore/__init__.py`
- Create: `tests/unit/test_cmaes_restarts.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Write failing restart tests**

Create `tests/unit/test_cmaes_restarts.py`:

```python
import pytest

from evocore import CMAESOptimizer, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import (
    FixedCMAESRestartPolicy,
    IPOPCMAESRestartPolicy,
    create_cmaes_restart,
)


def test_fixed_restart_derives_fresh_child_seed() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=4).decide(
        parent=parent,
        restart_index=1,
        reason="stall",
    )

    assert decision.restart_index == 1
    assert decision.reason == "stall"
    assert decision.population_size == 4
    assert decision.seed != parent.seed


def test_ipop_restart_grows_population() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = IPOPCMAESRestartPolicy(base_population_size=4, growth_factor=2).decide(
        parent=parent,
        restart_index=2,
        reason="stall",
    )

    assert decision.population_size == 16


def test_restart_rejects_pending_batch() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    parent.ask()

    with pytest.raises(ConfigurationError, match="pending"):
        FixedCMAESRestartPolicy(population_size=4).decide(
            parent=parent,
            restart_index=1,
            reason="stall",
        )


def test_create_restart_returns_fresh_optimizer() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=6).decide(
        parent=parent,
        restart_index=1,
        reason="stall",
    )
    child = create_cmaes_restart(parent=parent, decision=decision)

    assert child.population_size == 6
    assert child.seed == decision.seed
    assert child.generation == 0
```

- [ ] **Step 2: Implement restart helpers**

Create `evocore/optimizers/cmaes/restarts.py` with:

- frozen `CMAESRestartDecision`;
- protocol `CMAESRestartPolicy`;
- `FixedCMAESRestartPolicy`;
- `IPOPCMAESRestartPolicy`;
- `create_cmaes_restart(parent, decision)`.

Rules:

- Reject restarts when `parent.state_summary().pending_batch_ids` is non-empty.
- Derive seed with `derive_child_seed(parent_seed=parent.seed, candidate_hash=parent.gene_space.hash(), stage=f"cma_restart:{restart_index}:{reason}")`.
- Copy direction, gene space, initial sigma, callbacks, tracking, integer strategy, and margin probability into the fresh optimizer.
- Do not copy parent Rust state or pending batches.

- [ ] **Step 3: Export, test, and commit restart helpers**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_restarts.py tests/unit/test_package_init.py -v
```

Expected: selected tests pass.

Commit:

```powershell
git add evocore/optimizers/cmaes/restarts.py evocore/optimizers/cmaes/__init__.py evocore/__init__.py tests/unit/test_cmaes_restarts.py tests/unit/test_package_init.py
git commit -m "feat(cmaes): add restart planning helpers"
```

## Verification

- [ ] **Step 1: Run CMA regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_cmaes_external_state.py tests/integration/test_cmaes_rosenbrock.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run format, lint, extension, and property checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/property/test_integer_margin_properties.py -v
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

If `src/` changed, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all commands pass.

## Self-Review Notes

- Spec coverage: projected means, template mismatch rejection, non-invertible transform reporting, integer strategy config identity, margin sampling, exact resume, and fresh-run restarts are covered.
- Compatibility: `integer_strategy="round"` remains the default.
- Phase 3D dependency: docs and the expensive-system recipe should demonstrate these helpers without adding a formal hybrid engine.
