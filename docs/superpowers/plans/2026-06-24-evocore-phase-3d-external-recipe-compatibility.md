# EvoCore Phase 3D External Recipe Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document and verify a Trading-Algo-style external expensive optimization workflow using Phase 1, Phase 2, and Phase 3 public APIs.

**Architecture:** Add a synthetic integration test and docs recipe that compose outer templates, active projections, transforms, cached records, archives, penalties, CMA warm starts, family/specialist policies, staged refinement, and restart lineage. Keep the recipe generic; no trading-specific package dependency or public API belongs in EvoCore.

**Tech Stack:** Python integration tests, EvoCore public APIs, MkDocs, changelog, full unit/integration/property regression suite.

---

## Dependency

- Complete Phase 3A, 3B, and 3C first.
- Source design: `docs/superpowers/specs/2026-06-22-evocore-phase-3-projection-cma-design.md`

## File Structure

- Create: `tests/integration/test_phase3_expensive_projection_recipe.py`
  - Synthetic outer GA / projected inner CMA recipe.
- Modify: `docs/site/api.md`
  - API entries for projection, constraints, CMA projection, integer strategy, and restarts.
- Modify: `docs/site/mixed-variable-search.md`
  - Explain `round` default and opt-in `margin`.
- Modify: `docs/site/expensive-external-evaluations.md`
  - End-to-end expensive-system recipe.
- Modify: `CHANGELOG.md`
  - Public API, compatibility, and checkpoint/config identity notes.
- Modify: `tests/unit/test_checkpoint_golden_fixtures.py`
  - Only when Phase 3C changed checkpoint fixture expectations.

## Task 1: Synthetic External Integration Test

**Files:**
- Create: `tests/integration/test_phase3_expensive_projection_recipe.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_phase3_expensive_projection_recipe.py`:

```python
from evocore import (
    CMAESOptimizer,
    CandidateArchive,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
    derive_child_seed,
)
from evocore.lifecycle import constraint_penalty_record
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ConstraintViolation,
    ExponentialIntegerTransform,
)


def _template_projection(family: int) -> ActiveGeneProjection:
    return ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("family", "int", 0, 2),
                Gene("fast_log", "float", 1.0, 4.0),
                Gene("use_filter", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast_log", "use_filter"],
        structural_bindings={"family": family},
        transforms={
            "fast_log": ExponentialIntegerTransform(base=2.0),
            "use_filter": BinaryThresholdTransform(),
        },
        identity_keys=("family",),
        schema_id="synthetic-template",
        schema_version="1",
    )


def test_template_outer_ga_inner_cma_projection_recipe_is_deterministic() -> None:
    outer_space = GeneSpace([Gene("family", "int", 0, 2), Gene("mode", "int", 0, 1)])
    outer = GeneticAlgorithmOptimizer(outer_space, population_size=6, seed=44)
    archive = CandidateArchive(direction="maximize")
    outer_candidate = outer.ask(1)[0]
    family = int(outer_candidate.genes[0])
    projection = _template_projection(family)
    inner_seed = derive_child_seed(
        parent_seed=44,
        candidate_hash=outer_candidate.candidate_hash(outer_space),
        stage="inner_cma",
    )
    inner = CMAESOptimizer(
        projection.optimizer_space,
        population_size=4,
        seed=inner_seed,
        integer_strategy="margin",
    )

    prior_mean = [2.0, 1.0]
    prior = WarmStartRecord(values=tuple(prior_mean), score=4.0, confidence="cached")
    inner.warm_start([prior], mode="tracked")
    inner_batch = inner.ask()
    records = []
    for candidate in inner_batch:
        decoded = projection.reconstruct(candidate.genes)
        if decoded.parameters["fast_log"] < 3:
            records.append(
                constraint_penalty_record(
                    candidate=candidate,
                    stage="projection",
                    direction="maximize",
                    violations=[
                        ConstraintViolation(
                            code="min_fast_period",
                            message="fast period must be at least 3",
                            names=("fast_log",),
                        )
                    ],
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )
        else:
            score = float(decoded.parameters["fast_log"]) + float(decoded.parameters["family"])
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence="trusted_full",
                    stage="full",
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )

    update = inner.tell(records)
    trusted = inner.top_candidates(2)
    archive.add_snapshot(inner.candidate_snapshot(scope="scored"))

    assert update.state_accepted_count == 4
    assert inner.state_summary().pending_batch_ids == ()
    assert all(snapshot.confidence != "constraint_penalty" for snapshot in trusted)
    assert archive.snapshot(k=4).candidates
```

- [ ] **Step 2: Run test and verify expected failure before docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_phase3_expensive_projection_recipe.py -v
```

Expected before all prior phases are merged: fails on missing Phase 3 APIs. Expected after 3A-3C: passes.

## Task 2: Documentation

**Files:**
- Modify: `docs/site/api.md`
- Modify: `docs/site/mixed-variable-search.md`
- Modify: `docs/site/expensive-external-evaluations.md`

- [ ] **Step 1: Add API reference entries**

Add these entries to `docs/site/api.md` in the relevant sections:

```markdown
::: evocore.search_space.ActiveGeneProjection

::: evocore.search_space.ProjectionResult

::: evocore.search_space.ProjectionSnapshot

::: evocore.search_space.IdentityTransform

::: evocore.search_space.BinaryThresholdTransform

::: evocore.search_space.ExponentialIntegerTransform

::: evocore.search_space.ConstraintViolation

::: evocore.lifecycle.constraint_penalty_record

::: evocore.optimizers.cmaes.build_projected_cma_mean

::: evocore.optimizers.cmaes.FixedCMAESRestartPolicy

::: evocore.optimizers.cmaes.IPOPCMAESRestartPolicy
```

- [ ] **Step 2: Update mixed-variable CMA docs**

In `docs/site/mixed-variable-search.md`, add:

````markdown
## CMA-ES Integer Strategy

`CMAESOptimizer` keeps `integer_strategy="round"` as the default for backward compatibility. In this mode, CMA samples continuous latent values and EvoCore repairs the public candidate values into integer bounds.

Use `integer_strategy="margin"` when native integer coordinates need protected sampling probability:

```python
from evocore import CMAESOptimizer, Gene, GeneSpace

space = GeneSpace([Gene("period", "int", 2, 20), Gene("threshold", "float", 0.0, 1.0)])
optimizer = CMAESOptimizer(
    space,
    population_size=12,
    seed=42,
    integer_strategy="margin",
    integer_min_probability=0.02,
)
```

The margin strategy is opt-in, participates in optimizer config hashes, and is checkpointed for exact ask/tell resume.
````

- [ ] **Step 3: Add expensive external workflow recipe**

In `docs/site/expensive-external-evaluations.md`, add a section named `Projected Template Optimization` with:

````markdown
## Projected Template Optimization

External expensive systems often choose a structure first, then tune only the parameters active for that structure. Use an outer optimizer for structures and `ActiveGeneProjection` to compile the active inner coordinates.

```python
from evocore import CMAESOptimizer, Gene, GeneSpace, GeneticAlgorithmOptimizer, derive_child_seed
from evocore.search_space import ActiveGeneProjection, BinaryThresholdTransform, ExponentialIntegerTransform

outer_space = GeneSpace([Gene("family", "int", 0, 2), Gene("mode", "int", 0, 1)])
outer = GeneticAlgorithmOptimizer(outer_space, population_size=24, seed=100)

for template_candidate in outer.ask(4):
    family = int(template_candidate.genes[0])
    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("family", "int", 0, 2),
                Gene("lookback_log", "float", 1.0, 5.0),
                Gene("use_filter", "float", 0.0, 1.0),
            ]
        ),
        active_names=["lookback_log", "use_filter"],
        structural_bindings={"family": family},
        transforms={
            "lookback_log": ExponentialIntegerTransform(base=2.0),
            "use_filter": BinaryThresholdTransform(),
        },
        identity_keys=("family",),
        schema_id="template-family",
        schema_version="1",
    )
    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=template_candidate.candidate_hash(outer_space),
        stage="inner_cma",
    )
    inner = CMAESOptimizer(projection.optimizer_space, population_size=16, seed=inner_seed)
```

Cached records, archives, family quotas, specialist caps, and survivor selection remain lifecycle helpers. Projection only owns the boundary between named domain parameters and optimizer-native coordinates.
````

## Task 3: Changelog and Compatibility

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `tests/unit/test_checkpoint_golden_fixtures.py`

- [ ] **Step 1: Add changelog entry**

Under `CHANGELOG.md` `## [Unreleased]`, add:

```markdown
### Added

- Added Phase 3 projection APIs for active named parameter spaces, portable transforms, constraint-penalty records, projected CMA warm starts, opt-in integer-margin CMA sampling, and CMA restart planning.

### Compatibility

- `CMAESOptimizer` keeps `integer_strategy="round"` as the default. Opting into `integer_strategy="margin"` changes config hashes and checkpoint payloads for that run.
- `constraint_penalty` records are state-update eligible but excluded from trusted snapshots, warm starts, top-k defaults, archive promotion, and selection helpers.
- Existing flat `GeneSpace` schema version 1 remains unchanged.
```

- [ ] **Step 2: Update golden fixture expectations if checkpoint payloads changed**

If Phase 3C changed golden checkpoint payloads, update `tests/unit/test_checkpoint_golden_fixtures.py` by adding explicit assertions for:

```python
assert payload["optimizer_config"]["parameters"]["integer_strategy"] == "round"
assert payload["optimizer_config"]["parameters"]["integer_min_probability"] == 0.02
```

Regenerate fixture data only through the existing fixture workflow in that test file. Do not hand-edit generated checkpoint JSON.

## Task 4: Final Verification and Commit

**Files:**
- All docs, tests, and fixture files modified in Phase 3D.

- [ ] **Step 1: Run focused recipe and docs checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_phase3_expensive_projection_recipe.py -v
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: both commands pass.

- [ ] **Step 2: Run full public-surface verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

If Rust/PyO3 changed in Phase 3C, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all commands pass.

- [ ] **Step 3: Commit Phase 3D**

Run:

```powershell
git add tests/integration/test_phase3_expensive_projection_recipe.py docs/site/api.md docs/site/mixed-variable-search.md docs/site/expensive-external-evaluations.md CHANGELOG.md tests/unit/test_checkpoint_golden_fixtures.py
git commit -m "docs: add phase 3 expensive optimization recipe"
```

## Pull Request Update

- [ ] **Step 1: Push the complete Phase 3 branch**

Run:

```powershell
git push
```

- [ ] **Step 2: Update the draft PR description**

Include:

- Phase 3A projection API summary.
- Phase 3B penalty confidence semantics.
- Phase 3C CMA integer strategy, projected warm starts, and restart helpers.
- Phase 3D recipe and docs updates.
- Backward compatibility notes.
- Exact verification commands and results.

## Self-Review Notes

- Spec coverage: synthetic external workflow, expensive-system docs, public API reference, mixed-variable CMA docs, changelog, and full verification are covered.
- Compatibility: recipe remains generic and does not import Trading-Algo.
- Completion gate: Phase 3 is not complete until 3A, 3B, 3C, and 3D verification all pass on the same branch.
