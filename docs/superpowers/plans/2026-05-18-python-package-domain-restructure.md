# Python Package Domain Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace EvoCore's flat Python package with a domain-oriented package and new public vocabulary while preserving optimizer behavior.

**Architecture:** Move shared building blocks into `core`, `search_space`, `lifecycle`, and `results`; move algorithms into `optimizers.ga` and `optimizers.cmaes`; keep top-level `evocore` convenience exports with new names only. The migration is intentionally breaking for direct old flat module paths, but existing optimizer behavior, seeds, budget accounting, and Rust-backed operators must remain stable.

**Tech Stack:** Python 3.11+, Rust/PyO3 via maturin, pytest, hypothesis, ruff, MkDocs/mkdocstrings.

---

## Scope Check

This plan implements the approved full package restructure from `docs/superpowers/specs/2026-05-18-python-package-domain-restructure-design.md`. It is large, but it is one coherent breaking API migration. Each task leaves the tree testable against a focused test subset and commits one clear slice.

## File Structure

Create this final source structure:

```text
evocore/
  __init__.py
  _core.pyi
  py.typed
  core/
    __init__.py
    errors.py
    serialization.py
    parallel.py
  search_space/
    __init__.py
    genes.py
    solutions.py
    codec.py
  lifecycle/
    __init__.py
    records.py
    batches.py
    policies.py
    scheduler.py
    protocols.py
    telemetry.py
    events.py
  results/
    __init__.py
    generation.py
    reproducibility.py
    run.py
  optimizers/
    __init__.py
    ga/
      __init__.py
      engine.py
      ask_tell.py
      generation_loop.py
      checkpointing.py
      multi_run.py
      reproduction.py
    cmaes/
      __init__.py
      engine.py
      ask_tell.py
      mixed.py
  callbacks/
    __init__.py
  surrogates/
    __init__.py
```

Remove these old flat Python modules at the end of the migration:

```text
evocore/advisors.py
evocore/batches.py
evocore/callbacks.py
evocore/cmaes.py
evocore/evaluation.py
evocore/exceptions.py
evocore/exporting.py
evocore/ga.py
evocore/gene_space.py
evocore/individual.py
evocore/mixed_cma.py
evocore/operators.py
evocore/parallel.py
evocore/policies.py
evocore/protocols.py
evocore/scheduler.py
evocore/stats.py
```

## Naming Map

Use this map consistently in code, tests, docs, and exported JSON:

```text
AdvisorScore -> SurrogateScore
CandidateScore -> ScoreObservation
CMAESEngine -> CMAESOptimizer
CategoricalState -> CategoricalDistributionState
EngineStateSummary -> OptimizerStateSummary
EvaluationScheduler -> BudgetScheduler
GAEngine -> GeneticAlgorithmOptimizer
GeneDef -> Gene
Individual -> Solution
IntegerMargin -> IntegerMarginDistribution
InverseDistanceSurrogateAdvisor -> InverseDistanceAdvisor
LogEntry -> GenerationRecord
Logbook -> GenerationHistory
MultiFidelityPolicy -> BudgetPolicy
MultiRunResult -> OptimizationBatchResult
OperatorSet -> OperatorCodec
Population -> SolutionSet
Rung -> EvaluationStage
RunResult -> OptimizationResult
TellResult -> UpdateResult
```

Use score vocabulary for public results and JSON:

```text
best_fitness -> best_score
best_individual -> best_solution
diversity_history -> diversity_by_generation
elite_history -> elite_solutions
final_population -> final_solutions
fitness_summary -> score_summary
history -> events
logbook -> generations
mean_fitness -> mean_score
std_fitness -> std_score
```

---

### Task 1: Add Import Contract Tests For The New Public Surface

**Files:**
- Modify: `tests/unit/test_package_init.py`
- Create: `tests/unit/test_domain_imports.py`

- [ ] **Step 1: Add failing top-level export assertions**

Replace `tests/unit/test_package_init.py` with this file. This intentionally removes old top-level name checks and checks the new convenience API.

```python
def test_evocore_imports_without_error():
    import evocore  # noqa: F401


def test_core_extension_accessible():
    from evocore import _core

    assert hasattr(_core, "FloatIndividual")
    assert hasattr(_core, "IntegerIndividual")
    assert hasattr(_core, "BinaryIndividual")
    assert hasattr(_core, "py_derive_seed")


def test_errors_accessible_from_top_level():
    from evocore import (
        CheckpointError,
        ConfigurationError,
        ConfigurationWarning,
        ConvergenceError,
        EvocoreError,
        FitnessError,
        FitnessWarning,
        ParallelError,
    )

    assert issubclass(ConfigurationError, EvocoreError)
    assert issubclass(FitnessWarning, Warning)
    assert issubclass(ConfigurationWarning, Warning)
    assert issubclass(CheckpointError, EvocoreError)
    assert issubclass(ConvergenceError, EvocoreError)
    assert issubclass(FitnessError, EvocoreError)
    assert issubclass(ParallelError, EvocoreError)


def test_search_space_exports_accessible_from_top_level():
    from evocore import Gene, GeneSpace, Solution, SolutionSet

    space = GeneSpace.uniform(-1.0, 1.0, 2)
    solution = Solution([1.0, 0.0])
    solutions = SolutionSet([solution])

    assert Gene("x", "float", -1.0, 1.0).name == "x"
    assert space.length == 2
    assert solution.values == [1.0, 0.0]
    assert len(solutions) == 1


def test_optimizer_exports_accessible_from_top_level():
    from evocore import CMAESOptimizer, GeneticAlgorithmOptimizer

    assert CMAESOptimizer is not None
    assert GeneticAlgorithmOptimizer is not None


def test_result_exports_accessible_from_top_level():
    from evocore import GenerationHistory, GenerationRecord, OptimizationBatchResult, OptimizationResult

    assert GenerationHistory is not None
    assert GenerationRecord is not None
    assert OptimizationBatchResult is not None
    assert OptimizationResult is not None


def test_lifecycle_exports_accessible_from_top_level():
    import evocore

    assert evocore.Candidate.__name__ == "Candidate"
    assert evocore.EvaluationRecord.__name__ == "EvaluationRecord"
    assert evocore.EvaluationStage.__name__ == "EvaluationStage"
    assert evocore.BudgetPolicy.__name__ == "BudgetPolicy"
    assert evocore.BudgetScheduler.__name__ == "BudgetScheduler"
    assert evocore.OptimizationTelemetry.__name__ == "OptimizationTelemetry"
    assert evocore.UpdateResult.__name__ == "UpdateResult"
    assert evocore.OptimizerStateSummary.__name__ == "OptimizerStateSummary"
    assert evocore.EventRecord.__name__ == "EventRecord"
    assert evocore.EventHistory.__name__ == "EventHistory"
    assert evocore.ReproducibilityMetadata.__name__ == "ReproducibilityMetadata"
```

- [ ] **Step 2: Add failing direct domain import tests**

Create `tests/unit/test_domain_imports.py` with this content:

```python
import importlib


def test_new_domain_imports_are_available():
    modules = [
        "evocore.core.errors",
        "evocore.core.serialization",
        "evocore.core.parallel",
        "evocore.search_space",
        "evocore.search_space.genes",
        "evocore.search_space.solutions",
        "evocore.search_space.codec",
        "evocore.lifecycle",
        "evocore.lifecycle.records",
        "evocore.lifecycle.batches",
        "evocore.lifecycle.policies",
        "evocore.lifecycle.scheduler",
        "evocore.lifecycle.protocols",
        "evocore.lifecycle.telemetry",
        "evocore.lifecycle.events",
        "evocore.results",
        "evocore.results.generation",
        "evocore.results.reproducibility",
        "evocore.results.run",
        "evocore.optimizers",
        "evocore.optimizers.ga",
        "evocore.optimizers.cmaes",
        "evocore.callbacks",
        "evocore.surrogates",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_new_domain_symbols_are_importable():
    from evocore.lifecycle import BudgetPolicy, BudgetScheduler, EvaluationStage
    from evocore.optimizers.cmaes import CMAESOptimizer
    from evocore.optimizers.ga import GeneticAlgorithmOptimizer
    from evocore.results import OptimizationBatchResult, OptimizationResult
    from evocore.search_space import Gene, GeneSpace, Solution, SolutionSet
    from evocore.surrogates import InverseDistanceAdvisor, SurrogateScore

    assert BudgetPolicy is not None
    assert BudgetScheduler is not None
    assert EvaluationStage is not None
    assert CMAESOptimizer is not None
    assert GeneticAlgorithmOptimizer is not None
    assert OptimizationBatchResult is not None
    assert OptimizationResult is not None
    assert Gene is not None
    assert GeneSpace is not None
    assert Solution is not None
    assert SolutionSet is not None
    assert InverseDistanceAdvisor is not None
    assert SurrogateScore is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/unit/test_package_init.py tests/unit/test_domain_imports.py -v
```

Expected: FAIL with import errors for `Gene`, `Solution`, `GeneticAlgorithmOptimizer`, and the new domain modules.

- [ ] **Step 4: Commit the failing contract tests**

```bash
git add tests/unit/test_package_init.py tests/unit/test_domain_imports.py
git commit -m "test: define domain package import contract"
```

---

### Task 2: Move Core Error, Serialization, And Parallel Utilities

**Files:**
- Create: `evocore/core/__init__.py`
- Move: `evocore/exceptions.py` -> `evocore/core/errors.py`
- Move: `evocore/exporting.py` -> `evocore/core/serialization.py`
- Move: `evocore/parallel.py` -> `evocore/core/parallel.py`
- Modify: all Python imports that reference `evocore.exceptions`, `evocore.exporting`, or `evocore.parallel`
- Test: `tests/unit/test_package_init.py`
- Test: `tests/unit/test_exceptions.py`
- Test: `tests/unit/test_parallel.py`

- [ ] **Step 1: Move files**

Run:

```bash
mkdir evocore/core
git mv evocore/exceptions.py evocore/core/errors.py
git mv evocore/exporting.py evocore/core/serialization.py
git mv evocore/parallel.py evocore/core/parallel.py
```

Create `evocore/core/__init__.py`:

```python
"""Core EvoCore utilities."""

from evocore.core.errors import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    ConvergenceError,
    EvocoreError,
    FitnessError,
    FitnessWarning,
    ParallelError,
)
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.core.serialization import (
    canonical_json_hash,
    json_safe,
    package_version,
    stable_json_dumps,
)

__all__ = [
    "CheckpointError",
    "ConfigurationError",
    "ConfigurationWarning",
    "ConvergenceError",
    "EvocoreError",
    "FitnessError",
    "FitnessWarning",
    "ParallelError",
    "ProcessParallel",
    "ThreadParallel",
    "canonical_json_hash",
    "ensure_picklable",
    "json_safe",
    "package_version",
    "stable_json_dumps",
]
```

- [ ] **Step 2: Update imports mechanically**

Run these searches and replace each match:

```bash
rg -n "evocore\\.exceptions|evocore\\.exporting|evocore\\.parallel" evocore tests docs/site
```

Replacement map:

```text
from evocore.exceptions import -> from evocore.core.errors import
from evocore.exporting import -> from evocore.core.serialization import
from evocore.parallel import -> from evocore.core.parallel import
```

Also update `evocore/__init__.py` to import errors and parallel helpers from `evocore.core`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/unit/test_exceptions.py tests/unit/test_parallel.py tests/unit/test_package_init.py -v
```

Expected: PASS for errors and parallel tests; `tests/unit/test_package_init.py` may still FAIL on later domain imports until Task 3 and Task 7.

- [ ] **Step 4: Commit**

```bash
git add evocore/core evocore tests
git commit -m "refactor(core): move foundational utilities"
```

---

### Task 3: Move Search Space, Solution, And Codec Types

**Files:**
- Create: `evocore/search_space/__init__.py`
- Move: `evocore/gene_space.py` -> `evocore/search_space/genes.py`
- Move: `evocore/individual.py` -> `evocore/search_space/solutions.py`
- Move: `evocore/operators.py` -> `evocore/search_space/codec.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_gene_space.py`
- Test: `tests/unit/test_individual.py`
- Test: `tests/unit/test_operators.py`
- Test: `tests/property/test_gene_space_properties.py`
- Test: `tests/property/test_operator_properties.py`

- [ ] **Step 1: Move files and rename public classes**

Run:

```bash
mkdir evocore/search_space
git mv evocore/gene_space.py evocore/search_space/genes.py
git mv evocore/individual.py evocore/search_space/solutions.py
git mv evocore/operators.py evocore/search_space/codec.py
```

In `evocore/search_space/genes.py`, rename:

```text
GeneDef -> Gene
```

Keep the `GeneSpace` class name. Update constructor examples and error messages to use `Gene(...)`.

In `evocore/search_space/solutions.py`, rename:

```text
Individual -> Solution
Population -> SolutionSet
genes attribute -> values attribute
fitness attribute -> score attribute
fitness_valid attribute -> score_valid attribute
mean_fitness() -> mean_score()
std_fitness() -> std_score()
```

Use these property aliases only inside `Solution` and `SolutionSet` to keep internal migration manageable during the branch:

```python
    @property
    def genes(self) -> list[GeneValue]:
        """Return decoded values for internal Rust-boundary compatibility."""
        return self.values

    @property
    def fitness(self) -> float | None:
        """Return score for internal GA/CMA compatibility during migration."""
        return self.score

    @fitness.setter
    def fitness(self, value: float | None) -> None:
        self.score = value

    @property
    def fitness_valid(self) -> bool:
        """Return score_valid for internal GA/CMA compatibility during migration."""
        return self.score_valid

    @fitness_valid.setter
    def fitness_valid(self, value: bool) -> None:
        self.score_valid = value
```

These aliases are not top-level public exports and must not appear in docs.

In `evocore/search_space/codec.py`, rename:

```text
OperatorSet -> OperatorCodec
encode_genes() -> encode_values()
decode_genes() -> decode_values()
```

Keep these internal aliases on `OperatorCodec` during the branch:

```python
    encode_genes = encode_values
    decode_genes = decode_values
```

- [ ] **Step 2: Create package exports**

Create `evocore/search_space/__init__.py`:

```python
"""Search-space definitions and decoded solution containers."""

from evocore.search_space.codec import OperatorCodec
from evocore.search_space.genes import Gene, GeneKind, GeneSpace
from evocore.search_space.solutions import GeneValue, Solution, SolutionSet

__all__ = [
    "Gene",
    "GeneKind",
    "GeneSpace",
    "GeneValue",
    "OperatorCodec",
    "Solution",
    "SolutionSet",
]
```

- [ ] **Step 3: Update imports and tests**

Run:

```bash
rg -n "GeneDef|Individual|Population|OperatorSet|evocore\\.gene_space|evocore\\.individual|evocore\\.operators|fitness|genes" tests/unit tests/property evocore docs/site
```

Apply these replacements in tests and docs:

```text
GeneDef -> Gene
Individual -> Solution
Population -> SolutionSet
OperatorSet -> OperatorCodec
from evocore.gene_space import -> from evocore.search_space import
from evocore.individual import -> from evocore.search_space import
from evocore.operators import -> from evocore.search_space import
.genes in public tests -> .values
.fitness in public tests -> .score
.fitness_valid in public tests -> .score_valid
mean_fitness() -> mean_score()
std_fitness() -> std_score()
```

Do not change Rust extension tests that import `evocore._core.FloatIndividual`, `IntegerIndividual`, or `BinaryIndividual`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/unit/test_gene_space.py tests/unit/test_individual.py tests/unit/test_operators.py tests/property/test_gene_space_properties.py tests/property/test_operator_properties.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evocore/search_space evocore tests docs/site
git commit -m "refactor(search-space): move genes solutions and codec"
```

---

### Task 4: Move Lifecycle Records, Telemetry, Policies, Scheduler, And Protocols

**Files:**
- Create: `evocore/lifecycle/__init__.py`
- Move: `evocore/evaluation.py` -> split into `evocore/lifecycle/records.py` and `evocore/lifecycle/telemetry.py`
- Move: `evocore/batches.py` -> `evocore/lifecycle/batches.py`
- Move: `evocore/policies.py` -> `evocore/lifecycle/policies.py`
- Move: `evocore/scheduler.py` -> `evocore/lifecycle/scheduler.py`
- Move: `evocore/protocols.py` -> `evocore/lifecycle/protocols.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_vnext_evaluation.py`
- Test: `tests/unit/test_vnext_policy_scheduler.py`
- Test: `tests/unit/test_protocols.py`

- [ ] **Step 1: Move and split lifecycle files**

Run:

```bash
mkdir evocore/lifecycle
git mv evocore/evaluation.py evocore/lifecycle/records.py
git mv evocore/batches.py evocore/lifecycle/batches.py
git mv evocore/policies.py evocore/lifecycle/policies.py
git mv evocore/scheduler.py evocore/lifecycle/scheduler.py
git mv evocore/protocols.py evocore/lifecycle/protocols.py
```

In `evocore/lifecycle/records.py`, keep these definitions:

```text
Direction
CandidateOrigin
CandidateStatus
EvaluationConfidence
STATE_UPDATE_CONFIDENCES
is_state_update_confidence
score_for_direction
EvaluationStage
EvaluationContext
ScoreObservation
EvaluationRecord
Candidate
```

Rename:

```text
Rung -> EvaluationStage
CandidateScore -> ScoreObservation
```

In `Candidate`, rename public score helper methods:

```text
best_observed_score stays best_observed_score
comparison_score stays comparison_score
best_state_score stays best_state_score
state_comparison_score stays state_comparison_score
candidate_hash stays candidate_hash
```

In `evocore/lifecycle/telemetry.py`, move these definitions out of `records.py`:

```text
OptimizationTelemetry
UpdateResult
OptimizerStateSummary
```

Rename:

```text
TellResult -> UpdateResult
EngineStateSummary -> OptimizerStateSummary
```

Keep telemetry field names for this task so behavior tests stay focused:

```text
total_candidates_proposed
unique_candidate_hashes
candidates_screened
candidates_partial_evaluated
candidates_full_evaluated
candidates_cached
promoted_by_rung
eliminated_by_rung
cost_by_rung
```

- [ ] **Step 2: Rename policy and scheduler classes**

In `evocore/lifecycle/policies.py`, rename:

```text
MultiFidelityPolicy -> BudgetPolicy
Rung imports -> EvaluationStage
```

The `single_full` classmethod must return:

```python
return cls(
    stages=[
        EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")
    ],
    max_evaluations=max_evaluations,
    batch_size=batch_size,
    exploration_fraction=0.0,
    audit_fraction=0.0,
)
```

Rename the dataclass field:

```text
rungs -> stages
rung_names -> stage_names
final_rung -> final_stage
```

In `evocore/lifecycle/scheduler.py`, rename:

```text
EvaluationScheduler -> BudgetScheduler
rung_after() -> stage_after()
assign_rung() -> assign_stage()
promote(... completed_rung=...) -> promote(... completed_stage=...)
```

- [ ] **Step 3: Update protocols**

In `evocore/lifecycle/protocols.py`, update imports and return types:

```python
from evocore.lifecycle.records import Candidate, Direction, EvaluationContext, EvaluationRecord
from evocore.lifecycle.telemetry import OptimizerStateSummary, UpdateResult
```

The `Optimizer.tell()` signature must return `UpdateResult`, and `state_summary()` must return `OptimizerStateSummary`.

- [ ] **Step 4: Create lifecycle exports**

Create `evocore/lifecycle/__init__.py`:

```python
"""Ask/tell lifecycle contracts shared by EvoCore optimizers."""

from evocore.lifecycle.batches import CandidateBatch, batch_id_from_seed
from evocore.lifecycle.policies import BudgetPolicy
from evocore.lifecycle.protocols import Evaluator, Optimizer
from evocore.lifecycle.records import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    Direction,
    EvaluationConfidence,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    ScoreObservation,
    is_state_update_confidence,
    score_for_direction,
)
from evocore.lifecycle.scheduler import BudgetScheduler
from evocore.lifecycle.telemetry import OptimizationTelemetry, OptimizerStateSummary, UpdateResult

__all__ = [
    "BudgetPolicy",
    "BudgetScheduler",
    "Candidate",
    "CandidateBatch",
    "CandidateOrigin",
    "CandidateStatus",
    "Direction",
    "EvaluationConfidence",
    "EvaluationContext",
    "EvaluationRecord",
    "EvaluationStage",
    "Evaluator",
    "OptimizationTelemetry",
    "Optimizer",
    "OptimizerStateSummary",
    "ScoreObservation",
    "UpdateResult",
    "batch_id_from_seed",
    "is_state_update_confidence",
    "score_for_direction",
]
```

- [ ] **Step 5: Update imports and tests**

Run:

```bash
rg -n "Rung|MultiFidelityPolicy|EvaluationScheduler|TellResult|EngineStateSummary|CandidateScore|evocore\\.evaluation|evocore\\.policies|evocore\\.scheduler|evocore\\.protocols|rungs|rung_names|final_rung|assign_rung|rung_after|completed_rung" evocore tests docs/site
```

Apply the naming map from this task. Update docs examples from `rungs=[Rung(...)]` to `stages=[EvaluationStage(...)]`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_vnext_policy_scheduler.py tests/unit/test_protocols.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evocore/lifecycle evocore tests docs/site
git commit -m "refactor(lifecycle): move ask tell contracts"
```

---

### Task 5: Move Results, Generation History, Events, And Reproducibility

**Files:**
- Create: `evocore/results/__init__.py`
- Create: `evocore/results/run.py`
- Move: `evocore/stats.py` -> split into `evocore/results/generation.py`, `evocore/lifecycle/events.py`, and `evocore/results/reproducibility.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_stats.py`
- Test: `tests/property/test_result_export_properties.py`
- Test: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Split `stats.py`**

Run:

```bash
mkdir evocore/results
git mv evocore/stats.py evocore/results/generation.py
```

Move these definitions from `evocore/results/generation.py` into `evocore/lifecycle/events.py`:

```text
StopReason
EventRecord
EventHistory
append_run_stop_event
```

Move this definition and helpers into `evocore/results/reproducibility.py`:

```text
ReproducibilityMetadata
gene_space_signature
gene_space_hash
```

Keep these definitions in `evocore/results/generation.py`:

```text
GenerationRecord
GenerationHistory
```

Rename:

```text
LogEntry -> GenerationRecord
Logbook -> GenerationHistory
best_fitness -> best_score
mean_fitness -> mean_score
std_fitness -> std_score
best_fitnesses() -> best_scores()
nan_counts() stays nan_counts()
```

- [ ] **Step 2: Create result envelope module**

Create `evocore/results/run.py` by moving `RunResult` and `MultiRunResult` from the old GA module. Rename the classes:

```text
RunResult -> OptimizationResult
MultiRunResult -> OptimizationBatchResult
```

The `OptimizationResult` dataclass must use this public field shape:

```python
@dataclass
class OptimizationResult:
    """Store the outcome of one optimization run."""

    best_solution: Solution
    best_score: float
    final_solutions: SolutionSet
    generations: GenerationHistory
    wall_time_seconds: float
    n_evaluations: int
    elite_solutions: list[Solution]
    diversity_by_generation: list[list[float]]
    seed: int
    stop_reason: StopReason = "max_generations"
    max_generations: int | None = None
    max_evaluations: int | None = None
    telemetry: OptimizationTelemetry = field(default_factory=OptimizationTelemetry)
    direction: Direction = "maximize"
    optimizer_type: str = ""
    best_candidate_id: str | None = None
    best_observed_score: float | None = None
    events: EventHistory = field(default_factory=EventHistory)
    reproducibility: ReproducibilityMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

The `to_dict()` payload must set:

```python
"schema_version": 2
"optimizer_type": self.optimizer_type
"best": {
    "score": best_observed_score,
    "candidate_id": self.best_candidate_id,
    "values": list(self.best_solution.values),
    "params": self.best_solution.metadata.get("params"),
}
"budget": {
    "max_evaluations": self.max_evaluations,
    "max_generations": self.max_generations,
    "n_evaluations": self.n_evaluations,
}
"events": self.events.to_dict()
"generations": self.generations.to_dict()
```

Do not include `best.fitness`, `history`, or `logbook` keys.

The `OptimizationBatchResult` dataclass must use:

```python
@dataclass
class OptimizationBatchResult:
    """Store the aggregated outcome of multiple optimizer runs."""

    best: OptimizationResult
    all_runs: list[OptimizationResult]
    n_runs: int
    wall_time_seconds: float
    direction: Direction = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)
```

Rename `fitness_summary()` to `score_summary()` and export it under `score_summary`.

- [ ] **Step 3: Create results exports**

Create `evocore/results/__init__.py`:

```python
"""Completed-run result and reporting types."""

from evocore.lifecycle.events import EventHistory, EventRecord, StopReason, append_run_stop_event
from evocore.results.generation import GenerationHistory, GenerationRecord
from evocore.results.reproducibility import (
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)
from evocore.results.run import OptimizationBatchResult, OptimizationResult

__all__ = [
    "EventHistory",
    "EventRecord",
    "GenerationHistory",
    "GenerationRecord",
    "OptimizationBatchResult",
    "OptimizationResult",
    "ReproducibilityMetadata",
    "StopReason",
    "append_run_stop_event",
    "gene_space_hash",
    "gene_space_signature",
]
```

- [ ] **Step 4: Update result tests**

Update `tests/unit/test_stats.py` and `tests/property/test_result_export_properties.py` with these import replacements:

```text
from evocore.stats import -> from evocore.results import
LogEntry -> GenerationRecord
Logbook -> GenerationHistory
RunResult -> OptimizationResult
MultiRunResult -> OptimizationBatchResult
best_fitness -> best_score
mean_fitness -> mean_score
std_fitness -> std_score
fitness_summary -> score_summary
history -> events
logbook -> generations
```

Add this assertion to the result export tests:

```python
payload = result.to_dict()
assert payload["schema_version"] == 2
assert "fitness" not in payload["best"]
assert "score" in payload["best"]
assert "events" in payload
assert "generations" in payload
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/unit/test_stats.py tests/property/test_result_export_properties.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evocore/results evocore/lifecycle/events.py evocore tests
git commit -m "refactor(results): move optimization result envelopes"
```

---

### Task 6: Move Callbacks And Surrogate Advisors

**Files:**
- Move: `evocore/callbacks.py` -> `evocore/callbacks/__init__.py`
- Move: `evocore/advisors.py` -> `evocore/surrogates/__init__.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_callbacks.py`
- Test: `tests/unit/test_vnext_advisors.py`

- [ ] **Step 1: Move callbacks into a package**

Run:

```bash
mkdir evocore/callbacks
git mv evocore/callbacks.py evocore/callbacks/__init__.py
```

In `evocore/callbacks/__init__.py`, update type-checking imports:

```python
if TYPE_CHECKING:
    from evocore.results import OptimizationResult
    from evocore.search_space import SolutionSet
```

Update callback signatures:

```text
Population -> SolutionSet
RunResult -> OptimizationResult
best_fitness record key -> best_score
```

- [ ] **Step 2: Move surrogate advisor into a package**

Run:

```bash
mkdir evocore/surrogates
git mv evocore/advisors.py evocore/surrogates/__init__.py
```

Rename in `evocore/surrogates/__init__.py`:

```text
AdvisorScore -> SurrogateScore
InverseDistanceSurrogateAdvisor -> InverseDistanceAdvisor
GeneDef/GeneSpace imports -> evocore.search_space
Candidate/EvaluationRecord imports -> evocore.lifecycle
```

- [ ] **Step 3: Update imports and tests**

Run:

```bash
rg -n "evocore\\.callbacks|evocore\\.advisors|AdvisorScore|InverseDistanceSurrogateAdvisor|best_fitness|Population|RunResult" evocore tests docs/site
```

Apply replacements:

```text
from evocore.advisors import -> from evocore.surrogates import
AdvisorScore -> SurrogateScore
InverseDistanceSurrogateAdvisor -> InverseDistanceAdvisor
best_fitness -> best_score
Population -> SolutionSet
RunResult -> OptimizationResult
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/unit/test_callbacks.py tests/unit/test_vnext_advisors.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evocore/callbacks evocore/surrogates evocore tests docs/site
git commit -m "refactor(callbacks): move callbacks and surrogates"
```

---

### Task 7: Split The Genetic Algorithm Optimizer

**Files:**
- Create: `evocore/optimizers/__init__.py`
- Create: `evocore/optimizers/ga/__init__.py`
- Move and split: `evocore/ga.py` -> `evocore/optimizers/ga/*.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_ga_engine.py`
- Test: `tests/unit/test_ga_ask_tell_vnext.py`
- Test: `tests/unit/test_rng_reproducibility.py`
- Test: `tests/unit/test_runtime_observability.py`
- Test: `tests/integration/test_binary_onemax.py`
- Test: `tests/integration/test_mixed_gene_space.py`
- Test: `tests/integration/test_rastrigin.py`
- Test: `tests/integration/test_sphere_function.py`

- [ ] **Step 1: Create package shell and failing GA import test**

Update GA tests to import the new class:

```text
from evocore import GAEngine -> from evocore import GeneticAlgorithmOptimizer
from evocore.ga import RunResult, MultiRunResult -> from evocore.results import OptimizationResult, OptimizationBatchResult
GAEngine -> GeneticAlgorithmOptimizer
RunResult -> OptimizationResult
MultiRunResult -> OptimizationBatchResult
```

Run:

```bash
python -m pytest tests/unit/test_ga_engine.py::test_invalid_parallel_mode_rejected -v
```

Expected: FAIL because `GeneticAlgorithmOptimizer` is not implemented yet.

- [ ] **Step 2: Create optimizer package exports**

Create `evocore/optimizers/__init__.py`:

Run:

```bash
mkdir evocore/optimizers
mkdir evocore/optimizers/ga
```

Then create `evocore/optimizers/__init__.py`:

```python
"""Optimization algorithm implementations."""

from evocore.optimizers.ga import GeneticAlgorithmOptimizer

__all__ = ["GeneticAlgorithmOptimizer"]
```

Create `evocore/optimizers/ga/__init__.py`:

```python
"""Genetic algorithm optimizer."""

from evocore.optimizers.ga.engine import GeneticAlgorithmOptimizer
from evocore.optimizers.ga.multi_run import run_child_optimizer

__all__ = ["GeneticAlgorithmOptimizer", "run_child_optimizer"]
```

- [ ] **Step 3: Split `ga.py` by method ownership**

Create `evocore/optimizers/ga/engine.py` containing the public class definition, constructor, shared state helpers, state summary, metadata helpers, and inheritance from mixins:

```python
class GeneticAlgorithmOptimizer(
    GAAskTellMixin,
    GAGenerationLoopMixin,
    GAMultiRunMixin,
    GACheckpointMixin,
):
    """Run deterministic genetic algorithm optimization over a gene space."""
```

Move these existing methods into `engine.py` and rename types/imports:

```text
__init__
_reset_vnext_state
_pending_batch_ids
_best_candidate_id_and_score
_record_state_candidate
state_summary
_warn_if_large_int_gene_without_sigma
_optimizer_config
_reproducibility_metadata
```

Create `evocore/optimizers/ga/reproduction.py` and move:

```text
_normalise_fitness_result
_remaining_evaluations
_fitnesses_for_selection
_evaluate_with_budget
_evaluate_all
_compute_sigma_fraction
_initial_population
_clone_elites
_make_offspring
```

Expose them through:

```python
class GAReproductionMixin:
    ...
```

Make `GAGenerationLoopMixin` inherit `GAReproductionMixin` if it calls those methods, or make `GeneticAlgorithmOptimizer` inherit both mixins in this order:

```python
class GeneticAlgorithmOptimizer(
    GAAskTellMixin,
    GAGenerationLoopMixin,
    GAMultiRunMixin,
    GACheckpointMixin,
    GAReproductionMixin,
):
```

Create `evocore/optimizers/ga/generation_loop.py` and move:

```text
_bind_callbacks
_callbacks_should_stop
_generation_record
_record_generation
_run_generation
_run_from_solutions
```

Rename:

```text
_log_entry -> _generation_record
_run_from_population -> _run_from_solutions
Population -> SolutionSet
Individual -> Solution
Logbook -> GenerationHistory
LogEntry -> GenerationRecord
RunResult -> OptimizationResult
fitness_fn parameter -> objective_fn parameter on private legacy generation-loop methods
```

Create `evocore/optimizers/ga/ask_tell.py` and move:

```text
_candidate_from_genes
ask
tell
_evaluation_context
_validate_evaluator_records
_append_ask_events
_append_tell_event
run
```

Rename local variables from `rung` to `stage` in policy-driven execution:

```text
resolved_policy.rungs -> resolved_policy.stages
scheduler.assign_rung -> scheduler.assign_stage
scheduler.promote(... completed_rung=...) -> scheduler.promote(... completed_stage=...)
```

Create `evocore/optimizers/ga/multi_run.py` and move:

```text
_run_child_engine -> run_child_optimizer
run_multiple
_copy_with_seed
```

Create `evocore/optimizers/ga/checkpointing.py` and move:

```text
resume
```

In `checkpointing.py`, implement legacy checkpoint unpickling for old `evocore.individual.Individual` payloads:

```python
class _LegacyCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "evocore.individual" and name == "Individual":
            return Solution
        return super().find_class(module, name)
```

Use:

```python
payload = _LegacyCheckpointUnpickler(handle).load()
```

Validate checkpoint populations with:

```python
if not isinstance(solutions, list) or not all(isinstance(solution, Solution) for solution in solutions):
    raise CheckpointError(
        "checkpoint payload must contain a list[Solution] under key 'population'."
    )
```

Keep the pickle key `"population"` for compatibility with existing checkpoint files.

- [ ] **Step 4: Rename public result construction**

In all GA result construction, use:

```text
OptimizationResult
best_solution
best_score
final_solutions
generations
elite_solutions
diversity_by_generation
optimizer_type="GeneticAlgorithmOptimizer"
events
```

When constructing `Solution`, use:

```python
Solution(
    list(candidate.values if hasattr(candidate, "values") else candidate.genes),
    score=candidate.best_state_score(self.direction),
    score_valid=True,
    metadata={"params": candidate.params, "candidate_id": candidate.candidate_id},
)
```

Use direct `list(candidate.genes)` if `Candidate` still exposes `genes`; do not rename `Candidate.genes` in this task.

- [ ] **Step 5: Update GA tests for public score vocabulary**

In `tests/unit/test_ga_engine.py` and `tests/unit/test_ga_ask_tell_vnext.py`, replace:

```text
result.best_fitness -> result.best_score
result.best_individual -> result.best_solution
result.final_population -> result.final_solutions
result.logbook -> result.generations
result.history -> result.events
result.elite_history -> result.elite_solutions
result.diversity_history -> result.diversity_by_generation
fitness_summary() -> score_summary()
fitness_fn variable names -> objective_fn where the callable is part of public examples
```

Do not rename private test helpers that intentionally assert Rust fitness behavior.

- [ ] **Step 6: Run focused GA tests**

Run:

```bash
python -m pytest tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_rng_reproducibility.py tests/unit/test_runtime_observability.py -v
python -m pytest tests/integration/test_binary_onemax.py tests/integration/test_mixed_gene_space.py tests/integration/test_rastrigin.py tests/integration/test_sphere_function.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evocore/optimizers evocore tests docs/site
git commit -m "refactor(ga): split genetic algorithm optimizer"
```

---

### Task 8: Split The CMA-ES Optimizer

**Files:**
- Create: `evocore/optimizers/cmaes/__init__.py`
- Move and split: `evocore/cmaes.py` -> `evocore/optimizers/cmaes/engine.py` and `evocore/optimizers/cmaes/ask_tell.py`
- Move: `evocore/mixed_cma.py` -> `evocore/optimizers/cmaes/mixed.py`
- Modify: imports in `evocore/`, `tests/`, and `docs/site/`
- Test: `tests/unit/test_cmaes_engine.py`
- Test: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Test: `tests/unit/test_cmaes_rust.py`
- Test: `tests/unit/test_mixed_cma_vnext.py`
- Test: `tests/integration/test_cmaes_rosenbrock.py`

- [ ] **Step 1: Update CMA tests to fail on new imports**

Apply replacements:

```text
CMAESEngine -> CMAESOptimizer
from evocore.cmaes import CMAESEngine -> from evocore.optimizers.cmaes import CMAESOptimizer
from evocore.mixed_cma import -> from evocore.optimizers.cmaes import
IntegerMargin -> IntegerMarginDistribution
CategoricalState -> CategoricalDistributionState
```

Run:

```bash
python -m pytest tests/unit/test_cmaes_engine.py::test_invalid_parallel_process_message -v
```

Expected: FAIL because `CMAESOptimizer` is not implemented yet.

- [ ] **Step 2: Create CMA package exports**

Create `evocore/optimizers/cmaes/__init__.py`:

Run:

```bash
mkdir evocore/optimizers/cmaes
```

Then create `evocore/optimizers/cmaes/__init__.py`:

```python
"""CMA-ES optimizer."""

from evocore.optimizers.cmaes.engine import CMAESOptimizer
from evocore.optimizers.cmaes.mixed import (
    CategoricalDistributionState,
    IntegerMarginDistribution,
)

__all__ = [
    "CMAESOptimizer",
    "CategoricalDistributionState",
    "IntegerMarginDistribution",
]
```

Update `evocore/optimizers/__init__.py`:

```python
"""Optimization algorithm implementations."""

from evocore.optimizers.cmaes import CMAESOptimizer
from evocore.optimizers.ga import GeneticAlgorithmOptimizer

__all__ = ["CMAESOptimizer", "GeneticAlgorithmOptimizer"]
```

- [ ] **Step 3: Split `cmaes.py`**

Move the public class to `evocore/optimizers/cmaes/engine.py` and rename:

```text
CMAESEngine -> CMAESOptimizer
RunResult -> OptimizationResult
Individual -> Solution
Population -> SolutionSet
OperatorSet -> OperatorCodec
Logbook -> GenerationHistory
LogEntry -> GenerationRecord
fitness_fn parameter -> objective_fn in public run signature
```

Create `evocore/optimizers/cmaes/ask_tell.py` and move:

```text
_candidate_and_batch_for_record
_apply_record_confidence
_consume_complete_batch
_append_ask_events
_append_tell_event
ask
tell
```

Expose them through:

```python
class CMAESAskTellMixin:
    ...
```

Make `CMAESOptimizer` inherit `CMAESAskTellMixin`.

- [ ] **Step 4: Move mixed CMA helpers**

Run:

```bash
git mv evocore/mixed_cma.py evocore/optimizers/cmaes/mixed.py
```

Rename:

```text
IntegerMargin -> IntegerMarginDistribution
CategoricalState -> CategoricalDistributionState
```

- [ ] **Step 5: Update CMA result construction and logging**

Use:

```text
OptimizationResult
optimizer_type="CMAESOptimizer"
best_solution
best_score
final_solutions
generations
events
```

Update runtime log assertions in `tests/unit/test_runtime_observability.py`:

```text
record.name == "evocore.optimizers.cmaes.engine"
```

For GA runtime log assertions from Task 7, use:

```text
record.name == "evocore.optimizers.ga.generation_loop"
```

- [ ] **Step 6: Run focused CMA tests**

Run:

```bash
python -m pytest tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_rust.py tests/unit/test_mixed_cma_vnext.py tests/integration/test_cmaes_rosenbrock.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evocore/optimizers/cmaes evocore tests docs/site
git commit -m "refactor(cmaes): split cmaes optimizer"
```

---

### Task 9: Update Top-Level Exports And Remove Old Flat Modules

**Files:**
- Modify: `evocore/__init__.py`
- Delete: remaining old flat modules listed in the File Structure section
- Modify: tests that still import old module paths
- Test: `tests/unit/test_package_init.py`
- Test: `tests/unit/test_domain_imports.py`

- [ ] **Step 1: Replace `evocore/__init__.py` exports**

Replace imports in `evocore/__init__.py` with domain imports:

```python
"""Top-level evocore package exports."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("evocore")
except _metadata.PackageNotFoundError:
    __version__ = "0.7.0"

from evocore import _core
from evocore._core import (
    OP_CMAES_ASK,
    OP_CROSSOVER,
    OP_CROSSOVER_PROB,
    OP_INIT,
    OP_MULTI_RUN,
    OP_MUTATION,
    OP_SELECTION,
    BinaryIndividual,
    FloatIndividual,
    IntegerIndividual,
    py_derive_seed,
)
from evocore.callbacks import Callback, CheckpointCallback, EarlyStopping, GenerationInfo, MetricsLogger, ProgressBar
from evocore.core import (
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    ConvergenceError,
    EvocoreError,
    FitnessError,
    FitnessWarning,
    ParallelError,
    ProcessParallel,
    ThreadParallel,
)
from evocore.lifecycle import (
    BudgetPolicy,
    BudgetScheduler,
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    Direction,
    EvaluationConfidence,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    Evaluator,
    OptimizationTelemetry,
    Optimizer,
    OptimizerStateSummary,
    ScoreObservation,
    UpdateResult,
)
from evocore.optimizers.cmaes import CMAESOptimizer, CategoricalDistributionState, IntegerMarginDistribution
from evocore.optimizers.ga import GeneticAlgorithmOptimizer
from evocore.results import (
    EventHistory,
    EventRecord,
    GenerationHistory,
    GenerationRecord,
    OptimizationBatchResult,
    OptimizationResult,
    ReproducibilityMetadata,
)
from evocore.search_space import Gene, GeneKind, GeneSpace, GeneValue, OperatorCodec, Solution, SolutionSet
from evocore.surrogates import InverseDistanceAdvisor, SurrogateScore
```

Do not export old public names `GAEngine`, `CMAESEngine`, `RunResult`, `MultiRunResult`, `Individual`, `Population`, `Rung`, `MultiFidelityPolicy`, `EvaluationScheduler`, `TellResult`, `EngineStateSummary`, `Logbook`, `LogEntry`, `OperatorSet`, `AdvisorScore`, or `InverseDistanceSurrogateAdvisor`.

- [ ] **Step 2: Update `__all__`**

Set `__all__` to this exact list:

```python
__all__ = [
    "OP_CMAES_ASK",
    "OP_CROSSOVER",
    "OP_CROSSOVER_PROB",
    "OP_INIT",
    "OP_MULTI_RUN",
    "OP_MUTATION",
    "OP_SELECTION",
    "BinaryIndividual",
    "BudgetPolicy",
    "BudgetScheduler",
    "CMAESOptimizer",
    "Callback",
    "Candidate",
    "CandidateOrigin",
    "CandidateStatus",
    "CategoricalDistributionState",
    "CheckpointCallback",
    "CheckpointError",
    "ConfigurationError",
    "ConfigurationWarning",
    "ConvergenceError",
    "Direction",
    "EarlyStopping",
    "EvaluationConfidence",
    "EvaluationContext",
    "EvaluationRecord",
    "EvaluationStage",
    "Evaluator",
    "EventHistory",
    "EventRecord",
    "EvocoreError",
    "FitnessError",
    "FitnessWarning",
    "FloatIndividual",
    "Gene",
    "GeneKind",
    "GeneSpace",
    "GeneValue",
    "GenerationHistory",
    "GenerationInfo",
    "GenerationRecord",
    "GeneticAlgorithmOptimizer",
    "IntegerIndividual",
    "IntegerMarginDistribution",
    "InverseDistanceAdvisor",
    "MetricsLogger",
    "OptimizationBatchResult",
    "OptimizationResult",
    "OptimizationTelemetry",
    "Optimizer",
    "OptimizerStateSummary",
    "OperatorCodec",
    "ParallelError",
    "ProcessParallel",
    "ProgressBar",
    "ReproducibilityMetadata",
    "ScoreObservation",
    "Solution",
    "SolutionSet",
    "SurrogateScore",
    "ThreadParallel",
    "UpdateResult",
    "__version__",
    "_core",
    "py_derive_seed",
]
```

- [ ] **Step 3: Delete remaining old flat modules**

Run:

```bash
git rm -f evocore/advisors.py evocore/batches.py evocore/cmaes.py evocore/evaluation.py evocore/exceptions.py evocore/exporting.py evocore/ga.py evocore/gene_space.py evocore/individual.py evocore/mixed_cma.py evocore/operators.py evocore/parallel.py evocore/policies.py evocore/protocols.py evocore/scheduler.py evocore/stats.py
```

The command will skip files already moved if they no longer exist. If PowerShell stops on a missing path, rerun `git status --short`, remove only paths still present, and continue.

- [ ] **Step 4: Search for old paths and names**

Run:

```bash
rg -n "evocore\\.(ga|cmaes|gene_space|individual|evaluation|policies|scheduler|stats|parallel|advisors|mixed_cma|operators|exceptions|exporting|protocols)|\\b(GAEngine|CMAESEngine|RunResult|MultiRunResult|GeneDef|Individual|Population|Rung|MultiFidelityPolicy|EvaluationScheduler|TellResult|EngineStateSummary|Logbook|LogEntry|OperatorSet|AdvisorScore|InverseDistanceSurrogateAdvisor|IntegerMargin\\b|CategoricalState\\b|CandidateScore\\b)" evocore tests docs/site CHANGELOG.md
```

Expected: no matches in `evocore`, `tests`, or `docs/site` except:

```text
CHANGELOG.md historical release notes
docs/superpowers historical specs and plans
```

Do not edit historical `docs/superpowers` files for old names.

- [ ] **Step 5: Run import tests**

Run:

```bash
python -m pytest tests/unit/test_package_init.py tests/unit/test_domain_imports.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evocore tests docs/site
git commit -m "refactor(api): expose domain package names"
```

---

### Task 10: Update Documentation And Changelog

**Files:**
- Modify: `docs/site/api.md`
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/budget-aware-optimization.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/gene-space.md`
- Modify: `docs/site/mixed-variable-search.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `docs/site/parallelism.md`
- Modify: `docs/site/quickstart.md`
- Modify: `mkdocs.yml`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md` if implementation added `lifecycle/batches.py`, `optimizers/ga/multi_run.py`, or another final module not currently listed

- [ ] **Step 1: Update docs imports and prose**

Run:

```bash
rg -n "GAEngine|CMAESEngine|RunResult|MultiRunResult|GeneDef|Individual|Population|Rung|MultiFidelityPolicy|EvaluationScheduler|TellResult|EngineStateSummary|Logbook|LogEntry|OperatorSet|AdvisorScore|InverseDistanceSurrogateAdvisor|IntegerMargin\\b|CategoricalState\\b|fitness|evocore\\.(ga|cmaes|gene_space|individual|evaluation|policies|scheduler|stats|parallel|advisors|mixed_cma|operators|exceptions|exporting|protocols)" docs/site
```

Apply the naming map from this plan. Replace public `fitness` prose with `score` prose unless the docs are explaining GA internals or Rust operator internals.

- [ ] **Step 2: Update MkDocs API references**

In `docs/site/api.md`, use these references:

```markdown
# API Reference

::: evocore.search_space

::: evocore.lifecycle

::: evocore.results

::: evocore.optimizers.ga.GeneticAlgorithmOptimizer

::: evocore.optimizers.cmaes.CMAESOptimizer

::: evocore.callbacks

::: evocore.surrogates

::: evocore.core.errors

::: evocore.core.parallel
```

In `docs/site/ga.md`, update API references:

```markdown
::: evocore.optimizers.ga.GeneticAlgorithmOptimizer
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.results.OptimizationResult

::: evocore.results.OptimizationBatchResult
```

In `docs/site/cmaes.md`, update:

```markdown
::: evocore.optimizers.cmaes.CMAESOptimizer
```

In `docs/site/callbacks-checkpointing.md`, update:

```markdown
::: evocore.callbacks.Callback
::: evocore.callbacks.EarlyStopping
::: evocore.callbacks.ProgressBar
::: evocore.callbacks.CheckpointCallback
::: evocore.callbacks.MetricsLogger
```

- [ ] **Step 3: Update changelog**

Add this entry near the top of `CHANGELOG.md`:

```markdown
- Breaking: Reorganized the Python package into domain modules under
  `evocore.core`, `evocore.search_space`, `evocore.lifecycle`, `evocore.results`,
  and `evocore.optimizers`. Top-level convenience imports remain available with
  new public names such as `GeneticAlgorithmOptimizer`, `CMAESOptimizer`,
  `Gene`, `Solution`, `OptimizationResult`, `BudgetPolicy`, and
  `EvaluationStage`. Old flat module paths such as `evocore.ga`,
  `evocore.cmaes`, `evocore.gene_space`, and `evocore.stats` are no longer
  public import paths.
- Breaking: Result exports now use schema version `2` and score-oriented fields
  such as `best.score`, `generations`, `events`, and `score_summary` instead of
  fitness-oriented result keys.
```

- [ ] **Step 4: Build docs**

Run:

```bash
python -m mkdocs build
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/site mkdocs.yml CHANGELOG.md AGENTS.md
git commit -m "docs: document domain package api"
```

---

### Task 11: Run Full Verification And Fix Mechanical Breaks

**Files:**
- Modify: only files with failing imports, renamed attributes, or format/lint issues
- Test: full relevant suite

- [ ] **Step 1: Run formatting check**

Run:

```bash
python -m ruff format --check
```

Expected: PASS. If it fails with formatting diffs, run:

```bash
python -m ruff format
python -m ruff format --check
```

Then stage the formatted files in this task's final commit.

- [ ] **Step 2: Run lint**

Run:

```bash
python -m ruff check
```

Expected: PASS. Fix only issues introduced by this migration.

- [ ] **Step 3: Build extension**

Run:

```bash
python -m maturin develop --release
```

Expected: PASS.

- [ ] **Step 4: Run unit and integration tests**

Run:

```bash
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 5: Run property tests**

Run:

```bash
python -m pytest tests/property/ -v
```

Expected: PASS.

- [ ] **Step 6: Run benchmark smoke tests**

Run:

```bash
python -m pytest tests/benchmarks/ -v
```

Expected: PASS.

- [ ] **Step 7: Run Rust checks**

Run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: PASS.

- [ ] **Step 8: Run docs build**

Run:

```bash
python -m mkdocs build
```

Expected: PASS.

- [ ] **Step 9: Commit verification fixes**

If any files changed during verification:

```bash
git add evocore tests docs/site CHANGELOG.md AGENTS.md mkdocs.yml
git commit -m "fix: complete domain package migration"
```

If no files changed, do not create an empty commit.

---

### Task 12: Final Migration Audit

**Files:**
- Modify: only files with missed old public names or stale docs

- [ ] **Step 1: Audit old flat module imports**

Run:

```bash
rg -n "from evocore\\.(ga|cmaes|gene_space|individual|evaluation|policies|scheduler|stats|parallel|advisors|mixed_cma|operators|exceptions|exporting|protocols)|import evocore\\.(ga|cmaes|gene_space|individual|evaluation|policies|scheduler|stats|parallel|advisors|mixed_cma|operators|exceptions|exporting|protocols)" evocore tests docs/site
```

Expected: no matches.

- [ ] **Step 2: Audit old public names**

Run:

```bash
rg -n "\\b(GAEngine|CMAESEngine|RunResult|MultiRunResult|GeneDef|Individual|Population|Rung|MultiFidelityPolicy|EvaluationScheduler|TellResult|EngineStateSummary|Logbook|LogEntry|OperatorSet|AdvisorScore|InverseDistanceSurrogateAdvisor|IntegerMargin\\b|CategoricalState\\b|CandidateScore\\b)" evocore tests docs/site
```

Expected: no matches.

- [ ] **Step 3: Audit public result fitness vocabulary**

Run:

```bash
rg -n "best_fitness|mean_fitness|std_fitness|fitness_summary|best_individual|final_population|elite_history|diversity_history|logbook|history" evocore tests docs/site
```

Expected: no matches in public result code, tests, or docs. Matches inside Rust extension tests or private GA implementation comments are allowed only when they refer to internal Rust fitness arrays.

- [ ] **Step 4: Audit source file sizes**

Run:

```bash
Get-ChildItem -Path evocore -Filter *.py -Recurse | ForEach-Object { $lines = (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines; [PSCustomObject]@{ Lines = $lines; Path = $_.FullName.Substring((Get-Location).Path.Length + 1) } } | Sort-Object Lines -Descending | Format-Table -AutoSize
```

Expected: no Python source file in `evocore/` exceeds 600 lines. If a file exceeds 600 lines, split the largest coherent private helper group into a sibling module and rerun focused tests for that domain.

- [ ] **Step 5: Commit audit fixes**

If any files changed:

```bash
git add evocore tests docs/site CHANGELOG.md AGENTS.md
git commit -m "chore: finish domain migration audit"
```

If no files changed, do not create an empty commit.

---

## Final Verification

Before reporting completion, run:

```bash
python -m ruff format --check
python -m ruff check
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
python -m pytest tests/property/ -v
python -m pytest tests/benchmarks/ -v
python -m mkdocs build
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If any command fails, stop and report the failing command, the relevant error summary, and the likely files involved. Do not push or open a ready PR after failed verification.

If all commands pass, push the branch and update the existing draft PR with the verification results.
