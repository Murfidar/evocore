# Optimizer Configuration Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable, hook-aware optimizer configuration export, hashing, compatibility validation, and reproducibility metadata for GA and CMA-ES without bloating engine files.

**Architecture:** Keep optimizer engine classes thin and delegate config work to focused modules. Shared config primitives live in `evocore/optimizers/config.py`; optimizer-specific config assembly lives beside each optimizer in `evocore/optimizers/ga/config.py` and `evocore/optimizers/cmaes/config.py`. Result reproducibility metadata consumes those signatures and keeps runtime hooks separate from the core optimizer config hash.

**Tech Stack:** Python dataclasses and Protocols, existing `json_safe` and `canonical_json_hash` helpers, pytest, Hypothesis property tests, MkDocs markdown docs, repository-local `.venv` commands.

---

## File Structure

Create:

- `evocore/optimizers/config.py`
  Shared config dataclasses, protocols, config hashing, object identity helpers, callback hook classification, and reproducibility-status aggregation.

- `evocore/optimizers/ga/config.py`
  Genetic algorithm config assembly, runtime hook collection, reproducibility status, and explicit GA compatibility validation.

- `evocore/optimizers/cmaes/config.py`
  CMA-ES config assembly, runtime hook collection, reproducibility status, and explicit CMA-ES compatibility validation.

- `tests/unit/test_optimizer_config.py`
  Unit tests for shared config primitives, GA config export, CMA-ES config export, hook classification, config hashing, and public import surfaces.

- `tests/property/test_optimizer_config_properties.py`
  Property tests for JSON round-tripping, equivalent config hashes, and hash changes for reproducibility-critical fields.

Modify:

- `evocore/optimizers/__init__.py`
  Re-export public optimizer config helpers.

- `evocore/optimizers/ga/engine.py`
  Add thin public delegation methods: `config()`, `config_signature()`, `config_hash()`, `validate_compatibility()`. Update reproducibility metadata creation to include the new config hash, hook status, notes, and runtime hooks.

- `evocore/optimizers/cmaes/engine.py`
  Add the same thin public delegation methods and reproducibility metadata integration.

- `evocore/results/reproducibility.py`
  Extend `ReproducibilityMetadata` with `optimizer_config_hash`, `reproducibility_status`, `reproducibility_notes`, and `runtime_hooks`.

- `evocore/__init__.py`
  Re-export public config primitives that should be top-level convenience imports.

- `tests/unit/test_stats.py`
  Update reproducibility metadata expectations for the new fields.

- `tests/unit/test_ga_engine.py`
  Update existing flat `optimizer_config` assertions to the nested canonical signature and add result metadata hash/status assertions.

- `tests/unit/test_cmaes_engine.py`
  Update existing flat `optimizer_config` assertions to the nested canonical signature and add result metadata hash/status assertions.

- `tests/unit/test_domain_imports.py`
  Assert focused modules and public config symbols are importable from expected ownership locations.

- `docs/site/ga.md`
  Add a short config export/hash example.

- `docs/site/cmaes.md`
  Add a short config export/hash example.

- `docs/site/optimizer-telemetry.md`
  Add the new reproducibility metadata fields and hook status explanation.

- `docs/site/gene-space.md`
  Mention that gene-space hashes and optimizer config hashes are separate comparison tools.

- `docs/site/api.md`
  Include optimizer config primitives and new reproducibility fields.

- `CHANGELOG.md`
  Add an Unreleased entry for the optimizer configuration contract.

Do not move algorithm, lifecycle, reproduction, checkpoint, or ask/tell behavior into the config modules. If any implementation file starts combining unrelated responsibilities, split before it grows into a large multi-purpose module.

---

### Task 1: Shared Optimizer Config Primitives

**Files:**
- Create: `evocore/optimizers/config.py`
- Create: `tests/unit/test_optimizer_config.py`
- Modify: `evocore/optimizers/__init__.py`

- [ ] **Step 1: Write failing tests for shared config values and hook signatures**

Add this initial content to `tests/unit/test_optimizer_config.py`:

```python
import json

from evocore.callbacks import Callback, EarlyStopping, MetricsLogger, ProgressBar
from evocore.optimizers.config import (
    OptimizerConfig,
    RuntimeHookSignature,
    callback_hook_signature,
    config_hash,
    reproducibility_from_hooks,
    stable_object_identity,
)


class CustomCallback(Callback):
    pass


def sample_initializer() -> None:
    return None


def test_optimizer_config_to_dict_and_hash_are_deterministic():
    config = OptimizerConfig(
        optimizer_type="ExampleOptimizer",
        parameters={"seed": 42, "population_size": 8},
        components={
            "mutation": {
                "type": "gaussian",
                "parameters": {"sigma": 0.2, "probability": 0.1},
            }
        },
    )

    assert config.to_dict() == {
        "schema_version": 1,
        "optimizer_type": "ExampleOptimizer",
        "parameters": {"population_size": 8, "seed": 42},
        "components": {
            "mutation": {
                "parameters": {"probability": 0.1, "sigma": 0.2},
                "type": "gaussian",
            }
        },
    }
    assert config_hash(config.to_dict()) == config_hash(config.to_dict())
    assert json.loads(config.to_json()) == config.to_dict()


def test_runtime_hook_signature_to_dict_is_json_safe():
    hook = RuntimeHookSignature(
        hook_type="artifact",
        identity="evocore.callbacks.MetricsLogger",
        config={"path": {"b", "a"}},
        reproducibility="configured",
        notes=("writes metrics",),
    )

    assert hook.to_dict() == {
        "hook_type": "artifact",
        "identity": "evocore.callbacks.MetricsLogger",
        "config": {"path": ["a", "b"]},
        "reproducibility": "configured",
        "notes": ["writes metrics"],
    }


def test_stable_object_identity_uses_module_and_qualname():
    assert stable_object_identity(sample_initializer).endswith(
        "test_optimizer_config.sample_initializer"
    )


def test_callback_hook_signature_classifies_known_callbacks(tmp_path):
    metrics = callback_hook_signature(MetricsLogger(str(tmp_path / "metrics.jsonl")))
    progress = callback_hook_signature(ProgressBar())
    early_stop = callback_hook_signature(EarlyStopping(patience=3, min_delta=0.5))

    assert metrics.hook_type == "artifact"
    assert metrics.identity == "evocore.callbacks.metrics.MetricsLogger"
    assert metrics.config == {"path": str(tmp_path / "metrics.jsonl")}
    assert metrics.reproducibility == "configured"

    assert progress.hook_type == "artifact"
    assert progress.identity == "evocore.callbacks.progress.ProgressBar"
    assert progress.config == {}
    assert progress.reproducibility == "configured"

    assert early_stop.hook_type == "termination"
    assert early_stop.identity == "evocore.callbacks.stopping.EarlyStopping"
    assert early_stop.config == {"patience": 3, "min_delta": 0.5}
    assert early_stop.reproducibility == "configured"


def test_unknown_callback_marks_reproducibility_partial():
    hook = callback_hook_signature(CustomCallback())

    assert hook.hook_type == "termination"
    assert hook.reproducibility == "partial"
    assert hook.notes == (
        "CustomCallback may affect termination or side effects without a stable hook signature.",
    )


def test_reproducibility_from_hooks_reports_partial_notes():
    hooks = (
        RuntimeHookSignature(
            hook_type="environment",
            identity="tests.sample_initializer",
            config={},
            reproducibility="partial",
            notes=("process_initializer is opaque.",),
        ),
    )

    status, notes = reproducibility_from_hooks(hooks)

    assert status == "partial"
    assert notes == ("process_initializer is opaque.",)
```

- [ ] **Step 2: Run shared config tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py -v
```

Expected: fail during import because `evocore.optimizers.config` does not exist.

- [ ] **Step 3: Implement the shared config module**

Create `evocore/optimizers/config.py` with this content:

```python
"""Shared optimizer configuration export helpers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from evocore.core.serialization import canonical_json_hash, json_safe, stable_json_dumps

if TYPE_CHECKING:
    from evocore.callbacks import Callback

HookType = Literal["termination", "artifact", "environment"]
HookReproducibility = Literal["configured", "partial"]
ReproducibilityStatus = Literal["full", "partial"]


@runtime_checkable
class ConfigurableComponent(Protocol):
    """Protocol for custom algorithm components with stable config identity."""

    def config_signature(self) -> dict[str, Any]:
        """Return a JSON-safe canonical component signature."""

    def validate_compatibility(self, gene_space) -> None:
        """Raise ConfigurationError when incompatible with a gene space."""


@dataclass(frozen=True)
class OptimizerConfig:
    """Canonical optimizer configuration used for comparison and hashing."""

    optimizer_type: str
    parameters: Mapping[str, Any]
    components: Mapping[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Export this config as a deterministic JSON-safe signature."""
        return json_safe(
            {
                "schema_version": self.schema_version,
                "optimizer_type": self.optimizer_type,
                "parameters": self.parameters,
                "components": self.components,
            }
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Export this config as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)

    def hash(self) -> str:
        """Return the stable hash for this config signature."""
        return config_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeHookSignature:
    """Stable metadata for a non-core runtime hook."""

    hook_type: HookType
    identity: str
    config: Mapping[str, Any] = field(default_factory=dict)
    reproducibility: HookReproducibility = "configured"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Export this hook signature as JSON-safe metadata."""
        return json_safe(
            {
                "hook_type": self.hook_type,
                "identity": self.identity,
                "config": self.config,
                "reproducibility": self.reproducibility,
                "notes": self.notes,
            }
        )


def config_hash(signature: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a config signature."""
    return canonical_json_hash(signature)


def stable_object_identity(obj: object) -> str:
    """Return a stable module-qualified identity for a function, class, or object."""
    target = obj if inspect.isfunction(obj) or inspect.isclass(obj) else obj.__class__
    module = getattr(target, "__module__", target.__class__.__module__)
    qualname = getattr(target, "__qualname__", target.__class__.__qualname__)
    return f"{module}.{qualname}"


def callback_hook_signature(callback: Callback) -> RuntimeHookSignature:
    """Return the runtime hook signature for a callback instance."""
    from evocore.callbacks import CheckpointCallback, EarlyStopping, MetricsLogger, ProgressBar

    identity = stable_object_identity(callback)
    if isinstance(callback, EarlyStopping):
        return RuntimeHookSignature(
            hook_type="termination",
            identity=identity,
            config={"patience": callback.patience, "min_delta": callback.min_delta},
        )
    if isinstance(callback, CheckpointCallback):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={"path": callback.path, "every": callback.every},
        )
    if isinstance(callback, MetricsLogger):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={"path": callback.path},
        )
    if isinstance(callback, ProgressBar):
        return RuntimeHookSignature(
            hook_type="artifact",
            identity=identity,
            config={},
        )
    return RuntimeHookSignature(
        hook_type="termination",
        identity=identity,
        config={},
        reproducibility="partial",
        notes=(
            f"{callback.__class__.__name__} may affect termination or side effects without a stable hook signature.",
        ),
    )


def callback_hook_signatures(callbacks: list[Callback]) -> tuple[RuntimeHookSignature, ...]:
    """Return hook signatures for callback instances."""
    return tuple(callback_hook_signature(callback) for callback in callbacks)


def reproducibility_from_hooks(
    hooks: tuple[RuntimeHookSignature, ...],
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes derived from hook signatures."""
    notes: list[str] = []
    for hook in hooks:
        if hook.reproducibility == "partial":
            notes.extend(hook.notes)
    if notes:
        return "partial", tuple(notes)
    return "full", ()


__all__ = [
    "ConfigurableComponent",
    "HookReproducibility",
    "HookType",
    "OptimizerConfig",
    "ReproducibilityStatus",
    "RuntimeHookSignature",
    "callback_hook_signature",
    "callback_hook_signatures",
    "config_hash",
    "reproducibility_from_hooks",
    "stable_object_identity",
]
```

- [ ] **Step 4: Export shared config primitives from `evocore.optimizers`**

Modify `evocore/optimizers/__init__.py` to:

```python
"""Optimization algorithm implementations."""

from evocore.optimizers.cmaes import CMAESOptimizer
from evocore.optimizers.config import (
    ConfigurableComponent,
    OptimizerConfig,
    RuntimeHookSignature,
    config_hash,
)
from evocore.optimizers.ga import GeneticAlgorithmOptimizer

__all__ = [
    "CMAESOptimizer",
    "ConfigurableComponent",
    "GeneticAlgorithmOptimizer",
    "OptimizerConfig",
    "RuntimeHookSignature",
    "config_hash",
]
```

- [ ] **Step 5: Run shared config tests to verify they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py -v
```

Expected: all tests in `tests/unit/test_optimizer_config.py` pass.

- [ ] **Step 6: Commit shared config primitives**

Run:

```powershell
git add evocore/optimizers/config.py evocore/optimizers/__init__.py tests/unit/test_optimizer_config.py
git commit -m "feat(optimizers): add config signature primitives"
```

---

### Task 2: Reproducibility Metadata Extensions

**Files:**
- Modify: `evocore/results/reproducibility.py`
- Modify: `tests/unit/test_stats.py`

- [ ] **Step 1: Update metadata tests for config hash and hook-aware status**

In `tests/unit/test_stats.py`, update imports:

```python
from evocore.optimizers.config import RuntimeHookSignature, config_hash
```

Replace `test_reproducibility_metadata_to_dict_is_json_safe` with:

```python
def test_reproducibility_metadata_to_dict_is_json_safe():
    optimizer_config = {
        "schema_version": 1,
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "parameters": {"population_size": 8},
        "components": {"callbacks": {"not", "serialized"}},
    }
    hook = RuntimeHookSignature(
        hook_type="artifact",
        identity="evocore.callbacks.MetricsLogger",
        config={"path": {"b", "a"}},
        reproducibility="configured",
    )
    metadata = ReproducibilityMetadata(
        evocore_version="0.7.0",
        optimizer_type="GeneticAlgorithmOptimizer",
        seed=42,
        direction="maximize",
        gene_space_signature={"genes": [{"name": "x", "kind": "float"}]},
        gene_space_hash="abc123",
        optimizer_config=optimizer_config,
        runtime_hooks=(hook,),
    )

    assert metadata.to_dict() == {
        "evocore_version": "0.7.0",
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "seed": 42,
        "direction": "maximize",
        "gene_space_signature": {"genes": [{"kind": "float", "name": "x"}]},
        "gene_space_hash": "abc123",
        "optimizer_config": {
            "components": {"callbacks": ["not", "serialized"]},
            "optimizer_type": "GeneticAlgorithmOptimizer",
            "parameters": {"population_size": 8},
            "schema_version": 1,
        },
        "optimizer_config_hash": config_hash(optimizer_config),
        "reproducibility_status": "full",
        "reproducibility_notes": [],
        "runtime_hooks": [
            {
                "hook_type": "artifact",
                "identity": "evocore.callbacks.MetricsLogger",
                "config": {"path": ["a", "b"]},
                "reproducibility": "configured",
                "notes": [],
            }
        ],
        "extension": {},
    }
```

Add this test below it:

```python
def test_reproducibility_metadata_accepts_explicit_partial_status():
    metadata = ReproducibilityMetadata(
        evocore_version="0.7.0",
        optimizer_type="GeneticAlgorithmOptimizer",
        seed=42,
        direction="maximize",
        gene_space_signature={"genes": []},
        gene_space_hash="abc123",
        optimizer_config={"schema_version": 1},
        optimizer_config_hash="explicit-hash",
        reproducibility_status="partial",
        reproducibility_notes=("process_initializer is opaque.",),
    )

    payload = metadata.to_dict()

    assert payload["optimizer_config_hash"] == "explicit-hash"
    assert payload["reproducibility_status"] == "partial"
    assert payload["reproducibility_notes"] == ["process_initializer is opaque."]
```

- [ ] **Step 2: Run metadata tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_stats.py::test_reproducibility_metadata_to_dict_is_json_safe tests/unit/test_stats.py::test_reproducibility_metadata_accepts_explicit_partial_status -v
```

Expected: fail because `ReproducibilityMetadata` does not expose the new fields.

- [ ] **Step 3: Extend `ReproducibilityMetadata`**

Modify `evocore/results/reproducibility.py`:

```python
"""Reproducibility metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evocore.core.serialization import canonical_json_hash, json_safe
from evocore.lifecycle.records import Direction
from evocore.optimizers.config import (
    ReproducibilityStatus,
    RuntimeHookSignature,
    config_hash,
)
from evocore.search_space import GeneSpace


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Capture deterministic optimizer and environment identity for a result."""

    evocore_version: str
    optimizer_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    optimizer_config_hash: str | None = None
    reproducibility_status: ReproducibilityStatus = "full"
    reproducibility_notes: tuple[str, ...] = ()
    runtime_hooks: tuple[RuntimeHookSignature, ...] = ()
    extension: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optimizer_config_hash is None:
            object.__setattr__(
                self,
                "optimizer_config_hash",
                config_hash(self.optimizer_config),
            )

    def to_dict(self) -> dict[str, Any]:
        """Export reproducibility metadata as JSON-safe stable fields."""
        return json_safe(
            {
                "evocore_version": self.evocore_version,
                "optimizer_type": self.optimizer_type,
                "seed": self.seed,
                "direction": self.direction,
                "gene_space_signature": self.gene_space_signature,
                "gene_space_hash": self.gene_space_hash,
                "optimizer_config": self.optimizer_config,
                "optimizer_config_hash": self.optimizer_config_hash,
                "reproducibility_status": self.reproducibility_status,
                "reproducibility_notes": self.reproducibility_notes,
                "runtime_hooks": [
                    hook.to_dict() for hook in self.runtime_hooks
                ],
                "extension": self.extension,
            }
        )


def gene_space_signature(gene_space: GeneSpace) -> dict[str, Any]:
    """Return the canonical signature for a gene space."""
    return gene_space.signature()


def gene_space_hash(signature: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a gene-space signature."""
    return canonical_json_hash(signature)


__all__ = ["ReproducibilityMetadata", "gene_space_hash", "gene_space_signature"]
```

- [ ] **Step 4: Run metadata tests to verify they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_stats.py::test_reproducibility_metadata_to_dict_is_json_safe tests/unit/test_stats.py::test_reproducibility_metadata_accepts_explicit_partial_status -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit metadata extension**

Run:

```powershell
git add evocore/results/reproducibility.py tests/unit/test_stats.py
git commit -m "feat(results): add config hash reproducibility metadata"
```

---

### Task 3: Genetic Algorithm Config Export

**Files:**
- Create: `evocore/optimizers/ga/config.py`
- Modify: `evocore/optimizers/ga/engine.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing GA config tests**

Append to `tests/unit/test_optimizer_config.py`:

```python
import pytest

from evocore import ConfigurationError, Gene, GeneSpace, GeneticAlgorithmOptimizer
from evocore.callbacks import EarlyStopping


def test_ga_default_and_explicit_default_configs_match():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    default = GeneticAlgorithmOptimizer(space)
    explicit = GeneticAlgorithmOptimizer(
        space,
        population_size=100,
        max_generations=100,
        crossover="sbx",
        crossover_prob=0.9,
        crossover_eta=2.0,
        crossover_alpha=0.5,
        mutation="gaussian",
        mutation_prob=0.1,
        mutation_individual_prob=1.0,
        mutation_sigma=0.2,
        mutation_sigma_schedule="constant",
        mutation_sigma_end=0.02,
        selection="tournament",
        tournament_size=3,
        elitism=1,
        parallel="none",
        n_workers=None,
        seed=0,
        direction="maximize",
        max_evaluations=None,
        track_diversity=False,
    )

    assert default.config_signature() == explicit.config_signature()
    assert default.config_hash() == explicit.config_hash()
    assert default.config().hash() == default.config_hash()


def test_ga_config_signature_uses_nested_component_shape():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=8,
        max_generations=5,
        seed=42,
    )

    assert engine.config_signature() == {
        "schema_version": 1,
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "parameters": {
            "direction": "maximize",
            "elitism": 1,
            "max_evaluations": None,
            "max_generations": 5,
            "n_workers": None,
            "parallel": "none",
            "population_size": 8,
            "seed": 42,
            "track_diversity": False,
        },
        "components": {
            "crossover": {
                "type": "sbx",
                "parameters": {"alpha": 0.5, "eta": 2.0, "probability": 0.9},
            },
            "mutation": {
                "type": "gaussian",
                "parameters": {
                    "individual_probability": 1.0,
                    "probability": 0.1,
                    "sigma": 0.2,
                },
            },
            "mutation_schedule": {
                "type": "constant",
                "parameters": {"sigma_end": 0.02},
            },
            "selection": {
                "type": "tournament",
                "parameters": {"tournament_size": 3},
            },
        },
    }


def test_ga_algorithm_component_change_alters_hash():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    gaussian = GeneticAlgorithmOptimizer(space, mutation="gaussian")
    uniform = GeneticAlgorithmOptimizer(space, mutation="uniform")

    assert gaussian.config_hash() != uniform.config_hash()


def test_ga_artifact_hook_path_does_not_change_config_hash(tmp_path):
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    left = GeneticAlgorithmOptimizer(space, callbacks=[MetricsLogger(str(tmp_path / "a.jsonl"))])
    right = GeneticAlgorithmOptimizer(space, callbacks=[MetricsLogger(str(tmp_path / "b.jsonl"))])

    assert left.config_hash() == right.config_hash()


def test_ga_termination_hook_is_visible_in_reproducibility():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        callbacks=[EarlyStopping(patience=4, min_delta=0.25)],
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "full"
    assert payload["runtime_hooks"] == [
        {
            "hook_type": "termination",
            "identity": "evocore.callbacks.stopping.EarlyStopping",
            "config": {"patience": 4, "min_delta": 0.25},
            "reproducibility": "configured",
            "notes": [],
        }
    ]


def test_ga_process_initializer_marks_reproducibility_partial():
    def init_worker() -> None:
        return None

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        parallel="process",
        process_initializer=init_worker,
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "partial"
    assert payload["runtime_hooks"][0]["hook_type"] == "environment"
    assert payload["runtime_hooks"][0]["reproducibility"] == "partial"
    assert "process_initializer is opaque." in payload["reproducibility_notes"]


def test_ga_validate_compatibility_is_public():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace([Gene("a", "bool"), Gene("b", "bool")]),
        crossover="one_point",
        mutation="bit_flip",
    )

    assert engine.validate_compatibility() is None

    with pytest.raises(ConfigurationError, match="binary GeneSpace"):
        GeneticAlgorithmOptimizer(
            GeneSpace([Gene("a", "bool"), Gene("b", "bool")]),
            crossover="sbx",
            mutation="bit_flip",
        )
```

- [ ] **Step 2: Run GA config tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py -v
```

Expected: fail because `GeneticAlgorithmOptimizer` does not expose public config methods.

- [ ] **Step 3: Implement GA config assembly in a focused module**

Create `evocore/optimizers/ga/config.py`:

```python
"""Genetic algorithm optimizer configuration helpers."""

from __future__ import annotations

from typing import Any

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
    stable_object_identity,
)
from evocore.search_space import OperatorCodec


def build_ga_config(optimizer: Any) -> OptimizerConfig:
    """Build the canonical GA optimizer config."""
    return OptimizerConfig(
        optimizer_type="GeneticAlgorithmOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "max_generations": optimizer.max_generations,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "elitism": optimizer.elitism,
            "max_evaluations": optimizer.max_evaluations,
            "track_diversity": optimizer.track_diversity,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
        },
        components={
            "crossover": {
                "type": optimizer.crossover,
                "parameters": {
                    "probability": optimizer.crossover_prob,
                    "eta": optimizer.crossover_eta,
                    "alpha": optimizer.crossover_alpha,
                },
            },
            "mutation": {
                "type": optimizer.mutation,
                "parameters": {
                    "probability": optimizer.mutation_prob,
                    "individual_probability": optimizer.mutation_individual_prob,
                    "sigma": optimizer.mutation_sigma,
                },
            },
            "mutation_schedule": {
                "type": optimizer.mutation_sigma_schedule,
                "parameters": {"sigma_end": optimizer.mutation_sigma_end},
            },
            "selection": {
                "type": optimizer.selection,
                "parameters": {"tournament_size": optimizer.tournament_size},
            },
        },
    )


def ga_runtime_hooks(optimizer: Any) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a GA optimizer."""
    hooks = list(callback_hook_signatures(optimizer.callbacks))
    if optimizer.process_initializer is not None:
        hooks.append(
            RuntimeHookSignature(
                hook_type="environment",
                identity=stable_object_identity(optimizer.process_initializer),
                config={"process_initargs": optimizer.process_initargs},
                reproducibility="partial",
                notes=("process_initializer is opaque.",),
            )
        )
    return tuple(hooks)


def ga_reproducibility_status(
    optimizer: Any,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a GA optimizer."""
    return reproducibility_from_hooks(ga_runtime_hooks(optimizer))


def validate_ga_compatibility(optimizer: Any) -> None:
    """Validate GA optimizer, operator, and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for GeneticAlgorithmOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    OperatorCodec(optimizer.gene_space, optimizer.crossover, optimizer.mutation)
    if optimizer.parallel not in ("none", "thread", "process"):
        raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
    if optimizer.selection not in ("tournament", "roulette", "rank"):
        raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
    if optimizer.population_size < 2:
        raise ConfigurationError("population_size must be at least 2.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if optimizer.max_evaluations is not None and optimizer.max_evaluations <= 0:
        raise ConfigurationError("max_evaluations must be positive when provided.")
    if optimizer.elitism < 0 or optimizer.elitism >= optimizer.population_size:
        raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
    if not (0.0 <= optimizer.mutation_individual_prob <= 1.0):
        raise ConfigurationError("mutation_individual_prob must be in [0, 1].")
    if optimizer.mutation_sigma_schedule not in ("constant", "linear_decay", "cosine_decay"):
        raise ConfigurationError(
            "mutation_sigma_schedule must be 'constant', 'linear_decay', or 'cosine_decay'."
        )


__all__ = [
    "build_ga_config",
    "ga_reproducibility_status",
    "ga_runtime_hooks",
    "validate_ga_compatibility",
]
```

- [ ] **Step 4: Add thin GA engine delegation**

Modify imports in `evocore/optimizers/ga/engine.py`:

```python
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.ga.config import (
    build_ga_config,
    ga_reproducibility_status,
    ga_runtime_hooks,
    validate_ga_compatibility,
)
```

Add these methods to `GeneticAlgorithmOptimizer` near `_warn_if_large_int_gene_without_sigma`:

```python
    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_ga_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer, operator, and gene-space compatibility."""
        validate_ga_compatibility(self)
```

Replace `_optimizer_config` with:

```python
    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable GA constructor configuration."""
        return self.config_signature()
```

Update `_reproducibility_metadata`:

```python
    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        status, notes = ga_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="GeneticAlgorithmOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=ga_runtime_hooks(self),
        )
```

- [ ] **Step 5: Update existing GA result metadata assertions**

In `tests/unit/test_ga_engine.py`, update `test_ga_vnext_run_attaches_history_and_reproducibility_metadata`:

```python
    assert result.reproducibility.optimizer_config_hash == engine.config_hash()
    assert result.reproducibility.reproducibility_status == "full"
    assert result.reproducibility.runtime_hooks == ()
    assert result.reproducibility.optimizer_config["parameters"]["population_size"] == 4
    assert result.reproducibility.optimizer_config["parameters"]["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config["parameters"]
```

Remove the old flat assertions:

```python
    assert result.reproducibility.optimizer_config["population_size"] == 4
    assert result.reproducibility.optimizer_config["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config
```

- [ ] **Step 6: Run GA targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py::test_ga_vnext_run_attaches_history_and_reproducibility_metadata -v
```

Expected: targeted tests pass.

- [ ] **Step 7: Commit GA config export**

Run:

```powershell
git add evocore/optimizers/ga/config.py evocore/optimizers/ga/engine.py tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py
git commit -m "feat(ga): expose optimizer config signature"
```

---

### Task 4: CMA-ES Config Export

**Files:**
- Create: `evocore/optimizers/cmaes/config.py`
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Modify: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Add failing CMA-ES config tests**

Append to `tests/unit/test_optimizer_config.py`:

```python
from evocore import CMAESOptimizer


def test_cmaes_default_and_explicit_default_configs_match():
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    default = CMAESOptimizer(space)
    explicit = CMAESOptimizer(
        space,
        population_size=50,
        initial_mean=None,
        initial_sigma=0.3,
        max_generations=300,
        parallel="none",
        n_workers=None,
        callbacks=None,
        seed=0,
        direction="maximize",
        track_diversity=False,
    )

    assert default.config_signature() == explicit.config_signature()
    assert default.config_hash() == explicit.config_hash()
    assert default.config().hash() == default.config_hash()


def test_cmaes_config_signature_uses_nested_component_shape():
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        initial_mean=[0.0, 0.1, 0.2],
        initial_sigma=0.4,
        max_generations=8,
        seed=42,
    )

    assert engine.config_signature() == {
        "schema_version": 1,
        "optimizer_type": "CMAESOptimizer",
        "parameters": {
            "direction": "maximize",
            "initial_mean": [0.0, 0.1, 0.2],
            "initial_sigma": 0.4,
            "max_generations": 8,
            "n_workers": None,
            "parallel": "none",
            "population_size": 6,
            "seed": 42,
            "track_diversity": False,
        },
        "components": {
            "distribution": {
                "type": "cma_es",
                "parameters": {"initial_sigma": 0.4},
            }
        },
    }


def test_cmaes_strategy_parameter_change_alters_hash():
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    small_sigma = CMAESOptimizer(space, initial_sigma=0.2)
    large_sigma = CMAESOptimizer(space, initial_sigma=0.4)

    assert small_sigma.config_hash() != large_sigma.config_hash()


def test_cmaes_callback_hook_is_visible_in_reproducibility(tmp_path):
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        callbacks=[MetricsLogger(str(tmp_path / "metrics.jsonl"))],
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "full"
    assert payload["runtime_hooks"] == [
        {
            "hook_type": "artifact",
            "identity": "evocore.callbacks.metrics.MetricsLogger",
            "config": {"path": str(tmp_path / "metrics.jsonl")},
            "reproducibility": "configured",
            "notes": [],
        }
    ]
```

- [ ] **Step 2: Run CMA-ES config tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py -v
```

Expected: fail because `CMAESOptimizer` does not expose public config methods.

- [ ] **Step 3: Implement CMA-ES config assembly in a focused module**

Create `evocore/optimizers/cmaes/config.py`:

```python
"""CMA-ES optimizer configuration helpers."""

from __future__ import annotations

from typing import Any

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
)


def build_cmaes_config(optimizer: Any) -> OptimizerConfig:
    """Build the canonical CMA-ES optimizer config."""
    return OptimizerConfig(
        optimizer_type="CMAESOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "initial_mean": optimizer.initial_mean,
            "initial_sigma": optimizer.initial_sigma,
            "max_generations": optimizer.max_generations,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
            "track_diversity": optimizer.track_diversity,
        },
        components={
            "distribution": {
                "type": "cma_es",
                "parameters": {"initial_sigma": optimizer.initial_sigma},
            }
        },
    )


def cmaes_runtime_hooks(optimizer: Any) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a CMA-ES optimizer."""
    return callback_hook_signatures(optimizer.callbacks)


def cmaes_reproducibility_status(
    optimizer: Any,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a CMA-ES optimizer."""
    return reproducibility_from_hooks(cmaes_runtime_hooks(optimizer))


def validate_cmaes_compatibility(optimizer: Any) -> None:
    """Validate CMA-ES optimizer and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for CMAESOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    if "bool" in optimizer.gene_space.kinds:
        raise ConfigurationError(
            "CMAESOptimizer does not support bool genes; use float/int genes only."
        )
    if optimizer.gene_space.fixed_count:
        raise ConfigurationError(
            "CMAESOptimizer does not support fixed numeric genes yet. "
            "Use GeneticAlgorithmOptimizer for full-genome fixed genes, or remove fixed genes from the CMA-ES GeneSpace."
        )
    if optimizer.parallel == "process":
        raise ConfigurationError(
            "CMAESOptimizer does not support parallel='process'.\n"
            "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable.\n"
            "  Fix: use parallel='thread' if your objective function releases the GIL, or parallel='none'.\n"
            "  Note: parallel='process' is supported by GeneticAlgorithmOptimizer, not CMAESOptimizer."
        )
    if optimizer.parallel not in ("none", "thread"):
        raise ConfigurationError("CMAESOptimizer parallel must be 'none' or 'thread'.")
    if optimizer.population_size < 2:
        raise ConfigurationError("population_size must be at least 2.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if not (optimizer.initial_sigma > 0.0):
        raise ConfigurationError("initial_sigma must be > 0.")
    if optimizer.initial_mean is not None and len(optimizer.initial_mean) != optimizer.gene_space.length:
        raise ConfigurationError("initial_mean length must match gene_space.length.")


__all__ = [
    "build_cmaes_config",
    "cmaes_reproducibility_status",
    "cmaes_runtime_hooks",
    "validate_cmaes_compatibility",
]
```

- [ ] **Step 4: Add thin CMA-ES engine delegation**

Modify imports in `evocore/optimizers/cmaes/engine.py`:

```python
from evocore.optimizers.cmaes.config import (
    build_cmaes_config,
    cmaes_reproducibility_status,
    cmaes_runtime_hooks,
    validate_cmaes_compatibility,
)
from evocore.optimizers.config import OptimizerConfig
```

Add these methods to `CMAESOptimizer` near `_bounds_list`:

```python
    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_cmaes_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer and gene-space compatibility."""
        validate_cmaes_compatibility(self)
```

Replace `_optimizer_config` with:

```python
    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable CMA constructor configuration."""
        return self.config_signature()
```

Update `_reproducibility_metadata`:

```python
    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        status, notes = cmaes_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="CMAESOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=cmaes_runtime_hooks(self),
        )
```

- [ ] **Step 5: Update existing CMA result metadata assertions**

In `tests/unit/test_cmaes_engine.py`, update `test_cma_generation_loop_result_attaches_history_and_reproducibility`:

```python
    assert result.reproducibility.optimizer_config_hash == engine.config_hash()
    assert result.reproducibility.reproducibility_status == "full"
    assert result.reproducibility.runtime_hooks == ()
    assert result.reproducibility.optimizer_config["parameters"]["population_size"] == 6
    assert result.reproducibility.optimizer_config["parameters"]["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config["parameters"]
```

Remove the old flat assertions:

```python
    assert result.reproducibility.optimizer_config["population_size"] == 6
    assert result.reproducibility.optimizer_config["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config
```

- [ ] **Step 6: Run CMA-ES targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: targeted tests pass.

- [ ] **Step 7: Commit CMA-ES config export**

Run:

```powershell
git add evocore/optimizers/cmaes/config.py evocore/optimizers/cmaes/engine.py tests/unit/test_optimizer_config.py tests/unit/test_cmaes_engine.py
git commit -m "feat(cmaes): expose optimizer config signature"
```

---

### Task 5: Public Imports And Result Integration Sweep

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_domain_imports.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Add import ownership tests**

In `tests/unit/test_domain_imports.py`, add modules to `test_new_domain_imports_are_available`:

```python
        "evocore.optimizers.config",
        "evocore.optimizers.ga.config",
        "evocore.optimizers.cmaes.config",
```

In `test_new_domain_symbols_are_importable`, add:

```python
    from evocore.optimizers import ConfigurableComponent, OptimizerConfig, RuntimeHookSignature

    assert ConfigurableComponent is not None
    assert OptimizerConfig is not None
    assert RuntimeHookSignature is not None
```

In `test_domain_packages_export_symbols_owned_by_focused_modules`, add:

```python
    from evocore.optimizers import OptimizerConfig, RuntimeHookSignature, config_hash
    from evocore.optimizers.config import ConfigurableComponent

    assert OptimizerConfig.__module__ == "evocore.optimizers.config"
    assert RuntimeHookSignature.__module__ == "evocore.optimizers.config"
    assert ConfigurableComponent.__module__ == "evocore.optimizers.config"
    assert config_hash.__module__ == "evocore.optimizers.config"
```

- [ ] **Step 2: Add top-level convenience import tests**

In `tests/unit/test_domain_imports.py`, extend `test_new_domain_symbols_are_importable`:

```python
    from evocore import OptimizerConfig as TopLevelOptimizerConfig
    from evocore import RuntimeHookSignature as TopLevelRuntimeHookSignature

    assert TopLevelOptimizerConfig is OptimizerConfig
    assert TopLevelRuntimeHookSignature is RuntimeHookSignature
```

- [ ] **Step 3: Run import tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py -v
```

Expected: fail until top-level exports are added.

- [ ] **Step 4: Export public config primitives from package surfaces**

Modify `evocore/__init__.py` imports:

```python
from evocore.optimizers import (
    ConfigurableComponent,
    OptimizerConfig,
    RuntimeHookSignature,
    config_hash,
)
```

Add these names to `__all__` in `evocore/__init__.py`:

```python
    "ConfigurableComponent",
    "OptimizerConfig",
    "RuntimeHookSignature",
    "config_hash",
```

Leave `evocore/optimizers/ga/__init__.py` and `evocore/optimizers/cmaes/__init__.py`
focused on optimizer-class exports. The helper modules remain importable through
`evocore.optimizers.ga.config` and `evocore.optimizers.cmaes.config`.


- [ ] **Step 5: Strengthen result integration assertions**

In `tests/unit/test_ga_engine.py`, add to `test_ga_vnext_run_attaches_history_and_reproducibility_metadata`:

```python
    payload = result.to_dict()
    assert payload["reproducibility"]["optimizer_config_hash"] == engine.config_hash()
    assert payload["reproducibility"]["reproducibility_status"] == "full"
```

In `tests/unit/test_cmaes_engine.py`, add to `test_cma_generation_loop_result_attaches_history_and_reproducibility`:

```python
    payload = result.to_dict()
    assert payload["reproducibility"]["optimizer_config_hash"] == engine.config_hash()
    assert payload["reproducibility"]["reproducibility_status"] == "full"
```

- [ ] **Step 6: Run import and result integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py tests/unit/test_ga_engine.py::test_ga_vnext_run_attaches_history_and_reproducibility_metadata tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit public import and result integration sweep**

Run:

```powershell
git add evocore/__init__.py tests/unit/test_domain_imports.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py
git commit -m "feat(api): export optimizer config helpers"
```

---

### Task 6: Optimizer Config Property Tests

**Files:**
- Create: `tests/property/test_optimizer_config_properties.py`

- [ ] **Step 1: Add property tests for stable config signatures**

Create `tests/property/test_optimizer_config_properties.py`:

```python
import json

from hypothesis import given, strategies as st

from evocore import CMAESOptimizer, GeneSpace, GeneticAlgorithmOptimizer


@given(
    population_size=st.integers(min_value=2, max_value=20),
    max_generations=st.integers(min_value=0, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_ga_config_signature_round_trips_through_json(
    population_size: int,
    max_generations: int,
    seed: int,
) -> None:
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
    )

    payload = engine.config_signature()

    assert json.loads(engine.config().to_json()) == payload
    assert engine.config_hash() == engine.config_hash()


@given(
    population_size=st.integers(min_value=2, max_value=20),
    max_generations=st.integers(min_value=0, max_value=20),
    initial_sigma=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_cmaes_config_signature_round_trips_through_json(
    population_size: int,
    max_generations: int,
    initial_sigma: float,
) -> None:
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=population_size,
        max_generations=max_generations,
        initial_sigma=initial_sigma,
    )

    payload = engine.config_signature()

    assert json.loads(engine.config().to_json()) == payload
    assert engine.config_hash() == engine.config_hash()


@given(seed=st.integers(min_value=0, max_value=2**32 - 2))
def test_ga_seed_change_alters_hash(seed: int) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    left = GeneticAlgorithmOptimizer(space, seed=seed)
    right = GeneticAlgorithmOptimizer(space, seed=seed + 1)

    assert left.config_hash() != right.config_hash()


@given(initial_sigma=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False))
def test_cmaes_initial_sigma_change_alters_hash(initial_sigma: float) -> None:
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    left = CMAESOptimizer(space, initial_sigma=initial_sigma)
    right = CMAESOptimizer(space, initial_sigma=initial_sigma + 0.25)

    assert left.config_hash() != right.config_hash()
```

- [ ] **Step 2: Run property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/test_optimizer_config_properties.py -v
```

Expected: all property tests pass.

- [ ] **Step 3: Commit property tests**

Run:

```powershell
git add tests/property/test_optimizer_config_properties.py
git commit -m "test(optimizers): cover config signature properties"
```

---

### Task 7: Documentation And Changelog

**Files:**
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `docs/site/gene-space.md`
- Modify: `docs/site/api.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add GA config docs**

In `docs/site/ga.md`, add a section near the optimizer configuration example:

```markdown
## Configuration Identity

`GeneticAlgorithmOptimizer` exposes a stable configuration signature for comparing
reproducible optimizer setup:

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = GeneticAlgorithmOptimizer(space, population_size=64, seed=42)

signature = optimizer.config_signature()
config_hash = optimizer.config_hash()
optimizer.validate_compatibility()
```

The config hash covers algorithm-defining fields such as seed, direction, population
size, crossover, mutation, mutation schedule, selection, elitism, and budget caps.
Runtime hooks such as progress bars, metrics paths, checkpoint paths, and process
initializers are recorded in reproducibility metadata rather than mixed into the core
config hash.
```

- [ ] **Step 2: Add CMA-ES config docs**

In `docs/site/cmaes.md`, add:

```markdown
## Configuration Identity

`CMAESOptimizer` exposes the same config export surface as GA:

```python
from evocore import CMAESOptimizer, GeneSpace

space = GeneSpace.uniform(-2.0, 2.0, 4)
optimizer = CMAESOptimizer(space, population_size=24, initial_sigma=0.25, seed=42)

signature = optimizer.config_signature()
config_hash = optimizer.config_hash()
optimizer.validate_compatibility()
```

The CMA-ES config hash covers public strategy inputs such as population size, initial
mean, initial sigma, maximum generations, seed, direction, and supported parallel mode.
Gene-space identity remains separate through `space.signature()` and `space.hash()`.
```

- [ ] **Step 3: Add reproducibility metadata docs**

In `docs/site/optimizer-telemetry.md`, add:

```markdown
## Optimizer Config Reproducibility

Run results include hook-aware optimizer configuration metadata:

```python
result = optimizer.run(evaluator)
metadata = result.reproducibility

metadata.optimizer_config
metadata.optimizer_config_hash
metadata.reproducibility_status
metadata.reproducibility_notes
metadata.runtime_hooks
```

`optimizer_config_hash` hashes only the canonical optimizer configuration. Runtime hooks
are listed separately. Known artifact hooks such as metrics loggers and progress bars are
recorded as configured hooks. Opaque environment hooks such as process initializers mark
the metadata as partially reproducible because EvoCore cannot prove their behavior from
configuration alone.
```

- [ ] **Step 4: Add gene-space separation docs**

In `docs/site/gene-space.md`, add:

```markdown
## Gene-Space Hash Versus Optimizer Config Hash

`GeneSpace.hash()` identifies the search-space structure: gene order, names, kinds,
bounds, sigma values, fixed-gene metadata, and naming mode. Optimizers expose a separate
`config_hash()` for algorithm configuration.

Use both hashes when comparing runs:

```python
same_space = left_result.reproducibility.gene_space_hash == right_result.reproducibility.gene_space_hash
same_optimizer = (
    left_result.reproducibility.optimizer_config_hash
    == right_result.reproducibility.optimizer_config_hash
)
```
```

- [ ] **Step 5: Add API docs entries**

In `docs/site/api.md`, add optimizer config symbols near optimizer API entries:

```markdown
::: evocore.optimizers.OptimizerConfig

::: evocore.optimizers.RuntimeHookSignature

::: evocore.optimizers.ConfigurableComponent

::: evocore.optimizers.config_hash
```

- [ ] **Step 6: Add changelog entry**

In `CHANGELOG.md`, under `[Unreleased]` `### Added`, add:

```markdown
- Public optimizer configuration signatures and hashes for `GeneticAlgorithmOptimizer`
  and `CMAESOptimizer`, with hook-aware reproducibility metadata.
```

Under `[Unreleased]` `### Changed`, add:

```markdown
- Run reproducibility metadata now separates optimizer config hash, gene-space hash,
  reproducibility status, notes, and runtime hook signatures.
```

- [ ] **Step 7: Run docs-related checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
git diff --check
```

Expected: all checks pass.

- [ ] **Step 8: Commit docs and changelog**

Run:

```powershell
git add docs/site/ga.md docs/site/cmaes.md docs/site/optimizer-telemetry.md docs/site/gene-space.md docs/site/api.md CHANGELOG.md
git commit -m "docs: document optimizer config contract"
```

---

### Task 8: Final Verification And File-Size Guard

**Files:**
- Inspect: all files changed by this plan

- [ ] **Step 1: Run targeted unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py tests/unit/test_stats.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py tests/unit/test_domain_imports.py -v
```

Expected: all selected unit tests pass.

- [ ] **Step 2: Run property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
```

Expected: all property tests pass.

- [ ] **Step 3: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
git diff --check
```

Expected: all checks pass.

- [ ] **Step 4: Check focused module sizes**

Run:

```powershell
(Get-Content evocore/optimizers/config.py).Length
(Get-Content evocore/optimizers/ga/config.py).Length
(Get-Content evocore/optimizers/cmaes/config.py).Length
(Get-Content evocore/optimizers/ga/engine.py).Length
(Get-Content evocore/optimizers/cmaes/engine.py).Length
```

Expected: new config modules are comfortably below 400 lines each, and engine files grow only by thin delegation and metadata wiring. If a file is drifting toward a large multi-responsibility shape, split the responsibility before committing the final verification.

- [ ] **Step 5: Rebuild extension and run public Python suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: extension rebuild succeeds and unit/integration tests pass.

- [ ] **Step 6: Confirm no uncommitted verification drift remains**

Run:

```powershell
git status --short
```

Expected: no unstaged or staged changes remain after the task commits above. If this
shows changed files, return to the task that owns those files, make the smallest fix, run
that task's verification command, and use that task's commit command.

---

## Self-Review Notes

Spec coverage:

- Public config methods are covered in Tasks 1, 3, and 4.
- Stable defaults and canonical nested signatures are covered in Tasks 3 and 4.
- Config hashing is covered in Tasks 1, 3, 4, and 6.
- Hook classification and partial reproducibility are covered in Tasks 1, 3, 4, and 5.
- Reproducibility metadata fields are covered in Tasks 2 and 5.
- Explicit compatibility validation is covered in Tasks 3 and 4.
- Focused module structure and anti-bloat guard are covered in the file structure section and Task 8.
- Docs and changelog are covered in Task 7.

Implementation guardrails:

- Keep config assembly out of engine files.
- Keep runtime hooks outside `optimizer_config_hash`.
- Keep gene-space signature and hash separate from optimizer config signature and hash.
- Preserve existing constructor validation behavior while adding public validation methods.
- Do not add config loaders, plugin registries, custom operator behavior, or replay semantics in this implementation.
