# Operator Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public, typed GA operator contract for crossover, mutation, selection, bounds enforcement, compatibility validation, sigma semantics, and custom operator extension.

**Architecture:** Add focused public operator value objects in `evocore/optimizers/operators.py`, then make GA normalize legacy strings and typed specs into canonical operator specs before config export, validation, and reproduction. Keep Rust-backed built-ins on the fast path; route only custom operator combinations through a Python orchestration path that applies the same public bounds and validation contract.

**Tech Stack:** Python dataclasses and Protocols, existing `GeneSpace` validation and optimizer config helpers, PyO3 Rust built-ins through `evocore._core`, pytest, Hypothesis, MkDocs markdown docs, repository-local `.venv` commands.

---

## File Structure

Create:

- `evocore/optimizers/operators.py`
  Public operator specs, bounds policy, custom operator protocols, context dataclasses, normalization helpers, compatibility checks, conflict checks, signatures, and decoded-value bounds enforcement.

- `tests/unit/test_operator_contract.py`
  Unit tests for public operator factories, signatures, normalization, constructor conflict handling, compatibility matrix, bounds behavior, sigma semantics, and custom operator validation.

- `tests/property/test_operator_contract_properties.py`
  Property tests for JSON-safe signatures and bounds-policy output validity.

- `docs/site/operator-contract.md`
  User-facing operator contract documentation with compatibility tables, typed examples, sigma semantics, bounds policy, and custom operator example.

Modify:

- `evocore/optimizers/ga/engine.py`
  Accept typed operator specs and `bounds_policy`, normalize once in `__init__`, preserve legacy string attributes, store normalized specs, include custom operator runtime hooks in reproducibility metadata, and call compatibility validation after normalization.

- `evocore/optimizers/ga/config.py`
  Build GA config from normalized operator signatures, include `bounds_policy`, validate via the public operator contract, and add custom operator reproducibility hooks.

- `evocore/optimizers/ga/reproduction.py`
  Use normalized built-in specs for Rust reproduction, apply explicit sigma semantics, and add a Python reproduction path when any operator is custom.

- `evocore/search_space/codec.py`
  Keep encoding, decoding, and sigma helpers; delegate name and compatibility validation to `evocore.optimizers.operators`.

- `evocore/optimizers/__init__.py`
  Re-export public operator contract names without forcing heavy optimizer imports.

- `evocore/optimizers/ga/__init__.py`
  Re-export operator contract names for GA-local discoverability.

- `evocore/__init__.py`
  Re-export `BoundsPolicy`, `CrossoverOperator`, `MutationOperator`, `SelectionOperator`, and context/protocol names that are part of the public API.

- `tests/unit/test_optimizer_config.py`
  Update GA config signature expectations for `operator_type`, `domain`, and `bounds_policy`; add typed-spec config hash tests.

- `tests/unit/test_operators.py`
  Update `OperatorCodec` compatibility expectations to prove it delegates to the operator contract while preserving the existing import path.

- `tests/unit/test_ga_engine.py`
  Add typed constructor behavior, legacy string stability, and custom reproduction smoke tests.

- `tests/property/test_operator_properties.py`
  Keep existing Rust operator properties; add any necessary assertions that built-in GA reproduction remains deterministic through normalized specs.

- `docs/site/ga.md`
  Link the operator contract and show one typed operator example.

- `docs/site/gene-space.md`
  Link per-gene `sigma` semantics to the operator contract.

- `docs/site/api.md`
  Include the public operator classes and protocols.

- `mkdocs.yml`
  Add the Operator Contract page to navigation near Gene Spaces and Genetic Algorithms.

- `CHANGELOG.md`
  Add an Unreleased entry for the public operator contract.

Do not move algorithm lifecycle, evaluation, budget, checkpoint, or result-history behavior into the operator module. Keep `evocore/optimizers/operators.py` focused on component contracts and pure helpers.

---

### Task 0: Confirm Branch And Baseline

**Files:**
- Read-only: git worktree metadata

- [ ] **Step 1: Confirm branch and worktree**

Run:

```powershell
git status --short --branch
```

Expected: on `feature/general-optimizer-framework` or another task branch, not `main`. If unrelated uncommitted files are present, leave them untouched and stage only files listed in this plan.

- [ ] **Step 2: Confirm local Python environment**

Run:

```powershell
.\.venv\Scripts\python.exe --version
```

Expected: prints the repository-local Python version. If `.venv\Scripts\python.exe` is missing or broken, stop and report it before using another interpreter.

---

### Task 1: Public Operator Specs And Normalization

**Files:**
- Create: `evocore/optimizers/operators.py`
- Create: `tests/unit/test_operator_contract.py`

- [ ] **Step 1: Write failing tests for public factory signatures and string normalization**

Create `tests/unit/test_operator_contract.py` with this content:

```python
import pytest

from evocore import ConfigurationError, Gene, GeneSpace
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    normalize_bounds_policy,
    normalize_crossover_operator,
    normalize_mutation_operator,
    normalize_selection_operator,
    resolve_operator_domain,
)


def test_builtin_operator_factories_have_canonical_signatures():
    assert CrossoverOperator.sbx(eta=3.0, probability=0.8).signature() == {
        "type": "sbx",
        "operator_type": "crossover",
        "domain": "numeric",
        "parameters": {"eta": 3.0, "probability": 0.8},
    }
    assert MutationOperator.gaussian(
        probability=0.25,
        individual_probability=0.75,
        sigma=0.15,
    ).signature() == {
        "type": "gaussian",
        "operator_type": "mutation",
        "domain": "numeric",
        "parameters": {
            "individual_probability": 0.75,
            "probability": 0.25,
            "sigma": 0.15,
        },
    }
    assert SelectionOperator.tournament(size=5).signature() == {
        "type": "tournament",
        "operator_type": "selection",
        "domain": "score",
        "parameters": {"tournament_size": 5},
    }
    assert BoundsPolicy.clamp().signature() == {
        "type": "clamp",
        "operator_type": "bounds",
        "domain": "repair",
        "parameters": {},
    }


def test_legacy_strings_normalize_to_builtin_operator_specs():
    assert normalize_crossover_operator(
        "sbx",
        probability=0.9,
        eta=2.0,
        alpha=0.5,
    ) == CrossoverOperator.sbx(eta=2.0, probability=0.9)
    assert normalize_mutation_operator(
        "gaussian",
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    ) == MutationOperator.gaussian(
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    )
    assert normalize_selection_operator("tournament", tournament_size=3) == (
        SelectionOperator.tournament(size=3)
    )
    assert normalize_bounds_policy(None) == BoundsPolicy.clamp()


def test_uniform_crossover_resolves_domain_from_gene_space():
    numeric_space = GeneSpace([Gene("period", "int", 2, 20), Gene("x", "float", -1.0, 1.0)])
    binary_space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    assert resolve_operator_domain(CrossoverOperator.uniform(), numeric_space).signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "numeric",
        "parameters": {"probability": 0.9},
    }
    assert resolve_operator_domain(CrossoverOperator.uniform(), binary_space).signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "binary",
        "parameters": {"probability": 0.9},
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CrossoverOperator.sbx(probability=-0.1),
        lambda: CrossoverOperator.sbx(eta=0.0),
        lambda: CrossoverOperator.blx(alpha=-0.1),
        lambda: MutationOperator.gaussian(probability=1.2),
        lambda: MutationOperator.gaussian(individual_probability=-0.1),
        lambda: MutationOperator.gaussian(sigma=1.5),
        lambda: SelectionOperator.tournament(size=0),
    ],
)
def test_operator_factories_validate_parameters(factory):
    with pytest.raises(ConfigurationError):
        factory()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py -v
```

Expected: FAIL because `evocore.optimizers.operators` does not exist yet.

- [ ] **Step 3: Implement public operator specs and normalization helpers**

Create `evocore/optimizers/operators.py` with this implementation skeleton and keep the public names exactly as shown:

```python
"""Public operator contracts for EvoCore optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.optimizers.config import RuntimeHookSignature, stable_object_identity
from evocore.search_space.genes import GeneKind, GeneSpace
from evocore.search_space.solutions import GeneValue

OperatorType = Literal["crossover", "mutation", "selection", "bounds"]
OperatorDomain = Literal["numeric", "binary", "score", "repair", "auto", "custom"]

NUMERIC_GENE_KINDS: frozenset[GeneKind] = frozenset({"float", "int"})
BINARY_GENE_KINDS: frozenset[GeneKind] = frozenset({"bool"})
ALL_FLAT_GENE_KINDS: frozenset[GeneKind] = frozenset({"float", "int", "bool"})

DEFAULT_CROSSOVER_PROBABILITY = 0.9
DEFAULT_CROSSOVER_ETA = 2.0
DEFAULT_CROSSOVER_ALPHA = 0.5
DEFAULT_MUTATION_PROBABILITY = 0.1
DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY = 1.0
DEFAULT_MUTATION_SIGMA = 0.2
DEFAULT_TOURNAMENT_SIZE = 3


def _immutable_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(json_safe(dict(parameters))))


def _validate_probability(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1].")
    return numeric


def _validate_positive(value: float, name: str) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ConfigurationError(f"{name} must be > 0.")
    return numeric


def _validate_non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ConfigurationError(f"{name} must be >= 0.")
    return numeric


def _validate_sigma_fraction(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1].")
    return numeric


@dataclass(frozen=True)
class CrossoverOperator:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    supported_gene_kinds: frozenset[GeneKind] = field(default_factory=frozenset)
    domain: OperatorDomain = "numeric"
    operator_type: Literal["crossover"] = "crossover"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))
        object.__setattr__(self, "supported_gene_kinds", frozenset(self.supported_gene_kinds))

    @classmethod
    def sbx(
        cls,
        *,
        eta: float = DEFAULT_CROSSOVER_ETA,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        return cls(
            "sbx",
            {"eta": _validate_positive(eta, "crossover_eta"), "probability": _validate_probability(probability, "crossover_prob")},
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def blx(
        cls,
        *,
        alpha: float = DEFAULT_CROSSOVER_ALPHA,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        return cls(
            "blx",
            {"alpha": _validate_non_negative(alpha, "crossover_alpha"), "probability": _validate_probability(probability, "crossover_prob")},
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def uniform(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        return cls(
            "uniform",
            {"probability": _validate_probability(probability, "crossover_prob")},
            ALL_FLAT_GENE_KINDS,
            "auto",
        )

    @classmethod
    def one_point(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        return cls(
            "one_point",
            {"probability": _validate_probability(probability, "crossover_prob")},
            BINARY_GENE_KINDS,
            "binary",
        )

    @classmethod
    def two_point(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        return cls(
            "two_point",
            {"probability": _validate_probability(probability, "crossover_prob")},
            BINARY_GENE_KINDS,
            "binary",
        )

    def signature(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MutationOperator:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    supported_gene_kinds: frozenset[GeneKind] = field(default_factory=frozenset)
    domain: OperatorDomain = "numeric"
    operator_type: Literal["mutation"] = "mutation"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))
        object.__setattr__(self, "supported_gene_kinds", frozenset(self.supported_gene_kinds))

    @classmethod
    def gaussian(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
        sigma: float = DEFAULT_MUTATION_SIGMA,
    ) -> MutationOperator:
        return cls(
            "gaussian",
            {
                "individual_probability": _validate_probability(individual_probability, "mutation_individual_prob"),
                "probability": _validate_probability(probability, "mutation_prob"),
                "sigma": _validate_sigma_fraction(sigma, "mutation_sigma"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def uniform(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
    ) -> MutationOperator:
        return cls(
            "uniform",
            {
                "individual_probability": _validate_probability(individual_probability, "mutation_individual_prob"),
                "probability": _validate_probability(probability, "mutation_prob"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def bit_flip(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
    ) -> MutationOperator:
        return cls(
            "bit_flip",
            {
                "individual_probability": _validate_probability(individual_probability, "mutation_individual_prob"),
                "probability": _validate_probability(probability, "mutation_prob"),
            },
            BINARY_GENE_KINDS,
            "binary",
        )

    def signature(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class SelectionOperator:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    domain: OperatorDomain = "score"
    operator_type: Literal["selection"] = "selection"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))

    @classmethod
    def tournament(cls, *, size: int = DEFAULT_TOURNAMENT_SIZE) -> SelectionOperator:
        tournament_size = int(size)
        if tournament_size <= 0:
            raise ConfigurationError("tournament_size must be >= 1.")
        return cls("tournament", {"tournament_size": tournament_size})

    @classmethod
    def roulette(cls) -> SelectionOperator:
        return cls("roulette", {})

    @classmethod
    def rank(cls) -> SelectionOperator:
        return cls("rank", {})

    def signature(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class BoundsPolicy:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    domain: OperatorDomain = "repair"
    operator_type: Literal["bounds"] = "bounds"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))

    @classmethod
    def clamp(cls) -> BoundsPolicy:
        return cls("clamp", {})

    def signature(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }
```

Then append these normalization helpers to the same file:

```python
def normalize_crossover_operator(
    value: str | CrossoverOperator,
    *,
    probability: float,
    eta: float,
    alpha: float,
) -> CrossoverOperator:
    if isinstance(value, CrossoverOperator):
        return value
    if value == "sbx":
        return CrossoverOperator.sbx(eta=eta, probability=probability)
    if value == "blx":
        return CrossoverOperator.blx(alpha=alpha, probability=probability)
    if value in ("uniform", "uniform_xo"):
        return CrossoverOperator.uniform(probability=probability)
    if value == "one_point":
        return CrossoverOperator.one_point(probability=probability)
    if value == "two_point":
        return CrossoverOperator.two_point(probability=probability)
    raise ConfigurationError(
        "Unknown crossover operator: "
        f"{value!r}. Valid: 'sbx', 'blx', 'uniform', 'one_point', 'two_point'."
    )


def normalize_mutation_operator(
    value: str | MutationOperator,
    *,
    probability: float,
    individual_probability: float,
    sigma: float,
) -> MutationOperator:
    if isinstance(value, MutationOperator):
        return value
    if value == "gaussian":
        return MutationOperator.gaussian(
            probability=probability,
            individual_probability=individual_probability,
            sigma=sigma,
        )
    if value == "uniform":
        return MutationOperator.uniform(
            probability=probability,
            individual_probability=individual_probability,
        )
    if value == "bit_flip":
        return MutationOperator.bit_flip(
            probability=probability,
            individual_probability=individual_probability,
        )
    raise ConfigurationError(
        f"Unknown mutation operator: {value!r}. Valid: 'gaussian', 'uniform', 'bit_flip'."
    )


def normalize_selection_operator(value: str | SelectionOperator, *, tournament_size: int) -> SelectionOperator:
    if isinstance(value, SelectionOperator):
        return value
    if value == "tournament":
        return SelectionOperator.tournament(size=tournament_size)
    if value == "roulette":
        return SelectionOperator.roulette()
    if value == "rank":
        return SelectionOperator.rank()
    raise ConfigurationError(
        f"Unknown selection operator: {value!r}. Valid: 'tournament', 'roulette', 'rank'."
    )


def normalize_bounds_policy(value: str | BoundsPolicy | None) -> BoundsPolicy:
    if value is None:
        return BoundsPolicy.clamp()
    if isinstance(value, BoundsPolicy):
        return value
    if value == "clamp":
        return BoundsPolicy.clamp()
    raise ConfigurationError(f"Unknown bounds policy: {value!r}. Valid: 'clamp'.")


def gene_space_domain(gene_space: GeneSpace) -> Literal["numeric", "binary"]:
    kinds = set(gene_space.kinds)
    if "bool" in kinds and len(kinds) > 1:
        raise ConfigurationError(
            "GeneSpace contains bool genes alongside float/int genes. "
            "Use a binary-only space or encode booleans as int genes with low=0, high=1."
        )
    if kinds == {"bool"}:
        return "binary"
    return "numeric"


def resolve_operator_domain(
    operator: CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy,
    gene_space: GeneSpace,
) -> CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy:
    if not isinstance(operator, CrossoverOperator) or operator.domain != "auto":
        return operator
    resolved_domain = gene_space_domain(gene_space)
    resolved_kinds = NUMERIC_GENE_KINDS if resolved_domain == "numeric" else BINARY_GENE_KINDS
    return CrossoverOperator(
        operator.name,
        dict(operator.parameters),
        resolved_kinds,
        resolved_domain,
        custom=operator.custom,
        implementation=operator.implementation,
    )
```

Add `__all__` for the public names and helper functions used by tests:

```python
__all__ = [
    "ALL_FLAT_GENE_KINDS",
    "BINARY_GENE_KINDS",
    "BoundsPolicy",
    "CrossoverOperator",
    "DEFAULT_CROSSOVER_ALPHA",
    "DEFAULT_CROSSOVER_ETA",
    "DEFAULT_CROSSOVER_PROBABILITY",
    "DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY",
    "DEFAULT_MUTATION_PROBABILITY",
    "DEFAULT_MUTATION_SIGMA",
    "DEFAULT_TOURNAMENT_SIZE",
    "MutationOperator",
    "NUMERIC_GENE_KINDS",
    "SelectionOperator",
    "gene_space_domain",
    "normalize_bounds_policy",
    "normalize_crossover_operator",
    "normalize_mutation_operator",
    "normalize_selection_operator",
    "resolve_operator_domain",
]
```

- [ ] **Step 4: Run factory tests and fix syntax issues**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_builtin_operator_factories_have_canonical_signatures tests/unit/test_operator_contract.py::test_legacy_strings_normalize_to_builtin_operator_specs tests/unit/test_operator_contract.py::test_uniform_crossover_resolves_domain_from_gene_space tests/unit/test_operator_contract.py::test_operator_factories_validate_parameters -v
```

Expected: PASS.

- [ ] **Step 5: Commit public operator specs**

Run:

```powershell
git add evocore/optimizers/operators.py tests/unit/test_operator_contract.py
git commit -m "feat(operators): add public operator specs"
```

Expected: commit succeeds.

---

### Task 2: Wire Public Exports And GA Constructor Normalization

**Files:**
- Modify: `evocore/optimizers/__init__.py`
- Modify: `evocore/optimizers/ga/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `evocore/optimizers/ga/engine.py`
- Modify: `tests/unit/test_operator_contract.py`
- Modify: `tests/unit/test_optimizer_config.py`

- [ ] **Step 1: Add failing tests for public imports, typed constructor, and conflicts**

Append to `tests/unit/test_operator_contract.py`:

```python
from evocore import BoundsPolicy, CrossoverOperator, GeneticAlgorithmOptimizer, MutationOperator
from evocore.optimizers.ga import SelectionOperator


def test_operator_contract_names_are_public_imports():
    assert CrossoverOperator.sbx().name == "sbx"
    assert MutationOperator.gaussian().name == "gaussian"
    assert SelectionOperator.tournament().name == "tournament"
    assert BoundsPolicy.clamp().name == "clamp"


def test_ga_constructor_accepts_typed_operator_specs():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    engine = GeneticAlgorithmOptimizer(
        space,
        crossover=CrossoverOperator.sbx(eta=3.0, probability=0.75),
        mutation=MutationOperator.gaussian(
            probability=0.25,
            individual_probability=0.5,
            sigma=0.15,
        ),
        selection=SelectionOperator.tournament(size=5),
        bounds_policy=BoundsPolicy.clamp(),
    )

    assert engine.crossover == "sbx"
    assert engine.crossover_prob == 0.75
    assert engine.crossover_eta == 3.0
    assert engine.mutation == "gaussian"
    assert engine.mutation_prob == 0.25
    assert engine.mutation_individual_prob == 0.5
    assert engine.mutation_sigma == 0.15
    assert engine.selection == "tournament"
    assert engine.tournament_size == 5
    assert engine.bounds_policy == BoundsPolicy.clamp()


def test_typed_crossover_rejects_conflicting_legacy_scalar():
    with pytest.raises(ConfigurationError, match="crossover_eta conflicts"):
        GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 2),
            crossover=CrossoverOperator.sbx(eta=3.0),
            crossover_eta=4.0,
        )


def test_typed_mutation_rejects_conflicting_legacy_scalar():
    with pytest.raises(ConfigurationError, match="mutation_prob conflicts"):
        GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 2),
            mutation=MutationOperator.gaussian(probability=0.25),
            mutation_prob=0.3,
        )
```

Update `tests/unit/test_optimizer_config.py::test_ga_config_signature_uses_nested_component_shape` so the GA `components` expectation is:

```python
        "components": {
            "bounds_policy": {
                "type": "clamp",
                "operator_type": "bounds",
                "domain": "repair",
                "parameters": {},
            },
            "crossover": {
                "type": "sbx",
                "operator_type": "crossover",
                "domain": "numeric",
                "parameters": {"eta": 2.0, "probability": 0.9},
            },
            "mutation": {
                "type": "gaussian",
                "operator_type": "mutation",
                "domain": "numeric",
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
                "operator_type": "selection",
                "domain": "score",
                "parameters": {"tournament_size": 3},
            },
        },
```

Add this config-hash test to `tests/unit/test_optimizer_config.py`:

```python
def test_ga_typed_operator_parameters_change_config_hash():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    left = GeneticAlgorithmOptimizer(space, crossover=CrossoverOperator.sbx(eta=2.0))
    right = GeneticAlgorithmOptimizer(space, crossover=CrossoverOperator.sbx(eta=3.0))

    assert left.config_hash() != right.config_hash()
```

If `CrossoverOperator` is not imported in `tests/unit/test_optimizer_config.py`, extend the top import:

```python
from evocore import CMAESOptimizer, ConfigurationError, CrossoverOperator, Gene, GeneSpace, GeneticAlgorithmOptimizer
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_operator_contract_names_are_public_imports tests/unit/test_operator_contract.py::test_ga_constructor_accepts_typed_operator_specs tests/unit/test_operator_contract.py::test_typed_crossover_rejects_conflicting_legacy_scalar tests/unit/test_operator_contract.py::test_typed_mutation_rejects_conflicting_legacy_scalar tests/unit/test_optimizer_config.py::test_ga_config_signature_uses_nested_component_shape tests/unit/test_optimizer_config.py::test_ga_typed_operator_parameters_change_config_hash -v
```

Expected: FAIL because exports, constructor normalization, conflict checks, and config shape are not wired yet.

- [ ] **Step 3: Re-export public operator names**

Modify `evocore/optimizers/__init__.py` to import and export:

```python
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
)
```

Add these names to `__all__`:

```python
    "BoundsPolicy",
    "CrossoverOperator",
    "MutationOperator",
    "SelectionOperator",
```

Modify `evocore/optimizers/ga/__init__.py` to import and export the same four names.

Modify `evocore/__init__.py` to import these names from `evocore.optimizers` and add them to top-level `__all__`.

- [ ] **Step 4: Add GA constructor normalization and conflict checks**

In `evocore/optimizers/ga/engine.py`, add imports:

```python
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    normalize_bounds_policy,
    normalize_crossover_operator,
    normalize_mutation_operator,
    normalize_selection_operator,
    resolve_operator_domain,
)
```

Change the constructor type hints:

```python
        crossover: str | CrossoverOperator = "sbx",
        mutation: str | MutationOperator = "gaussian",
        selection: str | SelectionOperator = "tournament",
        bounds_policy: str | BoundsPolicy | None = None,
```

Add this private helper near `_reset_vnext_state`:

```python
    @staticmethod
    def _reject_typed_operator_scalar_conflicts(
        *,
        component: str,
        provided: object,
        scalar_values: dict[str, object],
        default_values: dict[str, object],
    ) -> None:
        if isinstance(provided, str):
            return
        conflicts = [
            name
            for name, value in scalar_values.items()
            if value != default_values[name]
        ]
        if conflicts:
            joined = ", ".join(conflicts)
            raise ConfigurationError(
                f"{joined} conflicts with typed {component} operator parameters."
            )
```

In `__init__`, call conflict checks before assigning attributes:

```python
        self._reject_typed_operator_scalar_conflicts(
            component="crossover",
            provided=crossover,
            scalar_values={
                "crossover_prob": crossover_prob,
                "crossover_eta": crossover_eta,
                "crossover_alpha": crossover_alpha,
            },
            default_values={
                "crossover_prob": 0.9,
                "crossover_eta": 2.0,
                "crossover_alpha": 0.5,
            },
        )
        self._reject_typed_operator_scalar_conflicts(
            component="mutation",
            provided=mutation,
            scalar_values={
                "mutation_prob": mutation_prob,
                "mutation_individual_prob": mutation_individual_prob,
                "mutation_sigma": mutation_sigma,
            },
            default_values={
                "mutation_prob": 0.1,
                "mutation_individual_prob": 1.0,
                "mutation_sigma": 0.2,
            },
        )
        self._reject_typed_operator_scalar_conflicts(
            component="selection",
            provided=selection,
            scalar_values={"tournament_size": tournament_size},
            default_values={"tournament_size": 3},
        )
```

Then normalize:

```python
        crossover_operator = resolve_operator_domain(
            normalize_crossover_operator(
                crossover,
                probability=crossover_prob,
                eta=crossover_eta,
                alpha=crossover_alpha,
            ),
            gene_space,
        )
        mutation_operator = resolve_operator_domain(
            normalize_mutation_operator(
                mutation,
                probability=mutation_prob,
                individual_probability=mutation_individual_prob,
                sigma=mutation_sigma,
            ),
            gene_space,
        )
        selection_operator = normalize_selection_operator(
            selection,
            tournament_size=tournament_size,
        )
        normalized_bounds_policy = normalize_bounds_policy(bounds_policy)
```

Delete the old constructor check:

```python
        if selection not in ("tournament", "roulette", "rank"):
            raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
```

`normalize_selection_operator(...)` now owns valid selection names and handles both strings and typed specs.

Set normalized attributes and derive legacy attributes from them:

```python
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.selection_operator = selection_operator
        self.bounds_policy = normalized_bounds_policy

        self.crossover = crossover_operator.name
        self.crossover_prob = float(crossover_operator.parameters.get("probability", 0.9))
        self.crossover_eta = float(crossover_operator.parameters.get("eta", crossover_eta))
        self.crossover_alpha = float(crossover_operator.parameters.get("alpha", crossover_alpha))
        self.mutation = mutation_operator.name
        self.mutation_prob = float(mutation_operator.parameters.get("probability", 0.1))
        self.mutation_individual_prob = float(
            mutation_operator.parameters.get("individual_probability", 1.0)
        )
        self.mutation_sigma = float(mutation_operator.parameters.get("sigma", mutation_sigma))
        self.selection = selection_operator.name
        self.tournament_size = int(
            selection_operator.parameters.get("tournament_size", tournament_size)
        )
```

Delete the existing direct assignments for `self.crossover`, `self.crossover_prob`, `self.crossover_eta`, `self.crossover_alpha`, `self.mutation`, `self.mutation_prob`, `self.mutation_individual_prob`, `self.mutation_sigma`, `self.selection`, and `self.tournament_size` from the old constructor assignment block. Keep the assignments for `self.gene_space`, `self.population_size`, `self.max_generations`, `self.mutation_sigma_schedule`, `self.mutation_sigma_end`, `self.elitism`, `self.parallel`, `self.n_workers`, `self.process_initializer`, `self.process_initargs`, `self.seed`, `self.direction`, `self.max_evaluations`, `self.track_diversity`, and `self.callbacks`.

Replace the existing `OperatorCodec` construction with normalized legacy names:

```python
        self.operators = OperatorCodec(gene_space, self.crossover, self.mutation)
```

- [ ] **Step 5: Update GA config builder to use normalized signatures**

In `evocore/optimizers/ga/config.py`, extend `_GAOptimizerLike` with:

```python
    crossover_operator: object
    mutation_operator: object
    selection_operator: object
    bounds_policy: object
```

Update `build_ga_config(...)` component construction:

```python
        components={
            "bounds_policy": optimizer.bounds_policy.signature(),
            "crossover": optimizer.crossover_operator.signature(),
            "mutation": optimizer.mutation_operator.signature(),
            "mutation_schedule": {
                "type": optimizer.mutation_sigma_schedule,
                "parameters": {"sigma_end": optimizer.mutation_sigma_end},
            },
            "selection": optimizer.selection_operator.signature(),
        },
```

- [ ] **Step 6: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_operator_contract_names_are_public_imports tests/unit/test_operator_contract.py::test_ga_constructor_accepts_typed_operator_specs tests/unit/test_operator_contract.py::test_typed_crossover_rejects_conflicting_legacy_scalar tests/unit/test_operator_contract.py::test_typed_mutation_rejects_conflicting_legacy_scalar tests/unit/test_optimizer_config.py::test_ga_config_signature_uses_nested_component_shape tests/unit/test_optimizer_config.py::test_ga_typed_operator_parameters_change_config_hash -v
```

Expected: PASS.

- [ ] **Step 7: Commit constructor and config integration**

Run:

```powershell
git add evocore/__init__.py evocore/optimizers/__init__.py evocore/optimizers/ga/__init__.py evocore/optimizers/ga/engine.py evocore/optimizers/ga/config.py tests/unit/test_operator_contract.py tests/unit/test_optimizer_config.py
git commit -m "feat(ga): accept typed operator specs"
```

Expected: commit succeeds.

---

### Task 3: Compatibility Matrix, Bounds Policy, And Sigma Semantics

**Files:**
- Modify: `evocore/optimizers/operators.py`
- Modify: `evocore/search_space/codec.py`
- Modify: `evocore/optimizers/ga/config.py`
- Modify: `evocore/optimizers/ga/reproduction.py`
- Modify: `tests/unit/test_operator_contract.py`
- Modify: `tests/unit/test_operators.py`

- [ ] **Step 1: Add failing compatibility, bounds, and sigma tests**

Append to `tests/unit/test_operator_contract.py`:

```python
from evocore.optimizers.operators import apply_bounds_policy, validate_operator_compatibility


def test_numeric_operator_matrix_accepts_mixed_float_int_space():
    space = GeneSpace([Gene("period", "int", 2, 20), Gene("x", "float", -1.0, 1.0)])

    validate_operator_compatibility(CrossoverOperator.sbx(), space)
    validate_operator_compatibility(CrossoverOperator.blx(), space)
    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.gaussian(), space)
    validate_operator_compatibility(MutationOperator.uniform(), space)


def test_binary_operator_matrix_accepts_bool_space():
    space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    validate_operator_compatibility(CrossoverOperator.one_point(), space)
    validate_operator_compatibility(CrossoverOperator.two_point(), space)
    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.bit_flip(), space)


def test_incompatible_operator_errors_name_domain_and_actual_kinds():
    space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    with pytest.raises(ConfigurationError, match="crossover='sbx'.*numeric.*bool"):
        validate_operator_compatibility(CrossoverOperator.sbx(), space)


def test_mixed_bool_numeric_space_is_rejected():
    space = GeneSpace([Gene("x", "float", 0.0, 1.0), Gene("flag", "bool")])

    with pytest.raises(ConfigurationError, match="bool genes alongside"):
        validate_operator_compatibility(CrossoverOperator.uniform(), space)


def test_bounds_policy_clamps_rounds_thresholds_and_preserves_fixed_values():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 0.5, 0.5),
        ]
    )

    assert apply_bounds_policy([5.0, 20.8, 0.2, 99.0], space, BoundsPolicy.clamp()) == [
        1.0,
        20,
        False,
        0.5,
    ]
    assert apply_bounds_policy([-5.0, 1.2, 0.8, -99.0], space, BoundsPolicy.clamp()) == [
        -1.0,
        2,
        True,
        0.5,
    ]


def test_per_gene_sigma_override_does_not_decay_with_global_schedule():
    space = GeneSpace(
        [
            Gene("override", "float", 0.0, 10.0, sigma=0.5),
            Gene("scheduled", "float", 0.0, 10.0),
        ]
    )
    engine = GeneticAlgorithmOptimizer(
        space,
        mutation_sigma=0.4,
        mutation_sigma_schedule="linear_decay",
        mutation_sigma_end=0.1,
        max_generations=3,
    )

    assert engine.operators.sigma_abs_list(engine._compute_sigma_fraction(2)) == [5.0, 1.0]
```

Update `tests/unit/test_operators.py::test_binary_space_rejects_sbx` to keep passing through `OperatorCodec`, and add:

```python
def test_operator_codec_accepts_normalized_typed_operators():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    ops = OperatorCodec(space, CrossoverOperator.sbx(), MutationOperator.gaussian())

    assert ops.crossover == "sbx"
    assert ops.mutation == "gaussian"
```

Add `CrossoverOperator` and `MutationOperator` imports to `tests/unit/test_operators.py`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_numeric_operator_matrix_accepts_mixed_float_int_space tests/unit/test_operator_contract.py::test_binary_operator_matrix_accepts_bool_space tests/unit/test_operator_contract.py::test_incompatible_operator_errors_name_domain_and_actual_kinds tests/unit/test_operator_contract.py::test_mixed_bool_numeric_space_is_rejected tests/unit/test_operator_contract.py::test_bounds_policy_clamps_rounds_thresholds_and_preserves_fixed_values tests/unit/test_operator_contract.py::test_per_gene_sigma_override_does_not_decay_with_global_schedule tests/unit/test_operators.py::test_operator_codec_accepts_normalized_typed_operators -v
```

Expected: FAIL because compatibility and bounds helpers are not implemented and `OperatorCodec` does not accept typed specs yet.

- [ ] **Step 3: Implement compatibility and bounds helpers**

Append to `evocore/optimizers/operators.py`:

```python
def validate_operator_compatibility(
    operator: CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy,
    gene_space: GeneSpace,
) -> None:
    resolved = resolve_operator_domain(operator, gene_space)
    if isinstance(resolved, (SelectionOperator, BoundsPolicy)):
        return
    actual_kinds = frozenset(gene_space.kinds)
    if not actual_kinds.issubset(resolved.supported_gene_kinds):
        supported = ", ".join(sorted(resolved.supported_gene_kinds))
        actual = ", ".join(sorted(actual_kinds))
        raise ConfigurationError(
            f"{resolved.operator_type}='{resolved.name}' supports {resolved.domain} "
            f"GeneSpace kinds {{{supported}}}, got {{{actual}}}."
        )


def validate_operator_set(
    *,
    gene_space: GeneSpace,
    crossover: CrossoverOperator,
    mutation: MutationOperator,
    selection: SelectionOperator,
    bounds_policy: BoundsPolicy,
) -> None:
    gene_space_domain(gene_space)
    validate_operator_compatibility(crossover, gene_space)
    validate_operator_compatibility(mutation, gene_space)
    validate_operator_compatibility(selection, gene_space)
    validate_operator_compatibility(bounds_policy, gene_space)


def apply_bounds_policy(
    values: Sequence[GeneValue | float | int],
    gene_space: GeneSpace,
    bounds_policy: BoundsPolicy,
) -> list[GeneValue]:
    if bounds_policy.name != "clamp":
        raise ConfigurationError(f"Unsupported bounds policy: {bounds_policy.name!r}.")
    if len(values) != gene_space.length:
        raise ConfigurationError(
            f"Bounds policy expected {gene_space.length} genes, got {len(values)}."
        )

    bounded: list[GeneValue] = []
    for value, gene in zip(values, gene_space.genes, strict=False):
        if gene.kind == "bool":
            if type(value) is bool:
                bounded.append(value)
            elif isinstance(value, int | float):
                bounded.append(float(value) >= 0.5)
            else:
                raise ConfigurationError(
                    f"Gene {gene.name!r} expects bool-compatible value, got {type(value).__name__}."
                )
            continue

        low = float(gene.low)
        high = float(gene.high)
        if not isinstance(value, int | float) or type(value) is bool:
            raise ConfigurationError(
                f"Gene {gene.name!r} expects numeric value, got {type(value).__name__}."
            )
        clamped = min(max(float(value), low), high)
        if gene.kind == "int":
            bounded.append(int(round(clamped)))
        else:
            bounded.append(float(clamped))

    gene_space.validate_genes(bounded)
    return bounded
```

Add these names to `__all__`:

```python
    "apply_bounds_policy",
    "validate_operator_compatibility",
    "validate_operator_set",
```

- [ ] **Step 4: Delegate `OperatorCodec` validation to operator contract**

In `evocore/search_space/codec.py`, replace the hard-coded sets and `_validate()` logic with imports:

```python
from evocore.optimizers.operators import (
    CrossoverOperator,
    MutationOperator,
    normalize_crossover_operator,
    normalize_mutation_operator,
    resolve_operator_domain,
    validate_operator_set,
)
```

Change the constructor signature:

```python
    def __init__(
        self,
        gene_space: GeneSpace,
        crossover: str | CrossoverOperator,
        mutation: str | MutationOperator,
    ) -> None:
```

Normalize and store:

```python
        self.crossover_operator = resolve_operator_domain(
            normalize_crossover_operator(
                crossover,
                probability=0.9,
                eta=2.0,
                alpha=0.5,
            ),
            gene_space,
        )
        self.mutation_operator = resolve_operator_domain(
            normalize_mutation_operator(
                mutation,
                probability=0.1,
                individual_probability=1.0,
                sigma=0.2,
            ),
            gene_space,
        )
        self.crossover = self.crossover_operator.name
        self.mutation = self.mutation_operator.name
        self._validate()
```

Implement `_validate()` as:

```python
    def _validate(self) -> None:
        from evocore.optimizers.operators import BoundsPolicy, SelectionOperator

        validate_operator_set(
            gene_space=self.gene_space,
            crossover=self.crossover_operator,
            mutation=self.mutation_operator,
            selection=SelectionOperator.tournament(),
            bounds_policy=BoundsPolicy.clamp(),
        )
```

Keep `sigma_abs_list(...)` unchanged because it already implements the desired per-gene override semantics.

- [ ] **Step 5: Use operator contract validation in GA config**

In `evocore/optimizers/ga/config.py`, import:

```python
from evocore.optimizers.operators import validate_operator_set
```

Replace the `OperatorCodec(optimizer.gene_space, optimizer.crossover, optimizer.mutation)` validation call with:

```python
    validate_operator_set(
        gene_space=optimizer.gene_space,
        crossover=optimizer.crossover_operator,
        mutation=optimizer.mutation_operator,
        selection=optimizer.selection_operator,
        bounds_policy=optimizer.bounds_policy,
    )
```

Remove the duplicate selection name check because `normalize_selection_operator(...)` owns valid names. Keep population, generation, budget, elitism, parallel, and schedule checks in `validate_ga_compatibility`.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py tests/unit/test_operators.py::test_operator_codec_accepts_normalized_typed_operators tests/unit/test_operators.py::test_binary_space_rejects_sbx -v
```

Expected: PASS.

- [ ] **Step 7: Commit compatibility and bounds contract**

Run:

```powershell
git add evocore/optimizers/operators.py evocore/search_space/codec.py evocore/optimizers/ga/config.py tests/unit/test_operator_contract.py tests/unit/test_operators.py
git commit -m "feat(operators): validate compatibility contract"
```

Expected: commit succeeds.

---

### Task 4: Built-In Reproduction Uses Normalized Specs

**Files:**
- Modify: `evocore/optimizers/ga/reproduction.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_optimizer_config.py`

- [ ] **Step 1: Add failing tests for typed built-in run parity and legacy stability**

Append to `tests/unit/test_ga_engine.py`:

```python
def test_ga_run_with_typed_builtin_operators_matches_legacy_strings():
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    legacy = GeneticAlgorithmOptimizer(
        space,
        population_size=8,
        max_generations=2,
        seed=42,
        crossover="sbx",
        mutation="gaussian",
        selection="tournament",
    )
    typed = GeneticAlgorithmOptimizer(
        space,
        population_size=8,
        max_generations=2,
        seed=42,
        crossover=CrossoverOperator.sbx(),
        mutation=MutationOperator.gaussian(),
        selection=SelectionOperator.tournament(),
    )

    legacy_initial = [solution.values for solution in legacy._initial_population()]
    typed_initial = [solution.values for solution in typed._initial_population()]

    assert typed_initial == legacy_initial

    fitnesses = [-sum(float(value) ** 2 for value in solution.values) for solution in legacy._initial_population()]
    legacy_offspring = [
        solution.values
        for solution in legacy._make_offspring(
            legacy._initial_population(),
            fitnesses,
            gen=1,
            offspring_count=6,
        )
    ]
    typed_offspring = [
        solution.values
        for solution in typed._make_offspring(
            typed._initial_population(),
            fitnesses,
            gen=1,
            offspring_count=6,
        )
    ]

    assert typed_offspring == legacy_offspring
```

Add these imports to `tests/unit/test_ga_engine.py`:

```python
from evocore import CrossoverOperator, MutationOperator, SelectionOperator
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_engine.py::test_ga_run_with_typed_builtin_operators_matches_legacy_strings -v
```

Expected: FAIL if reproduction still reads stale scalar attributes or missing imports.

- [ ] **Step 3: Update Rust-backed reproduction to read normalized specs**

In `evocore/optimizers/ga/reproduction.py`, update `_make_offspring(...)` before the `_core.reproduce_population(...)` call:

```python
        crossover_params = self.crossover_operator.parameters
        mutation_params = self.mutation_operator.parameters
        selection_params = self.selection_operator.parameters
        sigma_list = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
```

Then call `_core.reproduce_population(...)` with normalized values:

```python
            self.crossover_operator.name,
            float(crossover_params.get("probability", self.crossover_prob)),
            float(crossover_params.get("eta", self.crossover_eta)),
            float(crossover_params.get("alpha", self.crossover_alpha)),
            self.mutation_operator.name,
            float(mutation_params.get("probability", self.mutation_prob)),
            sigma_list,
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.selection_operator.name,
            int(selection_params.get("tournament_size", self.tournament_size)),
            offspring_count,
            self.seed,
            gen,
            float(
                mutation_params.get(
                    "individual_probability",
                    self.mutation_individual_prob,
                )
            ),
```

Do not change `_core.reproduce_population(...)` signature in this task.

- [ ] **Step 4: Run focused test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_engine.py::test_ga_run_with_typed_builtin_operators_matches_legacy_strings tests/unit/test_optimizer_config.py::test_ga_default_and_explicit_default_configs_match -v
```

Expected: PASS.

- [ ] **Step 5: Commit normalized built-in reproduction**

Run:

```powershell
git add evocore/optimizers/ga/reproduction.py tests/unit/test_ga_engine.py tests/unit/test_optimizer_config.py
git commit -m "refactor(ga): use normalized operator specs"
```

Expected: commit succeeds.

---

### Task 5: Custom Operator Protocols And Python Reproduction Path

**Files:**
- Modify: `evocore/optimizers/operators.py`
- Modify: `evocore/optimizers/ga/reproduction.py`
- Modify: `evocore/optimizers/ga/config.py`
- Modify: `tests/unit/test_operator_contract.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_optimizer_config.py`

- [ ] **Step 1: Add failing tests for custom operator signatures and metadata**

Append to `tests/unit/test_operator_contract.py`:

```python
from evocore.optimizers.operators import (
    custom_crossover_operator,
    custom_mutation_operator,
    custom_selection_operator,
)


class ShiftMutation:
    name = "shift"
    operator_type = "mutation"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": "shift", "amount": 0.25}

    def validate_compatibility(self, gene_space):
        return None

    def mutate(self, values, context):
        return [float(value) + 0.25 for value in values]


def test_custom_mutation_operator_uses_stable_signature_when_available():
    operator = custom_mutation_operator(ShiftMutation())

    assert operator.custom is True
    assert operator.signature() == {
        "type": "shift",
        "operator_type": "mutation",
        "domain": "custom",
        "parameters": {"name": "shift", "amount": 0.25},
    }


def test_custom_mutation_operator_without_signature_uses_identity_and_partial_note():
    class IdentityMutation:
        name = "identity"
        operator_type = "mutation"
        supported_gene_kinds = frozenset({"float"})

        def validate_compatibility(self, gene_space):
            return None

        def mutate(self, values, context):
            return list(values)

    operator = custom_mutation_operator(IdentityMutation())

    assert operator.signature()["type"] == "identity"
    assert operator.signature()["parameters"]["identity"].endswith("IdentityMutation")


class SwapCrossover:
    name = "swap"
    operator_type = "crossover"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": "swap"}

    def validate_compatibility(self, gene_space):
        return None

    def crossover(self, left, right, context):
        return right, left


class FirstParentSelection:
    name = "first_parent"
    operator_type = "selection"

    def config_signature(self):
        return {"name": "first_parent"}

    def validate_compatibility(self, gene_space):
        return None

    def select(self, scores, count, context):
        return [0 for _ in range(count)]


def test_custom_crossover_and_selection_operators_have_signatures():
    crossover = custom_crossover_operator(SwapCrossover())
    selection = custom_selection_operator(FirstParentSelection())

    assert crossover.signature() == {
        "type": "swap",
        "operator_type": "crossover",
        "domain": "custom",
        "parameters": {"name": "swap"},
    }
    assert selection.signature() == {
        "type": "first_parent",
        "operator_type": "selection",
        "domain": "custom",
        "parameters": {"name": "first_parent"},
    }
```

Append to `tests/unit/test_ga_engine.py`:

```python
def test_ga_custom_mutation_path_applies_bounds_and_runs():
    class OvershootMutation:
        name = "overshoot"
        operator_type = "mutation"
        supported_gene_kinds = frozenset({"float"})

        def config_signature(self):
            return {"name": "overshoot"}

        def validate_compatibility(self, gene_space):
            return None

        def mutate(self, values, context):
            return [999.0 for _ in values]

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=1,
        seed=42,
        mutation=custom_mutation_operator(OvershootMutation()),
        crossover_prob=0.0,
    )
    population = engine._initial_population()
    fitnesses = [1.0, 2.0, 3.0, 4.0]

    offspring = engine._make_offspring(population, fitnesses, gen=1, offspring_count=4)

    assert len(offspring) == 4
    assert all(all(-1.0 <= value <= 1.0 for value in solution.values) for solution in offspring)


def test_ga_custom_crossover_and_selection_path_runs():
    class CopyFirstCrossover:
        name = "copy_first"
        operator_type = "crossover"
        supported_gene_kinds = frozenset({"float"})

        def config_signature(self):
            return {"name": "copy_first"}

        def validate_compatibility(self, gene_space):
            return None

        def crossover(self, left, right, context):
            return left, left

    class FirstSelection:
        name = "first"
        operator_type = "selection"

        def config_signature(self):
            return {"name": "first"}

        def validate_compatibility(self, gene_space):
            return None

        def select(self, scores, count, context):
            return [0 for _ in range(count)]

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=1,
        seed=42,
        crossover=custom_crossover_operator(CopyFirstCrossover()),
        mutation=MutationOperator.gaussian(probability=0.0),
        selection=custom_selection_operator(FirstSelection()),
    )
    population = engine._initial_population()
    fitnesses = [1.0, 2.0, 3.0, 4.0]

    offspring = engine._make_offspring(population, fitnesses, gen=1, offspring_count=4)

    assert len(offspring) == 4
    assert all(solution.values == population[0].values for solution in offspring)
```

Add imports:

```python
from evocore.optimizers.operators import (
    custom_crossover_operator,
    custom_mutation_operator,
    custom_selection_operator,
)
```

Append to `tests/unit/test_optimizer_config.py`:

```python
from evocore.optimizers.operators import custom_mutation_operator


def test_ga_custom_operator_is_visible_in_reproducibility_metadata():
    class IdentityMutation:
        name = "identity"
        operator_type = "mutation"
        supported_gene_kinds = frozenset({"float"})

        def validate_compatibility(self, gene_space):
            return None

        def mutate(self, values, context):
            return list(values)

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        mutation=custom_mutation_operator(IdentityMutation()),
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "partial"
    assert any(
        hook["config"].get("component") == "mutation"
        and hook["reproducibility"] == "partial"
        for hook in payload["runtime_hooks"]
    )
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_custom_mutation_operator_uses_stable_signature_when_available tests/unit/test_operator_contract.py::test_custom_mutation_operator_without_signature_uses_identity_and_partial_note tests/unit/test_operator_contract.py::test_custom_crossover_and_selection_operators_have_signatures tests/unit/test_ga_engine.py::test_ga_custom_mutation_path_applies_bounds_and_runs tests/unit/test_ga_engine.py::test_ga_custom_crossover_and_selection_path_runs tests/unit/test_optimizer_config.py::test_ga_custom_operator_is_visible_in_reproducibility_metadata -v
```

Expected: FAIL because custom wrappers, contexts, and Python reproduction do not exist yet.

- [ ] **Step 3: Add custom protocols, context dataclasses, and wrapper factories**

Append to `evocore/optimizers/operators.py`:

```python
@dataclass(frozen=True)
class OperatorContext:
    gene_space: GeneSpace
    generation: int
    seed: int
    individual_index: int | None
    pair_index: int | None
    bounds_policy: BoundsPolicy


@dataclass(frozen=True)
class MutationContext(OperatorContext):
    probability: float
    mutation_sigma: float
    mutation_sigmas: tuple[float, ...]


@dataclass(frozen=True)
class CrossoverContext(OperatorContext):
    probability: float


@dataclass(frozen=True)
class SelectionContext(OperatorContext):
    tournament_size: int | None = None


@runtime_checkable
class CustomMutationProtocol(Protocol):
    name: str
    operator_type: Literal["mutation"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def mutate(self, values: Sequence[GeneValue], context: MutationContext) -> Sequence[GeneValue]: ...


def _custom_signature_payload(operator: object) -> dict[str, Any]:
    config_signature = getattr(operator, "config_signature", None)
    if callable(config_signature):
        payload = config_signature()
        if not isinstance(payload, Mapping):
            raise ConfigurationError("custom operator config_signature() must return a mapping.")
        return dict(json_safe(payload))
    return {"identity": stable_object_identity(operator)}


def custom_mutation_operator(operator: CustomMutationProtocol) -> MutationOperator:
    if getattr(operator, "operator_type", None) != "mutation":
        raise ConfigurationError("custom mutation operator must declare operator_type='mutation'.")
    if not hasattr(operator, "mutate") or not callable(operator.mutate):
        raise ConfigurationError("custom mutation operator must implement mutate(values, context).")
    name = getattr(operator, "name", operator.__class__.__name__)
    supported = frozenset(getattr(operator, "supported_gene_kinds", ALL_FLAT_GENE_KINDS))
    return MutationOperator(
        str(name),
        _custom_signature_payload(operator),
        supported,
        "custom",
        custom=True,
        implementation=operator,
    )
```

Add these public names to `__all__`:

```python
    "CrossoverContext",
    "CustomMutationProtocol",
    "MutationContext",
    "SelectionContext",
    "custom_mutation_operator",
```

Also add crossover and selection protocols plus wrapper factories:

```python
@runtime_checkable
class CustomCrossoverProtocol(Protocol):
    name: str
    operator_type: Literal["crossover"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def crossover(
        self,
        left: Sequence[GeneValue],
        right: Sequence[GeneValue],
        context: CrossoverContext,
    ) -> tuple[Sequence[GeneValue], Sequence[GeneValue]]: ...


@runtime_checkable
class CustomSelectionProtocol(Protocol):
    name: str
    operator_type: Literal["selection"]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def select(
        self,
        scores: Sequence[float],
        count: int,
        context: SelectionContext,
    ) -> Sequence[int]: ...


def custom_crossover_operator(operator: CustomCrossoverProtocol) -> CrossoverOperator:
    if getattr(operator, "operator_type", None) != "crossover":
        raise ConfigurationError("custom crossover operator must declare operator_type='crossover'.")
    if not hasattr(operator, "crossover") or not callable(operator.crossover):
        raise ConfigurationError(
            "custom crossover operator must implement crossover(left, right, context)."
        )
    name = getattr(operator, "name", operator.__class__.__name__)
    supported = frozenset(getattr(operator, "supported_gene_kinds", ALL_FLAT_GENE_KINDS))
    return CrossoverOperator(
        str(name),
        _custom_signature_payload(operator),
        supported,
        "custom",
        custom=True,
        implementation=operator,
    )


def custom_selection_operator(operator: CustomSelectionProtocol) -> SelectionOperator:
    if getattr(operator, "operator_type", None) != "selection":
        raise ConfigurationError("custom selection operator must declare operator_type='selection'.")
    if not hasattr(operator, "select") or not callable(operator.select):
        raise ConfigurationError("custom selection operator must implement select(scores, count, context).")
    name = getattr(operator, "name", operator.__class__.__name__)
    return SelectionOperator(
        str(name),
        _custom_signature_payload(operator),
        "custom",
        custom=True,
        implementation=operator,
    )
```

Add these names to `__all__`:

```python
    "CustomCrossoverProtocol",
    "CustomSelectionProtocol",
    "custom_crossover_operator",
    "custom_selection_operator",
```

- [ ] **Step 4: Route custom reproduction through Python**

In `evocore/optimizers/ga/reproduction.py`, import:

```python
import random

from evocore.optimizers.operators import (
    CrossoverContext,
    MutationContext,
    SelectionContext,
    apply_bounds_policy,
)
```

Add helper:

```python
    def _uses_custom_operator(self) -> bool:
        return any(
            getattr(operator, "custom", False)
            for operator in (
                self.crossover_operator,
                self.mutation_operator,
                self.selection_operator,
            )
        )
```

At the top of `_make_offspring(...)`, after the `offspring_count <= 0` guard, add:

```python
        if self._uses_custom_operator():
            return self._make_offspring_python(
                working_population,
                fitnesses,
                gen,
                offspring_count,
            )
```

Add Python selection helper:

```python
    def _select_parent_indices_python(
        self,
        fitnesses: Sequence[float],
        count: int,
        gen: int,
    ) -> list[int]:
        if self.selection_operator.custom:
            context = SelectionContext(
                gene_space=self.gene_space,
                generation=gen,
                seed=self.seed,
                individual_index=None,
                pair_index=None,
                bounds_policy=self.bounds_policy,
                tournament_size=self.tournament_size,
            )
            selected = list(self.selection_operator.implementation.select(fitnesses, count, context))
            if len(selected) != count or any(index < 0 or index >= len(fitnesses) for index in selected):
                raise ConfigurationError("custom selection returned invalid parent indices.")
            return [int(index) for index in selected]
        if self.selection_operator.name == "tournament":
            return _core.tournament_selection(fitnesses, count, self.tournament_size, self.seed, gen)
        if self.selection_operator.name == "roulette":
            return _core.roulette_selection(fitnesses, count, self.seed, gen)
        if self.selection_operator.name == "rank":
            return _core.rank_selection(fitnesses, count, self.seed, gen)
        raise ConfigurationError(f"unknown selection operator: {self.selection_operator.name!r}")
```

Add Python custom mutation helper:

```python
    def _mutate_child_python(
        self,
        values: Sequence[float | int | bool],
        *,
        gen: int,
        individual_index: int,
        mutation_sigmas: Sequence[float],
    ) -> list[float | int | bool]:
        if self.mutation_operator.custom:
            context = MutationContext(
                gene_space=self.gene_space,
                generation=gen,
                seed=self.seed,
                individual_index=individual_index,
                pair_index=None,
                bounds_policy=self.bounds_policy,
                probability=self.mutation_prob,
                mutation_sigma=self.mutation_sigma,
                mutation_sigmas=tuple(float(value) for value in mutation_sigmas),
            )
            mutated = list(self.mutation_operator.implementation.mutate(list(values), context))
            return apply_bounds_policy(mutated, self.gene_space, self.bounds_policy)

        rng = random.Random(int(_core.py_derive_seed(self.seed, gen, individual_index, _core.OP_MUTATION)))
        if self.mutation_individual_prob <= 0.0:
            return list(values)
        if self.mutation_individual_prob < 1.0 and rng.random() >= self.mutation_individual_prob:
            return list(values)

        mutated = list(values)
        for index, gene in enumerate(self.gene_space.genes):
            if rng.random() >= self.mutation_prob:
                continue
            if self.mutation_operator.name == "gaussian" and gene.kind in ("float", "int"):
                mutated[index] = float(mutated[index]) + rng.gauss(0.0, max(float(mutation_sigmas[index]), 1e-20))
            elif self.mutation_operator.name == "uniform" and gene.kind == "float":
                mutated[index] = rng.uniform(float(gene.low), float(gene.high))
            elif self.mutation_operator.name == "uniform" and gene.kind == "int":
                mutated[index] = rng.randint(int(gene.low), int(gene.high))
            elif self.mutation_operator.name == "bit_flip" and gene.kind == "bool":
                mutated[index] = not bool(mutated[index])
        return apply_bounds_policy(mutated, self.gene_space, self.bounds_policy)
```

Add `_make_offspring_python(...)`:

```python
    def _make_offspring_python(
        self,
        working_population: Sequence[Solution],
        fitnesses: Sequence[float],
        gen: int,
        offspring_count: int,
    ) -> list[Solution]:
        mutation_sigmas = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
        parent_count = offspring_count + (offspring_count % 2)
        parent_indices = self._select_parent_indices_python(fitnesses, parent_count, gen)
        offspring: list[Solution] = []
        child_index = 0
        for pair_index in range(parent_count // 2):
            left = list(working_population[parent_indices[pair_index * 2]].values)
            right = list(working_population[parent_indices[pair_index * 2 + 1]].values)
            if self.crossover_operator.custom:
                context = CrossoverContext(
                    gene_space=self.gene_space,
                    generation=gen,
                    seed=self.seed,
                    individual_index=None,
                    pair_index=pair_index,
                    bounds_policy=self.bounds_policy,
                    probability=self.crossover_prob,
                )
                child_left, child_right = self.crossover_operator.implementation.crossover(
                    left,
                    right,
                    context,
                )
                children = [list(child_left), list(child_right)]
            else:
                children = [left, right]

            for child_values in children:
                bounded = apply_bounds_policy(child_values, self.gene_space, self.bounds_policy)
                mutated = self._mutate_child_python(
                    bounded,
                    gen=gen,
                    individual_index=child_index,
                    mutation_sigmas=mutation_sigmas,
                )
                offspring.append(self.operators.decode_solution(self.operators.encode_values(mutated)))
                child_index += 1
                if len(offspring) >= offspring_count:
                    return offspring
        return offspring
```

This path intentionally preserves custom operator validation and bounds behavior over byte-for-byte parity with Rust RNG internals.

- [ ] **Step 5: Include custom operator hooks in reproducibility metadata**

In `evocore/optimizers/ga/config.py`, add helper:

```python
def operator_runtime_hook_signatures(optimizer: _GAOptimizerLike) -> tuple[RuntimeHookSignature, ...]:
    hooks: list[RuntimeHookSignature] = []
    for component_name, operator in (
        ("crossover", optimizer.crossover_operator),
        ("mutation", optimizer.mutation_operator),
        ("selection", optimizer.selection_operator),
    ):
        if getattr(operator, "custom", False):
            hooks.append(
                RuntimeHookSignature(
                    hook_type="environment",
                    identity=stable_object_identity(operator.implementation),
                    config={"component": component_name, "operator": operator.signature()},
                    reproducibility="partial",
                    notes=(f"custom {component_name} operator executes Python code.",),
                )
            )
    return tuple(hooks)
```

Update `ga_runtime_hooks(...)`:

```python
    hooks = list(callback_hook_signatures(optimizer.callbacks))
    hooks.extend(operator_runtime_hook_signatures(optimizer))
```

Add `operator_runtime_hook_signatures` to `__all__`.

- [ ] **Step 6: Run custom operator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py::test_custom_mutation_operator_uses_stable_signature_when_available tests/unit/test_operator_contract.py::test_custom_mutation_operator_without_signature_uses_identity_and_partial_note tests/unit/test_operator_contract.py::test_custom_crossover_and_selection_operators_have_signatures tests/unit/test_ga_engine.py::test_ga_custom_mutation_path_applies_bounds_and_runs tests/unit/test_ga_engine.py::test_ga_custom_crossover_and_selection_path_runs tests/unit/test_optimizer_config.py::test_ga_custom_operator_is_visible_in_reproducibility_metadata -v
```

Expected: PASS.

- [ ] **Step 7: Commit custom operator path**

Run:

```powershell
git add evocore/optimizers/operators.py evocore/optimizers/ga/reproduction.py evocore/optimizers/ga/config.py tests/unit/test_operator_contract.py tests/unit/test_ga_engine.py tests/unit/test_optimizer_config.py
git commit -m "feat(operators): support custom mutation protocol"
```

Expected: commit succeeds.

---

### Task 6: Property Tests And Documentation

**Files:**
- Create: `tests/property/test_operator_contract_properties.py`
- Create: `docs/site/operator-contract.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/gene-space.md`
- Modify: `docs/site/api.md`
- Modify: `mkdocs.yml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add property tests for signatures and bounds policy**

Create `tests/property/test_operator_contract_properties.py`:

```python
import json

from hypothesis import given
from hypothesis import strategies as st

from evocore import BoundsPolicy, Gene, GeneSpace
from evocore.core.serialization import stable_json_dumps
from evocore.optimizers.operators import (
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    apply_bounds_policy,
)


@given(
    eta=st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_crossover_signature_is_json_safe(eta, probability):
    payload = CrossoverOperator.sbx(eta=eta, probability=probability).signature()

    assert json.loads(stable_json_dumps(payload)) == payload


@given(
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    individual_probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_mutation_signature_is_json_safe(probability, individual_probability, sigma):
    payload = MutationOperator.gaussian(
        probability=probability,
        individual_probability=individual_probability,
        sigma=sigma,
    ).signature()

    assert json.loads(stable_json_dumps(payload)) == payload


@given(
    x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    period=st.integers(min_value=-100, max_value=100),
    flag=st.booleans(),
)
def test_bounds_policy_outputs_valid_decoded_values(x, period, flag):
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
        ]
    )

    bounded = apply_bounds_policy([x, period, flag], space, BoundsPolicy.clamp())

    space.validate_genes(bounded)


def test_selection_and_bounds_signatures_are_json_safe():
    for payload in [
        SelectionOperator.tournament(size=3).signature(),
        SelectionOperator.roulette().signature(),
        SelectionOperator.rank().signature(),
        BoundsPolicy.clamp().signature(),
    ]:
        assert json.loads(stable_json_dumps(payload)) == payload
```

- [ ] **Step 2: Run property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/test_operator_contract_properties.py -v
```

Expected: PASS.

- [ ] **Step 3: Add operator contract docs**

Create `docs/site/operator-contract.md`:

```markdown
# Operator Contract

EvoCore genetic algorithms expose a public operator contract for crossover,
mutation, selection, and bounds enforcement. The existing string API remains valid:

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = GeneticAlgorithmOptimizer(
    space,
    crossover="sbx",
    mutation="gaussian",
    selection="tournament",
)
```

Typed operator specs make the same setup explicit:

```python
from evocore import (
    BoundsPolicy,
    CrossoverOperator,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    MutationOperator,
    SelectionOperator,
)

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = GeneticAlgorithmOptimizer(
    space,
    crossover=CrossoverOperator.sbx(eta=2.0, probability=0.9),
    mutation=MutationOperator.gaussian(
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    ),
    selection=SelectionOperator.tournament(size=3),
    bounds_policy=BoundsPolicy.clamp(),
)
```

## Compatibility

| Operator | Type | Supported genes |
| --- | --- | --- |
| `sbx` | crossover | `float`, `int` |
| `blx` | crossover | `float`, `int` |
| `uniform` | crossover | `float`, `int`, or `bool` depending on the space |
| `one_point` | crossover | `bool` |
| `two_point` | crossover | `bool` |
| `gaussian` | mutation | `float`, `int` |
| `uniform` | mutation | `float`, `int` |
| `bit_flip` | mutation | `bool` |
| `tournament` | selection | any GA-supported space |
| `roulette` | selection | any GA-supported space |
| `rank` | selection | any GA-supported space |

Numeric spaces may mix `float` and `int` genes. Binary spaces contain only `bool`
genes. Mixed `bool` and numeric spaces are rejected in this contract.

## Bounds Policy

`BoundsPolicy.clamp()` is the v1 bounds policy:

- Float genes clamp to inclusive bounds.
- Int genes round, then clamp to inclusive bounds.
- Bool genes threshold to `False` or `True`.
- Fixed numeric genes remain fixed.

## Sigma Semantics

`mutation_sigma` is a global fraction of each numeric gene span. A `Gene(..., sigma=...)`
value overrides the global scheduled sigma for that gene. Per-gene sigma overrides do
not decay with `mutation_sigma_schedule`.

## Custom Operators

Custom operators are structured objects, not bare callables. They declare a name,
operator type, supported gene kinds, compatibility validation, and the execution method.

```python
from evocore.optimizers.operators import custom_mutation_operator


class ShiftMutation:
    name = "shift"
    operator_type = "mutation"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": self.name, "amount": 0.1}

    def validate_compatibility(self, gene_space):
        return None

    def mutate(self, values, context):
        return [float(value) + 0.1 for value in values]


mutation = custom_mutation_operator(ShiftMutation())
```

Custom operators execute in Python and are recorded in reproducibility metadata as
partial reproducibility runtime hooks.
```

- [ ] **Step 4: Update navigation and related docs**

Modify `mkdocs.yml` nav near Gene Spaces:

```yaml
  - Gene Spaces: gene-space.md
  - Operator Contract: operator-contract.md
  - Genetic Algorithms: ga.md
```

Add this short paragraph to `docs/site/ga.md` under Configuration Identity:

```markdown
For crossover, mutation, selection, bounds policy, and custom operator compatibility,
see [Operator Contract](operator-contract.md). Typed operator specs such as
`CrossoverOperator.sbx(...)` and `MutationOperator.gaussian(...)` normalize to the same
configuration identity as legacy strings.
```

Add this short paragraph to `docs/site/gene-space.md` after the Stable Signature section:

```markdown
Per-gene `sigma` values are consumed by GA mutation operators. They override the global
scheduled mutation sigma for that gene; see [Operator Contract](operator-contract.md)
for the full sigma semantics.
```

Add operator API entries to `docs/site/api.md`:

```markdown
::: evocore.optimizers.operators.BoundsPolicy

::: evocore.optimizers.operators.CrossoverOperator

::: evocore.optimizers.operators.MutationOperator

::: evocore.optimizers.operators.SelectionOperator
```

Add a `CHANGELOG.md` Unreleased bullet:

```markdown
- Added public GA operator contract specs for crossover, mutation, selection, bounds
  policy, compatibility validation, sigma semantics, and custom operator extension.
```

- [ ] **Step 5: Run docs and property checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/test_operator_contract_properties.py -v
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: both commands PASS.

- [ ] **Step 6: Commit docs and property tests**

Run:

```powershell
git add tests/property/test_operator_contract_properties.py docs/site/operator-contract.md docs/site/ga.md docs/site/gene-space.md docs/site/api.md mkdocs.yml CHANGELOG.md
git commit -m "docs: document operator contract"
```

Expected: commit succeeds.

---

### Task 7: Full Verification

**Files:**
- Read-only verification over the touched Python, docs, and Rust-backed operator surfaces

- [ ] **Step 1: Run Python formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both commands PASS.

- [ ] **Step 2: Run Rust formatting, linting, and tests**

Run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all commands PASS.

- [ ] **Step 3: Rebuild the Python extension**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: PASS and installs the local Rust extension into `.venv`.

- [ ] **Step 4: Run Python unit and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 5: Run property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
```

Expected: PASS.

- [ ] **Step 6: Run docs build**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: PASS.

- [ ] **Step 7: Inspect final git status**

Run:

```powershell
git status --short --branch
```

Expected: clean worktree on the task branch, ahead of origin by the new implementation commits.

If any verification command fails, stop. Do not commit, push, or open a PR. Report the failing command, the relevant error summary, and the likely files involved.
