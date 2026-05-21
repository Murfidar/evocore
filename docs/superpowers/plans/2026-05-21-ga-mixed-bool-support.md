# GA Mixed Bool Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `GeneticAlgorithmOptimizer` to accept flat mixed `float`/`int`/`bool` `GeneSpace` values with typed default operators, decoded bool values, lifecycle support, checkpoint coverage, and documentation while keeping CMA-ES bool support unsupported.

**Architecture:** Keep the change inside the existing GA operator contract and Python GA reproduction path. Add a gene-space profile helper in `evocore/optimizers/operators.py`, resolve omitted GA operator defaults in `evocore/optimizers/ga/engine.py`, and route mixed bool plus numeric spaces through the existing Python reproduction helpers in `evocore/optimizers/ga/reproduction.py` so Rust homogeneous fast paths stay stable.

**Tech Stack:** Python dataclasses and literals, existing `GeneSpace`/`OperatorCodec` helpers, PyO3 `_core` initialization and crossover primitives, pytest, Hypothesis, MkDocs markdown docs, repository-local `.venv` commands.

---

## File Structure

Modify:

- `evocore/optimizers/operators.py`
  Add the `mixed` operator domain/profile, replace mixed-space rejection with profile detection, resolve mixed-compatible built-in operators, and keep explicit incompatible operators failing with named gene kinds.

- `evocore/optimizers/ga/engine.py`
  Replace string defaults for `crossover` and `mutation` with an internal sentinel so omitted values can resolve by gene-space profile while explicit incompatible values still fail.

- `evocore/optimizers/ga/reproduction.py`
  Route mixed spaces through Python reproduction and make `gaussian`, `uniform`, and `bit_flip` mutate bool genes according to the per-gene mutation probability.

- `evocore/search_space/codec.py`
  Keep the boundary API unchanged, relying on the updated operator contract to accept compatible mixed operators and reject incompatible ones.

- `tests/unit/test_operator_contract.py`
  Replace the current mixed bool rejection test with profile, compatibility, rejection, and custom-operator coverage.

- `tests/unit/test_operators.py`
  Update `OperatorCodec` mixed bool tests so compatible mixed operators succeed and incompatible numeric-only crossover still fails.

- `tests/unit/test_optimizer_config.py`
  Add default-resolution and config-signature tests for bool-only and mixed spaces while preserving numeric default config identity.

- `tests/unit/test_ga_engine.py`
  Add focused reproduction tests for mixed Python routing and typed bool mutation behavior.

- `tests/unit/test_ga_ask_tell_vnext.py`
  Add ask/tell tests proving decoded bool genes and params survive initial ask and offspring ask.

- `tests/unit/test_ask_tell_checkpointing.py`
  Add stable ask/tell checkpoint resume coverage for mixed bool values.

- `tests/unit/test_checkpointing.py`
  Add generation-loop checkpoint export/load coverage for mixed bool population values.

- `tests/unit/test_cmaes_engine.py`
  Add mixed bool plus numeric CMA-ES rejection coverage.

- `tests/integration/test_mixed_gene_space.py`
  Add a mixed bool GA default-operator run using the public `run` API.

- `tests/property/test_operator_contract_properties.py`
  Add a property that mixed-compatible default operator signatures remain JSON-safe.

- `docs/site/gene-space.md`
  Document that GA supports bool genes mixed with numeric genes.

- `docs/site/operator-contract.md`
  Document mixed GA operator behavior and remaining incompatible operators.

- `docs/site/ga.md`
  Add a short mixed bool GA example using omitted operators.

- `docs/site/cmaes.md`
  Clarify that CMA-ES still rejects bool genes.

- `CHANGELOG.md`
  Add an Unreleased entry for additive GA mixed bool support.

Do not change Rust operator implementations, checkpoint fixture baselines, public optimizer names, `GeneSpace` structure, or CMA-ES bool behavior.

---

### Task 0: Confirm Branch And Environment

**Files:**
- Read-only: git worktree metadata
- Read-only: `.venv/Scripts/python.exe`

- [ ] **Step 1: Confirm branch and uncommitted work**

Run:

```powershell
git status --short --branch
```

Expected: on `codex/feature/ga-mixed-bool-design` or another task branch, not `main`. If unrelated uncommitted files are present, leave them untouched and stage only files changed by this plan.

- [ ] **Step 2: Confirm repository-local Python**

Run:

```powershell
.\.venv\Scripts\python.exe --version
```

Expected: prints the repository-local Python version. If `.venv\Scripts\python.exe` is missing or broken, stop and report that before using another interpreter.

---

### Task 1: Operator Profile And Compatibility Contract

**Files:**
- Modify: `evocore/optimizers/operators.py`
- Modify: `tests/unit/test_operator_contract.py`
- Modify: `tests/unit/test_operators.py`

- [ ] **Step 1: Write failing operator contract tests**

In `tests/unit/test_operator_contract.py`, add `gene_space_profile` to the import from `evocore.optimizers.operators`:

```python
from evocore.optimizers.operators import (
    apply_bounds_policy,
    custom_crossover_operator,
    custom_mutation_operator,
    custom_selection_operator,
    gene_space_profile,
    normalize_bounds_policy,
    normalize_crossover_operator,
    normalize_mutation_operator,
    normalize_selection_operator,
    resolve_operator_domain,
    validate_operator_compatibility,
)
```

Replace `test_mixed_bool_numeric_space_is_rejected` with these tests:

```python
def _mixed_bool_numeric_space():
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_gene_space_profile_distinguishes_numeric_binary_and_mixed():
    assert gene_space_profile(GeneSpace.uniform(-1.0, 1.0, 2)) == "numeric"
    assert gene_space_profile(GeneSpace([Gene("a", "bool"), Gene("b", "bool")])) == "binary"
    assert gene_space_profile(_mixed_bool_numeric_space()) == "mixed"


def test_mixed_bool_numeric_operator_matrix_accepts_typed_ga_defaults():
    space = _mixed_bool_numeric_space()

    crossover = resolve_operator_domain(CrossoverOperator.uniform(), space)
    gaussian = resolve_operator_domain(MutationOperator.gaussian(), space)
    uniform = resolve_operator_domain(MutationOperator.uniform(), space)
    bit_flip = resolve_operator_domain(MutationOperator.bit_flip(), space)

    assert crossover.signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "mixed",
        "parameters": {"probability": 0.9},
    }
    assert gaussian.signature() == {
        "type": "gaussian",
        "operator_type": "mutation",
        "domain": "mixed",
        "parameters": {
            "individual_probability": 1.0,
            "probability": 0.1,
            "sigma": 0.2,
        },
    }
    assert uniform.signature()["domain"] == "mixed"
    assert bit_flip.signature()["domain"] == "mixed"

    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.gaussian(), space)
    validate_operator_compatibility(MutationOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.bit_flip(), space)


@pytest.mark.parametrize(
    "operator, pattern",
    [
        (CrossoverOperator.sbx(), r"crossover='sbx'.*bool.*float.*int"),
        (CrossoverOperator.blx(), r"crossover='blx'.*bool.*float.*int"),
        (CrossoverOperator.one_point(), r"crossover='one_point'.*bool.*float.*int"),
        (CrossoverOperator.two_point(), r"crossover='two_point'.*bool.*float.*int"),
    ],
)
def test_mixed_bool_numeric_operator_matrix_rejects_incompatible_crossovers(
    operator,
    pattern,
):
    with pytest.raises(ConfigurationError, match=pattern):
        validate_operator_compatibility(operator, _mixed_bool_numeric_space())


def test_custom_mutation_operator_must_cover_mixed_bool_gene_kind():
    operator = custom_mutation_operator(ShiftMutation())

    with pytest.raises(ConfigurationError, match=r"mutation='shift'.*bool.*float.*int"):
        validate_operator_compatibility(operator, _mixed_bool_numeric_space())
```

- [ ] **Step 2: Write failing `OperatorCodec` mixed bool tests**

In `tests/unit/test_operators.py`, replace `test_mixed_bool_numeric_rejected` with these tests:

```python
def test_mixed_bool_numeric_accepts_uniform_gaussian_and_decodes_bool():
    space = GeneSpace(
        [
            Gene("x", "float", 0.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
        ]
    )

    ops = OperatorCodec(space, "uniform", "gaussian")

    assert ops.crossover == "uniform"
    assert ops.mutation == "gaussian"
    assert ops.crossover_operator.domain == "mixed"
    assert ops.mutation_operator.domain == "mixed"
    assert ops.encode_values([0.25, 10, True]) == [0.25, 10.0, 1.0]
    assert ops.decode_values([0.25, 10.2, 0.8]) == [0.25, 10, True]


def test_mixed_bool_numeric_rejects_numeric_only_crossover():
    space = GeneSpace([Gene("x", "float", 0.0, 1.0), Gene("flag", "bool")])

    with pytest.raises(ConfigurationError, match=r"crossover='sbx'.*bool.*float"):
        OperatorCodec(space, "sbx", "gaussian")
```

- [ ] **Step 3: Run failing focused operator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py tests/unit/test_operators.py -v
```

Expected: FAIL because `gene_space_profile` is missing and mixed bool numeric spaces still raise the old "bool genes alongside" configuration error.

- [ ] **Step 4: Add mixed profile helpers and operator domain resolution**

In `evocore/optimizers/operators.py`, replace the `OperatorDomain` definition with:

```python
OperatorType = Literal["crossover", "mutation", "selection", "bounds"]
OperatorDomain = Literal[
    "numeric",
    "binary",
    "mixed",
    "score",
    "repair",
    "auto",
    "custom",
]
GeneSpaceProfile = Literal["numeric", "binary", "mixed"]
```

Replace `gene_space_domain` with this pair of helpers:

```python
def gene_space_profile(gene_space: GeneSpace) -> GeneSpaceProfile:
    """Return the flat GA profile implied by a gene space."""
    kinds = set(gene_space.kinds)
    if kinds == {"bool"}:
        return "binary"
    if "bool" in kinds:
        return "mixed"
    return "numeric"


def gene_space_domain(gene_space: GeneSpace) -> GeneSpaceProfile:
    """Return the GA operator domain implied by a gene space."""
    return gene_space_profile(gene_space)


def _supported_gene_kinds_for_profile(profile: GeneSpaceProfile) -> frozenset[GeneKind]:
    if profile == "numeric":
        return NUMERIC_GENE_KINDS
    if profile == "binary":
        return BINARY_GENE_KINDS
    return ALL_FLAT_GENE_KINDS
```

Replace `resolve_operator_domain` with:

```python
def resolve_operator_domain(
    operator: CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy,
    gene_space: GeneSpace,
) -> CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy:
    """Resolve profile-sensitive operators against a concrete gene space."""
    profile = gene_space_profile(gene_space)

    if isinstance(operator, CrossoverOperator) and operator.domain == "auto":
        return CrossoverOperator(
            operator.name,
            dict(operator.parameters),
            _supported_gene_kinds_for_profile(profile),
            profile,
            custom=operator.custom,
            implementation=operator.implementation,
        )

    if (
        isinstance(operator, MutationOperator)
        and profile == "mixed"
        and operator.name in ("gaussian", "uniform", "bit_flip")
        and not operator.custom
    ):
        return MutationOperator(
            operator.name,
            dict(operator.parameters),
            ALL_FLAT_GENE_KINDS,
            "mixed",
            custom=operator.custom,
            implementation=operator.implementation,
        )

    return operator
```

Add `gene_space_profile` to `__all__`:

```python
    "gene_space_domain",
    "gene_space_profile",
```

- [ ] **Step 5: Run focused operator tests until they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py tests/unit/test_operators.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit operator contract changes**

Run:

```powershell
git add evocore/optimizers/operators.py tests/unit/test_operator_contract.py tests/unit/test_operators.py
git commit -m "feat(ga): accept mixed bool operator profile"
```

Expected: commit succeeds. If the focused tests fail, stop and do not commit.

---

### Task 2: Profile-Aware GA Defaults And Mixed Python Reproduction

**Files:**
- Modify: `evocore/optimizers/ga/engine.py`
- Modify: `evocore/optimizers/ga/reproduction.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Write failing default-resolution tests**

In `tests/unit/test_optimizer_config.py`, append:

```python
def _bool_space():
    return GeneSpace([Gene("a", "bool"), Gene("b", "bool")])


def _mixed_bool_space():
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_bool_only_default_ga_resolves_binary_operators():
    default = GeneticAlgorithmOptimizer(_bool_space(), population_size=4, max_generations=1)
    explicit = GeneticAlgorithmOptimizer(
        _bool_space(),
        population_size=4,
        max_generations=1,
        crossover="uniform",
        mutation="bit_flip",
    )

    assert default.crossover == "uniform"
    assert default.mutation == "bit_flip"
    assert default.config_signature() == explicit.config_signature()
    assert default.config_signature()["components"]["crossover"]["domain"] == "binary"
    assert default.config_signature()["components"]["mutation"]["domain"] == "binary"


def test_mixed_bool_default_ga_resolves_typed_defaults():
    default = GeneticAlgorithmOptimizer(_mixed_bool_space(), population_size=4, max_generations=1)
    explicit = GeneticAlgorithmOptimizer(
        _mixed_bool_space(),
        population_size=4,
        max_generations=1,
        crossover="uniform",
        mutation="gaussian",
    )

    assert default.crossover == "uniform"
    assert default.mutation == "gaussian"
    assert default.config_signature() == explicit.config_signature()
    assert default.config_signature()["components"]["crossover"]["domain"] == "mixed"
    assert default.config_signature()["components"]["mutation"]["domain"] == "mixed"


@pytest.mark.parametrize("crossover", ["sbx", "blx", "one_point", "two_point"])
def test_explicit_incompatible_crossovers_still_reject_mixed_bool_spaces(crossover):
    with pytest.raises(ConfigurationError, match=rf"crossover='{crossover}'.*bool"):
        GeneticAlgorithmOptimizer(_mixed_bool_space(), crossover=crossover)
```

- [ ] **Step 2: Write failing mixed reproduction tests**

In `tests/unit/test_ga_engine.py`, append:

```python
def _ga_mixed_bool_space():
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_ga_mixed_bool_default_uses_python_reproduction_path():
    engine = GeneticAlgorithmOptimizer(
        _ga_mixed_bool_space(),
        population_size=4,
        max_generations=1,
        seed=42,
    )
    population = [
        Solution([0.1, 5, False], score=1.0, score_valid=True),
        Solution([0.2, 10, True], score=2.0, score_valid=True),
        Solution([0.3, 15, False], score=3.0, score_valid=True),
        Solution([0.4, 20, True], score=4.0, score_valid=True),
    ]

    assert engine._uses_python_reproduction() is True
    offspring = engine._make_offspring(population, [1.0, 2.0, 3.0, 4.0], gen=1, offspring_count=4)

    assert len(offspring) == 4
    for solution in offspring:
        assert type(solution.values[0]) is float
        assert type(solution.values[1]) is int
        assert type(solution.values[2]) is bool
        engine.gene_space.validate_genes(solution.values)


def test_gaussian_mutation_flips_bool_gene_in_mixed_space():
    engine = GeneticAlgorithmOptimizer(
        _ga_mixed_bool_space(),
        crossover="uniform",
        mutation="gaussian",
        mutation_prob=1.0,
        mutation_individual_prob=1.0,
        seed=42,
    )

    mutated = engine._mutate_child_python(
        [0.25, 10, False],
        gen=1,
        individual_index=0,
        mutation_sigmas=[0.1, 2.0, 0.0],
    )

    assert mutated[2] is True
    engine.gene_space.validate_genes(mutated)


def test_uniform_mutation_flips_bool_gene_in_mixed_space():
    engine = GeneticAlgorithmOptimizer(
        _ga_mixed_bool_space(),
        crossover="uniform",
        mutation="uniform",
        mutation_prob=1.0,
        mutation_individual_prob=1.0,
        seed=42,
    )

    mutated = engine._mutate_child_python(
        [0.25, 10, True],
        gen=1,
        individual_index=0,
        mutation_sigmas=[0.1, 2.0, 0.0],
    )

    assert mutated[2] is False
    engine.gene_space.validate_genes(mutated)


def test_bit_flip_mutation_only_changes_bool_gene_in_mixed_space():
    engine = GeneticAlgorithmOptimizer(
        _ga_mixed_bool_space(),
        crossover="uniform",
        mutation="bit_flip",
        mutation_prob=1.0,
        mutation_individual_prob=1.0,
        seed=42,
    )

    mutated = engine._mutate_child_python(
        [0.25, 10, False],
        gen=1,
        individual_index=0,
        mutation_sigmas=[0.1, 2.0, 0.0],
    )

    assert mutated == [0.25, 10, True]
    engine.gene_space.validate_genes(mutated)
```

- [ ] **Step 3: Run failing GA default and reproduction tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py -v
```

Expected: FAIL because the constructor still treats omitted defaults as explicit `"sbx"`/`"gaussian"` strings, bool-only defaults do not auto-resolve, and `_uses_python_reproduction()` does not exist.

- [ ] **Step 4: Add profile-aware omitted defaults in GA engine**

In `evocore/optimizers/ga/engine.py`, add `gene_space_profile` to the operator imports:

```python
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    gene_space_profile,
    normalize_bounds_policy,
    normalize_crossover_operator,
    normalize_mutation_operator,
    normalize_selection_operator,
    resolve_operator_domain,
)
```

Add these helpers near `logger = logging.getLogger(__name__)`:

```python
_DEFAULT_GA_OPERATOR = object()


def _default_crossover_for_profile(profile: str) -> str:
    if profile == "numeric":
        return "sbx"
    return "uniform"


def _default_mutation_for_profile(profile: str) -> str:
    if profile == "binary":
        return "bit_flip"
    return "gaussian"
```

Change the constructor parameters for `crossover` and `mutation`:

```python
        crossover: str | CrossoverOperator | object = _DEFAULT_GA_OPERATOR,
```

```python
        mutation: str | MutationOperator | object = _DEFAULT_GA_OPERATOR,
```

After the `mutation_sigma_schedule` validation block and before the first `_reject_typed_operator_scalar_conflicts` call, add:

```python
        profile = gene_space_profile(gene_space)
        crossover_value = (
            _default_crossover_for_profile(profile)
            if crossover is _DEFAULT_GA_OPERATOR
            else crossover
        )
        mutation_value = (
            _default_mutation_for_profile(profile)
            if mutation is _DEFAULT_GA_OPERATOR
            else mutation
        )
```

In all three `_reject_typed_operator_scalar_conflicts` calls, pass `crossover_value`, `mutation_value`, and the unchanged `selection` value:

```python
            provided=crossover_value,
```

```python
            provided=mutation_value,
```

In normalization, use `crossover_value` and `mutation_value`:

```python
                crossover_value,
```

```python
                mutation_value,
```

Update the constructor docstring operator descriptions to:

```python
        crossover: Crossover operator name or spec. When omitted, numeric spaces use
            `"sbx"` and bool-only or mixed bool/numeric spaces use `"uniform"`.
```

```python
        mutation: Mutation operator name or spec. When omitted, numeric and mixed
            bool/numeric spaces use `"gaussian"` and bool-only spaces use `"bit_flip"`.
```

- [ ] **Step 5: Route mixed spaces through Python reproduction**

In `evocore/optimizers/ga/reproduction.py`, add `gene_space_profile` to the import from `evocore.optimizers.operators`:

```python
from evocore.optimizers.operators import (
    CrossoverContext,
    MutationContext,
    SelectionContext,
    apply_bounds_policy,
    gene_space_profile,
)
```

In `_make_offspring`, replace:

```python
        if self._uses_custom_operator():
```

with:

```python
        if self._uses_python_reproduction():
```

After `_uses_custom_operator`, add:

```python
    def _uses_mixed_gene_space(self) -> bool:
        return gene_space_profile(self.gene_space) == "mixed"

    def _uses_python_reproduction(self) -> bool:
        return self._uses_custom_operator() or self._uses_mixed_gene_space()
```

In `_mutate_child_python`, after the integer uniform mutation branch and before the final return, add the bool mutation branch:

```python
            elif (
                self.mutation_operator.name in ("gaussian", "uniform", "bit_flip")
                and gene.kind == "bool"
            ):
                mutated[index] = not bool(mutated[index])
```

- [ ] **Step 6: Run focused GA default and reproduction tests until they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit default and reproduction changes**

Run:

```powershell
git add evocore/optimizers/ga/engine.py evocore/optimizers/ga/reproduction.py tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py
git commit -m "feat(ga): support mixed bool reproduction defaults"
```

Expected: commit succeeds. If the focused tests fail, stop and do not commit.

---

### Task 3: Mixed Bool Lifecycle, Checkpoint, CMA-ES, And Property Coverage

**Files:**
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `tests/unit/test_ask_tell_checkpointing.py`
- Modify: `tests/unit/test_checkpointing.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/integration/test_mixed_gene_space.py`
- Modify: `tests/property/test_operator_contract_properties.py`

- [ ] **Step 1: Add ask/tell mixed bool tests**

In `tests/unit/test_ga_ask_tell_vnext.py`, append:

```python
def _mixed_bool_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_ga_ask_returns_mixed_bool_candidates_with_bool_params() -> None:
    engine = GeneticAlgorithmOptimizer(
        _mixed_bool_space(),
        population_size=6,
        max_generations=5,
        seed=321,
    )

    candidates = engine.ask(6)

    assert candidates
    assert all(type(candidate.genes[2]) is bool for candidate in candidates)
    assert all(type(candidate.params["enabled"]) is bool for candidate in candidates)
    assert all(candidate.origin == "random" for candidate in candidates)


def test_ga_ask_tell_next_generation_preserves_mixed_bool_types() -> None:
    engine = GeneticAlgorithmOptimizer(
        _mixed_bool_space(),
        population_size=6,
        max_generations=5,
        seed=321,
    )
    first = engine.ask(6)
    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            )
            for index, candidate in enumerate(first)
        ]
    )

    second = engine.ask(6)

    assert second
    assert all(type(candidate.genes[2]) is bool for candidate in second)
    assert all(type(candidate.params["enabled"]) is bool for candidate in second)
    assert all(candidate.origin == "mutation" for candidate in second)
```

- [ ] **Step 2: Add ask/tell checkpoint mixed bool resume test**

In `tests/unit/test_ask_tell_checkpointing.py`, append:

```python
def _mixed_ga() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(
        GeneSpace(
            [
                Gene("threshold", "float", 0.0, 1.0),
                Gene("period", "int", 2, 50),
                Gene("enabled", "bool"),
            ]
        ),
        population_size=4,
        max_generations=5,
        seed=123,
    )


def test_ga_mixed_bool_ask_tell_checkpoint_round_trip_preserves_bool_values(tmp_path) -> None:
    source = _mixed_ga()
    candidates = source.ask(4)
    source.tell([_record(candidates[0].candidate_id, batch_id=candidates[0].batch_id)])
    checkpoint_path = tmp_path / "ga-mixed-bool.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _mixed_ga()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    restored_candidate = restored._candidates_by_id[candidates[0].candidate_id]

    assert summary.best_candidate_id == candidates[0].candidate_id
    assert type(restored_candidate.genes[2]) is bool
    assert type(restored_candidate.params["enabled"]) is bool
    assert restored_candidate.genes == candidates[0].genes
```

Add `Gene` to the import from `evocore` at the top of the file:

```python
from evocore import CheckpointError, FitnessError, Gene, GeneSpace, GeneticAlgorithmOptimizer
```

- [ ] **Step 3: Add generation checkpoint mixed bool export/load test**

In `tests/unit/test_checkpointing.py`, add `Gene` to the import from `evocore`:

```python
from evocore import CheckpointError, Gene, GeneSpace, GeneticAlgorithmOptimizer
```

Append:

```python
def test_ga_generation_checkpoint_round_trip_preserves_mixed_bool_values(tmp_path) -> None:
    space = GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )
    engine = GeneticAlgorithmOptimizer(space, population_size=4, max_generations=2, seed=42)
    population = [
        Solution(
            [0.25, 10, True],
            score=1.0,
            score_valid=True,
            metadata={"params": {"threshold": 0.25, "period": 10, "enabled": True}},
        )
    ]
    path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"

    engine.save_checkpoint(path, engine.checkpoint(generation=0, population=population))
    loaded = load_checkpoint(path)

    row = loaded["state"]["payload"]["population"][0]
    assert row["values"] == [0.25, 10, True]
    assert type(row["values"][2]) is bool
    assert row["metadata"]["params"]["enabled"] is True
```

- [ ] **Step 4: Add CMA-ES mixed bool rejection test**

In `tests/unit/test_cmaes_engine.py`, append:

```python
def test_cmaes_rejects_mixed_bool_numeric_genes():
    space = GeneSpace([Gene("x", "float", 0.0, 1.0), Gene("flag", "bool")])

    with pytest.raises(ConfigurationError, match="bool"):
        CMAESOptimizer(space)
```

- [ ] **Step 5: Add integration run coverage for default mixed bool GA**

In `tests/integration/test_mixed_gene_space.py`, change the helper import to:

```python
from tests.vnext_helpers import IndividualEvaluator, full_policy
```

Append:

```python
def mixed_bool_target(ind):
    params = ind.params
    enabled_bonus = 5.0 if params["enabled"] else 0.0
    return (
        enabled_bonus
        - ((params["period"] - 20) ** 2)
        - ((params["threshold"] - 0.3) ** 2)
    )


def test_mixed_bool_gene_space_runs_with_default_operators():
    space = GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )
    engine = GeneticAlgorithmOptimizer(
        space,
        population_size=12,
        max_generations=4,
        seed=42,
    )

    result = engine.run(
        IndividualEvaluator(mixed_bool_target),
        policy=full_policy(48, batch_size=12),
    )

    assert result.n_evaluations == 48
    assert type(result.best_solution.params["enabled"]) is bool
    assert all(type(solution.values[2]) is bool for solution in result.final_solutions)
```

- [ ] **Step 6: Add property coverage for mixed default operator signatures**

In `tests/property/test_operator_contract_properties.py`, add this import:

```python
from evocore.optimizers.operators import (
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
    apply_bounds_policy,
    resolve_operator_domain,
)
```

Append:

```python
@given(
    probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    mutation_probability=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    sigma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_mixed_default_operator_signatures_are_json_safe(
    probability,
    mutation_probability,
    sigma,
):
    space = GeneSpace(
        [
            Gene("x", "float", 0.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
        ]
    )
    payloads = [
        resolve_operator_domain(
            CrossoverOperator.uniform(probability=probability),
            space,
        ).signature(),
        resolve_operator_domain(
            MutationOperator.gaussian(probability=mutation_probability, sigma=sigma),
            space,
        ).signature(),
        resolve_operator_domain(
            MutationOperator.bit_flip(probability=mutation_probability),
            space,
        ).signature(),
    ]

    for payload in payloads:
        assert payload["domain"] == "mixed"
        assert json.loads(stable_json_dumps(payload)) == payload
```

- [ ] **Step 7: Run lifecycle, checkpoint, CMA-ES, integration, and property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py tests/unit/test_cmaes_engine.py tests/integration/test_mixed_gene_space.py tests/property/test_operator_contract_properties.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit lifecycle coverage**

Run:

```powershell
git add tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py tests/unit/test_cmaes_engine.py tests/integration/test_mixed_gene_space.py tests/property/test_operator_contract_properties.py
git commit -m "test(ga): cover mixed bool lifecycle support"
```

Expected: commit succeeds. If the focused tests fail, stop and do not commit.

---

### Task 4: Documentation And Changelog

**Files:**
- Modify: `docs/site/gene-space.md`
- Modify: `docs/site/operator-contract.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update gene-space docs**

In `docs/site/gene-space.md`, after the named-space `params` example, add:

````markdown
`GeneticAlgorithmOptimizer` supports flat spaces that mix `float`, `int`, and
`bool` genes. This lets users model real boolean switches directly instead of
encoding them as integer genes.

```python
from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("period", "int", 2, 50),
        Gene("enabled", "bool"),
    ]
)

optimizer = GeneticAlgorithmOptimizer(space)
```
````

- [ ] **Step 2: Update operator contract docs**

In `docs/site/operator-contract.md`, replace the compatibility table and the paragraph below it with:

```markdown
| Operator | Type | Supported GA spaces |
| --- | --- | --- |
| `sbx` | crossover | numeric-only `float`/`int` spaces |
| `blx` | crossover | numeric-only `float`/`int` spaces |
| `uniform` | crossover | numeric-only, bool-only, and mixed `float`/`int`/`bool` spaces |
| `one_point` | crossover | bool-only spaces |
| `two_point` | crossover | bool-only spaces |
| `gaussian` | mutation | numeric-only and mixed `float`/`int`/`bool` spaces |
| `uniform` | mutation | numeric-only and mixed `float`/`int`/`bool` spaces |
| `bit_flip` | mutation | bool-only and mixed `float`/`int`/`bool` spaces |
| `tournament` | selection | any GA-supported space |
| `roulette` | selection | any GA-supported space |
| `rank` | selection | any GA-supported space |

Numeric spaces may mix `float` and `int` genes. Binary spaces contain only `bool`
genes. Mixed GA spaces contain at least one numeric gene and at least one `bool`
gene.

When GA operator arguments are omitted, EvoCore resolves defaults by space profile:

| Space profile | Default crossover | Default mutation |
| --- | --- | --- |
| Numeric-only `float`/`int` | `sbx` | `gaussian` |
| Bool-only `bool` | `uniform` | `bit_flip` |
| Mixed `float`/`int` plus `bool` | `uniform` | `gaussian` |

In mixed spaces, `gaussian` and `uniform` mutation mutate numeric genes using
their numeric behavior and flip bool genes with `mutation_prob`. `bit_flip`
mutation flips bool genes with `mutation_prob` and leaves numeric genes unchanged.

`sbx`, `blx`, `one_point`, and `two_point` remain incompatible with mixed spaces.
Choose `uniform` crossover for mixed GA spaces.
```

- [ ] **Step 3: Add a mixed bool GA example**

In `docs/site/ga.md`, after the introductory ask/tell checkpoint paragraphs and before `## Budgeted Evaluation`, add:

````markdown
## Mixed Bool Spaces

GA can optimize flat spaces that combine numeric parameters with boolean switches.
Omitted operators resolve to `uniform` crossover and typed `gaussian` mutation for
mixed spaces.

```python
from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("period", "int", 2, 50),
        Gene("enabled", "bool"),
    ]
)

optimizer = GeneticAlgorithmOptimizer(space, population_size=32, seed=42)
candidates = optimizer.ask(4)

assert isinstance(candidates[0].params["enabled"], bool)
```
````

- [ ] **Step 4: Clarify CMA-ES bool rejection**

In `docs/site/cmaes.md`, after the opening paragraph, add:

```markdown
CMA-ES supports numeric `float` and `int` genes. It rejects `bool` genes,
including spaces that mix booleans with numeric genes. Use
`GeneticAlgorithmOptimizer` for mixed flat spaces with boolean switches.
```

- [ ] **Step 5: Update changelog**

In `CHANGELOG.md`, under `## [Unreleased]`, add:

```markdown
### Added

- `GeneticAlgorithmOptimizer` now accepts mixed flat `float`/`int`/`bool`
  `GeneSpace` values with profile-aware default operators, typed bool mutation,
  ask/tell, run, and checkpoint coverage. CMA-ES continues to reject bool genes.
```

- [ ] **Step 6: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: PASS.

- [ ] **Step 7: Commit docs**

Run:

```powershell
git add docs/site/gene-space.md docs/site/operator-contract.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "docs(ga): document mixed bool gene support"
```

Expected: commit succeeds. If docs build fails, stop and do not commit.

---

### Task 5: Final Verification

**Files:**
- Read-only verification across touched source, tests, docs, and extension build

- [ ] **Step 1: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS.

- [ ] **Step 2: Rebuild the Python extension**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: PASS.

- [ ] **Step 3: Run focused unit, integration, and property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_operator_contract.py tests/unit/test_operators.py tests/unit/test_optimizer_config.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py tests/unit/test_cmaes_engine.py tests/integration/test_mixed_gene_space.py tests/property/test_operator_contract_properties.py -v
```

Expected: PASS.

- [ ] **Step 4: Run broader Python tests for public behavior**

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

- [ ] **Step 6: Build docs one final time**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: PASS.

- [ ] **Step 7: Inspect final branch state**

Run:

```powershell
git status --short --branch
git log --oneline -4
```

Expected: branch is clean except ignored caches. The latest commits should be:

```text
docs(ga): document mixed bool gene support
test(ga): cover mixed bool lifecycle support
feat(ga): support mixed bool reproduction defaults
feat(ga): accept mixed bool operator profile
```

If verification passes, push and open a draft PR using `.github/pull_request_template.md` per `AGENTS.md`. If any verification command fails, stop and report the failing command, the relevant error summary, and the likely files involved.
