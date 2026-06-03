# DE Built-In Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `best1bin`, `rand2bin`, and `current-to-best1bin` as opt-in stateless strategies for `DifferentialEvolutionOptimizer`.

**Architecture:** Build on the strategy contract from `feature/de-strategy-contract`. Extend `strategies.py` with strategy-specific donor selection and mutation formulas, keep config validation strategy-aware, and keep `ask_tell.py` strategy-agnostic.

**Tech Stack:** Python 3.14, EvoCore Python package, pytest, ruff, MkDocs documentation, existing Rust-backed `_core` deterministic seed helpers.

---

## Dependencies

This plan implements Gate 2 from:

```text
docs/superpowers/specs/2026-06-03-differential-evolution-strategy-adaptation-design.md
```

Start from the completed contract branch:

```powershell
git switch feature/de-strategy-contract
git pull --ff-only
git switch -c feature/de-built-in-strategies
```

This plan assumes `evocore/optimizers/de/strategies.py` already exports:

```python
DEStrategySpec
TrialContext
TrialProposal
repair_de_gene_value
rng_for_de_trial
strategy_spec_for
supported_strategy_names
trial_proposal_for_strategy
validate_strategy_population_size
```

## File Structure

- Modify `evocore/optimizers/de/strategies.py`: add stateless strategy specs, donor helpers, best-slot selection, and strategy dispatch for `best1bin`, `rand2bin`, and `current-to-best1bin`.
- Modify `tests/unit/test_de_strategies.py`: add unit tests for donor metadata, mixed repair, direction-aware best selection, and validation.
- Modify `tests/unit/test_de_engine.py`: add constructor/config tests for accepted names and population-size errors.
- Modify `tests/unit/test_de_multi_run.py`: verify `run_multiple()` works with a non-default strategy.
- Modify `tests/integration/test_de_mixed_gene_space.py`: add one mixed-space run with a non-default stateless strategy.
- Modify `docs/site/de.md`: document strategy names and selection guidance.
- Modify `CHANGELOG.md`: add a user-visible entry for new DE strategies.

## Task 1: Add Failing Tests For Strategy Names And Validation

**Files:**
- Modify: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Add constructor validation tests**

Append these tests after the existing DE configuration tests:

```python
@pytest.mark.parametrize(
    "strategy",
    ["rand1bin", "best1bin", "rand2bin", "current-to-best1bin"],
)
def test_de_accepts_supported_stateless_strategies(strategy: str) -> None:
    population_size = 6 if strategy == "rand2bin" else 4

    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=population_size,
        strategy=strategy,
        seed=42,
    )

    assert engine.strategy == strategy
    assert engine.config_signature()["parameters"]["strategy"] == strategy
    assert engine.config_signature()["components"]["strategy"]["type"] == strategy


@pytest.mark.parametrize(
    ("strategy", "population_size", "message"),
    [
        ("best1bin", 3, "at least 4"),
        ("rand2bin", 5, "at least 6"),
        ("current-to-best1bin", 3, "at least 4"),
    ],
)
def test_de_strategy_specific_population_size_validation(
    strategy: str,
    population_size: int,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        DifferentialEvolutionOptimizer(
            _space(),
            population_size=population_size,
            strategy=strategy,
            seed=42,
        )


def test_de_config_hash_changes_when_strategy_changes() -> None:
    rand1 = DifferentialEvolutionOptimizer(_space(), population_size=6, strategy="rand1bin")
    rand2 = DifferentialEvolutionOptimizer(_space(), population_size=6, strategy="rand2bin")

    assert rand1.config_hash() != rand2.config_hash()
```

- [ ] **Step 2: Run the new engine tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_accepts_supported_stateless_strategies tests/unit/test_de_engine.py::test_de_strategy_specific_population_size_validation tests/unit/test_de_engine.py::test_de_config_hash_changes_when_strategy_changes -v
```

Expected: FAIL because `best1bin`, `rand2bin`, and `current-to-best1bin` are not supported yet.

## Task 2: Extend Strategy Specs

**Files:**
- Modify: `evocore/optimizers/de/strategies.py`
- Test: `tests/unit/test_de_engine.py`

- [ ] **Step 1: Extend `SUPPORTED_DE_STRATEGIES`**

Replace the existing dictionary with:

```python
SUPPORTED_DE_STRATEGIES: dict[str, DEStrategySpec] = {
    "rand1bin": DEStrategySpec(name="rand1bin", min_population_size=4),
    "best1bin": DEStrategySpec(name="best1bin", min_population_size=4),
    "rand2bin": DEStrategySpec(name="rand2bin", min_population_size=6),
    "current-to-best1bin": DEStrategySpec(
        name="current-to-best1bin",
        min_population_size=4,
    ),
}
```

- [ ] **Step 2: Update the unknown-strategy test from the contract plan**

In `tests/unit/test_de_strategies.py`, replace the assertion in `test_strategy_spec_for_rejects_unknown_strategy` with:

```python
    with pytest.raises(
        ConfigurationError,
        match=(
            "strategy must be one of 'rand1bin', 'best1bin', "
            "'rand2bin', 'current-to-best1bin'"
        ),
    ):
        strategy_spec_for("jade")
```

- [ ] **Step 3: Run validation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_accepts_supported_stateless_strategies tests/unit/test_de_engine.py::test_de_strategy_specific_population_size_validation tests/unit/test_de_engine.py::test_de_config_hash_changes_when_strategy_changes tests/unit/test_de_strategies.py::test_strategy_spec_for_rejects_unknown_strategy -v
```

Expected: PASS for constructor/config validation.

- [ ] **Step 4: Commit validation support**

Run:

```powershell
git add evocore/optimizers/de/strategies.py tests/unit/test_de_engine.py tests/unit/test_de_strategies.py
git commit -m "feat(de): accept stateless strategy names"
```

## Task 3: Add Strategy Proposal Tests

**Files:**
- Modify: `tests/unit/test_de_strategies.py`

- [ ] **Step 1: Add helper for direction-aware populations**

Append this helper below `_trusted_population`:

```python
def _trusted_population_with_direction(direction: str):
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction=direction,
    )
    candidates = engine.ask()
    scores = [0, 1, 2, 3, 4, 5] if direction == "maximize" else [5, 4, 3, 2, 1, 0]
    engine.tell(_records(candidates, scores))
    return engine
```

- [ ] **Step 2: Add strategy proposal tests**

Append these tests:

```python
@pytest.mark.parametrize(
    "strategy",
    ["best1bin", "rand2bin", "current-to-best1bin"],
)
def test_stateless_strategy_proposals_have_required_metadata(strategy: str) -> None:
    engine = _trusted_population()
    population = [engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name=strategy,
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    assert proposal.metadata["strategy"] == strategy
    assert proposal.metadata["target_slot"] == 0
    assert proposal.metadata["donor_slots"]
    assert len(proposal.genes) == engine.gene_space.length
    engine.gene_space.validate_genes(proposal.genes)


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("strategy", ["best1bin", "current-to-best1bin"])
def test_best_based_strategies_record_direction_aware_best_slot(
    direction: str,
    strategy: str,
) -> None:
    engine = _trusted_population_with_direction(direction)
    population = [engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name=strategy,
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    assert proposal.metadata["best_slot"] == 5


def test_rand2bin_records_five_distinct_donor_slots() -> None:
    engine = _trusted_population()
    population = [engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name="rand2bin",
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    donor_slots = proposal.metadata["donor_slots"]
    assert len(donor_slots) == 5
    assert len(set(donor_slots)) == 5
    assert 0 not in donor_slots
```

- [ ] **Step 3: Run proposal tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py::test_stateless_strategy_proposals_have_required_metadata tests/unit/test_de_strategies.py::test_best_based_strategies_record_direction_aware_best_slot tests/unit/test_de_strategies.py::test_rand2bin_records_five_distinct_donor_slots -v
```

Expected: FAIL because the strategy dispatcher only implements `rand1bin`.

## Task 4: Implement Stateless Trial Proposals

**Files:**
- Modify: `evocore/optimizers/de/strategies.py`
- Test: `tests/unit/test_de_strategies.py`

- [ ] **Step 1: Extend `TrialContext` with direction**

Add a direction field to `TrialContext`:

```python
    direction: str = "maximize"
```

- [ ] **Step 2: Update the ask/tell call site**

In `evocore/optimizers/de/ask_tell.py`, add the direction argument when constructing `TrialContext`:

```python
                direction=self.direction,
```

- [ ] **Step 3: Add helper functions to `strategies.py`**

Add these helpers below `_forced_variable_index`:

```python
def _comparison_score(candidate: Candidate, direction: str) -> float:
    return candidate.state_comparison_score(direction)


def _best_slot(context: TrialContext) -> int:
    return max(
        range(len(context.population)),
        key=lambda slot: _comparison_score(context.population[slot], context.direction),
    )


def _sample_slots(
    *,
    context: TrialContext,
    count: int,
    excluded: set[int],
    op_offset: int = 0,
):
    choices = [slot for slot in range(len(context.population)) if slot not in excluded]
    rng = rng_for_de_trial(
        context.seed,
        context.generation,
        context.target_slot + op_offset,
        _core.OP_SELECTION,
    )
    selected = rng.sample(choices, count)
    return tuple(int(slot) for slot in selected)


def _selected_gene(context: TrialContext, index: int, forced_index: int, mask_rng: random.Random) -> bool:
    return index == forced_index or mask_rng.random() < context.crossover_rate


def _bool_from_difference_pairs(
    *,
    base: Candidate,
    pairs: Sequence[tuple[Candidate, Candidate]],
    gene_index: int,
    mutation_factor: float,
    bool_rng: random.Random,
) -> bool:
    value = bool(base.genes[gene_index])
    for left, right in pairs:
        if bool(left.genes[gene_index]) != bool(right.genes[gene_index]) and bool_rng.random() < min(
            1.0,
            mutation_factor,
        ):
            value = not value
    return value


def _mutant_value(
    *,
    context: TrialContext,
    gene_index: int,
    base: Candidate,
    pairs: Sequence[tuple[Candidate, Candidate]],
    target: Candidate | None = None,
    best: Candidate | None = None,
    bool_rng: random.Random,
) -> float | int | bool:
    gene = context.gene_space.genes[gene_index]
    if gene.kind == "bool":
        if target is not None and best is not None and bool(best.genes[gene_index]) != bool(
            target.genes[gene_index]
        ):
            return bool(best.genes[gene_index])
        return _bool_from_difference_pairs(
            base=base,
            pairs=pairs,
            gene_index=gene_index,
            mutation_factor=context.mutation_factor,
            bool_rng=bool_rng,
        )

    mutant = float(base.genes[gene_index])
    if target is not None and best is not None:
        mutant = float(target.genes[gene_index]) + context.mutation_factor * (
            float(best.genes[gene_index]) - float(target.genes[gene_index])
        )
    for left, right in pairs:
        mutant += context.mutation_factor * (
            float(left.genes[gene_index]) - float(right.genes[gene_index])
        )
    return repair_de_gene_value(mutant, gene)


def _proposal_from_recipe(
    *,
    context: TrialContext,
    strategy: str,
    base_slot: int,
    difference_pairs: Sequence[tuple[int, int]],
    best_slot: int | None = None,
    current_to_best: bool = False,
) -> TrialProposal:
    target = _target_candidate(context)
    base = context.population[base_slot]
    best = context.population[best_slot] if best_slot is not None else None
    pairs = [
        (context.population[left], context.population[right])
        for left, right in difference_pairs
    ]
    mask_rng = rng_for_de_trial(
        context.seed,
        context.generation,
        context.target_slot,
        _core.OP_CROSSOVER,
    )
    bool_rng = rng_for_de_trial(
        context.seed,
        context.generation,
        context.target_slot,
        _core.OP_MUTATION,
    )
    forced_index = _forced_variable_index(context, mask_rng)
    values: list[float | int | bool] = []

    for index, gene in enumerate(context.gene_space.genes):
        if gene.is_fixed:
            values.append(repair_de_gene_value(float(gene.low), gene))
            continue
        if not _selected_gene(context, index, forced_index, mask_rng):
            values.append(target.genes[index])
            continue
        values.append(
            _mutant_value(
                context=context,
                gene_index=index,
                base=base,
                pairs=pairs,
                target=target if current_to_best else None,
                best=best if current_to_best else None,
                bool_rng=bool_rng,
            )
        )

    context.gene_space.validate_genes(values)
    donor_slots = (base_slot, *[slot for pair in difference_pairs for slot in pair])
    metadata: dict[str, object] = {
        "strategy": strategy,
        "target_slot": context.target_slot,
        "donor_slots": tuple(donor_slots),
        "base_slot": base_slot,
        "difference_pairs": tuple(tuple(pair) for pair in difference_pairs),
    }
    if best_slot is not None:
        metadata["best_slot"] = best_slot
    return TrialProposal(genes=values, metadata=metadata)
```

- [ ] **Step 4: Rewrite `_rand1bin_trial` to use the shared recipe**

Replace `_rand1bin_trial` with:

```python
def _rand1bin_trial(context: TrialContext) -> TrialProposal:
    a_slot, b_slot, c_slot = _rand1bin_donor_slots(context)
    return _proposal_from_recipe(
        context=context,
        strategy="rand1bin",
        base_slot=a_slot,
        difference_pairs=((b_slot, c_slot),),
    )
```

- [ ] **Step 5: Add the three new strategy functions**

Add these below `_rand1bin_trial`:

```python
def _best1bin_trial(context: TrialContext) -> TrialProposal:
    best_slot = _best_slot(context)
    b_slot, c_slot = _sample_slots(
        context=context,
        count=2,
        excluded={context.target_slot, best_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="best1bin",
        base_slot=best_slot,
        difference_pairs=((b_slot, c_slot),),
        best_slot=best_slot,
    )


def _rand2bin_trial(context: TrialContext) -> TrialProposal:
    a_slot, b_slot, c_slot, d_slot, e_slot = _sample_slots(
        context=context,
        count=5,
        excluded={context.target_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="rand2bin",
        base_slot=a_slot,
        difference_pairs=((b_slot, c_slot), (d_slot, e_slot)),
    )


def _current_to_best1bin_trial(context: TrialContext) -> TrialProposal:
    best_slot = _best_slot(context)
    b_slot, c_slot = _sample_slots(
        context=context,
        count=2,
        excluded={context.target_slot, best_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="current-to-best1bin",
        base_slot=context.target_slot,
        difference_pairs=((b_slot, c_slot),),
        best_slot=best_slot,
        current_to_best=True,
    )
```

- [ ] **Step 6: Extend `trial_proposal_for_strategy` dispatch**

Replace the dispatch block with:

```python
    if spec.name == "rand1bin":
        return _rand1bin_trial(context)
    if spec.name == "best1bin":
        return _best1bin_trial(context)
    if spec.name == "rand2bin":
        return _rand2bin_trial(context)
    if spec.name == "current-to-best1bin":
        return _current_to_best1bin_trial(context)
    raise ConfigurationError(f"Unsupported DE strategy implementation: {spec.name!r}.")
```

- [ ] **Step 7: Run strategy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py -v
```

Expected: PASS.

## Task 5: Verify Optimizer Ask/Tell With Non-Default Strategies

**Files:**
- Modify: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Add an ask/tell smoke test for stateless strategies**

Append this test near the existing trial generation tests:

```python
@pytest.mark.parametrize(
    "strategy",
    ["best1bin", "rand2bin", "current-to-best1bin"],
)
def test_de_non_default_stateless_strategies_generate_valid_trials(strategy: str) -> None:
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        strategy=strategy,
        seed=42,
    )
    targets = engine.ask()
    engine.tell(_records(targets, [0, 1, 2, 3, 4, 5]))

    trials = engine.ask()

    assert len(trials) == 6
    assert {trial.metadata["strategy"] for trial in trials} == {strategy}
    assert {trial.metadata["target_slot"] for trial in trials} == set(range(6))
    for trial in trials:
        _mixed_space().validate_genes(trial.genes)
```

- [ ] **Step 2: Run the new ask/tell test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_non_default_stateless_strategies_generate_valid_trials -v
```

Expected: PASS.

- [ ] **Step 3: Run all DE ask/tell and strategy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_strategies.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit stateless proposal implementation**

Run:

```powershell
git add evocore/optimizers/de/strategies.py evocore/optimizers/de/ask_tell.py tests/unit/test_de_strategies.py tests/unit/test_de_ask_tell.py
git commit -m "feat(de): add stateless differential evolution strategies"
```

## Task 6: Add Run, Multi-Run, And Integration Coverage

**Files:**
- Modify: `tests/unit/test_de_engine.py`
- Modify: `tests/unit/test_de_multi_run.py`
- Modify: `tests/integration/test_de_mixed_gene_space.py`

- [ ] **Step 1: Add a policy-driven run smoke test**

In `tests/unit/test_de_engine.py`, append:

```python
@pytest.mark.parametrize(
    "strategy",
    ["best1bin", "rand2bin", "current-to-best1bin"],
)
def test_de_run_supports_non_default_stateless_strategy(strategy: str) -> None:
    result = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        max_generations=2,
        strategy=strategy,
        seed=42,
    ).run(SphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.reproducibility.optimizer_config["parameters"]["strategy"] == strategy
    assert result.final_solutions
```

- [ ] **Step 2: Add a multi-run smoke test**

In `tests/unit/test_de_multi_run.py`, append:

```python
def test_de_run_multiple_supports_non_default_strategy() -> None:
    result = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        strategy="rand2bin",
        seed=7,
    ).run_multiple(SphereEvaluator(), n_runs=3)

    assert result.n_runs == 3
    assert result.best is result.all_runs[0]
    assert {run.reproducibility.optimizer_config["parameters"]["strategy"] for run in result.all_runs} == {
        "rand2bin"
    }
```

- [ ] **Step 3: Add a mixed-space integration smoke test**

In `tests/integration/test_de_mixed_gene_space.py`, append:

```python
def test_de_non_default_strategy_runs_on_mixed_gene_space() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=8,
        max_generations=3,
        strategy="current-to-best1bin",
        seed=123,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.final_solutions
    for solution in result.final_solutions:
        _mixed_space().validate_genes(solution.values)
```

- [ ] **Step 4: Run focused coverage**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_engine.py::test_de_run_supports_non_default_stateless_strategy tests/unit/test_de_multi_run.py::test_de_run_multiple_supports_non_default_strategy tests/integration/test_de_mixed_gene_space.py::test_de_non_default_strategy_runs_on_mixed_gene_space -v
```

Expected: PASS.

## Task 7: Update Documentation And Changelog

**Files:**
- Modify: `docs/site/de.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a Strategies section to `docs/site/de.md`**

Insert this section after "When To Choose DE":

````markdown
## Strategies

`DifferentialEvolutionOptimizer` supports these built-in stateless strategies:

| Strategy | Use when |
| --- | --- |
| `rand1bin` | You want the stable default with broad exploration. |
| `best1bin` | You want stronger pull toward the best current target. |
| `rand2bin` | You want broader differential variation and can afford `population_size >= 6`. |
| `current-to-best1bin` | You want each target to move toward the best target while retaining one difference vector. |

```python
optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=12,
    strategy="rand2bin",
    seed=42,
)
```

All strategies use binomial crossover and the same mixed `float`/`int`/`bool`
repair behavior. The selected strategy is included in the optimizer config
signature and reproducibility metadata.
````

- [ ] **Step 2: Update the Current Limitations section**

Replace:

```markdown
DE does not yet expose custom strategy plugins or a Rust-backed variation
kernel.
```

with:

```markdown
DE does not yet expose custom strategy plugins, adaptive strategies, or a
Rust-backed variation kernel.
```

- [ ] **Step 3: Add changelog entry**

Under `## [Unreleased]` / `### Added`, add:

```markdown
- Added built-in Differential Evolution strategies `best1bin`, `rand2bin`, and
  `current-to-best1bin` with strategy-aware validation and reproducibility
  metadata.
```

- [ ] **Step 4: Run docs-related grep smoke check**

Run:

```powershell
rg -n "best1bin|rand2bin|current-to-best1bin|adaptive strategies" docs/site/de.md CHANGELOG.md
```

Expected: output includes the new strategy section and changelog entry.

- [ ] **Step 5: Commit docs**

Run:

```powershell
git add docs/site/de.md CHANGELOG.md
git commit -m "docs(de): document stateless strategies"
```

## Task 8: Full Verification For Built-In Strategies

**Files:**
- Verify: Python, docs, and changelog changes.

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
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/unit/test_de_multi_run.py tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 4: Run mixed-space integration tests**

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

Expected: clean working tree on `feature/de-built-in-strategies`.

## Self-Review Checklist

- [ ] Strategy names are accepted through the `DifferentialEvolutionOptimizer` constructor.
- [ ] `rand2bin` rejects `population_size < 6`.
- [ ] `best1bin` and `current-to-best1bin` use direction-aware best selection.
- [ ] All strategies preserve mixed `float`/`int`/`bool` repair.
- [ ] `ask_tell.py` still has no strategy-specific branch for built-in strategy formulas.
- [ ] Docs mention built-in strategies and still state that adaptive strategies are not part of this slice.
