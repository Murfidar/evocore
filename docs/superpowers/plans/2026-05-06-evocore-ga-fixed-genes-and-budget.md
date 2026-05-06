# Evocore GA Fixed Genes And Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-genome fixed numeric genes for GA and exact `max_evaluations` budget control with clear run diagnostics.

**Architecture:** Keep Evocore's public schema full-genome: fixed genes stay in `GeneSpace`, `Individual.genes`, and `Individual.params`. Preserve fixed slots through Rust initialization/reproduction by treating equal numeric bounds as valid and clamping after variation. Add budget enforcement in the Python GA loop, where fitness-call accounting already lives.

**Tech Stack:** Python 3.14, pytest, Rust 1.94, PyO3 0.28, maturin, Cargo.

---

## Source Spec

Read the approved design first:

- `docs/superpowers/specs/2026-05-06-evocore-ga-fixed-genes-and-budget-design.md`

## File Structure

- Modify `evocore/gene_space.py`: allow equal numeric bounds and expose fixed/variable metadata.
- Modify `evocore/operators.py`: no structural change expected; tests verify full-genome params and fixed values through existing encode/decode helpers.
- Modify `src/reproduce.rs`: handle equal bounds in initialization and uniform mutation.
- Modify `evocore/cmaes.py`: reject fixed numeric genes until CMA-ES fixed-dimension reconstruction is designed.
- Modify `evocore/ga.py`: add `max_evaluations`, budget-aware evaluation, and run diagnostics.
- Modify `tests/unit/test_gene_space.py`: Python-facing fixed-gene validation and metadata tests.
- Modify `tests/unit/test_operators.py`: fixed genes appear in decoded full-genome params.
- Modify `tests/unit/test_reproduce_rust.py`: PyO3 fixed-bound initialization and reproduction tests.
- Modify `tests/unit/test_cmaes_engine.py`: CMA-ES fixed-gene guard test.
- Modify `tests/unit/test_ga_engine.py`: GA fixed-gene and exact-budget behavior tests.

## Task 1: GeneSpace Fixed Numeric Metadata

**Files:**
- Modify: `evocore/gene_space.py`
- Test: `tests/unit/test_gene_space.py`

- [ ] **Step 1: Add failing GeneDef and GeneSpace tests**

Append these tests to `tests/unit/test_gene_space.py`:

```python
def test_fixed_numeric_genes_are_valid_and_report_fixed_metadata():
    fixed_float = GeneDef("threshold", "float", 0.5, 0.5)
    fixed_int = GeneDef("signal_mode", "int", 2, 2)
    variable = GeneDef("period", "int", 5, 20)

    space = GeneSpace([fixed_float, variable, fixed_int])

    assert fixed_float.is_fixed is True
    assert fixed_int.is_fixed is True
    assert variable.is_fixed is False
    assert space.fixed_indices == [0, 2]
    assert space.variable_indices == [1]
    assert space.fixed_count == 2
    assert space.variable_count == 1
    assert space.bounds == [(0.5, 0.5), (5, 20), (2, 2)]
    assert space.rust_bounds == [(0.5, 0.5), (5.0, 20.0), (2.0, 2.0)]


def test_reversed_numeric_bounds_are_still_rejected():
    with pytest.raises(ConfigurationError, match="requires low <= high"):
        GeneDef("threshold", "float", 1.0, 0.5)


def test_fixed_int_genes_still_require_integer_bounds():
    with pytest.raises(ConfigurationError, match="integer bounds"):
        GeneDef("signal_mode", "int", 2.0, 2.0)


def test_bool_genes_are_not_fixed_in_this_iteration():
    gene = GeneDef("flag", "bool")

    assert gene.is_fixed is False
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
py -m pytest tests/unit/test_gene_space.py -q
```

Expected: FAIL. The fixed numeric tests fail because `GeneDef` still requires `low < high`, and `is_fixed`/index metadata do not exist.

- [ ] **Step 3: Implement fixed metadata in `evocore/gene_space.py`**

In `GeneDef.__post_init__`, change the numeric-bound validation from `low >= high` to `low > high`, and use this complete property implementation in `GeneDef`:

```python
    @property
    def is_fixed(self) -> bool:
        """Return whether this gene is a fixed numeric value."""
        return self.kind in ("float", "int") and self.low == self.high
```

Use this exact validation message for reversed numeric bounds:

```python
            if self.low > self.high:
                raise ConfigurationError(f"GeneDef('{self.name}') requires low <= high.")
```

Add these properties to `GeneSpace` after `kinds`:

```python
    @property
    def fixed_indices(self) -> list[int]:
        """Return indices of fixed numeric genes."""
        return [index for index, gene in enumerate(self._genes) if gene.is_fixed]

    @property
    def variable_indices(self) -> list[int]:
        """Return indices of genes that participate in variation."""
        return [index for index, gene in enumerate(self._genes) if not gene.is_fixed]

    @property
    def fixed_count(self) -> int:
        """Return the number of fixed numeric genes."""
        return len(self.fixed_indices)

    @property
    def variable_count(self) -> int:
        """Return the number of variable genes."""
        return len(self.variable_indices)
```

- [ ] **Step 4: Run the GeneSpace tests**

Run:

```powershell
py -m pytest tests/unit/test_gene_space.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add evocore/gene_space.py tests/unit/test_gene_space.py
git commit -m "feat: support fixed numeric gene metadata"
```

Expected: commit succeeds.

## Task 2: Rust Fixed Bounds In Initialization And Reproduction

**Files:**
- Modify: `src/reproduce.rs`
- Test: `tests/unit/test_reproduce_rust.py`

- [ ] **Step 1: Add failing PyO3 fixed-bound tests**

Add this test to `TestInitPopulation` in `tests/unit/test_reproduce_rust.py`:

```python
    def test_fixed_float_and_int_bounds_initialize_to_fixed_values(self):
        bounds = [(1.25, 1.25), (2.0, 2.0), (-5.0, 5.0)]
        pop = init_population(bounds, ["float", "int", "float"], 12, 42)

        assert len(pop) == 12
        assert all(ind[0] == 1.25 for ind in pop)
        assert all(ind[1] == 2.0 for ind in pop)
        assert any(ind[2] != pop[0][2] for ind in pop[1:])
```

Add this test to `TestReproducePopulation`:

```python
    def test_fixed_numeric_bounds_survive_uniform_mutation_and_crossover(self):
        bounds = [(1.25, 1.25), (2.0, 2.0), (-5.0, 5.0)]
        kinds = ["float", "int", "float"]
        pop = init_population(bounds, kinds, 20, 42)
        new_pop = reproduce_population(
            pop,
            [float(i) for i in range(20)],
            "sbx",
            1.0,
            2.0,
            0.5,
            "uniform",
            1.0,
            [0.0, 0.0, 2.0],
            bounds,
            kinds,
            "tournament",
            3,
            20,
            42,
            0,
        )

        assert len(new_pop) == 20
        assert all(ind[0] == 1.25 for ind in new_pop)
        assert all(ind[1] == 2.0 for ind in new_pop)
        assert all(-5.0 <= ind[2] <= 5.0 for ind in new_pop)
```

- [ ] **Step 2: Run the new PyO3 tests and verify they fail**

Run:

```powershell
py -m pytest tests/unit/test_reproduce_rust.py::TestInitPopulation::test_fixed_float_and_int_bounds_initialize_to_fixed_values tests/unit/test_reproduce_rust.py::TestReproducePopulation::test_fixed_numeric_bounds_survive_uniform_mutation_and_crossover -q
```

Expected: FAIL. The float initialization or uniform mutation path panics or raises because Rust samples an empty `low..high` range.

- [ ] **Step 3: Update Rust initialization**

In `src/reproduce.rs`, update the `GeneKind::Float` branch inside `init_population`:

```rust
                        GeneKind::Float => {
                            if low == high {
                                low
                            } else {
                                rng.gen_range(low..high)
                            }
                        }
```

Leave the int branch as inclusive `low..=high`; fixed int bounds already work there.

- [ ] **Step 4: Update Rust uniform mutation**

In `src/reproduce.rs`, update the `(MutationType::Uniform, GeneKind::Float)` branch inside `apply_mutation`:

```rust
                (MutationType::Uniform, GeneKind::Float) => {
                    if rng.gen::<f64>() < config.mutation_prob {
                        let (low, high) = config.gene_bounds[idx];
                        if low == high {
                            low
                        } else {
                            rng.gen_range(low..high)
                        }
                    } else {
                        gene
                    }
                }
```

- [ ] **Step 5: Add Rust unit tests**

In `src/reproduce.rs`, add this test inside `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn test_init_population_fixed_float_bounds_return_fixed_value() {
        let bounds = vec![(1.25_f64, 1.25), (-5.0, 5.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Float];
        let pop = init_population(&bounds, &kinds, 12, 42);

        assert!(pop.iter().all(|ind| ind[0] == 1.25));
        assert!(pop.iter().all(|ind| ind[1] >= -5.0 && ind[1] < 5.0));
    }

    #[test]
    fn test_reproduce_fixed_numeric_bounds_are_preserved() {
        let bounds = vec![(1.25_f64, 1.25), (2.0, 2.0), (-5.0, 5.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Int, GeneKind::Float];
        let pop = init_population(&bounds, &kinds, 20, 42);
        let fitnesses = (0..20).map(|value| value as f64).collect::<Vec<_>>();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::Sbx,
            crossover_prob: 1.0,
            crossover_eta: 2.0,
            crossover_alpha: 0.5,
            mutation_type: MutationType::Uniform,
            mutation_prob: 1.0,
            mutation_sigmas: vec![0.0, 0.0, 2.0],
            gene_bounds: bounds.clone(),
            gene_kinds: kinds.clone(),
            selection_type: SelectionType::Tournament,
            tournament_size: 3,
            population_size: 20,
        };

        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);

        assert!(new_pop.iter().all(|ind| ind[0] == 1.25));
        assert!(new_pop.iter().all(|ind| ind[1] == 2.0));
        assert!(new_pop.iter().all(|ind| ind[2] >= -5.0 && ind[2] <= 5.0));
    }
```

- [ ] **Step 6: Run Rust and PyO3 fixed-bound tests**

Run:

```powershell
cargo test reproduce::tests::test_init_population_fixed_float_bounds_return_fixed_value reproduce::tests::test_reproduce_fixed_numeric_bounds_are_preserved
py -m pytest tests/unit/test_reproduce_rust.py::TestInitPopulation::test_fixed_float_and_int_bounds_initialize_to_fixed_values tests/unit/test_reproduce_rust.py::TestReproducePopulation::test_fixed_numeric_bounds_survive_uniform_mutation_and_crossover -q
```

Expected: PASS for both commands.

- [ ] **Step 7: Commit Task 2**

Run:

```powershell
git add src/reproduce.rs tests/unit/test_reproduce_rust.py
git commit -m "fix: preserve fixed numeric genes in rust reproduction"
```

Expected: commit succeeds.

## Task 3: Python Operator And CMA-ES Fixed-Gene Guard

**Files:**
- Modify: `evocore/cmaes.py`
- Test: `tests/unit/test_operators.py`
- Test: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Add fixed-param decode test**

Append this test to `tests/unit/test_operators.py`:

```python
def test_decode_individual_preserves_fixed_numeric_params():
    space = GeneSpace(
        [
            GeneDef("signal_mode", "int", 2, 2),
            GeneDef("threshold", "float", 0.5, 0.5),
            GeneDef("period", "int", 5, 20),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")

    individual = ops.decode_individual([2.0, 0.5, 12.0])

    assert individual.genes == [2, 0.5, 12]
    assert individual.params == {
        "signal_mode": 2,
        "threshold": 0.5,
        "period": 12,
    }
```

- [ ] **Step 2: Add CMA-ES guard test**

Append this test to `tests/unit/test_cmaes_engine.py`:

```python
def test_cmaes_rejects_fixed_numeric_genes_until_reconstruction_is_supported():
    space = GeneSpace([GeneDef("signal_mode", "int", 2, 2), GeneDef("x", "float", -1.0, 1.0)])

    with pytest.raises(ConfigurationError) as exc:
        CMAESEngine(space)

    message = str(exc.value)
    assert "fixed numeric genes" in message
    assert "GAEngine" in message
```

- [ ] **Step 3: Run tests and verify CMA-ES guard fails**

Run:

```powershell
py -m pytest tests/unit/test_operators.py::test_decode_individual_preserves_fixed_numeric_params tests/unit/test_cmaes_engine.py::test_cmaes_rejects_fixed_numeric_genes_until_reconstruction_is_supported -q
```

Expected: the operator test passes after Task 1, and the CMA-ES guard test fails because `CMAESEngine` accepts the fixed numeric gene.

- [ ] **Step 4: Implement CMA-ES guard**

In `evocore/cmaes.py`, add this validation after the bool-gene check and before the process-parallel check:

```python
        if gene_space.fixed_count:
            raise ConfigurationError(
                "CMAESEngine does not support fixed numeric genes yet. "
                "Use GAEngine for full-genome fixed genes, or remove fixed genes from the CMA-ES GeneSpace."
            )
```

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
py -m pytest tests/unit/test_operators.py::test_decode_individual_preserves_fixed_numeric_params tests/unit/test_cmaes_engine.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add evocore/cmaes.py tests/unit/test_operators.py tests/unit/test_cmaes_engine.py
git commit -m "feat: guard cmaes fixed numeric genes"
```

Expected: commit succeeds.

## Task 4: GA Fixed Genes In Full Genome

**Files:**
- Modify: `evocore/ga.py`
- Test: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add GA fixed-gene preservation test**

Append this test to `tests/unit/test_ga_engine.py`:

```python
def test_ga_run_preserves_fixed_numeric_genes_in_full_genome():
    space = GeneSpace(
        [
            GeneDef("signal_mode", "int", 2, 2),
            GeneDef("threshold", "float", 0.5, 0.5),
            GeneDef("period", "int", 5, 20),
            GeneDef("x", "float", -1.0, 1.0),
        ]
    )
    engine = GAEngine(
        space,
        population_size=20,
        generations=5,
        crossover_prob=1.0,
        mutation_prob=1.0,
        mutation="uniform",
        seed=42,
    )

    result = engine.run(lambda ind: -abs(ind.params["period"] - 12) - ind.genes[3] ** 2)

    assert result.n_evaluations == 20 + (19 * 5)
    for individual in result.final_population:
        assert individual.genes[0] == 2
        assert individual.genes[1] == 0.5
        assert individual.params["signal_mode"] == 2
        assert individual.params["threshold"] == 0.5
```

- [ ] **Step 2: Run the fixed-gene GA test**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_ga_run_preserves_fixed_numeric_genes_in_full_genome -q
```

Expected: PASS after Tasks 1 and 2. If it fails, inspect `OperatorSet.sigma_abs_list()` and Rust clamping. Fixed spans should produce zero sigma, and clamping should restore equal bounds.

- [ ] **Step 3: Commit Task 4**

Run:

```powershell
git add tests/unit/test_ga_engine.py
git commit -m "test: cover ga fixed numeric genomes"
```

Expected: commit succeeds.

## Task 5: GA Run Diagnostics And Constructor Plumbing

**Files:**
- Modify: `evocore/ga.py`
- Test: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add constructor and default diagnostics test**

Append this test to `tests/unit/test_ga_engine.py`:

```python
def test_ga_run_reports_default_generation_stop_diagnostics():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=2, seed=42)

    result = engine.run(module_sphere)

    assert result.max_evaluations is None
    assert result.stop_reason == "generations"
    assert result.budget_reached is False
    assert result.stopped_early is False
```

- [ ] **Step 2: Run the new diagnostics test and verify it fails**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_ga_run_reports_default_generation_stop_diagnostics -q
```

Expected: FAIL because `RunResult` does not expose `max_evaluations`, `stop_reason`, or `budget_reached`.

- [ ] **Step 3: Add RunResult fields and GAEngine constructor parameter**

In `evocore/ga.py`, update the imports:

```python
from typing import Literal
```

Add this type alias near the top of the file after `logger`:

```python
StopReason = Literal["generations", "max_evaluations", "callback"]
```

Update `RunResult`:

```python
@dataclass
class RunResult:
    """Store the outcome of one optimization run."""

    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
    n_evaluations: int
    elite_history: list[Individual]
    diversity_history: list[list[float]]
    seed: int
    stopped_early: bool
    max_evaluations: int | None = None
    stop_reason: StopReason = "generations"
    budget_reached: bool = False
```

Add `max_evaluations` to `GAEngine.__init__` before `track_diversity`:

```python
        max_evaluations: int | None = None,
```

Add validation after the `generations` validation:

```python
        if max_evaluations is not None and max_evaluations <= 0:
            raise ConfigurationError("max_evaluations must be positive when provided.")
```

Store it:

```python
        self.max_evaluations = max_evaluations
```

Pass it through `_copy_with_seed`:

```python
            max_evaluations=self.max_evaluations,
```

When building `RunResult` in `_run_from_population`, set:

```python
            max_evaluations=self.max_evaluations,
            stop_reason=stop_reason,
            budget_reached=(
                self.max_evaluations is not None and n_evaluations >= self.max_evaluations
            ),
```

Before this constructor call, initialize `stop_reason: StopReason = "generations"` near `stopped_early = False`.

- [ ] **Step 4: Set callback stop reasons**

In `_run_from_population`, whenever callback logic sets `stopped_early = True`, also set:

```python
                stop_reason = "callback"
```

There are two callback-stop locations: after `on_generation_start` and after `on_generation_end`.

- [ ] **Step 5: Run diagnostics and existing GA tests**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_ga_run_reports_default_generation_stop_diagnostics tests/unit/test_ga_engine.py::test_ga_run_returns_result_with_logbook_length tests/unit/test_ga_engine.py::test_run_multiple_sequential_returns_sorted_runs -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat: add ga run stop diagnostics"
```

Expected: commit succeeds.

## Task 6: Exact `max_evaluations` Enforcement

**Files:**
- Modify: `evocore/ga.py`
- Test: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add exact-budget tests**

Append these tests to `tests/unit/test_ga_engine.py`:

```python
def test_ga_max_evaluations_can_stop_during_initial_population():
    calls = []
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=5,
        max_evaluations=4,
        seed=42,
    )

    result = engine.run(lambda ind: calls.append(tuple(ind.genes)) or module_sphere(ind))

    assert len(calls) == 4
    assert result.n_evaluations == 4
    assert len(result.final_population) == 4
    assert all(ind.fitness_valid for ind in result.final_population)
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert result.stopped_early is True


def test_ga_max_evaluations_stops_exactly_after_partial_generation():
    calls = []
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        generations=5,
        elitism=2,
        max_evaluations=11,
        seed=42,
    )

    result = engine.run(lambda ind: calls.append(tuple(ind.genes)) or module_sphere(ind))

    assert len(calls) == 11
    assert result.n_evaluations == 11
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert result.stopped_early is True
    assert all(ind.fitness_valid for ind in result.final_population)
    assert len(result.final_population) <= engine.population_size


def test_ga_rejects_non_positive_max_evaluations():
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), max_evaluations=0)
```

- [ ] **Step 2: Run exact-budget tests and verify they fail**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_ga_max_evaluations_can_stop_during_initial_population tests/unit/test_ga_engine.py::test_ga_max_evaluations_stops_exactly_after_partial_generation tests/unit/test_ga_engine.py::test_ga_rejects_non_positive_max_evaluations -q
```

Expected: FAIL until `max_evaluations` changes the evaluation loop.

- [ ] **Step 3: Add budget helper methods to `GAEngine`**

In `evocore/ga.py`, add these methods before `_evaluate_all`:

```python
    def _remaining_evaluations(self, n_evaluations: int) -> int | None:
        if self.max_evaluations is None:
            return None
        return max(self.max_evaluations - n_evaluations, 0)

    @staticmethod
    def _fitnesses_for_selection(individuals: Sequence[Individual]) -> list[float]:
        return [
            float(ind.fitness) if ind.fitness is not None and ind.fitness_valid else float("-inf")
            for ind in individuals
        ]

    def _evaluate_with_budget(
        self,
        individuals: Sequence[Individual],
        fitness_fn: Callable,
        gen: int,
        n_evaluations: int,
    ) -> tuple[list[Individual], list[float], int, int]:
        working = list(individuals)
        pending = [ind for ind in working if not ind.fitness_valid]
        remaining = self._remaining_evaluations(n_evaluations)
        if remaining == 0:
            evaluated = [ind for ind in working if ind.fitness_valid]
            return evaluated, self._fitnesses_for_selection(evaluated), 0, 0

        to_evaluate = pending if remaining is None else pending[:remaining]
        nan_count = 0
        if to_evaluate:
            _, nan_count = self._evaluate_all(to_evaluate, fitness_fn, gen=gen)

        evaluated_now = len(to_evaluate)
        evaluated = [ind for ind in working if ind.fitness_valid]
        return evaluated, self._fitnesses_for_selection(evaluated), evaluated_now, nan_count
```

- [ ] **Step 4: Use the budget helper for initial evaluation**

In `_run_from_population`, replace the initial evaluation block:

```python
        initial_pending = sum(1 for ind in working_population if not ind.fitness_valid)
        fitnesses, _ = self._evaluate_all(working_population, fitness_fn, gen=start_generation - 1)
        n_evaluations = initial_pending
```

with:

```python
        working_population, fitnesses, evaluated_now, _ = self._evaluate_with_budget(
            working_population,
            fitness_fn,
            gen=start_generation - 1,
            n_evaluations=0,
        )
        n_evaluations = evaluated_now
```

After `stopped_early = False` and `stop_reason: StopReason = "generations"`, add:

```python
        if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
            stopped_early = True
            stop_reason = "max_evaluations"
```

Update the generation loop header:

```python
        for gen in range(start_generation, self.generations):
            if stop_reason == "max_evaluations":
                break
```

- [ ] **Step 5: Use the budget helper for generation evaluation**

In `_run_from_population`, replace:

```python
            eval_before = n_evaluations
            evaluated_now = sum(1 for ind in working_population if not ind.fitness_valid)
            fitnesses, nan_count = self._evaluate_all(working_population, fitness_fn, gen=gen)
            n_evaluations += evaluated_now
```

with:

```python
            eval_before = n_evaluations
            working_population, fitnesses, evaluated_now, nan_count = self._evaluate_with_budget(
                working_population,
                fitness_fn,
                gen=gen,
                n_evaluations=n_evaluations,
            )
            n_evaluations += evaluated_now
            if self.max_evaluations is not None and n_evaluations >= self.max_evaluations:
                stopped_early = True
                stop_reason = "max_evaluations"
```

Keep `pop_obj = Population(working_population)` after this replacement so the log and final population use only evaluated individuals.

- [ ] **Step 6: Protect against empty evaluated populations**

Before `final_population = Population(working_population)`, add:

```python
        if not working_population:
            raise FitnessError("GA run produced no evaluated individuals.")
```

This should be unreachable with validated positive `max_evaluations`, but it makes failures clear if future changes break the invariant.

- [ ] **Step 7: Run exact-budget tests**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_ga_max_evaluations_can_stop_during_initial_population tests/unit/test_ga_engine.py::test_ga_max_evaluations_stops_exactly_after_partial_generation tests/unit/test_ga_engine.py::test_ga_rejects_non_positive_max_evaluations -q
```

Expected: PASS.

- [ ] **Step 8: Run focused GA regression tests**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 6**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat: enforce exact ga evaluation budgets"
```

Expected: commit succeeds.

## Task 7: `run_multiple` Budget Propagation

**Files:**
- Modify: `evocore/ga.py`
- Test: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add per-run cap test**

Append this module-level helper and test to `tests/unit/test_ga_engine.py`:

```python
def module_counted_sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_run_multiple_applies_max_evaluations_per_child_run():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        generations=5,
        max_evaluations=7,
        seed=42,
    )

    result = engine.run_multiple(module_counted_sphere, n_runs=3, run_parallel=False)

    assert result.n_runs == 3
    assert [run.n_evaluations for run in result.all_runs] == [7, 7, 7]
    assert all(run.max_evaluations == 7 for run in result.all_runs)
    assert all(run.stop_reason == "max_evaluations" for run in result.all_runs)
    assert all(run.budget_reached is True for run in result.all_runs)
```

- [ ] **Step 2: Run the test and verify it fails if `_copy_with_seed` missed the cap**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_run_multiple_applies_max_evaluations_per_child_run -q
```

Expected: PASS if Task 5 already added `max_evaluations=self.max_evaluations` to `_copy_with_seed`; otherwise FAIL with child runs evaluating more than 7 candidates.

- [ ] **Step 3: Fix `_copy_with_seed` if needed**

If the test failed, ensure this argument is present in the `GAEngine(...)` call inside `_copy_with_seed`:

```python
            max_evaluations=self.max_evaluations,
```

- [ ] **Step 4: Run the test again**

Run:

```powershell
py -m pytest tests/unit/test_ga_engine.py::test_run_multiple_applies_max_evaluations_per_child_run -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "test: cover ga multi-run budget caps"
```

Expected: commit succeeds.

## Task 8: Public Typing And Package Regression Sweep

**Files:**
- Modify: `evocore/_core.pyi` only if PyO3 signatures change; this plan does not require signature changes.
- Test: existing unit, property, integration, and Rust suites.

- [ ] **Step 1: Run focused Python tests**

Run:

```powershell
py -m pytest tests/unit/test_gene_space.py tests/unit/test_operators.py tests/unit/test_reproduce_rust.py tests/unit/test_cmaes_engine.py tests/unit/test_ga_engine.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Rust tests**

Run:

```powershell
cargo test
```

Expected: PASS.

- [ ] **Step 3: Run broader Python tests**

Run:

```powershell
py -m pytest tests/unit tests/property tests/integration -q
```

Expected: PASS.

- [ ] **Step 4: Run formatting or lint commands if configured locally**

Run:

```powershell
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

Expected: PASS. If `cargo clippy` reports pre-existing warnings unrelated to this change, capture the exact warning output in the final implementation note and still keep the code changes minimal.

- [ ] **Step 5: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only intended source and test files are modified. If generated caches or benchmark artifacts appear, leave them unstaged.

- [ ] **Step 6: Commit final verification adjustments**

If Step 1 through Step 5 required any source, test, or formatting changes after Task 7, commit them:

```powershell
git add evocore src tests
git commit -m "chore: verify ga fixed genes and budgets"
```

Expected: commit succeeds when there are changes. If there are no changes, skip this commit and record that no final adjustment commit was needed.

## Final Verification Checklist

- [ ] Fixed numeric `GeneDef` accepts equal bounds and rejects reversed bounds.
- [ ] Fixed genes appear in full `Individual.genes` and `Individual.params`.
- [ ] Rust initialization and reproduction preserve fixed numeric slots.
- [ ] CMA-ES rejects fixed numeric genes clearly.
- [ ] `GAEngine(max_evaluations=N)` makes exactly `N` user fitness calls.
- [ ] `RunResult.max_evaluations`, `stop_reason`, and `budget_reached` are populated.
- [ ] Existing tests for callbacks, resume, `run_multiple`, mixed numeric genes, and Rust reproduction pass.

