# EvoCore GA Quality Gap Root Cause Notes

Date: 2026-05-07

## Benchmark Context

Latest pre-fix Trading-Algo snapshot report:

- Path: `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/latest/ga_benchmark_report.md`
- Config: `mode=snapshot`, `population_size=80`, `evaluation_budget=500`, `seed=42`, `gene_profile=default`, `strategy_family=general`, `elitism_size=8`, `tournament_size=4`
- Recommendation: `keep_deap`
- DEAP: 51.579346s, 500 evals, 9.694 eval/s, best_fitness=0.582811, peak_memory=19.171 MB
- EvoCore: 14.876216s, 500 evals, 33.611 eval/s, best_fitness=0.540123, peak_memory=1.200 MB

## Root Causes

1. Numeric crossover semantics were not DEAP-parity.
   Trading-Algo's DEAP path uses `tools.cxUniform(indpb=0.5)`, which swaps attributes between parents independently. EvoCore's benchmark adapter used `crossover="sbx"`, which creates averaged real-valued children. SBX is a strong continuous-space operator, but this chromosome is mixed float, integer, and categorical-by-int. Averaging and rounding categorical genes changes the search behavior and pushed EvoCore toward lower-quality, one-sided candidates.

2. Mutation probability had one fewer gate than DEAP.
   The DEAP benchmark applies an outer per-individual mutation decision and then a per-gene mutation probability inside the mutation operator. EvoCore applied the per-gene probability to every offspring. With `mutation_prob=0.20`, that made EvoCore mutate much more aggressively than the benchmark baseline.

3. Tournament selection had stronger selection pressure than DEAP.
   DEAP's `selTournament` samples tournament aspirants with replacement. EvoCore sampled without replacement and treated `tournament_size >= population_size` as a full-population tournament. That reduces sampling noise and increases selection pressure, which is consistent with the observed premature convergence and one-sided candidate behavior.

4. Small-budget overhead came from unused initial individuals.
   `GAEngine(max_evaluations=N)` capped evaluations correctly, but still initialized a full population even when `N < population_size`. That inflated isolated small-budget setup cost without affecting result quality.

5. The benchmark DEAP side was not isolated from module-level RNG state.
   DEAP's `cxUniform` and `selTournament` use Python's module-level `random` functions. The benchmark already used a local `random.Random(seed)` for initialization and mutation, but did not seed and restore the module RNG around DEAP execution. Repeated or prior runs could therefore perturb the DEAP trajectory.

## Changes Made

EvoCore:

- Allowed `crossover="uniform"` for float/int `GeneSpace`, reusing the Rust uniform allele-swap crossover for mixed numeric chromosomes.
- Added `GAEngine(mutation_individual_prob=1.0)` as a backwards-compatible DEAP-parity mutation gate. Existing users keep old behavior by default.
- Added the same optional `mutation_individual_prob` to the PyO3 `reproduce_population` boundary.
- Changed tournament selection to sample aspirants with replacement, matching DEAP.
- Capped initial population construction to `min(population_size, max_evaluations)` when a hard budget is set.

Trading-Algo benchmark adapter:

- Switched the EvoCore adapter to `crossover="uniform"`.
- Passed `mutation_individual_prob=config.mutation_prob` so EvoCore uses the same outer mutation gate as DEAP.
- Seeded and restored Python's module-level RNG around the DEAP benchmark run.

## After Evidence

Single seeded standard snapshot report:

- Path: `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-seeded/ga_benchmark_report.md`
- Config includes `elitism_size=8`; this matches the pre-fix baseline report.
- Recommendation: `migrate_to_evocore`
- DEAP: 65.049196s, 500 evals, 7.686 eval/s, best_fitness=0.569617, peak_memory=18.864 MB
- EvoCore: 18.702458s, 500 evals, 26.734 eval/s, best_fitness=0.598306, peak_memory=1.778 MB

Fresh verification report with explicit `--elitism-size 8`:

- Path: `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-verify-elitism8/ga_benchmark_report.md`
- Config includes `elitism_size=8`; an accidental rerun with the CLI default `elitism_size=2` is a different benchmark variant and recommended `keep_deap`.
- Recommendation: `migrate_to_evocore`
- DEAP: 67.006043s, 500 evals, 7.462 eval/s, best_fitness=0.569617, peak_memory=18.863 MB
- EvoCore: 18.449907s, 500 evals, 27.100 eval/s, best_fitness=0.598306, peak_memory=1.773 MB

After aligning the Trading-Algo benchmark CLI default to the standard report's `elitism_size=8`, the exact standard CLI invocation without `--elitism-size` also recommends migration:

- Path: `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-verify-default/ga_benchmark_report.md`
- Recommendation: `migrate_to_evocore`
- DEAP: 155.895431s, 500 evals, 3.207 eval/s, best_fitness=0.569617, peak_memory=18.865 MB
- EvoCore: 42.741957s, 500 evals, 11.698 eval/s, best_fitness=0.598306, peak_memory=1.768 MB

Three separate-process repeats with the same seeded standard snapshot config:

- Paths:
  - `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-seeded/ga_benchmark_report.json`
  - `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-seeded-r2/ga_benchmark_report.json`
  - `D:/Kerja/pribadi/Trading-Algo-Scalper-Gold/data/benchmarks/ga/evocore-parity-seeded-r3/ga_benchmark_report.json`
- Recommendations: all `migrate_to_evocore`
- DEAP medians: 67.146406s, best_fitness=0.569617, peak_memory=18.863812 MB, 7.446 eval/s
- EvoCore medians: 18.664268s, best_fitness=0.598306, peak_memory=1.775434 MB, 26.789 eval/s
- Median wall-time speedup: 3.60x

An in-process three-repeat report was intentionally not used for the memory decision because the second and third DEAP repeats reused warmed imports/caches and no longer measured the same memory surface as fresh CLI runs.

## Research And Reference Notes

- DEAP `cxUniform` swaps attributes independently by `indpb`; `selTournament` chooses aspirants via repeated random selection with replacement.
- Deb and Agrawal's SBX is designed for continuous search spaces. It remains appropriate for real-coded continuous optimization, but it was the wrong parity choice for this Trading-Algo mixed numeric/categorical chromosome.
- Premature convergence literature commonly points to loss of diversity and excessive selection pressure as causes. The fix here keeps the implementation conservative: match DEAP's stochastic pressure first before adding heavier diversity mechanisms such as crowding, niching, random immigrants, or adaptive mutation.
