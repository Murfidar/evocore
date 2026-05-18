# GA Benchmark Parity

EvoCore 0.6.1 includes GA parity fixes driven by the
Trading-Algo-Scalper-Gold snapshot benchmark.

## Root Cause

The quality gap was not caused by evaluation-budget accounting. Both backends used 500
fitness calls. The main differences were operator semantics:

- Trading-Algo's DEAP path used uniform crossover, while the EvoCore adapter used SBX.
  SBX is useful for continuous real-coded search, but it averages parent alleles and is a poor
  semantic match for mixed float, integer, and categorical-by-integer chromosomes.
- DEAP applied an outer per-Solution mutation decision before its per-gene mutation decision.
  EvoCore only had the per-gene probability, so the Trading benchmark over-mutated offspring.
- DEAP tournament selection samples aspirants with replacement. EvoCore sampled without
  replacement, increasing selection pressure and encouraging premature convergence.
- The DEAP benchmark used module-level `random` through DEAP operators, so benchmark runs were
  not isolated from prior Python RNG state.

## Standard Snapshot Evidence

Fresh standard snapshot command:

```bash
python -m research_engine.main benchmark-ga \
  --mode snapshot \
  --SolutionSet-size 80 \
  --evaluation-budget 500 \
  --seed 42 \
  --gene-profile default \
  --strategy-family general \
  --output-dir data/benchmarks/ga/evocore-parity-verify-default
```

Result:

| Backend | Seconds | Evaluations | Eval/s | Best Fitness | Peak MB | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEAP | 155.895431 | 500 | 3.207 | 0.569617 | 18.865 | 0 |
| EvoCore | 42.741957 | 500 | 11.698 | 0.598306 | 1.768 | 0 |

The report recommended `migrate_to_evocore`.

Three separate-process repeats with the same seeded snapshot config had median EvoCore speedup
of 3.60x, median best fitness of `0.598306 >= 0.569617`, and median peak memory of
`1.775 MB <= 18.864 MB`.
