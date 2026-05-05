# evocore

evocore is a Rust-native Python optimization library for Genetic Algorithms and CMA-ES.

Python owns the ergonomic API. Rust owns the hot paths exposed through `evocore._core`.

## Current Scope

- Genetic Algorithms over float, integer, binary, and mixed numeric gene spaces.
- CMA-ES over float and integer gene spaces.
- Deterministic reproducibility from explicit seed derivation.
- Optional thread and process parallelism for supported engines.
- Callbacks, checkpoints, logbooks, and metrics.
