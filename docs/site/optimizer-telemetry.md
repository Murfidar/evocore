# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Stable export fields are:

- `total_candidates_proposed`
- `unique_candidate_hashes`
- `unique_candidate_count`
- `candidates_screened`
- `candidates_partial_evaluated`
- `candidates_full_evaluated`
- `candidates_cached`
- `promoted_by_stage`
- `eliminated_by_stage`
- `cost_by_stage`

`unique_candidate_hashes` is exported as a sorted list and
`unique_candidate_count` is derived from that set. Use `to_dict()` for a JSON-safe
payload or `to_json()` for deterministic JSON with sorted keys.

Cached evaluation records are state-eligible but do not count as fresh full evaluations.
They are visible through `OptimizationTelemetry.candidates_cached`,
`UpdateResult.cached_count`, and event history rows with `confidence="cached"`.

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
