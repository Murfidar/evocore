# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Stable export fields are:

- `total_candidates_proposed`
- `unique_candidate_hashes`
- `unique_candidate_count`
- `candidates_screened`
- `candidates_partial_evaluated`
- `candidates_full_evaluated`
- `promoted_by_rung`
- `eliminated_by_rung`
- `cost_by_rung`

`unique_candidate_hashes` is exported as a sorted list and
`unique_candidate_count` is derived from that set. Use `to_dict()` for a JSON-safe
payload or `to_json()` for deterministic JSON with sorted keys.

Cached evaluation records are state-eligible and count toward full-budget accounting,
while remaining visible through `TellResult.cached_count` and event history rows with
`confidence="cached"`.
