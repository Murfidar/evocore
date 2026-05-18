# EvoCore Result History Telemetry Contract Design

**Date:** 2026-05-14
**Status:** Draft approved for specification
**Scope:** Stable public contracts for optimization results, histories, telemetry, JSON export, pandas export, and reproducibility metadata

## Summary

EvoCore should stabilize optimizer outputs around a layered result contract. `RunResult`
continues to represent the final run outcome, `Logbook` continues to represent
generation-level summaries, and a new append-only `EventHistory` represents ask/tell audit
events. `OptimizationTelemetry` becomes JSON-stable through explicit export helpers.

The default export path should be deterministic where practical. Wall-clock timing and
other runtime observations remain available, but callers opt into them with
`include_runtime=True` so snapshot tests and cross-seed comparisons stay stable.

This slice is general-purpose. It does not add trading-specific metrics, external
benchmark comparisons, multi-objective result objects, or resume-from-result semantics.

## Product Direction

Downstream users should be able to inspect, serialize, audit, and compare optimizer runs
without reaching into private engine state. The core user model becomes:

```python
result = engine.run(evaluator)
payload = result.to_dict()
json_text = result.to_json()
events = result.history.to_dataframe()
```

Manual ask/tell users should be able to rely on the same public event schema as
policy-driven `run(...)` users. Generation-oriented runs may keep using `Logbook` while
still exporting through `RunResult`.

## Goals

- Keep `RunResult` as the stable public envelope for one completed optimization run.
- Add append-only event history for proposal, evaluation, and generation observations.
- Preserve raw user scores and expose direction-aware comparison scores explicitly.
- Make `OptimizationTelemetry` JSON-safe with stable field names and deterministic order.
- Provide deterministic `to_dict()` and `to_json()` exports by default.
- Keep wall-clock timing available through explicit runtime export options.
- Provide optional pandas exports without making pandas a hard dependency.
- Preserve positional dataclass compatibility where practical.
- Add reproducibility metadata for version, engine, seed, direction, gene-space signature,
  and optimizer configuration.

## Non-Goals

- Do not implement `from_dict()` or `from_json()` until checkpoint or reload semantics are
  deliberately designed.
- Do not promise resume-from-result.
- Do not add domain-specific metrics or objective logic.
- Do not add external benchmark or comparison workflows.
- Do not make pandas a required dependency.
- Do not redesign checkpoint pickle compatibility beyond avoiding unnecessary breakage.
- Do not add multi-objective or Pareto result contracts in this slice.

## Recommended Architecture

Use a layered contract:

- `RunResult`: final run envelope and export entrypoint.
- `MultiRunResult`: aggregate envelope for repeated runs.
- `Logbook`: ordered generation summaries.
- `EventHistory`: append-only event observations.
- `OptimizationTelemetry`: stable aggregate accounting.
- `ReproducibilityMetadata`: deterministic configuration and environment identity.

This keeps generation summaries and candidate lifecycle events separate. `Logbook` remains
compact and backward-compatible. `EventHistory` carries the richer row-oriented audit data
needed for ask/tell replay, JSON export, and tabular analysis.

## Public API Shape

### RunResult

Keep existing positional fields intact:

- `best_individual`
- `best_fitness`
- `final_population`
- `logbook`
- `wall_time_seconds`
- `n_evaluations`
- `elite_history`
- `diversity_history`
- `seed`
- `stopped_early`
- `max_evaluations`
- `stop_reason`
- `budget_reached`
- `telemetry`

Add keyword-default fields after the current fields:

- `direction: Direction = "maximize"`
- `engine_type: str = ""`
- `best_candidate_id: str | None = None`
- `best_score: float | None = None`
- `history: EventHistory = field(default_factory=EventHistory)`
- `reproducibility: ReproducibilityMetadata | None = None`
- `metadata: dict[str, Any] = field(default_factory=dict)`

`best_fitness` remains the raw public best score for compatibility. `best_score` is the
same raw score when known and should exist to make the result envelope read naturally next
to ask/tell state summaries.

Add helpers:

```python
def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]: ...
def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str: ...
def to_dataframe(self): ...
```

`to_dataframe()` should return `history.to_dataframe()` when history has events. When
history is empty, it should fall back to `logbook.to_dataframe()` so legacy generation
runs remain useful.

### MultiRunResult

Keep existing fields:

- `best`
- `all_runs`
- `n_runs`
- `wall_time_seconds`

Add keyword-default fields:

- `direction: Direction = "maximize"`
- `metadata: dict[str, Any] = field(default_factory=dict)`

Add helpers:

```python
def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]: ...
def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str: ...
def to_dataframe(self): ...
```

The aggregate export should include run summaries in existing direction-aware order. It
should include aggregate runtime only when requested.

### EventHistory And EventRecord

Add these to `evocore.stats` unless implementation finds a cleaner split that does not
create import cycles:

```python
@dataclass(frozen=True)
class EventRecord:
    event_index: int
    event_type: Literal["ask", "tell", "generation"]
    batch_id: str | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    generation: int | None = None
    rung: str | None = None
    confidence: EvaluationConfidence | None = None
    raw_score: float | None = None
    comparison_score: float | None = None
    cost: float = 0.0
    status: CandidateStatus | None = None
    origin: CandidateOrigin | None = None
    parents: tuple[str, ...] = ()
    genes: tuple[GeneValue, ...] = ()
    params: dict[str, GeneValue] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

`EventHistory` should provide:

```python
def append(self, event: EventRecord) -> None: ...
def __len__(self) -> int: ...
def __iter__(self) -> Iterator[EventRecord]: ...
def __getitem__(self, index: int) -> EventRecord: ...
def to_rows(self) -> list[dict[str, Any]]: ...
def to_dataframe(self): ...
```

The history is append-only from the public API perspective. It does not need mutation or
filter helpers in the first slice.

## Event Semantics

### Ask Events

Every proposed candidate from `ask()` should append one `ask` event with:

- event index
- batch ID
- candidate ID
- candidate hash
- origin
- parents
- generation when available
- genes
- params
- candidate metadata

Ask events should not include scores.

### Tell Events

Every accepted `EvaluationRecord` should append one `tell` event with:

- event index
- batch ID
- candidate ID
- candidate hash
- generation when available
- rung
- confidence
- raw score
- comparison score
- cost
- resulting candidate status
- genes
- params
- metrics
- record metadata

`raw_score` is the evaluator-provided score. `comparison_score` is derived with
`score_for_direction(raw_score, direction)` when a finite score exists, otherwise `None`.
Rejected records may have `raw_score=None` and `comparison_score=None`.

### Generation Events

Generation-oriented GA and CMA loops may append one `generation` event per `LogEntry`.
These events should not require candidate IDs. They exist to give legacy runs a common
history surface without changing the meaning of `Logbook`.

## Serialization Rules

`to_dict()` output must be JSON-safe:

- Use plain dicts, lists, strings, booleans, integers, floats, and `None`.
- Convert tuples to lists.
- Convert sets to sorted lists.
- Convert dataclasses through explicit export helpers, not raw `asdict()` when ordering or
  special float handling matters.
- Preserve raw scores exactly as Python JSON can represent finite floats.
- Avoid storing callables or private engine objects.
- Use stable top-level keys.

Suggested `RunResult.to_dict()` keys:

- `schema_version`
- `engine_type`
- `direction`
- `seed`
- `best`
- `stop`
- `budget`
- `n_evaluations`
- `reproducibility`
- `telemetry`
- `history`
- `logbook`
- `metadata`
- `runtime` only when `include_runtime=True`

`to_json()` should call `json.dumps(..., sort_keys=True)` so key order is stable. The
default `indent` may be `None`; callers can pass an indent for readability.

Runtime fields should live under `runtime`, not mixed into deterministic metadata. The
first runtime fields should be:

- `wall_time_seconds`
- optional run start/end timestamps only if implementation already has them and they are
  excluded from deterministic exports by default

## Reproducibility Metadata

Add a small frozen dataclass:

```python
@dataclass(frozen=True)
class ReproducibilityMetadata:
    evocore_version: str
    engine_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    extension: dict[str, Any] = field(default_factory=dict)
```

`gene_space_signature` should serialize each gene in order:

- name
- kind
- low
- high
- sigma

`gene_space_hash` should be a SHA-256 hash over a canonical JSON payload using compact
separators and sorted keys. Float values should use a stable representation when practical.

`optimizer_config` should include public constructor configuration only. It must not store
callbacks, evaluator callables, process initializers, or other non-serializable objects.

`extension` can remain empty in the first slice unless current build metadata is already
available without adding a new dependency or runtime cost.

## Telemetry Contract

`OptimizationTelemetry` should gain:

```python
def to_dict(self) -> dict[str, Any]: ...
def to_json(self, *, indent: int | None = None) -> str: ...
```

Stable fields:

- `total_candidates_proposed`
- `unique_candidate_hashes`
- `unique_candidate_count`
- `candidates_screened`
- `candidates_partial_evaluated`
- `candidates_full_evaluated`
- `promoted_by_rung`
- `eliminated_by_rung`
- `cost_by_rung`

`unique_candidate_hashes` should export as a sorted list. `unique_candidate_count` should
be derived from the set size to make summaries easier without forcing callers to count.

Cached records remain full-budget and state-eligible in the current lifecycle contract,
but they should remain separately visible in `TellResult.cached_count` and event history
rows through `confidence="cached"`.

## Logbook Contract

`LogEntry` and `Logbook` should remain generation-oriented.

Add:

```python
def LogEntry.to_dict(self) -> dict[str, Any]: ...
def Logbook.to_dict(self) -> list[dict[str, Any]]: ...
def Logbook.to_json(self, *, indent: int | None = None) -> str: ...
```

Keep `to_rows()` as the tabular method used by `to_dataframe()`. `to_dict()` may return
the same row shape as `to_rows()` in the first slice, provided the docs call it a stable
generation-summary export.

Existing positional `LogEntry(...)` construction must continue working.

## Engine Integration

### GA Ask/Tell

`GAEngine.ask()` should append ask events after candidates are created and before returning
them. `GAEngine.tell()` should append tell events for accepted records after validation and
candidate state application.

`GAEngine.run(...)` should attach:

- direction
- engine type
- best candidate ID
- history
- telemetry
- reproducibility metadata

The vNext policy-driven path currently returns an empty `Logbook`; it should return a
populated `EventHistory`.

### CMA Ask/Tell

`CMAESEngine.ask()` and `CMAESEngine.tell()` should mirror GA event recording. CMA should
preserve raw scores in events and store direction-aware comparison scores separately.

The generation-loop `CMAESEngine.run(fitness_fn)` should attach result metadata and may
append generation events from the logbook. It does not need synthetic candidate IDs in this
slice.

### Legacy Generation Loop

The internal GA generation-loop path should keep existing behavior, add result metadata,
and optionally append generation events. It should not be forced to synthesize ask/tell
records.

## Documentation And Changelog

Required docs updates:

- `docs/site/api.md`: include `EventRecord`, `EventHistory`, and
  `ReproducibilityMetadata`.
- `docs/site/ask-tell-engines.md`: explain that ask/tell runs produce append-only event
  history.
- `docs/site/optimizer-telemetry.md`: document stable telemetry fields and JSON export.
- `docs/site/ga.md` and `docs/site/cmaes.md`: show result export examples where useful.
- `CHANGELOG.md`: note the new stable result/history/telemetry export contract.

Docs should be explicit that exports are not checkpoint resume files.

## Testing Strategy

Add tests before implementation.

Required unit tests:

- `RunResult` keeps existing positional construction working.
- `LogEntry` keeps existing positional construction working.
- `OptimizationTelemetry.to_dict()` exports sorted candidate hashes and derived unique
  counts.
- `EventHistory.to_rows()` preserves append order.
- `EventHistory.to_dataframe()` raises a clear pandas install error when pandas is absent.
- `RunResult.to_dict()` excludes runtime fields by default.
- `RunResult.to_dict(include_runtime=True)` includes `wall_time_seconds`.
- `RunResult.to_json()` is deterministic for repeated calls on the same result.
- Minimize runs preserve raw best scores and expose comparison scores separately in events.
- GA vNext `run(...)` returns non-empty history for ask/tell events.
- CMA ask/tell records ask and tell events.
- Reproducibility metadata includes version, engine type, seed, direction, gene-space
  signature, gene-space hash, and serializable optimizer config.
- `MultiRunResult.to_dict()` preserves direction-aware sorted run order.

Property tests should cover JSON round trips for generated JSON-safe event rows where
practical. They should not test `from_dict()` because import semantics are deferred.

## Verification

Implementation should run targeted tests first, then the project-required verification for
public API and documentation changes.

Expected targeted checks:

```powershell
python -m pytest tests/unit/test_stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py -v
python -m pytest tests/property/ -v
python -m ruff format --check
python -m ruff check
git diff --check
```

Because this contract touches public Python behavior, final verification should also run
the unit and integration suite after rebuilding the extension:

```powershell
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

Rust verification is required only if implementation touches Rust source or Rust-backed
contracts:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Rollout

Recommended implementation slices:

1. Add serialization helpers and compatibility tests for `Logbook`,
   `OptimizationTelemetry`, and JSON-safe value conversion.
2. Add `EventRecord`, `EventHistory`, and pandas/export tests.
3. Add reproducibility metadata helpers and gene-space signature hashing.
4. Extend `RunResult` and `MultiRunResult` export helpers while preserving construction
   compatibility.
5. Record GA ask/tell history and attach result metadata.
6. Record CMA ask/tell history and attach result metadata.
7. Refresh docs and changelog.
8. Run full verification and update PR #13.

## Acceptance Criteria

- EvoCore exposes a stable public result envelope with deterministic JSON export by
  default.
- Ask/tell optimizers produce append-only event history suitable for audit and tabular
  export.
- Telemetry exports use stable field names and JSON-safe values.
- Raw scores and direction-aware comparison scores are both visible where event-level
  scores are exported.
- Reproducibility metadata includes version, engine type, seed, direction, gene-space
  signature/hash, and serializable optimizer configuration.
- pandas remains optional with clear install errors.
- Existing positional `RunResult` and `LogEntry` construction remains compatible.
- Docs and changelog describe the result/history/telemetry contract and its limits.
- No trading-specific metrics, external comparisons, or resume-from-result promises are
  added.

## Deferred Follow-Ups

- `from_dict()` and `from_json()` once reload semantics are explicit.
- Resume-from-result or checkpoint migration.
- Stable problem declaration objects and categorical/permutation gene contracts.
- Multi-objective result and history schemas.
- External benchmark reporting or run comparison dashboards.
