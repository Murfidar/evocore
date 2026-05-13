# EvoCore vNext Explicit Batch Tokens Design

**Date:** 2026-05-09
**Status:** Draft
**Scope:** Explicit batch identity and asynchronous partial `tell()` semantics for vNext GA and CMA engines

## Purpose

EvoCore vNext is moving toward expensive black-box optimization where evaluations may run through
remote backtests, cached jobs, queues, or staged multi-fidelity workflows. In that setting, a
single `ask()` batch may return in pieces. Engines need a first-class way to group candidates,
accept partial `tell()` calls safely, and update optimizer state exactly once when enough trusted
records have arrived.

The current vNext implementation has three risks:

- `GAEngine.run()` can reuse stale vNext telemetry and trusted state across repeated runs.
- `GAEngine.run()` can hang if a synchronous evaluator returns fewer records than assigned.
- `CMAESEngine.tell()` only advances when one call contains the complete trusted population.

Explicit batch tokens solve the grouping problem directly and make these failure modes easier to
validate.

## Goals

- Add a public deterministic `batch_id` to every `Candidate`.
- Add an optional `batch_id` to `EvaluationRecord` for callers that want explicit record grouping.
- Treat asynchronous and partial `tell()` calls as a supported API for GA and CMA.
- Let engines validate candidate, batch, rung, duplicate, and completion state precisely.
- Update CMA state exactly once after a full trusted batch has arrived, even if it arrives across
  multiple `tell()` calls.
- Keep `GAEngine.run()` synchronous and strict: every evaluator call must return exactly one record
  for each assigned candidate.
- Make multi-fidelity policy full-rung placement unambiguous.

## Non-Goals

- Do not add a separate `Batch` object to the public API yet.
- Do not require callers to pass `batch_id` in every `EvaluationRecord`; candidate IDs can still
  imply the known batch.
- Do not add distributed job orchestration, persistence, retries, or queue integrations.
- Do not make CMA update from partial, surrogate, cached, or rejected records.
- Do not change legacy CMA `run(fitness_fn)` behavior in this design.

## Public API Design

`Candidate` gains:

```python
batch_id: str
```

All candidates returned by one `ask()` call share the same batch ID. Batch IDs are deterministic
for a given engine seed and ask event. A future implementation can move generation into Rust, but
the initial shape can derive it from the same seed and event-index inputs used for candidate IDs.

`EvaluationRecord` gains:

```python
batch_id: str | None = None
```

When `batch_id` is omitted, engines infer the batch from the stored candidate ledger. When it is
provided, the engine verifies that it matches the candidate's batch. A mismatch raises
`FitnessError`.

No method signature needs to change:

```python
candidates = engine.ask(16)
engine.tell(records[:4])
engine.tell(records[4:])
```

Callers that manage queues can include the token explicitly:

```python
EvaluationRecord(
    candidate_id=candidate.candidate_id,
    batch_id=candidate.batch_id,
    score=score,
    confidence="trusted_full",
    rung="full",
    cost=1.0,
)
```

## Batch Ledger Design

Each ask/tell engine keeps an internal ledger keyed by `batch_id`.

For each batch, the ledger stores:

- ordered candidate IDs
- candidate ID to candidate object
- candidate ID to received records by rung
- consumed state for optimizer-state updates
- engine-specific payloads such as CMA continuous samples

The ledger should enforce these invariants:

- Unknown candidate ID raises `FitnessError`.
- Unknown explicit batch ID raises `FitnessError`.
- Candidate ID whose stored batch does not match an explicit record batch raises `FitnessError`.
- Duplicate record for the same candidate and rung raises `FitnessError`.
- A consumed batch cannot update optimizer state again.

The ledger can be private implementation detail in this change. The public contract is the
candidate/record batch token and the accepted partial `tell()` behavior.

## GA Engine Semantics

`GAEngine.ask()` creates one ledger entry per ask batch and assigns the same `batch_id` to each
candidate in that batch.

`GAEngine.tell()` accepts any subset of records from known batches. Trusted full records update
the trusted population and best candidate as they arrive. Partial, cached, surrogate, and rejected
records update candidate state and telemetry but do not by themselves require a complete batch.

Repeated `GAEngine.run()` calls must not reuse stale vNext state. At the start of `run()`, reset
the run-scoped vNext state:

- event index
- candidate and batch ledgers
- trusted population
- telemetry
- best candidate

`GAEngine.run()` remains synchronous. For every `(assigned candidates, rung)` evaluation call, it
must validate returned records before calling `tell()`:

- returned candidate IDs equal assigned candidate IDs
- no missing candidates
- no duplicate candidates
- no unknown candidates
- explicit batch IDs, if present, match assigned candidates

If the evaluator response is invalid, raise `FitnessError`. This prevents infinite budget loops
when an evaluator silently drops records.

## CMA Engine Semantics

`CMAESEngine.ask()` creates one CMA batch ledger entry. It stores the rounded public candidates and
the continuous samples needed for Rust `PyCMAESState.tell()`.

`CMAESEngine.tell()` accepts any subset of records. For trusted full records, the ledger stores the
score for that candidate. Once all candidate IDs in one CMA batch have trusted full records, the
engine calls Rust `tell(samples, fitnesses)` exactly once for that batch.

The update order uses the original ask order from the batch ledger. This keeps sample and fitness
alignment deterministic regardless of record arrival order.

After a batch is consumed, additional trusted full records for that batch raise `FitnessError`.
Duplicate records before consumption also raise `FitnessError`.

Partial, cached, surrogate, and rejected records update candidate state and telemetry but do not
advance the CMA distribution. A batch with rejected candidates remains incomplete for CMA state
purposes unless a later design adds explicit failed-sample handling.

## Policy Semantics

`MultiFidelityPolicy` should require exactly one `trusted_full` rung and that rung must be the
final rung. This matches how policy-driven GA counts full evaluations and avoids ambiguous batch
completion behavior.

Invalid policies:

```python
# trusted full is not final
[Rung("full", 1.0, 1.0, "trusted_full"), Rung("audit", 1.0, 1.0, "partial")]

# multiple trusted full rungs
[Rung("cheap", 0.1, 0.5, "partial"), Rung("full_a", 1.0, 1.0, "trusted_full"), Rung("full_b", 1.0, 1.0, "trusted_full")]
```

Valid policy:

```python
[Rung("cheap", 0.1, 0.5, "partial"), Rung("full", 1.0, 1.0, "trusted_full")]
```

## Error Handling

Use `FitnessError` for invalid tell/evaluator data:

- unknown candidate ID
- unknown batch ID
- candidate and batch mismatch
- duplicate candidate/rung record
- record set missing assigned candidates in synchronous `run()`
- consumed CMA batch receives another trusted update

Use `ConfigurationError` for invalid policy shape:

- no trusted full rung
- trusted full rung is not final
- more than one trusted full rung

## Testing Strategy

Add focused unit tests before implementation:

- Candidate and evaluation record expose optional/public batch IDs.
- GA `ask()` assigns one batch ID per ask and distinct IDs across asks.
- GA `tell()` accepts records for one batch in multiple partial calls.
- GA `tell()` rejects duplicate candidate/rung records.
- GA `tell()` rejects explicit batch mismatch.
- GA `run()` raises `FitnessError` when an evaluator omits assigned records.
- Repeated `GAEngine.run()` calls on the same engine perform fresh evaluation.
- CMA `ask()` assigns one batch ID per population batch.
- CMA advances generation after two partial trusted tells complete one batch.
- CMA keeps sample/fitness order deterministic when records arrive out of order.
- CMA rejects duplicate trusted records after batch consumption.
- Policy validation rejects early or multiple trusted full rungs.

Then run the relevant slices and final verification:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_vnext_policy_scheduler.py -v
python -m ruff format --check
python -m ruff check
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

## Migration Notes

Existing callers that ignore `batch_id` can continue to use `candidate_id` only. Explicit batch
tokens are additive on `Candidate` and optional on `EvaluationRecord`.

The stricter `GAEngine.run()` evaluator validation may reveal evaluator bugs that previously led
to hangs or silent under-evaluation. This is intentional.

The stricter policy validation may reject policies that had a trusted full rung before the end.
This is also intentional because the previous shape was ambiguous.
