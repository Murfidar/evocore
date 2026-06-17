# Expensive External Evaluations

Expensive black-box systems often need to reuse previous work, checkpoint while
external jobs are running, and promote only the best survivors to a later stage.
The external-state helpers provide a stable public surface for those workflows
without reading optimizer private attributes.

GA, DE, and CMA-ES expose the same core methods:

- `warm_start(records, mode=...)` imports known scored candidates.
- `candidate_snapshot(scope=...)` returns read-only known, trusted, pending, or
  scored candidates.
- `top_candidates(k, confidence=...)` returns ranked read-only snapshots.
- `inject_candidates(records, mode=...)` adds proposed or tracked external
  candidates when the optimizer supports that mode.
- `external_state_capabilities()` reports optimizer-specific support.

## Warm Start From Prior Runs

Use `WarmStartRecord` for candidates from archives, search memory, manual seed
pools, or a previous production run.

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer, WarmStartRecord


space = GeneSpace.uniform(-5.0, 5.0, 3)
optimizer = GeneticAlgorithmOptimizer(space, population_size=12, seed=42)

seed_records = [
    WarmStartRecord(
        values=(-0.2, 0.1, 0.0),
        score=0.91,
        confidence="cached",
        stage="archive",
        metadata={"source": "search_memory", "family": "baseline"},
    ),
    WarmStartRecord(
        values=(0.4, -0.1, 0.2),
        score=0.84,
        confidence="trusted_full",
        stage="manual_seed",
        metadata={"source": "known_good"},
    ),
]

result = optimizer.warm_start(seed_records, mode="state")
print(result.accepted_count)
```

`mode="state"` uses the records to initialize or update optimizer state.
`mode="tracked"` records the candidates for reporting and scoring history
without making them part of trusted optimizer state.

## Cached Scores For Current Candidates

When a candidate has already been evaluated by a trusted external cache, convert
the cache hit into `EvaluationRecord(confidence="cached")` records and feed them
through normal `tell(...)`.

```python
from evocore import cached_records


candidates = optimizer.ask(8)
cache = {
    snapshot.candidate_hash: {
        "score": 0.75,
        "metadata": {"cache_reason": "same_params"},
    }
    for snapshot in optimizer.candidate_snapshot(scope="pending").candidates[:2]
}

records = cached_records(
    candidates,
    gene_space=space,
    cache=cache,
    stage="cache_lookup",
    metadata={"external_job_id": "batch-2026-06-17"},
)
optimizer.tell(records)
```

The helper only builds records. Budget and optimizer state change when those
records are passed to `tell(...)` or imported with `warm_start(...)`.

## Promote Survivors Without Private State

Use snapshots for reports, archives, and staged survivor selection.

```python
trusted = optimizer.candidate_snapshot(scope="trusted")
survivors = optimizer.top_candidates(5)

archive_rows = [
    {
        "candidate_hash": candidate.candidate_hash,
        "score": candidate.score,
        "family": candidate.metadata.get("family"),
        "values": candidate.values,
    }
    for candidate in survivors
]
```

Snapshots are detached read-only data objects. Mutating them does not mutate the
optimizer.

## Inject External Candidates

GA and DE can accept proposed candidates during an ask/tell run. This is useful
for random immigrants, repaired candidates, domain-generated seeds, or search
memory candidates.

```python
immigrants = [
    WarmStartRecord(
        values=(1.0, -0.5, 0.25),
        score=0.0,
        metadata={"source": "immigrant"},
    )
]

if optimizer.external_state_capabilities().proposed_candidate_injection:
    injection = optimizer.inject_candidates(immigrants, mode="proposed")
    print(len(injection.accepted))
```

For proposed injection, the score on `WarmStartRecord` is only a carrier value.
The injected candidate still needs evaluation through `tell(...)`.

CMA-ES supports tracked-only injection in this release line. That keeps external
candidates visible to snapshots and archives without perturbing the covariance
state.

## Hybrid Outer GA And Inner CMA-ES

A common expensive workflow is an outer optimizer over structures or templates
and an inner optimizer over active continuous parameters:

```python
outer = GeneticAlgorithmOptimizer(template_space, population_size=24, seed=100)

for template_candidate in outer.ask(4):
    template = decode_template(template_candidate.params)
    inner_space = template.active_parameter_space()
    inner = CMAESOptimizer(inner_space, population_size=16, seed=template_candidate.event_index)

    prior_records = lookup_template_archive(template.name)
    if prior_records:
        inner.warm_start(prior_records, mode="state", cma_mean_strategy="top_k_centroid")

    tuned = run_inner_backtests(inner, template)
    report_outer_score(template_candidate, tuned.best_score)
```

When resuming a CMA-ES checkpoint after a state warm start, construct the
optimizer with the same warm-started `initial_mean` context used for the saved
run before loading the checkpoint.

## Checkpoint Around External Work

For long-running queues, save a checkpoint after `ask(...)` and before jobs leave
the process. On resume, load the checkpoint, poll for completed jobs, and call
`tell(...)` with the completed subset. Records may arrive in any order.

```python
candidates = optimizer.ask(16)
checkpoint = optimizer.ask_tell_checkpoint(metadata={"queue_batch": "bt-407"})
optimizer.save_checkpoint("bt-407.evocore-checkpoint.json", checkpoint)

# Submit jobs with candidate_id and batch_id here.
```

The checkpoint contains pending batch IDs, event history, telemetry, and
candidate metadata, so external job IDs and domain labels remain available after
resume.
