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

## Archive Search Memory And Select Survivors

Use `CandidateArchive` to keep scored candidates outside optimizer checkpoints,
then export the best records back into `warm_start(...)`.

```python
from evocore import CandidateArchive, FamilyQuota, select_candidates


archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
trusted = optimizer.candidate_snapshot(scope="trusted")
archive.add_population(trusted, source="stage1")

selection = select_candidates(
    trusted.candidates,
    k=8,
    score_direction="maximize",
    quotas=[FamilyQuota(metadata_key="family", max_count=3)],
)

survivor_records = selection.to_warm_start_records(stage="stage2_seed")
archive_records = archive.to_warm_start_records(k=8, stage="archive_seed")
```

Select directly from optimizer snapshots when making immediate promotion
decisions. Use archive exports when you want durable search memory that can seed
future runs. Once an archive contains entries, later population snapshots must
use the same score direction. Candidate selection rejects non-finite scores.

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
and an inner optimizer over active continuous parameters.

```python
from evocore import (
    CMAESOptimizer,
    GeneticAlgorithmOptimizer,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)


outer = GeneticAlgorithmOptimizer(template_space, population_size=24, seed=100)

for template_candidate in outer.ask(4):
    template_hash = template_candidate.candidate_hash(template_space)
    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=template_hash,
        stage="inner_cma",
    )
    template = decode_template(template_candidate.params)
    inner_space = template.active_parameter_space()
    inner = CMAESOptimizer(inner_space, population_size=16, seed=inner_seed)

    prior_records = lookup_template_archive(template.name)
    if prior_records:
        inner.warm_start(prior_records, mode="state", cma_mean_strategy="top_k_centroid")

    tuned = run_inner_backtests(inner, template)
    metadata = lineage_metadata(
        outer_candidate=template_candidate,
        gene_space=template_space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=inner_seed,
        stage="inner_cma",
        metadata={"template_name": template.name},
    )
    outer.tell(
        [
            inner_result_record(
                outer_candidate=template_candidate,
                gene_space=template_space,
                score=tuned.best_score,
                confidence="trusted_full",
                stage="inner_cma",
                metadata=metadata,
            )
        ]
    )
```

When resuming a CMA-ES checkpoint after a state warm start, construct the
optimizer with the same warm-started `initial_mean` context used for the saved
run before loading the checkpoint.

Caller metadata may add domain fields such as `template_name`, but it cannot
override canonical lineage identity, seed, stage, batch, or checkpoint fields.

## Projected Template Optimization

External expensive systems often choose a structure first, then tune only the
parameters active for that structure. Use an outer optimizer for structures and
`ActiveGeneProjection` to compile the active inner coordinates.

```python
from evocore import (
    CMAESOptimizer,
    CandidateArchive,
    EvaluationRecord,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
    derive_child_seed,
)
from evocore.lifecycle import constraint_penalty_record
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ConstraintViolation,
    ExponentialIntegerTransform,
)


outer_space = GeneSpace([Gene("family", "int", 0, 2), Gene("mode", "int", 0, 1)])
outer = GeneticAlgorithmOptimizer(outer_space, population_size=24, seed=100)
archive = CandidateArchive(score_direction="maximize")

for template_candidate in outer.ask(4):
    family = int(template_candidate.genes[0])
    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("family", "int", 0, 2),
                Gene("lookback_log", "float", 1.0, 5.0),
                Gene("use_filter", "float", 0.0, 1.0),
            ]
        ),
        active_names=["lookback_log", "use_filter"],
        structural_bindings={"family": family},
        transforms={
            "lookback_log": ExponentialIntegerTransform(base=2.0),
            "use_filter": BinaryThresholdTransform(),
        },
        identity_keys=("family",),
        schema_id="template-family",
        schema_version="1",
    )
    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=template_candidate.candidate_hash(outer_space),
        stage="inner_cma",
    )
    inner = CMAESOptimizer(
        projection.optimizer_space,
        population_size=16,
        seed=inner_seed,
        integer_strategy="margin",
    )

    prior_records = [
        WarmStartRecord(
            params={"lookback_log": 3.0, "use_filter": 1.0},
            score=8.0,
            confidence="cached",
            stage="template_archive",
            metadata={"family": family, "source": "search_memory"},
        )
    ]
    inner.warm_start(prior_records, mode="state")

    records = []
    for candidate in inner.ask():
        decoded = projection.reconstruct(candidate.genes)
        if decoded.parameters["lookback_log"] < 3:
            records.append(
                constraint_penalty_record(
                    candidate=candidate,
                    stage="projection",
                    direction="maximize",
                    violations=[
                        ConstraintViolation(
                            code="min_lookback",
                            message="lookback must be at least 3",
                            names=("lookback_log",),
                        )
                    ],
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )
            continue

        score = expensive_backtest(decoded.parameters)
        records.append(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=score,
                confidence="trusted_full",
                stage="full",
                metadata={
                    "family": family,
                    "projection_hash": decoded.projection_hash,
                },
            )
        )

    inner.tell(records)
    archive.add_population(inner.candidate_snapshot(scope="trusted"), source="inner_cma")
```

Cached records, archives, family quotas, specialist caps, and survivor selection
remain lifecycle helpers. Projection only owns the boundary between named domain
parameters and optimizer-native coordinates. Use trusted snapshots for archive
promotion; scored snapshots may include `constraint_penalty` records that update
optimizer state but are intentionally not archive or warm-start evidence.

## Stop Long-Running Ask/Tell Loops

Stop policies are reusable helpers for external loops. They do not spend budget
and do not mutate optimizer state. Snapshot scoring follows each policy's
`score_direction`, and cumulative evaluation-limit counts remain monotonic until
`reset()` is called.

```python
from evocore import CompositeStopPolicy, EvaluationLimitPolicy, NoImprovementPolicy


stop_policy = CompositeStopPolicy(
    [
        EvaluationLimitPolicy(max_evaluations=500),
        NoImprovementPolicy(window=8, min_delta=0.001, score_direction="maximize"),
    ]
)

while True:
    candidates = optimizer.ask(16)
    records = expensive_evaluator(candidates)
    update = optimizer.tell(records)
    decision = stop_policy.observe(
        update,
        snapshot=optimizer.candidate_snapshot(scope="trusted"),
    )
    if decision.stop:
        break
```

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
