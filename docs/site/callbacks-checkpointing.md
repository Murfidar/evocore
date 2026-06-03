# Callbacks And Checkpointing

Callbacks observe or influence optimization runs.

::: evocore.callbacks.Callback

::: evocore.callbacks.EarlyStopping

::: evocore.callbacks.ProgressBar

::: evocore.callbacks.CheckpointCallback

::: evocore.callbacks.MetricsLogger

## Stable Checkpoints

`CheckpointCallback` writes stable JSON checkpoint files when `format="stable"`:

```python
from evocore import CheckpointCallback, GeneSpace, GeneticAlgorithmOptimizer

optimizer = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-1.0, 1.0, 3),
    population_size=20,
    max_generations=10,
    seed=42,
    callbacks=[CheckpointCallback(path="./checkpoints", every=1, format="stable")],
)
```

Stable checkpoint files are named
`checkpoint_gen_{generation}.evocore-checkpoint.json`. They are optimizer state
snapshots for continuation. They are separate from `OptimizationResult.to_dict()`
and `EventHistory.to_rows()`, which are analysis and audit exports.

GA generation-loop checkpoints validate optimizer type, seed, direction,
gene-space hash, optimizer config hash, and seed derivation version before
resuming. Resume fails with `CheckpointError` when the receiving optimizer does
not match the checkpoint identity.

## Compatibility Baseline

Stable JSON checkpoints produced by EvoCore 0.8.0 are the forward compatibility
baseline for checkpoint schema v1 across GA generation-loop, GA ask/tell, and
CMA-ES ask/tell workflows. Differential Evolution ask/tell checkpoints join the
stable checkpoint surface with the EvoCore 0.9.0 DE fixture baseline.
Compatible patch and minor releases should continue to load these stable
checkpoint files, or fail with an explicit `CheckpointError` when a documented
incompatibility is introduced.

The guarantee covers stable JSON checkpoint files only. Legacy GA pickle
checkpoints remain legacy support, but they are not part of the forward
compatibility guarantee. `OptimizationResult.to_dict()` exports and
`EventHistory.to_rows()` exports are not checkpoint files and are not replayed
to rebuild optimizer state.

## Legacy Pickle Checkpoints

The old pickle format remains the checkpoint v1 default for compatibility and is
also available explicitly:

```python
CheckpointCallback(path="./checkpoints", every=1, format="legacy_pickle")
```

Legacy pickle files are named `checkpoint_gen_{generation}.pkl` and contain the
population, generation, and seed. They are retained for GA compatibility, but the
stable JSON checkpoint format is the forward contract.

## Unsupported Checkpoint Surfaces

Policy-driven `run(evaluator, policy=...)` mid-loop resume is not part of
checkpoint v1 for GA or DE. `EventHistory` remains audit data and is not
replayed to rebuild optimizer state.

CMA-ES generation-loop resume and policy-driven `run(evaluator, policy=...)`
resume remain unsupported in checkpoint v1. Manual CMA-ES ask/tell checkpoints
are supported through `CMAESOptimizer.ask_tell_checkpoint()` and
`resume_ask_tell_checkpoint(...)`.

## GA Ask/Tell Checkpoints

Stable checkpoints also cover manual GA ask/tell workflows. This is the
recommended checkpoint boundary when evaluation work happens outside EvoCore,
for example in a job queue or remote worker pool.

```python
from evocore import EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer

gene_space = GeneSpace.uniform(-1.0, 1.0, 3)
optimizer = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "ga-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("ga-ask-tell.evocore-checkpoint.json")

scores = [-sum(float(value) ** 2 for value in candidate.genes) for candidate in candidates]
records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="trusted_full",
        stage="full",
    )
    for candidate, score in zip(candidates, scores, strict=False)
]
restored.tell(records)
```

Pending batches and partial tells are valid checkpoint state. Resume restores
candidate and batch ledgers directly; event history is audit data and is not
replayed to rebuild optimizer state.

## CMA-ES Ask/Tell Checkpoints

CMA-ES ask/tell checkpoints use the same stable envelope as GA and additionally
store the Rust-backed CMA-ES state snapshot. This preserves covariance
adaptation, pending batches, partial records, telemetry, and audit events for
manual external-evaluation workflows.

```python
from evocore import CMAESOptimizer, EvaluationRecord, GeneSpace

gene_space = GeneSpace.uniform(-1.0, 1.0, 3)
optimizer = CMAESOptimizer(gene_space, population_size=8, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "cmaes-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = CMAESOptimizer(gene_space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("cmaes-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

Events are restored as audit history. Resume restores structured optimizer
state directly and does not replay events to rebuild CMA-ES state.

## Differential Evolution Ask/Tell Checkpoints

Differential Evolution ask/tell checkpoints store target slots and pending trial
mappings in addition to candidates, batches, telemetry, and events. This lets a
restored optimizer compare returned trial records against the same target
candidate after resume.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, GeneSpace

gene_space = GeneSpace.uniform(-5.0, 5.0, 3)
optimizer = DifferentialEvolutionOptimizer(gene_space, population_size=6, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "de-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = DifferentialEvolutionOptimizer(gene_space, population_size=6, seed=42)
summary = restored.resume_ask_tell_checkpoint("de-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

DE checkpoint identity validation covers optimizer type, seed, direction,
gene-space hash, optimizer config hash, checkpoint state kind, schema version,
and trial target mappings.

For synchronous DE `run(...)`, callbacks can observe generation start,
generation end, and run end. Manual ask/tell checkpoints remain the stable DE
resume path; `CheckpointCallback` is not advertised as a DE policy-run resume
mechanism.
