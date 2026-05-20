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

GA ask/tell checkpointing is not part of checkpoint v1. `EventHistory` remains
audit data and is not replayed to rebuild optimizer state.

`CMAESOptimizer` checkpoint/resume is unsupported until the Rust-backed CMA-ES
state exposes a stable export/import contract. CMA-ES result export and event
audit history remain available.

## GA Ask/Tell Checkpoints

Stable checkpoints also cover manual GA ask/tell workflows. This is the
recommended checkpoint boundary when evaluation work happens outside EvoCore,
for example in a job queue or remote worker pool.

```python
from evocore import EvaluationRecord, GeneticAlgorithmOptimizer

optimizer = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "ga-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("ga-ask-tell.evocore-checkpoint.json")

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