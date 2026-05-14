# Objective, Budget, And Optimizer Config Stabilization Handoff

**Date:** 2026-05-14
**Branch:** `feature/general-optimizer-framework`
**Purpose:** Start the next stabilization discussion after lifecycle, result telemetry, and GeneSpace.

## Current State

EvoCore is being moved toward a general evolutionary optimization package comparable in
scope to libraries such as PyGAD and DEAP, while keeping a clean ask/tell-first public
framework.

Recently stabilized surfaces:

- Optimizer lifecycle protocol: `Optimizer`, `Evaluator`, `Candidate`, `EvaluationRecord`,
  `EvaluationContext`, `TellResult`, and `EngineStateSummary`.
- Result/history/telemetry contract: stable result envelopes, append-only event history,
  reproducibility metadata, deterministic JSON-safe exports.
- GeneSpace contract: flat `float`/`int`/`bool` spaces, fixed numeric genes, named params,
  canonical `signature()`, `hash()`, `to_dict()`, `to_json()`, and `validate_genes(...)`.

Important existing docs/specs:

- `docs/superpowers/specs/2026-05-13-optimizer-lifecycle-protocols-design.md`
- `docs/superpowers/specs/2026-05-14-result-history-telemetry-contract-design.md`
- `docs/superpowers/specs/2026-05-14-genespace-contract-design.md`
- `docs/superpowers/plans/2026-05-14-genespace-contract.md`
- `docs/site/ask-tell-engines.md`
- `docs/site/optimizer-telemetry.md`
- `docs/site/gene-space.md`

The next session should stay in brainstorming/design mode first. Do not jump directly into
implementation until the user approves a focused design.

## Next Stabilization Areas

### 1. Objective And Evaluation Semantics

This is the next most important layer after GeneSpace. EvoCore needs a stable answer for
how objective observations are interpreted and accounted for across optimizers.

Questions to settle:

- How `maximize` and `minimize` behave at every public surface.
- How raw scores and direction-aware comparison scores are represented.
- Which non-finite scores raise, warn, sanitize, or become rejected observations.
- How failed evaluations are represented: exception wrapping, rejected records, metrics,
  or metadata.
- Whether noisy objectives are explicitly unsupported, tolerated, or represented through
  repeated observations.
- What repeated evaluation of the same candidate means for state updates, telemetry, and
  result history.
- What cached evaluations mean and when they are state-eligible.
- What counts toward evaluation budget: proposed candidates, screened candidates, partial
  records, trusted full records, cached records, rejected records, evaluator errors.
- Whether constraints remain metadata-only for now or become a core result/evaluation
  concept.

Likely design direction:

- Keep the stable single-objective core.
- Preserve raw evaluator scores.
- Keep direction comparison explicit through a separate comparison score.
- Treat constraints as metadata-only in this slice unless the user explicitly expands
  scope.
- Define noisy/repeated evaluation semantics conservatively before adding algorithmic
  support.

Potential output:

- A spec named something like
  `docs/superpowers/specs/2026-05-14-objective-evaluation-semantics-design.md`.

### 2. Budget And Termination Contract

Before island models or broader orchestration, EvoCore needs shared stop and budget
semantics that are not GA-only.

Questions to settle:

- Stable meaning of `max_evaluations`.
- Whether `generations` should become, or be documented as, `max_generations`.
- Whether `population_size` is only optimizer configuration or also part of budget
  semantics.
- Whether `target_score` belongs in the shared contract now.
- Whether early stopping and patience belong in the shared contract now, and what they
  observe.
- Whether wall-clock limits are deferred or included as a future-compatible stop reason.
- Shared stop reason vocabulary across optimizers.
- Whether stop reasons should appear on `EngineStateSummary`, `RunResult`, `EventHistory`,
  or all of them.

Likely design direction:

- Define shared vocabulary before adding new controls.
- Keep wall-clock budget deferred unless the user wants it now.
- Avoid promising island-model behavior yet, but choose names that will compose with
  islands later.

Potential output:

- A spec named something like
  `docs/superpowers/specs/2026-05-14-budget-termination-contract-design.md`.

### 3. Optimizer Configuration Contract

Each optimizer needs an exportable, comparable public config so reproducibility metadata
can be trusted and users can understand what makes two runs equivalent.

Questions to settle:

- Which constructor args are part of reproducibility.
- Which defaults are stable public behavior.
- Which values may be callables or hooks, and how those are excluded or represented.
- How optimizer config appears in `RunResult.reproducibility.optimizer_config`.
- Whether config export should be a method, for example `engine.config()` or
  `engine.optimizer_config()`.
- How compatibility validation works between optimizer, operators, and `GeneSpace`.
- Whether GA and CMA should share a config protocol or only converge on export shape.

Likely design direction:

- Keep callable hooks out of deterministic config payloads.
- Export only public constructor configuration.
- Keep engine-specific config fields allowed, but make the envelope shape stable.
- Make compatibility validation explicit and testable.

Potential output:

- A spec named something like
  `docs/superpowers/specs/2026-05-14-optimizer-configuration-contract-design.md`.

## Recommended Decomposition

Do not combine all three into one implementation spec unless the user asks for a high-level
roadmap only. These are related but separable contract layers.

Recommended order:

1. Objective and evaluation semantics.
2. Budget and termination contract.
3. Optimizer configuration contract.

Reasoning:

- Objective semantics define what observations mean.
- Budget/termination depends on what observations count.
- Optimizer configuration then records the public controls and compatibility choices.

## First Question For The Next Chat

Ask the user which slice to brainstorm first:

```text
Do you want to start with objective/evaluation semantics first, or draft a short roadmap
that decomposes objective, budget, and optimizer config into separate specs?
```

Recommended answer to guide them toward:

- Start with objective/evaluation semantics first.

## Constraints For The Next Session

- Use the brainstorming skill before design or implementation.
- Stay design-first until the user approves a spec.
- Keep scope focused; do not implement multi-objective, island model, categorical genes,
  checkpoint reload, or custom operators as part of these stabilization slices.
- Preserve deterministic seed behavior and current checkpoint compatibility unless the
  user explicitly changes them.
- Keep docs and changelog aligned for any public API or behavior change.
