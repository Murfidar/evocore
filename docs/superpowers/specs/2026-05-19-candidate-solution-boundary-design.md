# Candidate/Solution Boundary Design

**Date:** 2026-05-19
**Status:** Draft approved for specification
**Scope:** Stabilize the boundary between lifecycle-facing `Candidate` records and population/result-facing `Solution` records, including conversion rules and search-space value hash semantics

## Summary

EvoCore should make the `Candidate` and `Solution` boundary explicit before adding
features such as islands, archives, elites, migration, surrogate memory, and checkpoint
resume. The current branch has already replaced the old public `Individual` vocabulary
with `Solution`, but some code still treats lifecycle records and population records as
nearly interchangeable. That ambiguity will become expensive as more orchestration
features need to compare, store, replay, or move search points.

The approved direction is conservative: keep `Candidate.genes` and `Candidate.params` for
evaluator ergonomics, but define them as proposal payload carried by a lifecycle record.
`Solution` remains the population and result container. Conversions become explicit
helpers, and value hashing becomes owned by `GeneSpace` so duplicate decoded values hash
consistently under the same schema.

## Product Direction

The boundary should reinforce EvoCore's current shape:

- `Candidate` supports ask/tell lifecycle, scheduler state, event history, and evaluator
  interaction.
- `Solution` supports optimizer populations, elite sets, result exports, and user-facing
  scored search points.
- `GeneSpace` owns schema-aware value validation, encoding, and value identity.

The important identity split is:

- `candidate_id` answers: which lifecycle proposal is this?
- `candidate_hash` answers: which search-space value is this?
- `gene_space_hash` answers: which search-space schema gives that value meaning?
- `Solution` answers: what scored search point appears in a population or result?

Two candidates may have different `candidate_id` values and the same `candidate_hash`.
The same raw values in incompatible gene spaces should not share a value hash.

## Goals

- Define `Candidate` as lifecycle-facing and `Solution` as population/result-facing.
- Keep evaluator ergonomics by preserving decoded values and params on `Candidate`.
- Add explicit conversion helpers instead of ad hoc GA/CMA conversion code.
- Make score transfer from `Candidate` to `Solution` depend on state-eligible
  observations only.
- Prevent `Solution` from silently carrying scheduler state as first-class fields.
- Make candidate value hashing align with `GeneSpace` validation and encoding.
- Include `gene_space_hash` in value hash payloads to avoid cross-schema collisions.
- Preserve a temporary zero-argument `Candidate.candidate_hash()` fallback for external
  compatibility while internal engine code moves to GeneSpace-backed hashing.
- Update docs and tests so future features can lean on the boundary.

## Non-Goals

- Do not reintroduce `Individual` as a public Python API name.
- Do not remove `Candidate.genes` or `Candidate.params` in this slice.
- Do not redesign objective semantics, budget policy, event history, or checkpoint
  formats beyond the fields needed for provenance.
- Do not add islands, archives, migration, surrogate memory, or elite persistence in this
  slice.
- Do not change Rust extension individual structs.
- Do not make partial or surrogate scores state-eligible.

## Boundary Model

`Candidate` is the lifecycle proposal record. It owns:

- `candidate_id`
- `batch_id`
- `origin`
- `parents`
- `event_index`
- `generation`
- `stage`
- `status`
- `confidence`
- `cost`
- `scores`
- decoded proposal `genes`
- decoded proposal `params`
- lifecycle metadata

`Candidate.genes` and `Candidate.params` are not population ownership. They are the
decoded payload supplied to evaluators and recorded in lifecycle events.

`Solution` is the population/result record. It owns:

- decoded `values`
- `score`
- `score_valid`
- result-facing metadata

`Solution.metadata` may carry provenance such as `candidate_id`, `candidate_hash`,
`batch_id`, `origin`, and `generation`, but lifecycle state such as `stage`, `status`,
`confidence`, scheduler records, and observation history should not become first-class
`Solution` fields.

The old conceptual `Candidate`/`Individual` split maps to the current public
`Candidate`/`Solution` split. Documentation should use `Solution` as the active term and
mention `Individual` only as historical context where useful.

## Conversion API

Add a focused module:

```text
evocore/lifecycle/conversion.py
```

Recommended public or semi-public helpers:

```python
def candidate_to_solution(
    candidate: Candidate,
    *,
    direction: Direction,
    gene_space: GeneSpace | None = None,
    include_provenance: bool = True,
) -> Solution: ...

def solution_to_candidate(
    solution: Solution,
    *,
    gene_space: GeneSpace,
    candidate_id: str,
    batch_id: str,
    origin: CandidateOrigin,
    event_index: int,
    parents: Sequence[str] = (),
    generation: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Candidate: ...
```

The exact signatures may be adjusted during implementation, but conversion must live in
one shared place rather than being duplicated across GA and CMA-ES.

## Candidate To Solution Rules

`Candidate -> Solution` is used when lifecycle observations become population or result
material.

Rules:

- `Solution.values = list(candidate.genes)`.
- `Solution.score = candidate.best_state_score(direction)` only when the candidate has a
  state-eligible observation.
- `Solution.score_valid = True` only when the candidate has at least one state-eligible
  observation, currently `trusted_full` or `cached`.
- `Solution.metadata["params"] = candidate.params` when params are present.
- `Solution.metadata["candidate_id"] = candidate.candidate_id`.
- If `gene_space` is supplied, include
  `Solution.metadata["candidate_hash"] = gene_space.value_hash(candidate.genes)`.
- Optional provenance may include `batch_id`, `origin`, and `generation`.
- Do not copy `stage`, `status`, `scores`, `confidence`, `cost`, scheduler state, or
  observation history into first-class `Solution` fields.

If a candidate has only partial, surrogate, or rejected observations, conversion may still
produce a `Solution`, but `score_valid` should be false unless the caller explicitly opts
into a different future behavior.

## Solution To Candidate Rules

`Solution -> Candidate` is used for optimizer-internal proposal construction.

Rules:

- The caller must provide fresh lifecycle identity: `candidate_id`, `batch_id`,
  `event_index`, and `origin`.
- `Candidate.genes = list(solution.values)`.
- `Candidate.params` must be recomputed from `GeneSpace.params_for(solution.values)`.
- `Solution.metadata["params"]` must not be blindly trusted as the candidate params.
- `Solution.score` must not automatically become a candidate observation.
- If a score needs to re-enter lifecycle, it must come through an `EvaluationRecord`.
- Caller-provided metadata may be copied to `Candidate.metadata`, but result metadata is
  not a substitute for lifecycle records.

This keeps population state and lifecycle observations from bleeding across the boundary.

## GeneSpace Value Identity

Add GeneSpace-owned value identity helpers:

```python
class GeneSpace:
    def value_signature(self, values: Sequence[GeneValue]) -> dict[str, Any]: ...
    def value_hash(self, values: Sequence[GeneValue]) -> str: ...
```

Rules:

- Validate values with `GeneSpace.validate_genes(values)` before hashing.
- Encode values according to declared gene kinds, not incidental Python runtime types.
- Float genes hash by canonical numeric representation, likely `float(value).hex()`.
- Int genes hash as integers.
- Bool genes hash as booleans.
- Include `gene_space_hash = self.hash()` in the payload.
- Include gene names and kinds in value rows for readable debug signatures.
- Export through existing canonical JSON hashing helpers.

Recommended payload:

```python
{
    "schema_version": 1,
    "gene_space_hash": space.hash(),
    "values": [
        {"name": "x", "kind": "float", "value": "0x1.0000000000000p+0"},
        {"name": "n", "kind": "int", "value": 3},
        {"name": "enabled", "kind": "bool", "value": True},
    ],
}
```

Including the gene-space hash is intentional. Identical raw values under differently
named, ordered, bounded, or typed spaces should not be merged by archives, surrogate
memory, checkpoints, or migration systems.

## Candidate Hash Semantics

`Candidate.candidate_hash()` currently owns independent type-tag hashing. That should
become compatibility behavior rather than the preferred internal contract.

Recommended transition:

```python
def candidate_hash(self, gene_space: GeneSpace | None = None) -> str:
    if gene_space is not None:
        return gene_space.value_hash(self.genes)
    return legacy_candidate_hash(self.genes)
```

Internal engine code should always pass `self.gene_space` when recording events,
telemetry, and provenance metadata. External zero-argument calls remain available during
this stabilization phase to avoid unnecessary breakage.

This creates clear semantics:

- `candidate_id` is lifecycle identity.
- `candidate_hash(gene_space)` is schema-aware search-point identity.
- zero-argument `candidate_hash()` is legacy value hashing for callers that have not yet
  adopted the schema-aware form.

Docs should recommend the GeneSpace-backed form.

## Engine Integration

GA and CMA-ES should use the shared conversion and hash helpers where they currently
construct records by hand.

Recommended updates:

- Candidate creation from decoded optimizer values goes through `solution_to_candidate`
  or a small shared helper that applies the same rules.
- Result construction for best, final, and elite solutions goes through
  `candidate_to_solution`.
- Ask/tell event rows record `candidate_hash = candidate.candidate_hash(self.gene_space)`.
- Telemetry unique candidate hashes use GeneSpace-backed hashes.
- Result `Solution.metadata` includes provenance without copying lifecycle state.

This change should reduce duplicated conversion logic in:

```text
evocore/optimizers/ga/ask_tell.py
evocore/optimizers/cmaes/ask_tell.py
```

and any result construction paths that materialize `Solution` values from trusted
`Candidate` records.

## Public Documentation

Update `docs/site/ask-tell-engines.md` with:

- A short `Candidate` versus `Solution` section.
- The identity split between `candidate_id`, `candidate_hash`, and `gene_space_hash`.
- Guidance that evaluators consume `Candidate` records and results expose `Solution`
  records.
- Guidance that users should compare search points with GeneSpace-backed candidate
  hashes, not candidate IDs.

Update API docs if the conversion helpers are exported from `evocore.lifecycle`.

Update `CHANGELOG.md` because the preferred candidate hash semantics and conversion
helpers are public behavior.

## Testing Plan

Unit tests should cover:

- `GeneSpace.value_signature(values)` validates length and kind.
- `GeneSpace.value_hash(values)` is deterministic.
- Float, int, and bool values use kind-aware canonical encoding.
- Equivalent spaces and equivalent decoded values produce the same value hash.
- Different gene-space signatures produce different value hashes for the same raw values.
- Two candidates with different IDs and duplicate genes share
  `candidate.candidate_hash(space)`.
- `Candidate.candidate_hash()` without a gene space remains available.
- `candidate_to_solution` copies values, state-eligible score, params, and provenance.
- `candidate_to_solution` does not copy lifecycle state as first-class `Solution` fields.
- `candidate_to_solution` leaves `score_valid=False` for partial, surrogate, and rejected
  candidates.
- `solution_to_candidate` recomputes params from `GeneSpace`.
- `solution_to_candidate` does not convert `Solution.score` into an observation.
- GA ask events and telemetry use `gene_space.value_hash(candidate.genes)`.
- CMA-ES ask events and telemetry use `gene_space.value_hash(candidate.genes)`.
- Result construction uses shared conversion semantics.

Property tests should cover:

- JSON round-trip stability for value signatures across generated flat gene spaces.
- Stable value hashes for equivalent spaces and generated valid values.
- Hash differences when gene kind, order, name, bounds, or `has_names` changes.

## Migration And Compatibility

This is a behavior-stabilization change on the feature branch. It may change event and
telemetry candidate hash values because the preferred hash now includes `gene_space_hash`.
That is acceptable before release because it makes archives and checkpoints safer.

Compatibility rules:

- Keep `Candidate.candidate_hash()` callable without arguments for now.
- Use the GeneSpace-backed form in all EvoCore engines.
- Document the zero-argument form as legacy compatibility.
- Do not add old public `Individual` aliases.
- Do not rename `Solution` back to `Individual`.

## Risks And Decisions

The main API risk is changing the meaning of candidate hashes in event history and
telemetry. Including `gene_space_hash` is the safer long-term choice because future
archive, migration, surrogate, and checkpoint systems should not merge values across
incompatible schemas.

The main ergonomics risk is making `Candidate` feel too population-like because it still
has decoded genes and params. The design accepts that trade-off for evaluator usability,
but uses docs, conversion helpers, and tests to make ownership explicit.

The main implementation risk is partial adoption. If GA and CMA-ES each keep local
conversion shortcuts, the boundary can drift again. The implementation should make shared
helpers the default path and add tests around engine event/result behavior.

## Approval Notes

During brainstorming, the approved direction was:

- Stabilize `Candidate` and `Solution`, not old public `Individual`.
- Preserve `Candidate.genes` and `Candidate.params` for evaluator ergonomics.
- Add explicit conversion helpers.
- Keep candidate identity and search-point identity separate.
- Make value hashing GeneSpace-owned.
- Include `gene_space_hash` in value hashes.
- Preserve zero-argument `Candidate.candidate_hash()` as temporary compatibility while
  internal code moves to GeneSpace-backed hashes.
