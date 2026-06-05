# DE Rust Kernel Adapter Design

**Date:** 2026-06-05
**Status:** Draft for review
**Scope:** Move Differential Evolution Rust-kernel marshalling out of ask/tell
lifecycle code and into a focused Python adapter.

## Summary

DE proposal math now lives in Rust through `_core.de_generate_trials(...)`.
That preserves the preferred EvoCore architecture: Rust handles deterministic
inner-loop math, while Python owns public optimizer lifecycle. The current
Python integration still performs raw kernel marshalling inside DE ask/tell
code: population encoding, Rust argument construction, jDE state export, raw
dictionary validation, proposal decoding, and `TrialProposal` construction.

This design introduces a focused Python adapter so DE lifecycle code calls a
small domain-level interface instead of manually assembling and decoding Rust
payloads.

## Goals

- Keep `DifferentialEvolutionOptimizer` public behavior stable.
- Keep DE lifecycle, candidates, target population, replacement decisions,
  jDE commit/discard, events, telemetry, policies, evaluator integration, and
  checkpoints in Python.
- Keep Rust returning encoded trial proposals and metadata, not Python
  `Candidate` objects.
- Move Rust boundary encoding, argument construction, payload validation, and
  decoding into a focused adapter.
- Reuse shared Python search-space codec helpers for encoding and decoding.
- Preserve deterministic seed/checkpoint behavior.

## Non-Goals

- Do not change `_core.de_generate_trials(...)` unless the adapter uncovers a
  necessary contract issue.
- Do not move target replacement or jDE state ownership into Rust.
- Do not add a second Python implementation of DE strategy math.
- Do not introduce a public DE strategy plugin system.
- Do not broaden lifecycle helper consolidation in this slice.

## Proposed Adapter

Add a small Python module:

```text
evocore/optimizers/de/kernel.py
```

The module should define an adapter class or function with an internal API
similar to:

```python
class DERustKernelAdapter:
    def generate_trials(
        self,
        *,
        target_population: Sequence[Candidate],
        scores: Sequence[float],
        gene_space: GeneSpace,
        strategy: str,
        mutation_factor: float,
        crossover_rate: float,
        seed: int,
        generation: int,
        target_slots: Sequence[int],
        direction: str,
        jde_state: Mapping[str, Sequence[float]] | None,
    ) -> list[TrialProposal]: ...
```

The exact shape can be adjusted to match existing DE style. The important
boundary is responsibility, not the class name.

## Responsibilities

The adapter should own:

- encoding target population genes with the shared search-space codec;
- validating score and population lengths before calling Rust where Python has
  better context;
- passing `gene_space.rust_bounds` and `gene_space.kinds`;
- passing strategy, F, CR, seed, generation, direction, target slots, and jDE
  committed state;
- validating each Rust proposal dictionary contains required fields;
- decoding Rust `genes` with shared search-space codec;
- converting metadata into Python `TrialProposal` objects.

The adapter should not own:

- candidate ID generation;
- pending candidate maps;
- jDE pending/commit/discard decisions;
- target population replacement;
- events;
- telemetry;
- checkpoint payloads;
- evaluator integration.

## DE Ask/Tell Integration

`evocore/optimizers/de/ask_tell.py` should become easier to read:

- initial population sampling still calls the existing Rust initialization
  helper and shared decoding;
- trial proposal generation delegates to the adapter;
- ask/tell lifecycle continues to create candidates, register trial-target
  mappings, append events, and apply replacement logic.

The adapter should stay private to the DE package unless another optimizer
needs the same pattern later.

## Testing

Add adapter-focused tests with `_core.de_generate_trials` monkeypatched:

- the adapter encodes bools as `1.0`/`0.0` and numerics as floats;
- the adapter passes seed, generation, target slots, direction, strategy, F,
  CR, bounds, kinds, and jDE state exactly;
- the adapter decodes int and bool proposal genes through shared codec helpers;
- malformed Rust payloads raise clear `ValueError` or `RuntimeError` messages.

Keep existing direct Rust kernel tests. Keep existing DE ask/tell behavior
tests, but make them assert lifecycle behavior rather than raw marshalling
details.

## Compatibility

This should be a behavior-preserving refactor. Public API, checkpoint payloads,
event schemas, telemetry fields, and seeded sequences should remain stable.
If adapter validation introduces clearer errors for malformed Rust payloads,
those errors are internal-facing and should be covered by tests.
