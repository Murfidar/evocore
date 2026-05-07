# EvoCore vNext Expensive Optimizer Design

**Date:** 2026-05-07
**Status:** Draft approved for specification
**Scope:** Full vNext optimizer architecture for expensive black-box strategy search

## Summary

EvoCore vNext should stop treating DEAP parity as the product center. The next
architecture should fully override the current DEAP-shaped GA/CMA-ES run loops and
make expensive black-box optimization the default design target.

The new engine family should be ask/tell-native, multi-fidelity, racing-aware,
surrogate-assisted, mixed-variable ready, and telemetry-rich. Trading-Algo-Scalper-Gold
is the first reference workload, especially snapshot backtests where full fitness calls
dominate runtime.

## Latest Benchmark Implications

The latest Trading-Algo GA benchmark report recommends migration to EvoCore, but it also
shows the next bottleneck clearly:

- `stress_isolated`: EvoCore evaluates 768 candidates in `1.243s` at `617.884 eval/s`.
- `stress_snapshot`: EvoCore evaluates the same 768-candidate budget in `22.653s` at
  `33.903 eval/s` on a `706254` row XAUUSD snapshot.
- `stress_snapshot`: DEAP takes `62.080s`, so EvoCore is already about `2.74x` faster,
  but full snapshot fitness remains the scarce resource.
- `stress_multiprocessing` is a reliability/pressure scenario; it does not prove
  optimizer intelligence because both backends share the same pressure harness timing.

The capability gap is no longer "make DEAP faster." The gap is that every candidate is
still treated as if it deserves a full-cost fitness call. EvoCore vNext should instead
manage evaluation budget as a first-class optimization resource.

## Goals

- Redesign GA and CMA-ES around ask/tell-native engine state.
- Make multi-fidelity evaluation, racing, early elimination, and promotion first-class.
- Add surrogate/model advisor seams for candidate ranking, screening, and proposal.
- Support mixed-variable search across continuous, integer, and categorical-by-integer
  genes.
- Track full trial lineage and anti-overfitting telemetry for downstream financial gates.
- Keep Python ergonomics high while moving repeated ranking, scheduling, encoding, and
  numerical hot loops into Rust.
- Treat Trading-Algo snapshot/fold optimization as the first high-value downstream
  workload.
- Include release hygiene in the implementation plan: version bump, changelog, MkDocs,
  examples, and public docstrings.

## Non-Goals

- Do not preserve DEAP parity as a vNext acceptance criterion.
- Do not keep DEAP-style generation semantics as the default GA architecture.
- Do not treat current integer-rounded CMA-ES as sufficient mixed-variable support.
- Do not let surrogate or cheap-rung scores silently masquerade as full backtest truth.
- Do not require Trading-Algo-specific code inside EvoCore; Trading-Algo provides domain
  evaluator rungs through public hooks.
- Do not implement the entire vision in one code slice. The architecture is large, but
  the rollout must be incremental.

## Research Grounding

This design combines four research streams:

- Multi-fidelity resource allocation: Successive Halving and Hyperband show that adaptive
  resource allocation can evaluate many configurations cheaply and promote only promising
  survivors.
- Racing/configuration: F-Race and irace motivate statistically eliminating weak
  candidates under noisy repeated evaluations.
- Surrogate-assisted evolutionary optimization: expensive EA literature emphasizes model
  management, uncertainty, and periodic true-fitness calibration.
- Mixed-variable evolution strategies: CMA-ES with Margin, CatCMA, and CatCMAwM motivate
  explicit handling for integer and categorical variables instead of naive post-hoc
  rounding.

Financial anti-overfitting references also shape the telemetry design: White's Reality
Check, Deflated Sharpe Ratio, Hansen SPA, and Model Confidence Set all require careful
accounting of how many alternatives were tried and which evidence was selected.

Reference anchors:

- Successive Halving: https://proceedings.mlr.press/v51/jamieson16.html
- Hyperband: https://www.jmlr.org/beta/papers/v18/16-558.html
- BOHB: https://proceedings.mlr.press/v80/falkner18a.html
- irace: https://www.sciencedirect.com/science/article/pii/S2214716015300270
- Surrogate-assisted expensive optimization survey:
  https://link.springer.com/article/10.1007/s11633-022-1317-4
- CMA-ES with Margin: https://arxiv.org/abs/2205.13482
- CatCMA: https://arxiv.org/abs/2405.09962
- CatCMAwM: https://arxiv.org/abs/2504.07884
- White's Reality Check:
  https://econpapers.repec.org/article/ecmemetrp/v_3a68_3ay_3a2000_3ai_3a5_3ap_3a1097-1126.htm
- Deflated Sharpe Ratio:
  https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2460551_code87814.pdf?abstractid=2460551&mirid=1
- Hansen SPA: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569
- Model Confidence Set: https://pure.au.dk/portal/en/publications/the-model-confidence-set/

## Product Direction

EvoCore vNext becomes an expensive black-box optimizer for real strategy search:

```text
ask/tell-native engines
+ multi-fidelity evaluation
+ racing / early elimination
+ surrogate-assisted candidate ranking
+ mixed-variable search
+ anti-overfit trial accounting
+ Rust hot loops
+ Python ergonomic policies
```

The current parity benchmark is historical evidence, not a design constraint. Public docs
should move away from "GA Benchmark Parity" as a central page and toward "Budget-Aware
Optimization" and "Mixed-Variable Search."

## Core Architecture

The vNext engine stack has five layers:

```text
Candidate Generator
  GA, mixed-variable CMA, future optimizers

Evaluation Scheduler
  cheap/full rungs, racing, promotion, budget caps

Advisor Layer
  surrogate models, search-memory priors, novelty pressure, exploration quotas

Validation And Risk Layer
  trial accounting, fold lineage, anti-overfit telemetry

User Fitness Backends
  Trading-Algo snapshot/backtest evaluators first
```

EvoCore owns candidate generation, budget allocation, promotion decisions, optimizer state,
and telemetry. Trading-Algo owns domain-specific evaluator rungs such as cheap IS slice,
recent slice, full snapshot, and OOS audit.

## Candidate State

Every candidate should be represented by a stable state object rather than an anonymous
list of genes.

Required fields:

- `candidate_id`: deterministic stable ID for this optimization run.
- `genome`: encoded optimizer genome.
- `params`: optional decoded parameter dictionary.
- `origin`: random, crossover, mutation, CMA sample, surrogate proposal, memory seed, or
  restart.
- `parents`: zero or more parent candidate IDs.
- `event_index`: monotonically increasing ask/tell event index.
- `generation`: optional generation number when a policy uses generations.
- `rung`: current evaluation rung.
- `status`: proposed, screened, racing, promoted, trusted, eliminated, archived.
- `confidence`: surrogate, partial, cached, trusted_full, rejected.
- `cost`: evaluator-reported cost units such as rows, folds, seconds, or full-call units.
- `scores`: all observed scores keyed by rung and confidence.
- `metadata`: user-facing diagnostics and evaluator details.

Rust should own compact candidate arrays and repeated ranking/sorting operations. Python
should expose ergonomic dataclasses or typed wrappers.

## Evaluation Records

`tell()` should receive evaluation records, not raw floats.

```python
EvaluationRecord(
    candidate_id=...,
    score=...,
    confidence="trusted_full",
    rung="full_snapshot",
    cost=...,
    metrics={...},
)
```

Confidence levels must be explicit:

- `surrogate`: model estimate or advisor score.
- `partial`: cheap fold, short slice, partial rows, or reduced backtest.
- `cached`: reused known result with compatible fingerprint/version.
- `trusted_full`: full evaluator result suitable for default optimizer-state updates.
- `rejected`: structural or runtime rejection with reason.

Non-finite, failed, or rejected evaluations should keep the existing strong error semantics
where appropriate, but they should also be representable as rejected records when the policy
chooses to continue.

## Ask/Tell Engine Core

Both GA and CMA-ES should become ask/tell-native.

Core methods:

```python
engine.ask(n: int | None = None) -> list[Candidate]
engine.tell(records: Sequence[EvaluationRecord]) -> EngineStateSummary
engine.run(evaluator: Evaluator, policy: OptimizationPolicy | None = None) -> RunResult
```

`run()` becomes orchestration over `ask()`, scheduler assignment, evaluator execution, and
`tell()`. It is no longer the primary algorithm boundary.

## GA vNext

GA vNext should be event-based rather than DEAP-generation-based.

Default GA behavior:

- Maintain a trusted population built from `trusted_full` records by default.
- Allow partial/surrogate records to influence promotion and parent preselection through
  policy, not through silent full-fitness substitution.
- Use candidate age, diversity, lineage, and evaluator confidence in replacement decisions.
- Support family/category diversity quotas so sparse Trading-Algo strategy families are not
  eliminated prematurely.
- Keep elitism as one replacement policy option, not the core semantic anchor.

Rust hot paths:

- candidate encoding/decoding support
- mutation/crossover batches
- tournament/rank/novelty selection with confidence masks
- population replacement
- diversity metrics
- rung-aware sorting and top-k selection

## CMA vNext

CMA vNext should not be limited to continuous samples plus integer rounding.

Default CMA behavior:

- Expose ask/tell state.
- Update covariance/distribution from trusted records by default.
- Allow explicitly policy-approved partial records for aggressive modes.
- Support fixed genes through reconstruction or inactive coordinates.
- Add mixed-variable support inspired by CMA-ES with Margin and CatCMA-style categorical
  distributions.

Phased mixed-variable representation:

- continuous genes: CMA covariance state
- integer genes: margin-aware discretization with probability mass protection
- categorical-by-integer genes: categorical distribution with natural-gradient-style update
  or CatCMA-inspired update
- fixed genes: reconstructed after sampling and excluded from adaptive dimensions

This may become a new `MixedCMAEngine` if the implementation boundary is cleaner than
mutating the current `CMAESEngine` in place.

## Evaluation Scheduler

The scheduler assigns candidate rungs and decides promotion/elimination.

Initial rungs for Trading-Algo:

```text
surrogate_or_memory_prior
-> cheap_is_slice
-> fold_race
-> full_snapshot
-> oos_audit_metadata
```

Scheduler policies:

- successive halving: promote a fixed fraction each rung
- racing: eliminate candidates when enough cheap/fold evidence says they are dominated
- trust-region surrogate mode: let the surrogate advise only near calibrated regions
- exploration quota: reserve full or partial budget for diversity and rare families
- audit quota: periodically fully evaluate candidates that the surrogate would reject

The first implementation should ship one simple successive-halving/racing policy, then add
surrogate trust management later.

## Advisor Layer

Advisors are optional, composable sources of candidate ranking or proposal.

Advisor types:

- surrogate model advisor
- search-memory prior advisor
- novelty/diversity advisor
- family quota advisor
- uncertainty advisor

Advisor output must include:

- ranking score
- confidence
- explanation/reason
- feature vector or metadata used for training/audit when applicable

Surrogates must be calibrated with true evaluations. The policy should keep an audit sample
of rejected candidates to estimate false-negative risk.

## Anti-Overfit Telemetry

EvoCore should not implement all financial-statistics gates itself in vNext phase 1, but it
must export the evidence those gates need.

Required telemetry:

- total candidates proposed
- unique candidates by hash
- candidates screened by surrogate
- candidates partial-evaluated
- candidates full-evaluated
- candidates promoted/eliminated per rung
- evaluator cost per rung
- strategy family/category metadata
- fold/window identifiers when provided by the evaluator
- selected candidate lineage
- optimizer seed and policy configuration
- audit sample outcomes

Trading-Algo can then feed trial counts and lineage into Deflated Sharpe Ratio, White's
Reality Check, Hansen SPA, and Model Confidence Set workflows without undercounting the
search breadth.

## Python API Shape

The public API should favor policy objects over a large constructor full of unrelated
parameters.

Sketch:

```python
from evocore import (
    GeneDef,
    GeneSpace,
    GAEngine,
    MultiFidelityPolicy,
    Rung,
)

space = GeneSpace([...])

policy = MultiFidelityPolicy(
    rungs=[
        Rung("cheap_is_slice", budget=0.10, promote_fraction=0.35),
        Rung("fold_race", budget=0.35, promote_fraction=0.25),
        Rung("full_snapshot", budget=1.00, promote_fraction=1.00),
    ],
    full_evaluation_budget=256,
    exploration_fraction=0.10,
)

engine = GAEngine(space, population_size=128, seed=42)
result = engine.run(evaluator, policy=policy)
```

Evaluator protocol:

```python
class Evaluator(Protocol):
    def evaluate(self, candidates: Sequence[Candidate], rung: Rung) -> Sequence[EvaluationRecord]:
        ...
```

## Rust/Python Boundary

Python owns:

- public API
- policy configuration
- evaluator protocols
- callbacks and user-facing records
- Trading-Algo integration examples
- docs and examples

Rust owns:

- compact candidate representation
- repeated selection/replacement/ranking loops
- deterministic seed derivation
- mutation/crossover batches
- CMA numerical state
- mixed-variable sampling/update math
- efficient telemetry aggregation where it is hot

This follows the existing project decision: Python for ergonomics and orchestration, Rust
for repeated numerical/population work.

## Trading-Algo Reference Integration

Trading-Algo should be the first proving ground.

Reference evaluator rungs:

- `cheap_is_slice`: small deterministic row or date slice using compact backtest metrics.
- `fold_race`: one or more fold-aware evaluations that can eliminate candidates early.
- `full_snapshot`: existing snapshot benchmark backtest path.
- `oos_audit_metadata`: optional metadata export for downstream gate analysis, not optimizer
  training by default.

Reference success metrics:

- reduce full snapshot evaluations needed to reach equal or better finalist quality
- improve wall-clock per useful finalist
- preserve or improve full-snapshot best-fitness distribution
- increase visibility of eliminated/promoted candidates
- export trial counts compatible with Trading-Algo's gate accounting

## Error Handling

- Unknown rungs, duplicate rung names, or invalid promotion fractions raise configuration
  errors before a run starts.
- `tell()` rejects records for unknown candidate IDs unless policy explicitly allows external
  seeded records.
- `tell()` rejects conflicting duplicate trusted records unless the evaluator marks them as
  repeated observations.
- Surrogate records cannot update trusted optimizer state unless the policy explicitly opts
  into aggressive surrogate updates.
- A run that exhausts full-evaluation budget without any trusted candidate returns a clear
  failed result or raises a fitness error depending on configured failure policy.
- Evaluator exceptions should preserve current wrapped-error clarity but include candidate ID,
  rung name, and policy event index.

## Testing

Required unit and integration tests:

- candidate lifecycle transitions
- deterministic `candidate_id` and lineage
- ask/tell determinism under fixed seed
- scheduler promotion and elimination behavior
- full vs partial vs surrogate confidence handling
- GA state updates from trusted records only by default
- CMA state updates from trusted records only by default
- mixed-variable encoding for float, integer, categorical, and fixed genes
- surrogate advisor cannot silently mark scores as trusted
- audit quota sends some rejected candidates to full evaluation
- evaluator exception context includes candidate ID and rung
- result telemetry counts proposed, screened, partial, full, promoted, and eliminated candidates
- Trading-Algo-style mocked rungs reproduce expected budget savings
- docs examples execute as smoke tests where practical

Rust tests:

- confidence-mask ranking
- rung-aware top-k promotion
- deterministic candidate ID seed derivation
- mixed-variable sampling bounds
- integer margin behavior
- categorical distribution update invariants
- fixed-gene reconstruction

Benchmark tests:

- compare single-fidelity baseline against multi-fidelity policy on deterministic synthetic
  expensive functions
- compare false-negative audit behavior on known deceptive cheap-fitness functions
- run a small Trading-Algo-style mock evaluator before requiring real snapshot benchmark runs

## Release, Docs, And Versioning

The implementation plan must include a release hygiene slice.

Required release tasks:

- Update `pyproject.toml` version.
- Update `Cargo.toml` version.
- Update `CHANGELOG.md` with a breaking vNext section.
- Update `README.md` to describe EvoCore as an expensive black-box optimization library,
  not primarily as a DEAP replacement.
- Update MkDocs navigation in `mkdocs.yml`.
- Add new MkDocs pages for budget-aware optimization, ask/tell engines, mixed-variable
  search, and optimizer telemetry.
- Remove or demote the public `GA Benchmark Parity` page from primary navigation.
- Add or refresh public docstrings for all new user-facing classes and protocols.
- Update examples to demonstrate vNext multi-fidelity optimization.
- Ensure type stubs and `py.typed` coverage stay current for new public APIs.

Recommended version target:

- Use `0.7.0` if this remains a pre-1.0 breaking optimizer architecture.
- Use `1.0.0a1` only if the implementation plan also commits to a public API stabilization
  track.

## Rollout Phases

### Phase 1: vNext Primitives And Ask/Tell GA

- Candidate and evaluation record types.
- Ask/tell GA engine.
- Scheduler skeleton.
- Basic trusted-only state update semantics.
- Telemetry in `RunResult`.
- Version, changelog, MkDocs, examples, and docstrings.

### Phase 2: Racing And Trading-Algo Rungs

- Successive-halving/racing policy.
- Mocked Trading-Algo-style evaluator.
- Reference downstream integration guide.
- Budget-savings benchmark.

### Phase 3: Surrogate Advisor

- Advisor API.
- Simple baseline surrogate.
- Calibration/audit sampling.
- Surrogate telemetry and false-negative reporting.

### Phase 4: Mixed-Variable CMA

- Ask/tell CMA state.
- Fixed-gene reconstruction.
- Integer margin support.
- Categorical distribution support.
- Trusted-record distribution update semantics.

### Phase 5: Anti-Overfit Export

- Trial accounting export.
- Fold/window lineage export.
- Gate-oriented report schema.
- Integration notes for DSR, White Reality Check, Hansen SPA, and Model Confidence Set.

## Acceptance Criteria

- EvoCore vNext no longer uses DEAP parity as an acceptance criterion.
- GA supports ask/tell and multi-fidelity scheduler integration.
- CMA has an ask/tell seam and a clear path to mixed-variable support.
- Partial, surrogate, cached, rejected, and trusted-full evaluations are distinguishable in
  state and results.
- The scheduler can reduce full evaluations on a mocked expensive benchmark while preserving
  finalist quality within a documented tolerance.
- Result telemetry is sufficient for downstream anti-overfit trial accounting.
- Trading-Algo reference examples can express cheap slice, fold race, and full snapshot rungs.
- Version, changelog, MkDocs, README, examples, type stubs, and public docstrings are updated
  as part of the implementation.

## Risks And Mitigations

- Risk: Surrogate or cheap-rung false negatives discard rare good candidates.
  Mitigation: require exploration and audit quotas, and keep full false-negative telemetry.
- Risk: CMA updates from partial scores corrupt the distribution.
  Mitigation: default CMA updates to trusted records only.
- Risk: The architecture is too large for one branch.
  Mitigation: implement phases in separate reviewable slices, starting with primitives and
  ask/tell GA.
- Risk: Trading-Algo-specific assumptions leak into EvoCore.
  Mitigation: keep rungs as user-provided evaluator protocols and use Trading-Algo only as
  reference examples/tests.
- Risk: Removing DEAP parity disrupts current users.
  Mitigation: document the breaking version clearly and maintain release notes; vNext is a
  product direction change, not a silent patch release.
