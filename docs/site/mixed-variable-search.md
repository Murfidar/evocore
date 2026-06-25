# Mixed-Variable Search

EvoCore vNext starts moving beyond continuous-only CMA by separating continuous, integer,
categorical, and fixed-gene behavior.

`IntegerMarginDistribution` protects integer probability mass so integer genes do not collapse too
quickly. `CategoricalDistributionState` tracks categorical-by-integer probability updates for future
mixed CMA engines.

## CMA-ES Integer Strategy

`CMAESOptimizer` keeps `integer_strategy="round"` as the default for backward
compatibility. In this mode, CMA samples continuous latent values and EvoCore
repairs the public candidate values into integer bounds.

Use `integer_strategy="margin"` when native integer coordinates need protected
sampling probability:

```python
from evocore import CMAESOptimizer, Gene, GeneSpace


space = GeneSpace([Gene("period", "int", 2, 20), Gene("threshold", "float", 0.0, 1.0)])
optimizer = CMAESOptimizer(
    space,
    population_size=12,
    seed=42,
    integer_strategy="margin",
    integer_min_probability=0.02,
)
```

The margin strategy is opt-in, participates in optimizer config hashes, and is
checkpointed for exact ask/tell resume.
