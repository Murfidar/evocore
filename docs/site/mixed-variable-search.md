# Mixed-Variable Search

EvoCore vNext starts moving beyond continuous-only CMA by separating continuous, integer,
categorical, and fixed-gene behavior.

`IntegerMarginDistribution` protects integer probability mass so integer genes do not collapse too
quickly. `CategoricalDistributionState` tracks categorical-by-integer probability updates for future
mixed CMA engines.
