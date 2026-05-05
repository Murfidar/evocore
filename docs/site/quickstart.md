# Quickstart

```python
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


engine = GAEngine(
    GeneSpace.uniform(-5.0, 5.0, 10),
    population_size=100,
    generations=100,
    seed=42,
)
result = engine.run(sphere)

print(result.best_fitness)
print(result.best_individual.genes)
```
