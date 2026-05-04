# evocore v5 - Part 5: Python API Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the user-facing Python foundation that Parts 6 and 7 depend on: `GeneSpace`, `Individual`, `Population`, operator-boundary helpers, statistics/logbook, callbacks with `GenerationInfo`, and parallel evaluation wrappers.

**Architecture:** The Python layer owns API ergonomics, validation, typed gene decoding, callback orchestration, logbook objects, warnings, and process/thread worker setup. The Rust extension remains the hot path for initialization, reproduction, selection, operators, and CMA-ES state. All genes crossing into Rust are encoded as `list[list[float]]`; Python `OperatorSet` is the canonical encoder/decoder and the source of per-gene sigma lists.

**Tech Stack:** Python 3.11+, pytest, optional pandas/matplotlib/tqdm, existing `evocore._core` from Parts 1-4

**Prerequisite:** Parts 1-4 complete. In particular, `_core.init_population`, `_core.reproduce_population`, `_core.evaluate_sequential`, `_core.evaluate_parallel_rayon`, and `_core.PyCMAESState` are importable.

---

## Review Notes from Parts 1-4

- Part 4 was patched during review: `ask(seed, gen)` is deterministic only for a fixed CMA-ES state. After `tell()` changes mean/covariance, asking with the same seed/generation may correctly produce different samples.
- Part 3's `_core.evaluate_*` functions accept encoded gene lists and return floats. The public Python API still accepts fitness functions that receive `Individual` and may return `(fitness, metrics)`. Parts 5-7 therefore keep public evaluation wrappers in Python and use Rust for population initialization/reproduction/CMA-ES state. The low-level Rust evaluation functions remain tested and available for raw gene-list workloads.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evocore/gene_space.py` | Create | `GeneDef`, `GeneSpace`, validation, bounds/kind helpers |
| `evocore/individual.py` | Create | `Individual`, `Population`, params metadata, fitness summaries, diversity |
| `evocore/operators.py` | Create | `OperatorSet` validation, f64 boundary encoding/decoding, per-gene sigma list |
| `evocore/stats.py` | Create | `LogEntry`, `Logbook`, optional dataframe/plot helpers |
| `evocore/callbacks.py` | Create | `GenerationInfo`, callback base, early stopping, progress, checkpoint, metrics logger |
| `evocore/parallel.py` | Create | `ThreadParallel`, `ProcessParallel`, picklability probe, v5 teardown |
| `evocore/__init__.py` | Modify | Add top-level exports for Part 5 public classes |
| `tests/unit/test_gene_space.py` | Create | Gene definition and space validation |
| `tests/unit/test_individual.py` | Create | Individual/Population behavior |
| `tests/unit/test_operators.py` | Create | OperatorSet validation, encoding, decoding, sigma |
| `tests/unit/test_stats.py` | Create | Logbook basics and optional dependency errors |
| `tests/unit/test_callbacks.py` | Create | GenerationInfo and built-in callbacks |
| `tests/unit/test_parallel.py` | Create/extend | Spawn context, picklability probe, teardown-friendly process wrapper |

---

## Task 1: `evocore/gene_space.py`

**Files:**
- Create: `evocore/gene_space.py`
- Create: `tests/unit/test_gene_space.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneDef, GeneSpace


def test_gene_def_float_requires_bounds():
    with pytest.raises(ConfigurationError, match="bounds required"):
        GeneDef("x", "float")


def test_gene_def_int_requires_integer_bounds():
    with pytest.raises(ConfigurationError, match="integer bounds"):
        GeneDef("period", "int", 1.5, 10)


def test_gene_def_bool_rejects_bounds():
    with pytest.raises(ConfigurationError, match="bool genes do not use bounds"):
        GeneDef("flag", "bool", 0, 1)


def test_gene_def_sigma_range():
    with pytest.raises(ConfigurationError, match="sigma"):
        GeneDef("x", "float", -1.0, 1.0, sigma=1.5)


def test_uniform_space_properties():
    space = GeneSpace.uniform(-5.0, 5.0, 3)
    assert space.length == 3
    assert space.kinds == ["float", "float", "float"]
    assert space.bounds == [(-5.0, 5.0)] * 3
    assert space.has_names is False
    assert space.params_for([1.0, 2.0, 3.0]) is None


def test_named_space_params():
    space = GeneSpace([
        GeneDef("fast", "int", 5, 50),
        GeneDef("threshold", "float", 0.0, 1.0),
        GeneDef("enabled", "bool"),
    ])
    assert space.has_names is True
    assert space.names == ["fast", "threshold", "enabled"]
    assert space.params_for([10, 0.25, True]) == {
        "fast": 10,
        "threshold": 0.25,
        "enabled": True,
    }


def test_duplicate_names_rejected():
    with pytest.raises(ConfigurationError, match="Duplicate gene name"):
        GeneSpace([GeneDef("x", "float", 0.0, 1.0), GeneDef("x", "float", 0.0, 1.0)])


def test_rust_bounds_encode_bool_as_zero_one():
    space = GeneSpace([GeneDef("flag", "bool")])
    assert space.rust_bounds == [(0.0, 1.0)]
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/unit/test_gene_space.py -v`

Expected: `ModuleNotFoundError: No module named 'evocore.gene_space'`

- [ ] **Step 3: Implement `evocore/gene_space.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from evocore.exceptions import ConfigurationError

GeneKind = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class GeneDef:
    name: str
    kind: GeneKind
    low: float | int | None = None
    high: float | int | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("GeneDef name must be a non-empty string.")
        if self.kind not in ("float", "int", "bool"):
            raise ConfigurationError("GeneDef kind must be 'float', 'int', or 'bool'.")
        if self.kind == "bool":
            if self.low is not None or self.high is not None:
                raise ConfigurationError("bool genes do not use bounds; pass GeneDef(name, 'bool').")
        else:
            if self.low is None or self.high is None:
                raise ConfigurationError(f"bounds required for {self.kind} gene '{self.name}'.")
            if self.low >= self.high:
                raise ConfigurationError(f"GeneDef('{self.name}') requires low < high.")
            if self.kind == "int" and (not isinstance(self.low, int) or not isinstance(self.high, int)):
                raise ConfigurationError(f"GeneDef('{self.name}') with kind='int' requires integer bounds.")
        if self.sigma is not None and not (0.0 < self.sigma <= 1.0):
            raise ConfigurationError("GeneDef sigma must be in (0, 1].")


class GeneSpace:
    def __init__(self, genes: Sequence[GeneDef], *, has_names: bool = True) -> None:
        if not genes:
            raise ConfigurationError("GeneSpace requires at least one GeneDef.")
        self._genes = tuple(genes)
        self._has_names = bool(has_names)
        if self._has_names:
            seen: set[str] = set()
            for gene in self._genes:
                if gene.name in seen:
                    raise ConfigurationError(f"Duplicate gene name: {gene.name!r}.")
                seen.add(gene.name)

    @classmethod
    def uniform(cls, low: float, high: float, length: int) -> "GeneSpace":
        if length <= 0:
            raise ConfigurationError("GeneSpace.uniform length must be positive.")
        if low >= high:
            raise ConfigurationError("GeneSpace.uniform requires low < high.")
        return cls(
            [GeneDef(f"gene_{i}", "float", float(low), float(high)) for i in range(length)],
            has_names=False,
        )

    @property
    def genes(self) -> tuple[GeneDef, ...]:
        return self._genes

    @property
    def length(self) -> int:
        return len(self._genes)

    @property
    def names(self) -> list[str]:
        return [g.name for g in self._genes]

    @property
    def kinds(self) -> list[str]:
        return [g.kind for g in self._genes]

    @property
    def bounds(self) -> list[tuple[float | int, float | int] | None]:
        return [None if g.kind == "bool" else (g.low, g.high) for g in self._genes]

    @property
    def rust_bounds(self) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for gene in self._genes:
            if gene.kind == "bool":
                result.append((0.0, 1.0))
            else:
                result.append((float(gene.low), float(gene.high)))
        return result

    @property
    def has_names(self) -> bool:
        return self._has_names

    def params_for(self, genes: Sequence[float | int | bool]) -> dict[str, float | int | bool] | None:
        if len(genes) != self.length:
            raise ConfigurationError(
                f"Expected {self.length} genes for params mapping, got {len(genes)}."
            )
        if not self._has_names:
            return None
        return dict(zip(self.names, genes))
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/test_gene_space.py -v`

Expected: all tests pass.

```bash
git add evocore/gene_space.py tests/unit/test_gene_space.py
git commit -m "feat(python): GeneDef and GeneSpace foundation"
```

---

## Task 2: `evocore/individual.py`

**Files:**
- Create: `evocore/individual.py`
- Create: `tests/unit/test_individual.py`

- [ ] **Step 1: Write failing tests**

```python
from evocore.individual import Individual, Population


def test_individual_params_property():
    ind = Individual([10, 0.5], metadata={"params": {"fast": 10, "threshold": 0.5}})
    assert ind.params == {"fast": 10, "threshold": 0.5}


def test_population_best_ignores_none_fitness():
    pop = Population([
        Individual([1.0], fitness=None),
        Individual([2.0], fitness=5.0),
        Individual([3.0], fitness=2.0),
    ])
    assert pop.best()[0].genes == [2.0]


def test_population_mean_and_std():
    pop = Population([Individual([0], fitness=1.0), Individual([1], fitness=3.0)])
    assert pop.mean_fitness() == 2.0
    assert pop.std_fitness() == 1.0


def test_population_diversity_bool_as_numeric():
    pop = Population([Individual([False, 0.0]), Individual([True, 2.0])])
    div = pop.diversity()
    assert len(div) == 2
    assert div[0] > 0.0
    assert div[1] > 0.0
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/unit/test_individual.py -v`

Expected: `ModuleNotFoundError: No module named 'evocore.individual'`

- [ ] **Step 3: Implement `evocore/individual.py`**

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Iterator, Sequence

GeneValue = float | int | bool


@dataclass
class Individual:
    genes: list[GeneValue]
    fitness: float | None = None
    fitness_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, GeneValue] | None:
        return self.metadata.get("params")

    def clone(self) -> "Individual":
        return Individual(
            genes=list(self.genes),
            fitness=self.fitness,
            fitness_valid=self.fitness_valid,
            metadata=dict(self.metadata),
        )


class Population(Sequence[Individual]):
    def __init__(self, individuals: Sequence[Individual]) -> None:
        self._individuals = list(individuals)

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self._individuals[idx]

    def as_list(self) -> list[Individual]:
        return list(self._individuals)

    @staticmethod
    def _fitness_key(ind: Individual) -> float:
        value = ind.fitness
        if value is None or math.isnan(value):
            return float("-inf")
        if value == float("inf"):
            return float("inf")
        return value

    def best(self, n: int = 1) -> list[Individual]:
        if n <= 0:
            return []
        return sorted(self._individuals, key=self._fitness_key, reverse=True)[:n]

    def _finite_fitnesses(self) -> list[float]:
        values: list[float] = []
        for ind in self._individuals:
            if ind.fitness is not None and math.isfinite(ind.fitness):
                values.append(float(ind.fitness))
        return values

    def mean_fitness(self) -> float:
        values = self._finite_fitnesses()
        return mean(values) if values else float("nan")

    def std_fitness(self) -> float:
        values = self._finite_fitnesses()
        return pstdev(values) if len(values) > 1 else 0.0

    def diversity(self) -> list[float]:
        if not self._individuals:
            return []
        gene_len = len(self._individuals[0].genes)
        result: list[float] = []
        for gene_idx in range(gene_len):
            values = [float(ind.genes[gene_idx]) for ind in self._individuals]
            result.append(pstdev(values) if len(values) > 1 else 0.0)
        return result

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Population.to_dataframe() requires pandas. Install with: pip install pandas") from exc
        rows = []
        for ind in self._individuals:
            row = {f"gene_{i}": value for i, value in enumerate(ind.genes)}
            row["fitness"] = ind.fitness
            row["fitness_valid"] = ind.fitness_valid
            rows.append(row)
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/test_individual.py -v`

Expected: all tests pass.

```bash
git add evocore/individual.py tests/unit/test_individual.py
git commit -m "feat(python): Individual and Population containers"
```

---

## Task 3: `evocore/operators.py`

**Files:**
- Create: `evocore/operators.py`
- Create: `tests/unit/test_operators.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual
from evocore.operators import OperatorSet


def test_numeric_space_accepts_sbx_gaussian():
    ops = OperatorSet(GeneSpace.uniform(-1.0, 1.0, 2), "sbx", "gaussian")
    assert ops.gene_kinds == ["float", "float"]


def test_binary_space_rejects_sbx():
    space = GeneSpace([GeneDef("a", "bool"), GeneDef("b", "bool")])
    with pytest.raises(ConfigurationError, match="binary"):
        OperatorSet(space, "sbx", "bit_flip")


def test_mixed_bool_numeric_rejected():
    space = GeneSpace([GeneDef("x", "float", 0.0, 1.0), GeneDef("flag", "bool")])
    with pytest.raises(ConfigurationError, match="bool genes alongside"):
        OperatorSet(space, "sbx", "gaussian")


def test_encode_decode_roundtrip_named_mixed_numeric():
    space = GeneSpace([GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)])
    ops = OperatorSet(space, "sbx", "gaussian")
    encoded = ops.encode_genes([10, 0.25])
    assert encoded == [10.0, 0.25]
    decoded = ops.decode_genes([10.2, 0.25])
    assert decoded == [10, 0.25]


def test_decode_individual_adds_params_metadata():
    space = GeneSpace([GeneDef("period", "int", 5, 20)])
    ops = OperatorSet(space, "sbx", "gaussian")
    ind = ops.decode_individual([12.0], fitness=3.0, fitness_valid=True)
    assert ind.genes == [12]
    assert ind.fitness == 3.0
    assert ind.fitness_valid is True
    assert ind.params == {"period": 12}


def test_sigma_override_takes_precedence():
    space = GeneSpace([GeneDef("wide", "int", 0, 1000, sigma=0.01), GeneDef("x", "float", -1.0, 1.0)])
    ops = OperatorSet(space, "sbx", "gaussian")
    assert ops.sigma_abs_list(0.2) == [10.0, 0.4]
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/unit/test_operators.py -v`

Expected: `ModuleNotFoundError: No module named 'evocore.operators'`

- [ ] **Step 3: Implement `evocore/operators.py`**

```python
from __future__ import annotations

from typing import Sequence

from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneSpace
from evocore.individual import Individual

NUMERIC_CROSSOVERS = {"sbx", "blx"}
BINARY_CROSSOVERS = {"one_point", "two_point", "uniform"}
NUMERIC_MUTATIONS = {"gaussian", "uniform"}
BINARY_MUTATIONS = {"bit_flip"}


class OperatorSet:
    def __init__(self, gene_space: GeneSpace, crossover: str, mutation: str) -> None:
        self.gene_space = gene_space
        self.crossover = crossover
        self.mutation = mutation
        self._validate()

    def _validate(self) -> None:
        kinds = set(self.gene_space.kinds)
        if "bool" in kinds and len(kinds) > 1:
            raise ConfigurationError(
                "GeneSpace contains bool genes alongside float/int genes. "
                "Use a binary-only space or encode booleans as int genes with low=0, high=1."
            )
        if kinds == {"bool"}:
            if self.crossover not in BINARY_CROSSOVERS:
                raise ConfigurationError(
                    "binary GeneSpace requires crossover='one_point', 'two_point', or 'uniform'."
                )
            if self.mutation not in BINARY_MUTATIONS:
                raise ConfigurationError("binary GeneSpace requires mutation='bit_flip'.")
        else:
            if self.crossover not in NUMERIC_CROSSOVERS:
                raise ConfigurationError("float/int GeneSpace requires crossover='sbx' or 'blx'.")
            if self.mutation not in NUMERIC_MUTATIONS:
                raise ConfigurationError("float/int GeneSpace requires mutation='gaussian' or 'uniform'.")

    @property
    def gene_kinds(self) -> list[str]:
        return self.gene_space.kinds

    @property
    def gene_bounds(self) -> list[tuple[float, float]]:
        return self.gene_space.rust_bounds

    def encode_genes(self, genes: Sequence[float | int | bool]) -> list[float]:
        if len(genes) != self.gene_space.length:
            raise ConfigurationError(f"Expected {self.gene_space.length} genes, got {len(genes)}.")
        encoded: list[float] = []
        for value, gene in zip(genes, self.gene_space.genes):
            if gene.kind == "bool":
                encoded.append(1.0 if bool(value) else 0.0)
            elif gene.kind == "int":
                encoded.append(float(int(value)))
            else:
                encoded.append(float(value))
        return encoded

    def decode_genes(self, genes_f64: Sequence[float]) -> list[float | int | bool]:
        if len(genes_f64) != self.gene_space.length:
            raise ConfigurationError(f"Expected {self.gene_space.length} encoded genes, got {len(genes_f64)}.")
        decoded: list[float | int | bool] = []
        for value, gene in zip(genes_f64, self.gene_space.genes):
            if gene.kind == "bool":
                decoded.append(bool(value >= 0.5))
            elif gene.kind == "int":
                decoded.append(int(round(value)))
            else:
                decoded.append(float(value))
        return decoded

    def encode_population(self, population: Sequence[Individual]) -> list[list[float]]:
        return [self.encode_genes(ind.genes) for ind in population]

    def decode_individual(
        self,
        genes_f64: Sequence[float],
        *,
        fitness: float | None = None,
        fitness_valid: bool = False,
        metadata: dict | None = None,
    ) -> Individual:
        genes = self.decode_genes(genes_f64)
        md = dict(metadata or {})
        params = self.gene_space.params_for(genes)
        if params is not None:
            md["params"] = params
        return Individual(list(genes), fitness=fitness, fitness_valid=fitness_valid, metadata=md)

    def decode_population(self, population_f64: Sequence[Sequence[float]]) -> list[Individual]:
        return [self.decode_individual(genes) for genes in population_f64]

    def sigma_abs_list(self, global_sigma_fraction: float) -> list[float]:
        if not (0.0 <= global_sigma_fraction <= 1.0):
            raise ConfigurationError("mutation_sigma must be in [0, 1].")
        sigmas: list[float] = []
        for gene in self.gene_space.genes:
            if gene.kind == "bool":
                sigmas.append(0.0)
                continue
            lo = float(gene.low)
            hi = float(gene.high)
            fraction = gene.sigma if gene.sigma is not None else global_sigma_fraction
            sigmas.append(float(fraction) * (hi - lo))
        return sigmas
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/unit/test_operators.py -v`

Expected: all tests pass.

```bash
git add evocore/operators.py tests/unit/test_operators.py
git commit -m "feat(python): OperatorSet encoding and validation"
```

---

## Task 4: `evocore/stats.py`

**Files:**
- Create: `evocore/stats.py`
- Create: `tests/unit/test_stats.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from evocore.stats import LogEntry, Logbook


def test_logbook_append_len_iter_getitem():
    book = Logbook()
    entry = LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 2, [], {"sharpe": 1.2})
    book.append(entry)
    assert len(book) == 1
    assert list(book) == [entry]
    assert book[0].custom["sharpe"] == 1.2


def test_logbook_fitness_lists():
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {}))
    book.append(LogEntry(1, 2.0, 1.0, 0.2, 15.0, 10, 1, 0, [], {}))
    assert book.best_fitnesses() == [1.0, 2.0]
    assert book.nan_counts() == [0, 1]


def test_to_dataframe_missing_pandas_message(monkeypatch):
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {}))
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="pip install pandas"):
        book.to_dataframe()
```

- [ ] **Step 2: Implement `evocore/stats.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LogEntry:
    gen: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    wall_time_ms: float
    n_evaluations: int
    nan_fitness_count: int
    cached_count: int
    diversity: list[float] = field(default_factory=list)
    custom: dict = field(default_factory=dict)


class Logbook:
    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def append(self, entry: LogEntry) -> None:
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._entries)

    def __getitem__(self, idx: int) -> LogEntry:
        return self._entries[idx]

    def best_fitnesses(self) -> list[float]:
        return [entry.best_fitness for entry in self._entries]

    def nan_counts(self) -> list[int]:
        return [entry.nan_fitness_count for entry in self._entries]

    def to_rows(self) -> list[dict]:
        rows: list[dict] = []
        for entry in self._entries:
            row = {
                "gen": entry.gen,
                "best_fitness": entry.best_fitness,
                "mean_fitness": entry.mean_fitness,
                "std_fitness": entry.std_fitness,
                "wall_time_ms": entry.wall_time_ms,
                "n_evaluations": entry.n_evaluations,
                "nan_fitness_count": entry.nan_fitness_count,
                "cached_count": entry.cached_count,
            }
            row.update(entry.custom)
            rows.append(row)
        return rows

    def print(self) -> None:
        for row in self.to_rows():
            print(row)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Logbook.to_dataframe() requires pandas. Install with: pip install pandas") from exc
        return pd.DataFrame(self.to_rows())

    def plot(self, metrics: list[str] | None = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Logbook.plot() requires matplotlib. Install with: pip install matplotlib") from exc
        metrics = metrics or ["best_fitness", "mean_fitness"]
        rows = self.to_rows()
        fig, ax = plt.subplots()
        xs = [row["gen"] for row in rows]
        for metric in metrics:
            ax.plot(xs, [row.get(metric) for row in rows], label=metric)
        ax.set_xlabel("generation")
        ax.legend()
        return fig
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_stats.py -v`

Expected: all tests pass.

```bash
git add evocore/stats.py tests/unit/test_stats.py
git commit -m "feat(python): LogEntry and Logbook"
```

---

## Task 5: `evocore/callbacks.py`

**Files:**
- Create: `evocore/callbacks.py`
- Create: `tests/unit/test_callbacks.py`

- [ ] **Step 1: Write failing tests**

```python
import json
import pickle
from evocore.callbacks import CheckpointCallback, EarlyStopping, GenerationInfo, MetricsLogger
from evocore.individual import Individual, Population


def test_generation_info_fields():
    info = GenerationInfo(generation=2, nan_fitness_count=1, cached_count=3)
    assert info.generation == 2
    assert info.nan_fitness_count == 1
    assert info.cached_count == 3


def test_early_stopping_sets_should_stop():
    cb = EarlyStopping(patience=2, min_delta=0.01)
    pop = Population([Individual([0.0], fitness=1.0)])
    info = GenerationInfo(0, 0, 0)
    cb.on_generation_end(0, pop, info)
    cb.on_generation_end(1, pop, info)
    cb.on_generation_end(2, pop, info)
    assert cb.should_stop is True


def test_checkpoint_callback_writes_pickle(tmp_path):
    cb = CheckpointCallback(path=str(tmp_path), every=1)
    cb.bind_context(seed=42)
    pop = Population([Individual([1.0], fitness=2.0)])
    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))
    payload = pickle.loads((tmp_path / "checkpoint_gen_3.pkl").read_bytes())
    assert payload["generation"] == 3
    assert payload["seed"] == 42


def test_metrics_logger_uses_utf8_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    cb = MetricsLogger(str(path))
    pop = Population([Individual([1.0], fitness=2.0)])
    cb.on_generation_end(0, pop, GenerationInfo(0, 1, 2))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["nan_fitness_count"] == 1
    assert record["cached_count"] == 2
```

- [ ] **Step 2: Implement `evocore/callbacks.py`**

```python
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evocore.ga import RunResult
    from evocore.individual import Population


@dataclass
class GenerationInfo:
    generation: int
    nan_fitness_count: int
    cached_count: int


class Callback:
    should_stop: bool = False

    def bind_context(self, **kwargs) -> None:
        pass

    def on_generation_start(self, gen: int, pop: "Population") -> None:
        pass

    def on_generation_end(self, gen: int, pop: "Population", info: GenerationInfo) -> None:
        pass

    def on_run_end(self, result: "RunResult") -> None:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.should_stop = False
        self._best = float("-inf")
        self._no_improve_count = 0

    def on_generation_end(self, gen: int, pop: "Population", info: GenerationInfo) -> None:
        best = pop.best(1)
        if not best or best[0].fitness is None:
            return
        fitness = float(best[0].fitness)
        if fitness - self._best > self.min_delta:
            self._best = fitness
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
            if self._no_improve_count >= self.patience:
                self.should_stop = True


class ProgressBar(Callback):
    def __init__(self) -> None:
        self._bar = None
        self._total = None

    def bind_context(self, **kwargs) -> None:
        self._total = kwargs.get("generations")

    def on_generation_start(self, gen: int, pop: "Population") -> None:
        if self._bar is None:
            try:
                from tqdm import tqdm
            except ImportError:
                self._bar = False
                return
            self._bar = tqdm(total=self._total)

    def on_generation_end(self, gen: int, pop: "Population", info: GenerationInfo) -> None:
        if not self._bar:
            return
        best = pop.best(1)
        postfix = {"best": best[0].fitness if best else None}
        if info.nan_fitness_count:
            postfix["nan"] = info.nan_fitness_count
        self._bar.set_postfix(**postfix)
        self._bar.update(1)

    def on_run_end(self, result: "RunResult") -> None:
        if self._bar:
            self._bar.close()


class CheckpointCallback(Callback):
    def __init__(self, path: str = "./checkpoints", every: int = 10) -> None:
        self.path = path
        self.every = every
        self._seed: int | None = None

    def bind_context(self, **kwargs) -> None:
        self._seed = kwargs.get("seed")

    def on_generation_end(self, gen: int, pop: "Population", info: GenerationInfo) -> None:
        if self.every <= 0 or gen % self.every != 0:
            return
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.pkl")
        with open(filename, "wb") as f:
            pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, f)


class MetricsLogger(Callback):
    def __init__(self, path: str = "./metrics.jsonl") -> None:
        self.path = path

    def on_generation_end(self, gen: int, pop: "Population", info: GenerationInfo) -> None:
        best = pop.best(1)
        record = {
            "generation": gen,
            "best_fitness": best[0].fitness if best else None,
            "nan_fitness_count": info.nan_fitness_count,
            "cached_count": info.cached_count,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_callbacks.py -v`

Expected: all tests pass.

```bash
git add evocore/callbacks.py tests/unit/test_callbacks.py
git commit -m "feat(python): callbacks and GenerationInfo"
```

---

## Task 6: `evocore/parallel.py`

**Files:**
- Create: `evocore/parallel.py`
- Create/extend: `tests/unit/test_parallel.py`

- [ ] **Step 1: Write failing tests**

```python
import pickle
import pytest
from evocore.exceptions import ConfigurationError
from evocore.individual import Individual
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable


def module_level_fitness(ind):
    return sum(ind.genes)


def test_ensure_picklable_rejects_lambda():
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        ensure_picklable(lambda ind: 1.0, context="parallel='process'")


def test_process_parallel_forces_spawn_context():
    pp = ProcessParallel(n_workers=2)
    assert pp._ctx.get_start_method() == "spawn"


def test_thread_parallel_evaluates_population():
    tp = ThreadParallel(n_workers=2)
    pop = [Individual([1.0]), Individual([2.0])]
    assert tp.evaluate(pop, module_level_fitness) == [1.0, 2.0]


def test_process_parallel_evaluates_population():
    pp = ProcessParallel(n_workers=2)
    pop = [Individual([1.0]), Individual([2.0])]
    assert pp.evaluate(pop, module_level_fitness) == [1.0, 2.0]
```

- [ ] **Step 2: Implement `evocore/parallel.py`**

```python
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import pickle
from collections.abc import Callable, Sequence

from evocore.exceptions import ConfigurationError
from evocore.individual import Individual


def ensure_picklable(obj, *, context: str) -> None:
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ConfigurationError(
            f"fitness_fn cannot be pickled, required for {context}.\n"
            f"  Error: {exc}\n"
            "  Fix: define fitness_fn at module level, not as a lambda or nested function."
        ) from exc


class ThreadParallel:
    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = n_workers or os.cpu_count() or 1

    def evaluate(self, population: Sequence[Individual], fitness_fn: Callable[[Individual], object]) -> list[object]:
        if not population:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(fitness_fn, population))


class ProcessParallel:
    """
    ProcessPoolExecutor wrapper using spawn everywhere.

    KeyboardInterrupt behavior: queued futures are cancelled and the pool is
    asked to shut down without waiting for already-running evaluations.
    """

    def __init__(self, n_workers: int | None = None, initializer=None, initargs=()) -> None:
        self.n_workers = n_workers or os.cpu_count() or 1
        self.initializer = initializer
        self.initargs = initargs
        self._ctx = multiprocessing.get_context("spawn")

    def evaluate(self, population: Sequence[Individual], fitness_fn: Callable[[Individual], object]) -> list[object]:
        if not population:
            return []
        ensure_picklable(fitness_fn, context="parallel='process'")
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=self._ctx,
            initializer=self.initializer,
            initargs=self.initargs,
        )
        try:
            return list(pool.map(fitness_fn, population))
        finally:
            pool.shutdown(cancel_futures=True, wait=False)
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_parallel.py -v`

Expected: all tests pass. If process tests are flaky on an interactive Windows shell, rerun with `pytest tests/unit/test_parallel.py::test_process_parallel_evaluates_population -v -s`.

```bash
git add evocore/parallel.py tests/unit/test_parallel.py
git commit -m "feat(python): parallel evaluation wrappers with spawn teardown"
```

---

## Task 7: Update `evocore/__init__.py`

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Extend package init tests**

```python
def test_part5_exports_accessible_from_top_level():
    from evocore import (
        GeneDef,
        GeneSpace,
        Individual,
        Population,
        OperatorSet,
        LogEntry,
        Logbook,
        GenerationInfo,
        Callback,
        EarlyStopping,
        CheckpointCallback,
        MetricsLogger,
        ThreadParallel,
        ProcessParallel,
    )
    assert GeneSpace.uniform(-1.0, 1.0, 2).length == 2
    assert Individual([1.0]).genes == [1.0]
```

- [ ] **Step 2: Update exports**

Add these imports and `__all__` entries to `evocore/__init__.py`:

```python
from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.stats import LogEntry, Logbook
from evocore.callbacks import (
    GenerationInfo,
    Callback,
    EarlyStopping,
    ProgressBar,
    CheckpointCallback,
    MetricsLogger,
)
from evocore.parallel import ThreadParallel, ProcessParallel
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_package_init.py tests/unit/test_gene_space.py tests/unit/test_individual.py tests/unit/test_operators.py tests/unit/test_stats.py tests/unit/test_callbacks.py tests/unit/test_parallel.py -v`

Expected: all Part 5 unit tests pass.

```bash
git add evocore/__init__.py tests/unit/test_package_init.py
git commit -m "feat(python): export Part 5 API foundation"
```

---

## Task 8: Full Part 5 Verification

- [ ] **Step 1: Build extension is still importable**

Run: `python -c "import evocore; import evocore._core; print('imports ok')"`

Expected: `imports ok`

- [ ] **Step 2: Run all unit tests**

Run: `pytest tests/unit/ -v`

Expected: all unit tests from Parts 1-5 pass.

- [ ] **Step 3: Run a Python foundation smoke test**

```bash
python - << 'EOF'
from evocore import GeneDef, GeneSpace, Individual, OperatorSet, Population

space = GeneSpace([
    GeneDef("period", "int", 5, 20, sigma=0.05),
    GeneDef("threshold", "float", 0.0, 1.0),
])
ops = OperatorSet(space, "sbx", "gaussian")
ind = ops.decode_individual([10.2, 0.25], fitness=1.5, fitness_valid=True)
assert ind.genes == [10, 0.25]
assert ind.params == {"period": 10, "threshold": 0.25}
pop = Population([ind, Individual([5, 0.1], fitness=0.5)])
assert pop.best()[0].fitness == 1.5
assert ops.sigma_abs_list(0.2) == [0.75, 0.2]
print("Part 5 complete - Python foundation ok")
EOF
```

Expected: `Part 5 complete - Python foundation ok`

- [ ] **Step 4: Final commit and tag**

```bash
git add .
git commit -m "chore: Part 5 complete - Python API foundation"
git tag part5-complete
```

---

## Part 5 Exit Criteria Checklist

- [ ] `GeneDef` validates kind, bounds, and `sigma` with `ConfigurationError`
- [ ] `GeneSpace.uniform()` creates unnamed uniform float spaces
- [ ] Named `GeneSpace` produces `ind.params` dictionaries through `OperatorSet.decode_individual()`
- [ ] `Individual.fitness_valid` exists only in Python
- [ ] `Population.best()`, `mean_fitness()`, `std_fitness()`, and `diversity()` work on finite data
- [ ] `OperatorSet` rejects bool+numeric mixed spaces and invalid operator combinations
- [ ] `OperatorSet` encodes int/bool genes as f64 for Rust and decodes them back to Python types
- [ ] Per-gene `GeneDef.sigma` overrides engine-level mutation sigma
- [ ] `GenerationInfo` exists and callback `on_generation_end(gen, pop, info)` signature is implemented
- [ ] `MetricsLogger` opens JSONL files with `encoding="utf-8"`
- [ ] `ProcessParallel` forces `spawn` and always calls `shutdown(cancel_futures=True, wait=False)` in `finally`
- [ ] Part 5 public classes are exported from top-level `evocore`
