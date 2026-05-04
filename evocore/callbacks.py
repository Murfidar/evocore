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
        with open(filename, "wb") as handle:
            pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, handle)


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
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
