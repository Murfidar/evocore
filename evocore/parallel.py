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

    def evaluate(
        self, population: Sequence[Individual], fitness_fn: Callable[[Individual], object]
    ) -> list[object]:
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

    def evaluate(
        self, population: Sequence[Individual], fitness_fn: Callable[[Individual], object]
    ) -> list[object]:
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
