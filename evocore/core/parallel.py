"""Python parallel evaluation helpers and pickle validation."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import pickle
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Self

from evocore.core.errors import ConfigurationError
from evocore.search_space import Solution


def ensure_picklable(obj, *, context: str) -> None:
    """Raise a configuration error if an object is not picklable."""
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ConfigurationError(
            f"fitness_fn cannot be pickled, required for {context}.\n"
            f"  Error: {exc}\n"
            "  Fix: define fitness_fn at module level, not as a lambda or nested function."
        ) from exc


class ThreadParallel:
    """Evaluate individuals in a thread pool."""

    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = n_workers or os.cpu_count() or 1

    def evaluate(
        self, solutions: Sequence[Solution], fitness_fn: Callable[[Solution], object]
    ) -> list[object]:
        """Evaluate a SolutionSet with a thread pool."""
        if not solutions:
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(fitness_fn, solutions))


class ProcessParallel:
    """Evaluate individuals in a spawn-based process pool."""

    def __init__(self, n_workers: int | None = None, initializer=None, initargs=()) -> None:
        self.n_workers = n_workers or os.cpu_count() or 1
        self.initializer = initializer
        self.initargs = initargs
        self._ctx = multiprocessing.get_context("spawn")
        self._pool: concurrent.futures.ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close(wait=False)

    def _executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._pool is None:
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=self._ctx,
                initializer=self.initializer,
                initargs=self.initargs,
            )
        return self._pool

    def close(self, *, wait: bool = True, cancel_futures: bool = True) -> None:
        """Shut down the persistent process pool."""
        if self._pool is None:
            return
        self._pool.shutdown(cancel_futures=cancel_futures, wait=wait)
        self._pool = None

    def evaluate(
        self, solutions: Sequence[Solution], fitness_fn: Callable[[Solution], object]
    ) -> list[object]:
        """Evaluate a SolutionSet with a process pool."""
        if not solutions:
            return []

        ensure_picklable(fitness_fn, context="parallel='process'")
        return list(self._executor().map(fitness_fn, solutions))
