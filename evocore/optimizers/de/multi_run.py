from __future__ import annotations

import copy
import logging
import os
import time
from typing import TYPE_CHECKING, Protocol

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.core.parallel import ensure_picklable
from evocore.lifecycle import Evaluator, score_for_direction
from evocore.results import OptimizationBatchResult, OptimizationResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer


class _ChildOptimizer(Protocol):
    def _copy_with_seed(self, seed: int) -> _ChildOptimizer: ...

    def run(self, evaluator: Evaluator) -> OptimizationResult: ...


def run_child_optimizer(
    engine: _ChildOptimizer,
    seed: int,
    evaluator: Evaluator,
) -> OptimizationResult:
    """Run one child optimizer with a derived seed for process-pool execution."""
    return engine._copy_with_seed(seed).run(evaluator)


class DifferentialEvolutionMultiRunMixin:
    """Seed derivation, optimizer copying, and multi-run execution for DE."""

    def _copy_with_seed(self, seed: int) -> DifferentialEvolutionOptimizer:
        from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer

        return DifferentialEvolutionOptimizer(
            gene_space=self.gene_space,
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_factor=self.mutation_factor,
            crossover_rate=self.crossover_rate,
            strategy=self.strategy,
            parallel=self.parallel,
            n_workers=self.n_workers,
            process_initializer=self.process_initializer,
            process_initargs=self.process_initargs,
            seed=int(seed),
            direction=self.direction,
            max_evaluations=self.max_evaluations,
            track_diversity=self.track_diversity,
            callbacks=copy.deepcopy(self.callbacks),
        )

    def run_multiple(
        self,
        evaluator: Evaluator,
        n_runs: int = 10,
        aggregate: str = "best",
        run_parallel: bool = False,
    ) -> OptimizationBatchResult:
        """Run multiple deterministic child DE runs from derived seeds."""
        if n_runs <= 0:
            raise ConfigurationError("n_runs must be positive.")
        if aggregate not in ("best", "all"):
            raise ConfigurationError("aggregate must be 'best' or 'all'.")

        child_seeds = [
            int(_core.py_derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
            for run_idx in range(n_runs)
        ]
        logger.debug("DE run_multiple n_runs=%s child_seeds=%s", n_runs, child_seeds)

        started = time.perf_counter()
        if run_parallel:
            ensure_picklable(evaluator, context="run_multiple(run_parallel=True)")
            ensure_picklable(self, context="run_multiple(run_parallel=True) engine")

            import concurrent.futures
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_runs, self.n_workers or os.cpu_count() or 1),
                mp_context=ctx,
            )
            try:
                futures = [
                    pool.submit(run_child_optimizer, self, seed, evaluator) for seed in child_seeds
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            finally:
                pool.shutdown(cancel_futures=True, wait=False)
        else:
            results = [self._copy_with_seed(seed).run(evaluator) for seed in child_seeds]

        results.sort(
            key=lambda run: score_for_direction(run.best_score, self.direction),
            reverse=True,
        )
        return OptimizationBatchResult(
            best=results[0],
            all_runs=results,
            n_runs=n_runs,
            wall_time_seconds=time.perf_counter() - started,
            direction=self.direction,
        )


__all__ = ["DifferentialEvolutionMultiRunMixin", "run_child_optimizer"]
