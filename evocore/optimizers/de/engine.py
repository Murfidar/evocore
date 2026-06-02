"""Differential Evolution optimizer engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evocore.callbacks import Callback
from evocore.core.serialization import package_version
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    Direction,
    OptimizationTelemetry,
    OptimizerStateSummary,
)
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.de.ask_tell import DifferentialEvolutionAskTellMixin
from evocore.optimizers.de.config import (
    build_de_config,
    de_reproducibility_status,
    de_runtime_hooks,
    validate_de_compatibility,
)
from evocore.results import EventHistory, ReproducibilityMetadata
from evocore.search_space import GeneSpace


class DifferentialEvolutionOptimizer(DifferentialEvolutionAskTellMixin):
    """Run Differential Evolution over a flat EvoCore GeneSpace."""

    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        max_generations: int = 300,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.9,
        strategy: str = "rand1bin",
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: object | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            from evocore.core.errors import ConfigurationError

            raise ConfigurationError(
                f"DifferentialEvolutionOptimizer got unexpected argument(s): {unknown}."
            )
        self.gene_space = gene_space
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.strategy = str(strategy)
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.track_diversity = bool(track_diversity)
        self.callbacks = list(callbacks or [])
        validate_de_compatibility(self)
        self._reset_vnext_state()

    def _reset_vnext_state(self) -> None:
        """Reset state used by DE ask/tell and run APIs."""
        self._event_index = 0
        self.generation = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._target_candidate_ids: list[str] = []
        self._trial_target_slots: dict[str, int] = {}
        self._trial_target_candidate_ids: dict[str, str] = {}
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
        self.events = EventHistory()

    def _trusted_count(self) -> int:
        return len(self._target_candidate_ids)

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return self.best_candidate.candidate_id, self.best_candidate.best_state_score(self.direction)

    def state_summary(self) -> OptimizerStateSummary:
        """Return a stable read-only DE state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return OptimizerStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            telemetry=self.vnext_telemetry,
        )

    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_de_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer and gene-space compatibility."""
        validate_de_compatibility(self)

    def _optimizer_config(self) -> dict[str, Any]:
        return self.config_signature()

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        status, notes = de_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="DifferentialEvolutionOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=de_runtime_hooks(self),
        )
