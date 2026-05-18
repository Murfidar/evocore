"""Genetic algorithm engine and run result containers."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from evocore.callbacks import Callback
from evocore.core.errors import (
    ConfigurationError,
    ConfigurationWarning,
)
from evocore.core.serialization import package_version
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    Direction,
    OptimizationTelemetry,
    OptimizerStateSummary,
    score_for_direction,
)
from evocore.optimizers.config import OptimizerConfig
from evocore.optimizers.ga.ask_tell import GeneticAlgorithmAskTellMixin
from evocore.optimizers.ga.checkpointing import GeneticAlgorithmCheckpointingMixin
from evocore.optimizers.ga.config import (
    build_ga_config,
    ga_reproducibility_status,
    ga_runtime_hooks,
    validate_ga_compatibility,
)
from evocore.optimizers.ga.generation_loop import GeneticAlgorithmGenerationLoopMixin
from evocore.optimizers.ga.multi_run import GeneticAlgorithmMultiRunMixin
from evocore.optimizers.ga.reproduction import GeneticAlgorithmReproductionMixin
from evocore.results import (
    EventHistory,
    EventRecord,
    GenerationHistory,
    ReproducibilityMetadata,
)
from evocore.search_space import GeneSpace, OperatorCodec

logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer(
    GeneticAlgorithmAskTellMixin,
    GeneticAlgorithmGenerationLoopMixin,
    GeneticAlgorithmCheckpointingMixin,
    GeneticAlgorithmMultiRunMixin,
    GeneticAlgorithmReproductionMixin,
):
    """Run deterministic genetic algorithm optimization over a gene space.

    Args:
        gene_space: Gene definitions for individuals.
        population_size: Number of individuals per generation.
        max_generations: Maximum number of generations to run.
        crossover: Crossover operator name. Numeric spaces support `"sbx"`, `"blx"`,
            and `"uniform"`; binary spaces support `"one_point"`, `"two_point"`, and
            `"uniform"`.
        crossover_prob: Probability of applying crossover.
        crossover_eta: Eta parameter for simulated binary crossover.
        crossover_alpha: Alpha parameter for blend crossover.
        mutation: Mutation operator name.
        mutation_prob: Per-gene mutation probability once an offspring is selected for mutation.
        mutation_individual_prob: Per-offspring probability of applying mutation.
        mutation_sigma: Global mutation sigma fraction.
        mutation_sigma_schedule: Sigma schedule name.
        mutation_sigma_end: Final sigma fraction for decay schedules.
        selection: Selection operator name.
        tournament_size: Number of candidates per tournament.
        elitism: Number of best individuals copied into each generation.
        parallel: Evaluation mode: `"none"`, `"thread"`, or `"process"`.
        n_workers: Worker count for parallel modes.
        process_initializer: Optional initializer for process workers.
        process_initargs: Arguments passed to the process initializer.
        seed: Master seed for deterministic reproducibility.
        track_diversity: Whether to record per-gene diversity.
        callbacks: Optional callbacks invoked during the run.
        max_evaluations: Optional hard cap on objective calls.

    Raises:
        ConfigurationError: If engine configuration is invalid.
    """

    def __init__(  # noqa: PLR0915
        self,
        gene_space: GeneSpace,
        population_size: int = 100,
        max_generations: int = 100,
        crossover: str = "sbx",
        crossover_prob: float = 0.9,
        crossover_eta: float = 2.0,
        crossover_alpha: float = 0.5,
        mutation: str = "gaussian",
        mutation_prob: float = 0.1,
        mutation_individual_prob: float = 1.0,
        mutation_sigma: float = 0.2,
        mutation_sigma_schedule: str = "constant",
        mutation_sigma_end: float = 0.02,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 1,
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer: Callable[..., object] | None = None,
        process_initargs: tuple[object, ...] = (),
        seed: int = 0,
        direction: Direction = "maximize",
        max_evaluations: int | None = None,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for GeneticAlgorithmOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if "generations" in legacy_kwargs:
            raise ConfigurationError(
                "GeneticAlgorithmOptimizer uses max_generations, not generations"
            )
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(
                f"GeneticAlgorithmOptimizer got unexpected argument(s): {unknown}."
            )
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if max_generations < 0:
            raise ConfigurationError("max_generations must be >= 0.")
        if max_evaluations is not None and max_evaluations <= 0:
            raise ConfigurationError("max_evaluations must be positive when provided.")
        if elitism < 0 or elitism >= population_size:
            raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
        if parallel not in ("none", "thread", "process"):
            raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
        if selection not in ("tournament", "roulette", "rank"):
            raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
        if not (0.0 <= mutation_individual_prob <= 1.0):
            raise ConfigurationError("mutation_individual_prob must be in [0, 1].")
        if mutation_sigma_schedule not in ("constant", "linear_decay", "cosine_decay"):
            raise ConfigurationError(
                "mutation_sigma_schedule must be 'constant', 'linear_decay', or 'cosine_decay'."
            )

        self.gene_space = gene_space
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.crossover_alpha = crossover_alpha
        self.mutation = mutation
        self.mutation_prob = mutation_prob
        self.mutation_individual_prob = mutation_individual_prob
        self.mutation_sigma = mutation_sigma
        self.mutation_sigma_schedule = mutation_sigma_schedule
        self.mutation_sigma_end = mutation_sigma_end
        self.selection = selection
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        if direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        self.direction = direction
        self.max_evaluations = max_evaluations
        self.track_diversity = track_diversity
        self.callbacks = list(callbacks or [])
        self.operators = OperatorCodec(gene_space, crossover, mutation)
        self._fitness_warning_emitted = False
        self._reset_vnext_state()

        self._warn_if_large_int_gene_without_sigma()

    def _reset_vnext_state(self) -> None:
        """Reset state used by the vNext ask/tell and run APIs."""
        self._event_index = 0
        self._candidates_by_id: dict[str, Candidate] = {}
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._trusted_population_vnext: list[Candidate] = []
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None
        self.events = EventHistory()

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if len(batch.records_by_key) < len(batch.candidate_ids)
        )

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return (
            self.best_candidate.candidate_id,
            self.best_candidate.best_state_score(self.direction),
        )

    def _record_state_candidate(self, candidate: Candidate) -> None:
        if not any(
            existing.candidate_id == candidate.candidate_id
            for existing in self._trusted_population_vnext
        ):
            self._trusted_population_vnext.append(candidate)
        if self.best_candidate is None or candidate.state_comparison_score(
            self.direction
        ) > self.best_candidate.state_comparison_score(self.direction):
            self.best_candidate = candidate

    def state_summary(self) -> OptimizerStateSummary:
        """Return a stable read-only vNext state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return OptimizerStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=len(self._trusted_population_vnext),
            telemetry=self.vnext_telemetry,
        )

    def config(self) -> OptimizerConfig:
        """Return the public optimizer configuration object."""
        return build_ga_config(self)

    def config_signature(self) -> dict[str, Any]:
        """Return the canonical JSON-safe optimizer configuration signature."""
        return self.config().to_dict()

    def config_hash(self) -> str:
        """Return the stable hash for this optimizer configuration."""
        return self.config().hash()

    def validate_compatibility(self) -> None:
        """Validate optimizer, operator, and gene-space compatibility."""
        validate_ga_compatibility(self)

    def _warn_if_large_int_gene_without_sigma(self) -> None:
        for gene in self.gene_space.genes:
            if gene.kind == "int" and gene.sigma is None and (gene.high - gene.low) > 100:
                sigma_abs = self.mutation_sigma * (gene.high - gene.low)
                warnings.warn(
                    f'Gene("{gene.name}", "int", {gene.low}, {gene.high}) has range '
                    f"{gene.high - gene.low} and no per-gene sigma. With mutation_sigma={self.mutation_sigma}, "
                    f"sigma_abs={sigma_abs:g} may prevent fine-tuning. Consider sigma=0.03.",
                    category=ConfigurationWarning,
                    stacklevel=2,
                )

    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable GA constructor configuration."""
        return self.config_signature()

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        status, notes = ga_reproducibility_status(self)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="GeneticAlgorithmOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
            optimizer_config_hash=self.config_hash(),
            reproducibility_status=status,
            reproducibility_notes=notes,
            runtime_hooks=ga_runtime_hooks(self),
        )

    def _generation_history(self, generation_history: GenerationHistory) -> EventHistory:
        """Convert generation GenerationHistory entries into generation events."""
        history = EventHistory()
        for entry in generation_history:
            history.append(
                EventRecord(
                    event_index=len(history),
                    event_type="generation",
                    generation=entry.gen,
                    raw_score=entry.best_score,
                    comparison_score=score_for_direction(entry.best_score, self.direction),
                    metrics=entry.to_dict(),
                )
            )
        return history
