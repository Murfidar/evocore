"""CMA-ES engine backed by Rust covariance state."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.parallel import ThreadParallel
from evocore.core.serialization import json_safe, package_version
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    Direction,
    EvaluationRecord,
    OptimizationTelemetry,
    OptimizerStateSummary,
    UpdateResult,
    batch_id_from_seed,
    is_state_update_confidence,
    score_for_direction,
)
from evocore.results import (
    EventHistory,
    EventRecord,
    GenerationHistory,
    GenerationRecord,
    OptimizationResult,
    ReproducibilityMetadata,
    StopReason,
    append_run_stop_event,
)
from evocore.search_space import GeneSpace, OperatorCodec, Solution, SolutionSet

logger = logging.getLogger(__name__)


class CMAESOptimizer:
    """Run covariance matrix adaptation evolution strategy optimization.

    Args:
        gene_space: Float or integer gene definitions.
        population_size: Number of sampled candidates per generation.
        initial_mean: Optional encoded initial mean.
        initial_sigma: Initial sigma fraction relative to gene bounds.
        max_generations: Maximum number of generations to run.
        parallel: Evaluation mode: `"none"` or `"thread"`.
        n_workers: Worker count for thread mode.
        callbacks: Optional callbacks invoked during the run.
        seed: Master seed for deterministic sampling.
        track_diversity: Whether to record per-gene diversity.

    Raises:
        ConfigurationError: If configuration is invalid or process parallelism is requested.
    """

    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        initial_mean: list[float] | None = None,
        initial_sigma: float = 0.3,
        max_generations: int = 300,
        parallel: str = "none",
        n_workers: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        seed: int = 0,
        direction: Direction = "maximize",
        track_diversity: bool = False,
        **legacy_kwargs: object,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for CMAESOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if "generations" in legacy_kwargs:
            raise ConfigurationError("CMAESOptimizer uses max_generations, not generations")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(f"CMAESOptimizer got unexpected argument(s): {unknown}.")
        if "bool" in gene_space.kinds:
            raise ConfigurationError(
                "CMAESOptimizer does not support bool genes; use float/int genes only."
            )
        if gene_space.fixed_count:
            raise ConfigurationError(
                "CMAESOptimizer does not support fixed numeric genes yet. "
                "Use GeneticAlgorithmOptimizer for full-genome fixed genes, or remove fixed genes from the CMA-ES GeneSpace."
            )
        if parallel == "process":
            raise ConfigurationError(
                "CMAESOptimizer does not support parallel='process'.\n"
                "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable.\n"
                "  Fix: use parallel='thread' if your fitness function releases the GIL, or parallel='none'.\n"
                "  Note: parallel='process' is supported by GeneticAlgorithmOptimizer, not CMAESOptimizer."
            )
        if parallel not in ("none", "thread"):
            raise ConfigurationError("CMAESOptimizer parallel must be 'none' or 'thread'.")
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if max_generations < 0:
            raise ConfigurationError("max_generations must be >= 0.")
        if not (initial_sigma > 0.0):
            raise ConfigurationError("initial_sigma must be > 0.")
        if initial_mean is not None and len(initial_mean) != gene_space.length:
            raise ConfigurationError("initial_mean length must match gene_space.length.")

        self.gene_space = gene_space
        self.population_size = population_size
        self.initial_mean = initial_mean
        self.initial_sigma = initial_sigma
        self.max_generations = max_generations
        self.parallel = parallel
        self.n_workers = n_workers
        self.callbacks = list(callbacks or [])
        self.seed = int(seed)
        if direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        self.direction = direction
        self.track_diversity = track_diversity
        self.operators = OperatorCodec(gene_space, "sbx", "gaussian")
        self._fitness_warning_emitted = False
        self._state: _core.PyCMAESState | None = None
        self._event_index = 0
        self._batches_by_id: dict[str, CandidateBatch] = {}
        self._candidates_by_id: dict[str, Candidate] = {}
        self.events = EventHistory()
        self.vnext_telemetry = OptimizationTelemetry()
        self.best_candidate: Candidate | None = None

    @property
    def generation(self) -> int:
        """Return the current CMA generation."""
        return 0 if self._state is None else int(self._state.generation)

    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if not batch.consumed and len(batch.records_by_key) < len(batch.candidate_ids)
        )

    def _best_candidate_id_and_score(self) -> tuple[str | None, float | None]:
        if self.best_candidate is None:
            return None, None
        return (
            self.best_candidate.candidate_id,
            self.best_candidate.best_state_score(self.direction),
        )

    def _trusted_count(self) -> int:
        return sum(
            1
            for candidate in self._candidates_by_id.values()
            if any(
                is_state_update_confidence(score.confidence) for score in candidate.scores.values()
            )
        )

    def state_summary(self) -> OptimizerStateSummary:
        """Return a stable read-only vNext state summary."""
        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return OptimizerStateSummary(
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            event_index=self._event_index,
            pending_batch_ids=self._pending_batch_ids(),
            trusted_count=self._trusted_count(),
            telemetry=self.vnext_telemetry,
        )

    @property
    def _bounds_list(self) -> list[tuple[float, float]]:
        return self.operators.gene_bounds

    def _initial_mean_encoded(self) -> list[float]:
        if self.initial_mean is not None:
            return [float(value) for value in self.initial_mean]
        return _core.init_population(self._bounds_list, self.operators.gene_kinds, 1, self.seed)[0]

    def _sigma_abs(self) -> float:
        spans = [high - low for low, high in self._bounds_list]
        return self.initial_sigma * (sum(spans) / len(spans))

    def _apply_bounds_and_round(self, genes_f64: Sequence[float]) -> list[float]:
        rounded: list[float] = []
        for value, gene, (low, high) in zip(genes_f64, self.gene_space.genes, self._bounds_list):
            clamped = max(low, min(high, float(value)))
            if gene.kind == "int":
                clamped = float(round(clamped))
                clamped = max(low, min(high, clamped))
            rounded.append(clamped)
        return rounded

    def _decode_individual(
        self,
        genes_f64: Sequence[float],
        fitness: float | None = None,
    ) -> Solution:
        return self.operators.decode_individual(
            genes_f64,
            fitness=fitness,
            fitness_valid=fitness is not None,
        )

    def _normalise_fitness_result(
        self,
        result,
        ind: Solution,
        gen: int,
        idx: int,
    ) -> tuple[float, int]:
        metrics = {}
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise FitnessError("fitness_fn tuple return must be (float, dict).")
            result, metrics = result

        try:
            fitness = float(result)
        except (TypeError, ValueError) as exc:
            raise FitnessError(
                f"fitness_fn must return a float, got {type(result)!r} at generation {gen}, index {idx}."
            ) from exc

        ind.metadata["metrics"] = dict(metrics)
        if not math.isfinite(fitness):
            raise FitnessError(
                f"fitness_fn must return a finite float at generation {gen}, index {idx}; "
                f"got {fitness!r}."
            )

        ind.fitness = fitness
        ind.fitness_valid = True
        return fitness, 0

    def _fitness_comparison_score(self, fitness: float | None) -> float:
        if fitness is None:
            return float("-inf")
        fitness = float(fitness)
        if not math.isfinite(fitness):
            return float("-inf")
        return score_for_direction(fitness, self.direction)

    def _best_population_individual(self, solutions: SolutionSet) -> Solution:
        return max(
            solutions,
            key=lambda solution: self._fitness_comparison_score(solution.fitness),
        )

    def _evaluate_all(
        self,
        individuals: Sequence[Solution],
        fitness_fn: Callable[[Solution], float | tuple[float, dict]],
        gen: int,
    ) -> tuple[list[float], int]:
        if self.parallel == "thread":
            logger.debug(
                "CMA-ES thread evaluation generation=%s n_workers=%s SolutionSet=%s",
                gen,
                self.n_workers,
                len(individuals),
            )
            try:
                raw_results = ThreadParallel(self.n_workers).evaluate(individuals, fitness_fn)
            except Exception as exc:
                raise FitnessError(
                    f"fitness_fn raised {type(exc).__name__} during thread evaluation at generation {gen}. "
                    f"Original error: {exc}"
                ) from exc
        else:
            raw_results = []
            for idx, ind in enumerate(individuals):
                try:
                    raw_results.append(fitness_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"fitness_fn raised {type(exc).__name__} for Solution at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc

        fitnesses: list[float] = []
        nan_count = 0
        for idx, (ind, raw) in enumerate(zip(individuals, raw_results, strict=False)):
            fitness, bad_count = self._normalise_fitness_result(raw, ind, gen, idx)
            fitnesses.append(fitness)
            nan_count += bad_count

        return fitnesses, nan_count

    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(seed=self.seed, max_generations=self.max_generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    def _ensure_state(self) -> _core.PyCMAESState:
        if self._state is None:
            self._state = _core.PyCMAESState(
                self._initial_mean_encoded(),
                self._sigma_abs(),
                self.population_size,
                self._bounds_list,
            )
        return self._state

    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        candidate = self._candidates_by_id.get(record.candidate_id)
        if candidate is None:
            raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
        if record.batch_id is not None and record.batch_id not in self._batches_by_id:
            raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
        batch = self._batches_by_id.get(candidate.batch_id)
        if batch is None:
            raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
        return candidate, batch

    def _apply_record_confidence(
        self,
        candidate: Candidate,
        record: EvaluationRecord,
        trusted_records: list[EvaluationRecord],
    ) -> str:
        if is_state_update_confidence(record.confidence) and (
            self.best_candidate is None
            or candidate.state_comparison_score(self.direction)
            > self.best_candidate.state_comparison_score(self.direction)
        ):
            self.best_candidate = candidate
        if record.confidence == "trusted_full":
            trusted_records.append(record)
            self.vnext_telemetry.record_full(1, stage=record.stage, cost=record.cost)
            return "trusted"
        if record.confidence == "cached":
            self.vnext_telemetry.record_cached(1, stage=record.stage, cost=record.cost)
            return "cached"
        if record.confidence == "partial":
            self.vnext_telemetry.record_partial(1, stage=record.stage, cost=record.cost)
            return "partial"
        if record.confidence == "surrogate":
            self.vnext_telemetry.record_screened(1)
            return "surrogate"
        self.vnext_telemetry.record_eliminated(1, stage=record.stage)
        return "rejected"

    def _consume_complete_batch(self, batch: CandidateBatch) -> bool:
        ordered_records = batch.ordered_state_update_records()
        if ordered_records is None or batch.consumed:
            return False
        samples = []
        fitnesses = []
        for record in ordered_records:
            sample = batch.continuous_samples_by_id.get(record.candidate_id)
            if sample is None:
                raise FitnessError(
                    f"missing continuous sample for candidate_id {record.candidate_id!r}."
                )
            samples.append(sample)
            if record.score is None:
                raise FitnessError(
                    f"trusted_full record for candidate_id {record.candidate_id!r} is missing score."
                )
            fitnesses.append(score_for_direction(float(record.score), self.direction))
        self._ensure_state().tell(samples, fitnesses)
        batch.consumed = True
        return True

    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed CMA candidates."""
        for candidate in candidates:
            self.events.append(
                EventRecord(
                    event_index=len(self.events),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(),
                    generation=candidate.generation,
                    origin=candidate.origin,
                    parents=tuple(candidate.parents),
                    genes=tuple(candidate.genes),
                    params=dict(candidate.params) if candidate.params is not None else None,
                    metadata=dict(candidate.metadata),
                )
            )

    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        raw_score = float(record.score) if record.score is not None else None
        comparison_score = (
            score_for_direction(raw_score, self.direction)
            if raw_score is not None and math.isfinite(raw_score)
            else None
        )
        self.events.append(
            EventRecord(
                event_index=len(self.events),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(),
                generation=candidate.generation,
                stage=record.stage,
                confidence=record.confidence,
                raw_score=raw_score,
                comparison_score=comparison_score,
                cost=record.cost,
                status=candidate.status,
                origin=candidate.origin,
                parents=tuple(candidate.parents),
                genes=tuple(candidate.genes),
                params=dict(candidate.params) if candidate.params is not None else None,
                metrics=dict(record.metrics),
                metadata=dict(record.metadata),
            )
        )

    def ask(self, n: int | None = None) -> list[Candidate]:
        """Return a CMA candidate batch."""
        if n is not None and int(n) != self.population_size:
            raise ConfigurationError(
                "CMAESOptimizer.ask currently requires n to equal population_size."
            )
        state = self._ensure_state()
        event_index = self._event_index
        batch_id = batch_id_from_seed(self.seed, event_index)
        samples_continuous = state.ask(self.seed, event_index)
        samples_discrete = [self._apply_bounds_and_round(sample) for sample in samples_continuous]
        candidates: list[Candidate] = []
        continuous_samples_by_id: dict[str, list[float]] = {}
        for index, sample in enumerate(samples_discrete):
            solution = self._decode_individual(sample)
            candidate_id = _core.candidate_id(self.seed, event_index, index)
            candidate = Candidate(
                candidate_id=candidate_id,
                genes=list(solution.genes),
                batch_id=batch_id,
                params=solution.params,
                origin="cma_sample",
                event_index=event_index,
            )
            continuous_samples_by_id[candidate_id] = list(samples_continuous[index])
            self._candidates_by_id[candidate_id] = candidate
            candidates.append(candidate)
        self._batches_by_id[batch_id] = CandidateBatch(
            batch_id=batch_id,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            continuous_samples_by_id=continuous_samples_by_id,
        )
        self._event_index += 1
        self.vnext_telemetry.record_proposed_candidates(candidates)
        self._append_ask_events(candidates)
        return candidates

    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
        """Update CMA state from trusted evaluation records."""
        trusted_records: list[EvaluationRecord] = []
        counts = {"partial": 0, "surrogate": 0, "cached": 0, "rejected": 0}
        touched_batch_ids: set[str] = set()
        consumed_batch_ids: set[str] = set()
        for record in records:
            candidate, batch = self._candidate_and_batch_for_record(record)
            batch.accept_record(record, reject_consumed_state_record=True)
            touched_batch_ids.add(batch.batch_id)
            candidate.apply_record(record)
            self._append_tell_event(candidate, record)
            confidence = self._apply_record_confidence(candidate, record, trusted_records)
            if confidence in counts:
                counts[confidence] += 1

        for batch_id in touched_batch_ids:
            batch = self._batches_by_id[batch_id]
            if self._consume_complete_batch(batch):
                consumed_batch_ids.add(batch.batch_id)

        best_candidate_id, best_score = self._best_candidate_id_and_score()
        return UpdateResult(
            accepted_count=len(records),
            trusted_count=len(trusted_records),
            partial_count=counts["partial"],
            surrogate_count=counts["surrogate"],
            cached_count=counts["cached"],
            rejected_count=counts["rejected"],
            best_candidate_id=best_candidate_id,
            best_score=best_score,
            consumed_batch_ids=tuple(sorted(consumed_batch_ids)),
            pending_batch_ids=self._pending_batch_ids(),
            telemetry=self.vnext_telemetry,
        )

    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable CMA constructor configuration."""
        return json_safe(
            {
                "population_size": self.population_size,
                "initial_mean": self.initial_mean,
                "initial_sigma": self.initial_sigma,
                "max_generations": self.max_generations,
                "parallel": self.parallel,
                "n_workers": self.n_workers,
                "track_diversity": self.track_diversity,
            }
        )

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            optimizer_type="CMAESOptimizer",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
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

    def run(
        self, fitness_fn: Callable[[Solution], float | tuple[float, dict]]
    ) -> OptimizationResult:
        """Run one CMA-ES optimization.

        Args:
            fitness_fn: Callable receiving an `Solution` and returning either a fitness
                float or `(fitness, metrics_dict)`.

        Returns:
            Run result containing the best Solution, final SolutionSet, GenerationHistory, and timing.

        Raises:
            FitnessError: If the fitness function raises or returns an invalid value.

        Warns:
            FitnessWarning: When NaN or Inf fitness values are assigned `-inf`.
        """
        self._fitness_warning_emitted = False
        self._bind_callbacks()

        start = time.perf_counter()
        state = _core.PyCMAESState(
            self._initial_mean_encoded(),
            self._sigma_abs(),
            self.population_size,
            self._bounds_list,
        )
        generation_history = GenerationHistory()
        elite_history: list[Solution] = []
        diversity_by_generation: list[list[float]] = []
        final_solutions = SolutionSet([])
        n_evaluations = 0
        stop_reason: StopReason = "max_generations"

        for gen in range(self.max_generations):
            for callback in self.callbacks:
                callback.on_generation_start(gen, final_solutions)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break

            gen_start = time.perf_counter()
            samples_continuous = state.ask(self.seed, gen)
            samples_discrete = [
                self._apply_bounds_and_round(sample) for sample in samples_continuous
            ]
            individuals = [self._decode_individual(sample) for sample in samples_discrete]
            fitnesses, nan_count = self._evaluate_all(individuals, fitness_fn, gen)
            n_evaluations += len(individuals)
            state.tell(
                samples_continuous,
                [self._fitness_comparison_score(fitness) for fitness in fitnesses],
            )

            final_solutions = SolutionSet(individuals)
            info = GenerationInfo(gen, nan_count, 0)
            best = self._best_population_individual(final_solutions)
            diversity = final_solutions.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_by_generation.append(diversity)
            elite_history.append(best.clone())
            generation_history.append(
                GenerationRecord(
                    gen=gen,
                    best_score=float(best.fitness),
                    mean_score=final_solutions.mean_fitness(),
                    std_score=final_solutions.std_fitness(),
                    wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
                    n_evaluations=len(individuals),
                    nan_score_count=nan_count,
                    cached_count=0,
                    diversity=diversity,
                    custom=dict(best.metadata.get("metrics", {})),
                )
            )
            logger.info(
                "CMA-ES generation=%s best_score=%s mean_fitness=%s nan_fitness_count=%s",
                gen,
                float(best.fitness),
                final_solutions.mean_fitness(),
                nan_count,
            )
            for callback in self.callbacks:
                callback.on_generation_end(gen, final_solutions, info)
            if self._callbacks_should_stop():
                stop_reason = "callback"
                break

        if len(final_solutions):
            best = self._best_population_individual(final_solutions)
            best_solution = best.clone()
            best_score = float(best.fitness)
        else:
            best_solution = Solution([], fitness=float("-inf"), fitness_valid=True)
            best_score = float("-inf")

        result = OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            final_solutions=final_solutions,
            generations=generation_history,
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_solutions=elite_history,
            diversity_by_generation=diversity_by_generation,
            seed=self.seed,
            stop_reason=stop_reason,
            max_generations=self.max_generations,
            max_evaluations=None,
            direction=self.direction,
            optimizer_type="CMAESOptimizer",
            events=self._generation_history(generation_history),
            reproducibility=self._reproducibility_metadata(),
        )
        append_run_stop_event(
            result.events,
            stop_reason=result.stop_reason,
            max_evaluations=result.max_evaluations,
            max_generations=result.max_generations,
            n_evaluations=result.n_evaluations,
        )
        for callback in self.callbacks:
            callback.on_run_end(result)
        return result
