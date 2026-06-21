"""Stop and stall policies for external ask/tell workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import PopulationSnapshot
from evocore.lifecycle.records import Direction, score_for_direction
from evocore.lifecycle.telemetry import OptimizationTelemetry, UpdateResult


@dataclass(frozen=True)
class StopDecision:
    """Decision returned by a stop policy observation."""

    stop: bool
    reason: str | None = None
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop and not self.reason:
            raise ConfigurationError("StopDecision reason is required when stop=True.")
        payload = json_safe(dict(self.metadata))
        if not isinstance(payload, dict):
            raise ConfigurationError("StopDecision metadata must be JSON-safe.")
        object.__setattr__(self, "metadata", payload)


class StopPolicy(Protocol):
    """Protocol for reusable external ask/tell stop policies."""

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        """Observe optimizer progress and return a stop decision."""
        ...

    def reset(self) -> None:
        """Reset policy-local counters."""
        ...


def _ok(metadata: dict[str, object] | None = None) -> StopDecision:
    return StopDecision(stop=False, metadata=metadata or {})


def _validate_direction(direction: Direction) -> Direction:
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
    return direction


def _best_snapshot_score(
    snapshot: PopulationSnapshot,
    score_direction: Direction,
) -> float | None:
    scored = [candidate.score for candidate in snapshot.candidates if candidate.score is not None]
    if not scored:
        return None
    return max(scored) if score_direction == "maximize" else min(scored)


class EvaluationLimitPolicy:
    """Stop after a configured number of counted evaluations."""

    def __init__(
        self,
        *,
        max_evaluations: int,
        include_cached: bool = True,
        include_partial: bool = False,
        include_surrogate: bool = False,
    ) -> None:
        if int(max_evaluations) <= 0:
            raise ConfigurationError("max_evaluations must be positive.")
        self.max_evaluations = int(max_evaluations)
        self.include_cached = bool(include_cached)
        self.include_partial = bool(include_partial)
        self.include_surrogate = bool(include_surrogate)
        self._observed_evaluations = 0

    def reset(self) -> None:
        """Reset the accumulated evaluation count."""
        self._observed_evaluations = 0

    def _count_from_telemetry(self, telemetry: OptimizationTelemetry) -> int:
        count = int(telemetry.candidates_full_evaluated)
        if self.include_cached:
            count += int(telemetry.candidates_cached)
        if self.include_partial:
            count += int(telemetry.candidates_partial_evaluated)
        if self.include_surrogate:
            count += int(telemetry.candidates_screened)
        return count

    def _count_from_update(self, update: UpdateResult) -> int:
        count = int(update.trusted_count)
        if self.include_cached:
            count += int(update.cached_count)
        if self.include_partial:
            count += int(update.partial_count)
        if self.include_surrogate:
            count += int(update.surrogate_count)
        return count

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        """Count observed evaluations and stop when the configured cap is reached."""
        source = telemetry or (snapshot.telemetry if snapshot is not None else None)
        if source is not None:
            observed = self._count_from_telemetry(source)
            self._observed_evaluations = max(self._observed_evaluations, observed)
        elif update is not None:
            self._observed_evaluations += self._count_from_update(update)

        observed = self._observed_evaluations
        metadata = {
            "observed_evaluations": observed,
            "max_evaluations": self.max_evaluations,
        }
        if observed >= self.max_evaluations:
            return StopDecision(
                stop=True,
                reason="evaluation_limit",
                message="Evaluation limit reached.",
                metadata=metadata,
            )
        return _ok(metadata)


class NoImprovementPolicy:
    """Stop after a fixed window without meaningful best-score improvement."""

    def __init__(
        self,
        *,
        window: int,
        min_delta: float = 0.0,
        score_direction: Direction = "maximize",
    ) -> None:
        if int(window) <= 0:
            raise ConfigurationError("window must be positive.")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0.0:
            raise ConfigurationError("min_delta must be finite and >= 0.")
        self.window = int(window)
        self.min_delta = float(min_delta)
        self.score_direction = _validate_direction(score_direction)
        self._best_comparison_score: float | None = None
        self._best_raw_score: float | None = None
        self._stale_count = 0

    def reset(self) -> None:
        """Reset the best-score and stale-window counters."""
        self._best_comparison_score = None
        self._best_raw_score = None
        self._stale_count = 0

    def _best_score(
        self,
        update: UpdateResult | None,
        snapshot: PopulationSnapshot | None,
    ) -> float | None:
        if update is not None and update.best_score is not None:
            return float(update.best_score)
        if snapshot is not None:
            return _best_snapshot_score(snapshot, self.score_direction)
        return None

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        """Observe the latest best score and stop after a stale window."""
        raw_score = self._best_score(update, snapshot)
        if raw_score is None:
            return _ok({"stale_count": self._stale_count, "best_score": self._best_raw_score})

        comparison = score_for_direction(raw_score, self.score_direction)
        improved = (
            self._best_comparison_score is None
            or comparison > self._best_comparison_score + self.min_delta
        )
        if improved:
            self._best_comparison_score = comparison
            self._best_raw_score = raw_score
            self._stale_count = 0
        else:
            self._stale_count += 1

        metadata = {"stale_count": self._stale_count, "best_score": self._best_raw_score}
        if self._stale_count >= self.window:
            return StopDecision(
                stop=True,
                reason="no_improvement",
                message="No improvement window reached.",
                metadata=metadata,
            )
        return _ok(metadata)


class ConvergencePolicy:
    """Stop when the best score reaches a target threshold."""

    def __init__(
        self,
        *,
        target_score: float,
        score_direction: Direction = "maximize",
        tolerance: float = 0.0,
    ) -> None:
        if not math.isfinite(float(target_score)):
            raise ConfigurationError("target_score must be finite.")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ConfigurationError("tolerance must be finite and >= 0.")
        self.target_score = float(target_score)
        self.score_direction = _validate_direction(score_direction)
        self.tolerance = float(tolerance)

    def reset(self) -> None:
        """Reset the policy state."""

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        """Stop once the best score reaches the configured convergence target."""
        best_score = update.best_score if update is not None else None
        if best_score is None and snapshot is not None:
            best_score = _best_snapshot_score(snapshot, self.score_direction)
        if best_score is None:
            return _ok()

        best = float(best_score)
        target = self.target_score
        if self.score_direction == "maximize":
            reached = best >= target - self.tolerance
        else:
            reached = best <= target + self.tolerance
        metadata = {"best_score": best, "target_score": target, "tolerance": self.tolerance}
        if reached:
            return StopDecision(
                stop=True,
                reason="convergence",
                message="Convergence target reached.",
                metadata=metadata,
            )
        return _ok(metadata)


class CompositeStopPolicy:
    """Evaluate stop policies in order and return the first stop decision."""

    def __init__(self, policies: Sequence[StopPolicy]) -> None:
        self.policies = tuple(policies)
        if not self.policies:
            raise ConfigurationError("CompositeStopPolicy requires at least one policy.")

    def reset(self) -> None:
        """Reset every child policy in order."""
        for policy in self.policies:
            policy.reset()

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        """Return the first stopping child policy decision."""
        last_metadata: dict[str, object] = {}
        for policy in self.policies:
            decision = policy.observe(update, snapshot=snapshot, telemetry=telemetry)
            if decision.stop:
                return decision
            last_metadata[policy.__class__.__name__] = decision.metadata
        return _ok(last_metadata)


__all__ = [
    "CompositeStopPolicy",
    "ConvergencePolicy",
    "EvaluationLimitPolicy",
    "NoImprovementPolicy",
    "StopDecision",
    "StopPolicy",
]
