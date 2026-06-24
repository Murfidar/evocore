"""Projected warm-start helpers for CMA-ES active subspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Direction, WarmStartRecord, score_for_direction
from evocore.search_space import ParameterProjection


@dataclass(frozen=True)
class ProjectedWarmStartResult:
    """Summarize historical records projected into a CMA optimizer space."""

    initial_mean: list[float] | None
    accepted_count: int
    rejected: tuple[Mapping[str, object], ...]
    source_candidate_hashes: tuple[str, ...]
    metadata: Mapping[str, object]


def build_projected_cma_mean(
    *,
    projection: ParameterProjection,
    records: Sequence[WarmStartRecord],
    direction: Direction,
    strategy: Literal["best", "top_k_centroid"] = "best",
    top_k: int | None = None,
) -> ProjectedWarmStartResult:
    """Build a CMA initial mean by projecting trusted historical records."""
    if strategy not in ("best", "top_k_centroid"):
        raise ConfigurationError("strategy must be 'best' or 'top_k_centroid'.")
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
    if top_k is not None and int(top_k) <= 0:
        raise ConfigurationError("top_k must be positive when provided.")

    ranked = []
    rejected: list[Mapping[str, object]] = []
    for index, record in enumerate(records):
        params = dict(record.params or {})
        mismatch = _structural_mismatch(projection, params)
        if mismatch is not None:
            rejected.append(
                {
                    "record_index": index,
                    "reason": "projection_mismatch",
                    "message": mismatch,
                }
            )
            continue
        try:
            projected = projection.project(params)
        except ConfigurationError as exc:
            if "encode" in str(exc):
                raise
            rejected.append(
                {"record_index": index, "reason": "projection_mismatch", "message": str(exc)}
            )
            continue
        ranked.append((score_for_direction(record.score, direction), projected))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ProjectedWarmStartResult(
            None,
            0,
            tuple(rejected),
            (),
            {"strategy": strategy, "top_k": top_k},
        )

    selected_count = len(ranked) if top_k is None else int(top_k)
    selected = ranked[:selected_count]
    if strategy == "best":
        mean = [float(value) for value in selected[0][1].optimizer_values]
    else:
        mean = [
            sum(float(item[1].optimizer_values[index]) for item in selected) / len(selected)
            for index in range(projection.optimizer_space.length)
        ]
    projection.optimizer_space.validate_genes(mean)
    return ProjectedWarmStartResult(
        mean,
        len(ranked),
        tuple(rejected),
        tuple(projected.projection_hash for _, projected in selected),
        {"strategy": strategy, "top_k": top_k},
    )


def _structural_mismatch(
    projection: ParameterProjection,
    params: Mapping[str, object],
) -> str | None:
    bindings = getattr(projection, "structural_bindings", {})
    if not isinstance(bindings, Mapping):
        return None
    for name, expected in bindings.items():
        if name in params and params[name] != expected:
            return (
                f"record parameter {name!r}={params[name]!r} does not match "
                f"projection binding {expected!r}."
            )
    return None


__all__ = ["ProjectedWarmStartResult", "build_projected_cma_mean"]
