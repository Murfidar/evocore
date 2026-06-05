from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.optimizers.de.strategies import TrialProposal
from evocore.search_space import GeneSpace, GeneValue, decode_gene_values, encode_gene_values


def _require_mapping(raw: object) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigurationError(
            f"DE Rust kernel returned {type(raw).__name__}, expected mapping."
        )
    return raw


def _metadata_from_raw(raw: Mapping[str, Any]) -> dict[str, object]:
    if "metadata" not in raw:
        raise ConfigurationError("DE Rust kernel proposal is missing metadata.")
    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("DE Rust kernel proposal metadata must be a mapping.")
    return dict(metadata)


def _genes_from_raw(gene_space: GeneSpace, raw: Mapping[str, Any]) -> list[GeneValue]:
    if "genes" not in raw:
        raise ConfigurationError("DE Rust kernel proposal is missing genes.")
    genes = raw["genes"]
    if not isinstance(genes, Sequence) or isinstance(genes, str | bytes):
        raise ConfigurationError("DE Rust kernel proposal genes must be a sequence.")
    return decode_gene_values(gene_space, genes)


def _coerce_target_slot(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ConfigurationError(f"DE Rust kernel proposal {label} must be an int.")
    return int(value)


def _validate_target_slot(
    raw: Mapping[str, Any],
    metadata: Mapping[str, object],
    expected_slot: int,
) -> None:
    if "target_slot" not in raw:
        raise ConfigurationError("DE Rust kernel proposal is missing target_slot.")
    raw_slot = _coerce_target_slot(raw["target_slot"], label="target_slot")
    if raw_slot != expected_slot:
        raise ConfigurationError(
            "DE Rust kernel proposal target_slot mismatch: "
            f"expected {expected_slot}, got {raw_slot}."
        )

    if "target_slot" not in metadata:
        raise ConfigurationError("DE Rust kernel proposal metadata is missing target_slot.")
    metadata_slot = _coerce_target_slot(
        metadata["target_slot"],
        label="metadata target_slot",
    )
    if metadata_slot != expected_slot:
        raise ConfigurationError(
            "DE Rust kernel proposal metadata target_slot mismatch: "
            f"expected {expected_slot}, got {metadata_slot}."
        )


class DERustKernelAdapter:
    """Convert Python DE state to and from the Rust proposal kernel."""

    def generate_trials(
        self,
        *,
        target_population: Sequence[Candidate],
        scores: Sequence[float],
        gene_space: GeneSpace,
        strategy: str,
        mutation_factor: float,
        crossover_rate: float,
        seed: int,
        generation: int,
        target_slots: Sequence[int],
        direction: str,
        jde_state: Mapping[str, Sequence[float]] | None,
    ) -> list[TrialProposal]:
        """Generate decoded DE trial proposals from the Rust encoded kernel."""
        if len(scores) != len(target_population):
            raise ConfigurationError(
                "DE Rust kernel scores length must match target population length."
            )

        population_encoded = [
            encode_gene_values(gene_space, candidate.genes) for candidate in target_population
        ]
        raw_proposals = _core.de_generate_trials(
            population_encoded,
            [float(score) for score in scores],
            gene_space.rust_bounds,
            gene_space.kinds,
            strategy,
            mutation_factor,
            crossover_rate,
            seed,
            generation,
            list(target_slots),
            direction,
            jde_state,
        )
        if len(raw_proposals) != len(target_slots):
            raise ConfigurationError(
                "DE Rust kernel proposal count mismatch: "
                f"expected {len(target_slots)}, got {len(raw_proposals)}."
            )

        proposals: list[TrialProposal] = []
        for expected_slot, raw_item in zip(target_slots, raw_proposals, strict=True):
            raw = _require_mapping(raw_item)
            metadata = _metadata_from_raw(raw)
            _validate_target_slot(raw, metadata, int(expected_slot))
            genes = _genes_from_raw(gene_space, raw)
            proposals.append(
                TrialProposal(
                    genes=genes,
                    metadata=metadata,
                )
            )
        return proposals


__all__ = ["DERustKernelAdapter"]
