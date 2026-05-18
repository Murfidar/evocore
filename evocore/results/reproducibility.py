"""Reproducibility metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evocore.core.serialization import canonical_json_hash, json_safe
from evocore.lifecycle.records import Direction
from evocore.search_space import GeneSpace


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Capture deterministic optimizer and environment identity for a result."""

    evocore_version: str
    optimizer_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    extension: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export reproducibility metadata as JSON-safe stable fields."""
        return json_safe(
            {
                "evocore_version": self.evocore_version,
                "optimizer_type": self.optimizer_type,
                "seed": self.seed,
                "direction": self.direction,
                "gene_space_signature": self.gene_space_signature,
                "gene_space_hash": self.gene_space_hash,
                "optimizer_config": self.optimizer_config,
                "extension": self.extension,
            }
        )


def gene_space_signature(gene_space: GeneSpace) -> dict[str, Any]:
    """Return the canonical signature for a gene space."""
    return gene_space.signature()


def gene_space_hash(signature: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a gene-space signature."""
    return canonical_json_hash(signature)


__all__ = ["ReproducibilityMetadata", "gene_space_hash", "gene_space_signature"]
