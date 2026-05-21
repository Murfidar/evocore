"""Stable optimizer checkpoint envelope helpers."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.core.serialization import json_safe, package_version, stable_json_dumps
from evocore.lifecycle import Direction

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "optimizer_state"
SEED_DERIVATION_ALGORITHM = "py_derive_seed"
SEED_DERIVATION_VERSION = 1


def _created_by() -> dict[str, str]:
    """Return stable writer metadata for checkpoint diagnostics."""
    return {
        "evocore_version": package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _available_checkpoints(path: Path) -> list[str]:
    """Return nearby checkpoint-like files for missing-file diagnostics."""
    directory = path.parent
    if not directory.is_dir():
        return []
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.endswith(".evocore-checkpoint.json")
        or entry.name.startswith("checkpoint_gen_")
    )


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Represent one stable optimizer checkpoint envelope."""

    optimizer_type: str
    optimizer_config: Mapping[str, Any]
    optimizer_config_hash: str
    gene_space_signature: Mapping[str, Any]
    gene_space_hash: str
    direction: Direction
    seed: int
    position: Mapping[str, Any]
    state: Mapping[str, Any]
    audit: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_by: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export this checkpoint as a JSON-safe stable dictionary."""
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND,
            "created_by": dict(self.created_by or _created_by()),
            "optimizer": {
                "optimizer_type": self.optimizer_type,
                "optimizer_config": dict(self.optimizer_config),
                "optimizer_config_hash": self.optimizer_config_hash,
                "gene_space_signature": dict(self.gene_space_signature),
                "gene_space_hash": self.gene_space_hash,
                "direction": self.direction,
                "seed": int(self.seed),
                "seed_derivation": {
                    "algorithm": SEED_DERIVATION_ALGORITHM,
                    "version": SEED_DERIVATION_VERSION,
                },
            },
            "position": dict(self.position),
            "state": dict(self.state),
            "audit": dict(self.audit),
            "metadata": dict(self.metadata),
        }
        return json_safe(payload)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint field {key!r} must be an object.")
    return value


def _validate_seed_derivation(seed_derivation: object) -> None:
    if not isinstance(seed_derivation, Mapping):
        raise CheckpointError("checkpoint optimizer.seed_derivation must be an object.")
    if seed_derivation.get("algorithm") != SEED_DERIVATION_ALGORITHM:
        raise CheckpointError(
            "checkpoint seed_derivation.algorithm "
            f"{seed_derivation.get('algorithm')!r} is unsupported."
        )
    if seed_derivation.get("version") != SEED_DERIVATION_VERSION:
        raise CheckpointError(
            "checkpoint seed_derivation.version "
            f"{seed_derivation.get('version')!r} is unsupported."
        )


def _validate_optimizer_section(optimizer: Mapping[str, Any]) -> None:
    if not optimizer.get("optimizer_type"):
        raise CheckpointError("checkpoint optimizer.optimizer_type is required.")
    if not optimizer.get("optimizer_config_hash"):
        raise CheckpointError("checkpoint optimizer.optimizer_config_hash is required.")
    if not optimizer.get("gene_space_hash"):
        raise CheckpointError("checkpoint optimizer.gene_space_hash is required.")
    if optimizer.get("direction") not in ("maximize", "minimize"):
        raise CheckpointError("checkpoint optimizer.direction must be 'maximize' or 'minimize'.")
    if "seed" not in optimizer:
        raise CheckpointError("checkpoint optimizer.seed is required.")
    _validate_seed_derivation(optimizer.get("seed_derivation"))


def _validate_state_section(state: Mapping[str, Any], optimizer_type: object) -> None:
    if state.get("optimizer_type") != optimizer_type:
        raise CheckpointError(
            "checkpoint state.optimizer_type must match optimizer.optimizer_type."
        )
    if state.get("schema_version") != 1:
        raise CheckpointError("checkpoint state.schema_version must be 1.")
    if not isinstance(state.get("payload"), Mapping):
        raise CheckpointError("checkpoint state.payload must be an object.")


def validate_checkpoint_envelope(payload: object) -> dict[str, Any]:
    """Validate the shared checkpoint envelope and return it as a dict."""
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a JSON object.")
    data = dict(payload)
    version = data.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "unsupported checkpoint_schema_version "
            f"{version!r}; supported version is {CHECKPOINT_SCHEMA_VERSION}."
        )
    if data.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointError(
            f"checkpoint_kind must be {CHECKPOINT_KIND!r}, got {data.get('checkpoint_kind')!r}."
        )
    created_by = _require_mapping(data, "created_by")
    if not created_by.get("evocore_version"):
        raise CheckpointError("checkpoint created_by.evocore_version is required.")
    optimizer = _require_mapping(data, "optimizer")
    _require_mapping(data, "position")
    state = _require_mapping(data, "state")
    _validate_optimizer_section(optimizer)
    _validate_state_section(state, optimizer.get("optimizer_type"))
    return data


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    optimizer_type: str,
    gene_space_hash: str,
    optimizer_config_hash: str,
    seed: int,
    direction: Direction,
) -> None:
    """Raise when checkpoint identity does not match the receiving optimizer."""
    data = validate_checkpoint_envelope(payload)
    optimizer = _require_mapping(data, "optimizer")
    state = _require_mapping(data, "state")
    if optimizer.get("optimizer_type") != optimizer_type:
        raise CheckpointError(
            "checkpoint optimizer_type "
            f"{optimizer.get('optimizer_type')!r} does not match {optimizer_type!r}."
        )
    if state.get("optimizer_type") != optimizer_type:
        raise CheckpointError(
            "checkpoint state.optimizer_type "
            f"{state.get('optimizer_type')!r} does not match {optimizer_type!r}."
        )
    if optimizer.get("gene_space_hash") != gene_space_hash:
        raise CheckpointError(
            "checkpoint gene_space_hash "
            f"{optimizer.get('gene_space_hash')!r} does not match {gene_space_hash!r}."
        )
    if optimizer.get("seed") != seed:
        raise CheckpointError(
            f"checkpoint seed {optimizer.get('seed')!r} does not match {seed!r}."
        )
    if optimizer.get("direction") != direction:
        raise CheckpointError(
            f"checkpoint direction {optimizer.get('direction')!r} does not match {direction!r}."
        )
    if optimizer.get("optimizer_config_hash") != optimizer_config_hash:
        raise CheckpointError(
            "checkpoint optimizer_config_hash "
            f"{optimizer.get('optimizer_config_hash')!r} does not match "
            f"{optimizer_config_hash!r}."
        )


def save_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: CheckpointSnapshot | Mapping[str, Any],
) -> None:
    """Write a checkpoint envelope as deterministic UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.to_dict() if isinstance(checkpoint, CheckpointSnapshot) else checkpoint
    data = validate_checkpoint_envelope(payload)
    target.write_text(stable_json_dumps(data, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a stable checkpoint JSON file."""
    source = Path(path)
    if not source.exists():
        available = _available_checkpoints(source)
        raise CheckpointError(
            f"checkpoint file {str(source)!r} not found. Available checkpoints: "
            f"{', '.join(available) or 'none'}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint file {str(source)!r} is corrupt or incompatible: {exc}"
        ) from exc
    return validate_checkpoint_envelope(payload)


__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "SEED_DERIVATION_ALGORITHM",
    "SEED_DERIVATION_VERSION",
    "CheckpointSnapshot",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_envelope",
    "validate_checkpoint_identity",
]
