"""Named parameter projection for optimizer-native flat search spaces."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import canonical_json_hash, json_safe
from evocore.search_space.constraints import ConstraintViolation, RepairRecord
from evocore.search_space.genes import GeneSpace
from evocore.search_space.solutions import GeneValue
from evocore.search_space.transforms import IdentityTransform, ParameterTransform


@dataclass(frozen=True)
class ProjectionSnapshot:
    """JSON-safe behavior snapshot for a parameter projection."""

    schema_version: int
    schema_id: str
    user_schema_version: str
    source_space_signature: Mapping[str, object]
    source_space_hash: str
    optimizer_space_signature: Mapping[str, object]
    optimizer_space_hash: str
    active_names: tuple[str, ...]
    structural_bindings: Mapping[str, object]
    transform_signatures: Mapping[str, Mapping[str, object]]
    identity_keys: tuple[str, ...]
    checkpointable: bool
    signature_hash: str


@dataclass(frozen=True)
class ProjectionResult:
    """Detached result of projecting or reconstructing one parameter mapping."""

    parameters: Mapping[str, object]
    optimizer_values: tuple[GeneValue, ...]
    active_names: tuple[str, ...]
    structural_bindings: Mapping[str, object]
    repairs: tuple[RepairRecord, ...] = ()
    violations: tuple[ConstraintViolation, ...] = ()
    projection_hash: str = ""
    parameter_hash: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    valid: bool = True
    checkpointable: bool = True


@runtime_checkable
class ParameterProjection(Protocol):
    """Protocol for translating domain parameters into optimizer coordinates."""

    optimizer_space: GeneSpace
    checkpointable: bool

    def project(self, parameters: Mapping[str, object]) -> ProjectionResult:
        """Encode domain parameters into optimizer-native values."""
        raise NotImplementedError

    def reconstruct(self, values: Sequence[GeneValue]) -> ProjectionResult:
        """Reconstruct domain parameters from optimizer-native values."""
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        """Return a JSON-safe projection behavior signature."""
        raise NotImplementedError

    def snapshot(self) -> ProjectionSnapshot:
        """Return a detached checkpointable projection snapshot."""
        raise NotImplementedError

    def value_hash(self, parameters: Mapping[str, object]) -> str:
        """Return the canonical projected identity hash for domain parameters."""
        raise NotImplementedError


class ActiveGeneProjection:
    """Compile active named source genes into a flat optimizer-native space."""

    def __init__(
        self,
        *,
        source_space: GeneSpace,
        active_names: Sequence[str],
        structural_bindings: Mapping[str, object] | None = None,
        transforms: Mapping[str, ParameterTransform] | None = None,
        identity_keys: Sequence[str] = (),
        schema_id: str = "active_gene_projection",
        schema_version: str = "1",
        checkpointable: bool = True,
    ) -> None:
        if not source_space.has_names:
            raise ConfigurationError("ActiveGeneProjection requires a named GeneSpace.")
        if not schema_id:
            raise ConfigurationError("ActiveGeneProjection schema_id must be non-empty.")
        if not schema_version:
            raise ConfigurationError("ActiveGeneProjection schema_version must be non-empty.")

        requested_names = tuple(str(name) for name in active_names)
        if not requested_names:
            raise ConfigurationError("ActiveGeneProjection requires at least one active name.")
        if len(set(requested_names)) != len(requested_names):
            raise ConfigurationError("ActiveGeneProjection active names must be unique.")

        source_names = tuple(source_space.names)
        unknown = sorted(set(requested_names) - set(source_names))
        if unknown:
            raise ConfigurationError(
                f"ActiveGeneProjection contains unknown active name(s): {unknown!r}."
            )

        active_name_set = set(requested_names)
        ordered_active_names = tuple(name for name in source_names if name in active_name_set)
        active_genes = [gene for gene in source_space.genes if gene.name in active_name_set]
        safe_bindings = _json_mapping(structural_bindings or {}, field_name="structural_bindings")

        transform_map = dict(transforms or {})
        unknown_transforms = sorted(set(transform_map) - set(ordered_active_names))
        if unknown_transforms:
            raise ConfigurationError(
                f"ActiveGeneProjection transform name(s) must be active: {unknown_transforms!r}."
            )
        resolved_transforms: dict[str, ParameterTransform] = {
            name: transform_map.get(name, IdentityTransform()) for name in ordered_active_names
        }

        ordered_identity_keys = tuple(str(key) for key in identity_keys)
        missing_identity_keys = [
            key
            for key in ordered_identity_keys
            if key not in source_names and key not in safe_bindings
        ]
        if missing_identity_keys:
            raise ConfigurationError(
                "ActiveGeneProjection identity key(s) must be source or structural names: "
                f"{missing_identity_keys!r}."
            )

        self.source_space = source_space
        self.active_names = ordered_active_names
        self.optimizer_space = GeneSpace(active_genes, has_names=True)
        self.structural_bindings = safe_bindings
        self.transforms = resolved_transforms
        self.identity_keys = ordered_identity_keys
        self.schema_id = str(schema_id)
        self.schema_version = str(schema_version)
        self.checkpointable = bool(checkpointable) and all(
            bool(getattr(transform, "checkpointable", False))
            for transform in self.transforms.values()
        )

    def project(self, parameters: Mapping[str, object]) -> ProjectionResult:
        """Encode a domain parameter mapping into optimizer-native coordinates."""
        params = dict(parameters)
        values: list[GeneValue] = []
        for name in self.active_names:
            if name not in params:
                raise ConfigurationError(f"ActiveGeneProjection missing parameter {name!r}.")
            try:
                values.append(self.transforms[name].encode(params[name]))
            except Exception as exc:
                if isinstance(exc, ConfigurationError):
                    raise
                raise ConfigurationError(
                    f"ActiveGeneProjection could not encode parameter {name!r}: {exc}"
                ) from exc

        self.optimizer_space.validate_genes(values)
        return self.reconstruct(values)

    def reconstruct(self, values: Sequence[GeneValue]) -> ProjectionResult:
        """Decode optimizer-native coordinates into canonical domain parameters."""
        optimizer_values = tuple(values)
        if len(optimizer_values) != self.optimizer_space.length:
            raise ConfigurationError(
                "ActiveGeneProjection expected "
                f"{self.optimizer_space.length} optimizer values, got {len(optimizer_values)}."
            )
        self.optimizer_space.validate_genes(optimizer_values)

        parameters: dict[str, object] = dict(self.structural_bindings)
        for name, value in zip(self.active_names, optimizer_values, strict=True):
            try:
                parameters[name] = self.transforms[name].decode(value)
            except Exception as exc:
                if isinstance(exc, ConfigurationError):
                    raise
                raise ConfigurationError(
                    f"ActiveGeneProjection could not decode parameter {name!r}: {exc}"
                ) from exc

        safe_parameters = _json_mapping(parameters, field_name="parameters")
        parameter_hash = canonical_json_hash(
            {
                "schema_version": 1,
                "parameters": safe_parameters,
            }
        )
        projection_hash = canonical_json_hash(
            {
                "schema_version": 1,
                "signature_hash": self.snapshot().signature_hash,
                "active_values": {name: safe_parameters[name] for name in self.active_names},
                "identity_values": {
                    key: safe_parameters[key]
                    for key in self.identity_keys
                    if key in safe_parameters
                },
            }
        )

        return ProjectionResult(
            parameters=copy.deepcopy(safe_parameters),
            optimizer_values=copy.deepcopy(optimizer_values),
            active_names=self.active_names,
            structural_bindings=copy.deepcopy(self.structural_bindings),
            repairs=(),
            violations=(),
            projection_hash=projection_hash,
            parameter_hash=parameter_hash,
            metadata={"projection_signature_hash": self.snapshot().signature_hash},
            valid=True,
            checkpointable=self.checkpointable,
        )

    def signature(self) -> Mapping[str, object]:
        """Return a JSON-safe projection behavior signature."""
        snapshot = self.snapshot()
        return {
            "schema_version": snapshot.schema_version,
            "schema_id": snapshot.schema_id,
            "user_schema_version": snapshot.user_schema_version,
            "source_space_hash": snapshot.source_space_hash,
            "optimizer_space_hash": snapshot.optimizer_space_hash,
            "active_names": list(snapshot.active_names),
            "structural_identity": self._structural_identity(),
            "transform_signatures": {
                key: dict(value) for key, value in snapshot.transform_signatures.items()
            },
            "identity_keys": list(snapshot.identity_keys),
            "checkpointable": snapshot.checkpointable,
        }

    def snapshot(self) -> ProjectionSnapshot:
        """Return a detached JSON-safe snapshot of projection behavior."""
        if not self.checkpointable:
            raise ConfigurationError(
                "ActiveGeneProjection contains runtime-only transforms and cannot snapshot."
            )

        transform_signatures = {
            name: _json_mapping(
                self.transforms[name].signature(),
                field_name=f"transform {name!r} signature",
            )
            for name in self.active_names
        }
        signature_payload = {
            "schema_version": 1,
            "schema_id": self.schema_id,
            "user_schema_version": self.schema_version,
            "source_space_hash": self.source_space.hash(),
            "optimizer_space_hash": self.optimizer_space.hash(),
            "active_names": list(self.active_names),
            "structural_identity": self._structural_identity(),
            "transform_signatures": transform_signatures,
            "identity_keys": list(self.identity_keys),
            "checkpointable": self.checkpointable,
        }
        signature_hash = canonical_json_hash(signature_payload)
        return ProjectionSnapshot(
            schema_version=1,
            schema_id=self.schema_id,
            user_schema_version=self.schema_version,
            source_space_signature=copy.deepcopy(self.source_space.signature()),
            source_space_hash=self.source_space.hash(),
            optimizer_space_signature=copy.deepcopy(self.optimizer_space.signature()),
            optimizer_space_hash=self.optimizer_space.hash(),
            active_names=self.active_names,
            structural_bindings=copy.deepcopy(self.structural_bindings),
            transform_signatures=copy.deepcopy(transform_signatures),
            identity_keys=self.identity_keys,
            checkpointable=self.checkpointable,
            signature_hash=signature_hash,
        )

    def value_hash(self, parameters: Mapping[str, object]) -> str:
        """Return the canonical projected identity hash for domain parameters."""
        return self.project(parameters).projection_hash

    def _structural_identity(self) -> dict[str, object]:
        return {
            key: self.structural_bindings[key]
            for key in self.identity_keys
            if key in self.structural_bindings
        }


def _json_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    payload = json_safe(dict(value))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


__all__ = [
    "ActiveGeneProjection",
    "ParameterProjection",
    "ProjectionResult",
    "ProjectionSnapshot",
]
