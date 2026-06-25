"""Named parameter projection for optimizer-native flat search spaces."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import canonical_json_hash, json_safe
from evocore.search_space.constraints import (
    ConstraintViolation,
    ParameterRepair,
    ParameterValidator,
    RepairRecord,
)
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
    repair_signatures: tuple[Mapping[str, object], ...]
    validator_signatures: tuple[Mapping[str, object], ...]
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
        repairs: Sequence[ParameterRepair | Callable[..., object]] = (),
        validators: Sequence[ParameterValidator | Callable[..., object]] = (),
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
        self.repairs = tuple(repairs)
        self.validators = tuple(validators)
        self.identity_keys = ordered_identity_keys
        self.schema_id = str(schema_id)
        self.schema_version = str(schema_version)
        self.checkpointable = (
            bool(checkpointable)
            and all(
                bool(getattr(transform, "checkpointable", False))
                for transform in self.transforms.values()
            )
            and all(_hook_checkpointable(hook) for hook in (*self.repairs, *self.validators))
        )

    def project(self, parameters: Mapping[str, object]) -> ProjectionResult:
        """Encode a domain parameter mapping into optimizer-native coordinates."""
        params = dict(parameters)
        self._validate_structural_bindings(params)
        values = self._encode_active_parameters(params)

        self.optimizer_space.validate_genes(values)
        result = self.reconstruct(values)
        repaired_values = self._encode_active_parameters(result.parameters)
        self.optimizer_space.validate_genes(repaired_values)
        if tuple(repaired_values) == result.optimizer_values:
            return result
        return ProjectionResult(
            parameters=result.parameters,
            optimizer_values=tuple(repaired_values),
            active_names=result.active_names,
            structural_bindings=result.structural_bindings,
            repairs=result.repairs,
            violations=result.violations,
            projection_hash=result.projection_hash,
            parameter_hash=result.parameter_hash,
            metadata=result.metadata,
            valid=result.valid,
            checkpointable=result.checkpointable,
        )

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
        safe_parameters, repairs = self._apply_repairs(safe_parameters)
        self._require_active_parameters(safe_parameters)
        violations = self._validate_parameters(safe_parameters)
        signature_hash = self._signature_hash(require_checkpointable=False)
        parameter_hash = canonical_json_hash(
            {
                "schema_version": 1,
                "parameters": safe_parameters,
            }
        )
        projection_hash = canonical_json_hash(
            {
                "schema_version": 1,
                "signature_hash": signature_hash,
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
            repairs=repairs,
            violations=violations,
            projection_hash=projection_hash,
            parameter_hash=parameter_hash,
            metadata={"projection_signature_hash": signature_hash},
            valid=not violations,
            checkpointable=self.checkpointable,
        )

    def signature(self) -> Mapping[str, object]:
        """Return a JSON-safe projection behavior signature."""
        return self._signature_payload(require_checkpointable=True)

    def snapshot(self) -> ProjectionSnapshot:
        """Return a detached JSON-safe snapshot of projection behavior."""
        if not self.checkpointable:
            raise ConfigurationError(
                "ActiveGeneProjection contains runtime-only transforms or hooks and cannot snapshot."
            )

        signature_payload = self._signature_payload(require_checkpointable=True)
        transform_signatures = signature_payload["transform_signatures"]
        repair_signatures = signature_payload["repair_signatures"]
        validator_signatures = signature_payload["validator_signatures"]
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
            repair_signatures=copy.deepcopy(repair_signatures),
            validator_signatures=copy.deepcopy(validator_signatures),
            identity_keys=self.identity_keys,
            checkpointable=self.checkpointable,
            signature_hash=signature_hash,
        )

    def value_hash(self, parameters: Mapping[str, object]) -> str:
        """Return the canonical projected identity hash for domain parameters."""
        return self.project(parameters).projection_hash

    def _validate_structural_bindings(self, parameters: Mapping[str, object]) -> None:
        identity_key_set = set(self.identity_keys)
        for name, expected in self.structural_bindings.items():
            if name not in identity_key_set:
                continue
            if name in parameters and parameters[name] != expected:
                raise ConfigurationError(
                    "ActiveGeneProjection structural binding mismatch for "
                    f"{name!r}: expected {expected!r}, got {parameters[name]!r}."
                )

    def _require_active_parameters(self, parameters: Mapping[str, object]) -> None:
        missing = [name for name in self.active_names if name not in parameters]
        if missing:
            raise ConfigurationError(
                f"ActiveGeneProjection missing active parameter(s): {missing!r}."
            )

    def _encode_active_parameters(self, parameters: Mapping[str, object]) -> list[GeneValue]:
        values: list[GeneValue] = []
        self._require_active_parameters(parameters)
        for name in self.active_names:
            try:
                values.append(self.transforms[name].encode(parameters[name]))
            except Exception as exc:
                if isinstance(exc, ConfigurationError):
                    raise
                raise ConfigurationError(
                    f"ActiveGeneProjection could not encode parameter {name!r}: {exc}"
                ) from exc
        return values

    def _structural_identity(self) -> dict[str, object]:
        return {
            key: self.structural_bindings[key]
            for key in self.identity_keys
            if key in self.structural_bindings
        }

    def _apply_repairs(
        self,
        parameters: Mapping[str, object],
    ) -> tuple[dict[str, object], tuple[RepairRecord, ...]]:
        repaired_parameters = dict(parameters)
        repairs: list[RepairRecord] = []
        for hook in self.repairs:
            result = (
                hook.repair(repaired_parameters)
                if hasattr(hook, "repair")
                else hook(repaired_parameters)
            )
            if (
                not isinstance(result, tuple)
                or len(result) != 2
                or not isinstance(result[0], Mapping)
            ):
                raise ConfigurationError(
                    "ActiveGeneProjection repair hooks must return (parameters, repairs)."
                )
            repaired_parameters = _json_mapping(result[0], field_name="repaired parameters")
            for record in result[1]:
                if not isinstance(record, RepairRecord):
                    raise ConfigurationError(
                        "ActiveGeneProjection repair hooks must return RepairRecord entries."
                    )
                repairs.append(record)
        return repaired_parameters, tuple(repairs)

    def _validate_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> tuple[ConstraintViolation, ...]:
        violations: list[ConstraintViolation] = []
        for hook in self.validators:
            result = hook.validate(parameters) if hasattr(hook, "validate") else hook(parameters)
            for violation in result:
                if not isinstance(violation, ConstraintViolation):
                    raise ConfigurationError(
                        "ActiveGeneProjection validators must return ConstraintViolation entries."
                    )
                violations.append(violation)
        return tuple(violations)

    def _signature_payload(self, *, require_checkpointable: bool) -> dict[str, object]:
        transform_signatures = {
            name: _json_mapping(
                self.transforms[name].signature(),
                field_name=f"transform {name!r} signature",
            )
            for name in self.active_names
        }
        return {
            "schema_version": 1,
            "schema_id": self.schema_id,
            "user_schema_version": self.schema_version,
            "source_space_hash": self.source_space.hash(),
            "optimizer_space_hash": self.optimizer_space.hash(),
            "active_names": list(self.active_names),
            "structural_identity": self._structural_identity(),
            "transform_signatures": transform_signatures,
            "repair_signatures": _hook_signatures(
                self.repairs,
                hook_kind="repair",
                require_checkpointable=require_checkpointable,
            ),
            "validator_signatures": _hook_signatures(
                self.validators,
                hook_kind="validator",
                require_checkpointable=require_checkpointable,
            ),
            "identity_keys": list(self.identity_keys),
            "checkpointable": self.checkpointable,
        }

    def _signature_hash(self, *, require_checkpointable: bool) -> str:
        return canonical_json_hash(
            self._signature_payload(require_checkpointable=require_checkpointable)
        )


def _json_mapping(value: Mapping[str, object], *, field_name: str) -> dict[str, object]:
    payload = json_safe(dict(value))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


def _hook_checkpointable(hook: object) -> bool:
    return bool(getattr(hook, "checkpointable", False)) and hasattr(hook, "signature")


def _hook_signatures(
    hooks: Sequence[object],
    *,
    hook_kind: str,
    require_checkpointable: bool,
) -> tuple[Mapping[str, object], ...]:
    signatures: list[Mapping[str, object]] = []
    for index, hook in enumerate(hooks):
        if _hook_checkpointable(hook):
            signatures.append(
                _json_mapping(
                    hook.signature(),
                    field_name=f"{hook_kind} hook {index} signature",
                )
            )
            continue
        if require_checkpointable:
            raise ConfigurationError(
                f"ActiveGeneProjection {hook_kind} hook {index} is runtime-only."
            )
        signatures.append(
            {
                "hook_kind": hook_kind,
                "index": index,
                "checkpointable": False,
                "runtime_only": True,
            }
        )
    return tuple(signatures)


__all__ = [
    "ActiveGeneProjection",
    "ParameterProjection",
    "ProjectionResult",
    "ProjectionSnapshot",
]
