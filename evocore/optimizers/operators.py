"""Public operator contracts for EvoCore optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.optimizers.config import stable_object_identity
from evocore.search_space.genes import GeneKind, GeneSpace
from evocore.search_space.solutions import GeneValue

OperatorType = Literal["crossover", "mutation", "selection", "bounds"]
OperatorDomain = Literal[
    "numeric",
    "binary",
    "mixed",
    "score",
    "repair",
    "auto",
    "custom",
]
GeneSpaceProfile = Literal["numeric", "binary", "mixed"]

NUMERIC_GENE_KINDS: frozenset[GeneKind] = frozenset({"float", "int"})
BINARY_GENE_KINDS: frozenset[GeneKind] = frozenset({"bool"})
ALL_FLAT_GENE_KINDS: frozenset[GeneKind] = frozenset({"float", "int", "bool"})

DEFAULT_CROSSOVER_PROBABILITY = 0.9
DEFAULT_CROSSOVER_ETA = 2.0
DEFAULT_CROSSOVER_ALPHA = 0.5
DEFAULT_MUTATION_PROBABILITY = 0.1
DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY = 1.0
DEFAULT_MUTATION_SIGMA = 0.2
DEFAULT_TOURNAMENT_SIZE = 3


def _immutable_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(json_safe(dict(parameters))))


def _validate_probability(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1].")
    return numeric


def _validate_positive(value: float, name: str) -> float:
    numeric = float(value)
    if numeric <= 0.0:
        raise ConfigurationError(f"{name} must be > 0.")
    return numeric


def _validate_non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if numeric < 0.0:
        raise ConfigurationError(f"{name} must be >= 0.")
    return numeric


def _validate_sigma_fraction(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1].")
    return numeric


@dataclass(frozen=True)
class CrossoverOperator:
    """Describe a crossover operator and its stable public signature."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    supported_gene_kinds: frozenset[GeneKind] = field(default_factory=frozenset)
    domain: OperatorDomain = "numeric"
    operator_type: Literal["crossover"] = "crossover"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))
        object.__setattr__(self, "supported_gene_kinds", frozenset(self.supported_gene_kinds))

    @classmethod
    def sbx(
        cls,
        *,
        eta: float = DEFAULT_CROSSOVER_ETA,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        """Return simulated binary crossover for numeric gene spaces."""
        return cls(
            "sbx",
            {
                "eta": _validate_positive(eta, "crossover_eta"),
                "probability": _validate_probability(probability, "crossover_prob"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def blx(
        cls,
        *,
        alpha: float = DEFAULT_CROSSOVER_ALPHA,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        """Return blend crossover for numeric gene spaces."""
        return cls(
            "blx",
            {
                "alpha": _validate_non_negative(alpha, "crossover_alpha"),
                "probability": _validate_probability(probability, "crossover_prob"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def uniform(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        """Return uniform crossover resolved against the gene-space domain."""
        return cls(
            "uniform",
            {"probability": _validate_probability(probability, "crossover_prob")},
            ALL_FLAT_GENE_KINDS,
            "auto",
        )

    @classmethod
    def one_point(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        """Return one-point crossover for binary gene spaces."""
        return cls(
            "one_point",
            {"probability": _validate_probability(probability, "crossover_prob")},
            BINARY_GENE_KINDS,
            "binary",
        )

    @classmethod
    def two_point(
        cls,
        *,
        probability: float = DEFAULT_CROSSOVER_PROBABILITY,
    ) -> CrossoverOperator:
        """Return two-point crossover for binary gene spaces."""
        return cls(
            "two_point",
            {"probability": _validate_probability(probability, "crossover_prob")},
            BINARY_GENE_KINDS,
            "binary",
        )

    def signature(self) -> dict[str, Any]:
        """Return the JSON-safe canonical operator signature."""
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MutationOperator:
    """Describe a mutation operator and its stable public signature."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    supported_gene_kinds: frozenset[GeneKind] = field(default_factory=frozenset)
    domain: OperatorDomain = "numeric"
    operator_type: Literal["mutation"] = "mutation"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))
        object.__setattr__(self, "supported_gene_kinds", frozenset(self.supported_gene_kinds))

    @classmethod
    def gaussian(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
        sigma: float = DEFAULT_MUTATION_SIGMA,
    ) -> MutationOperator:
        """Return Gaussian mutation for numeric gene spaces."""
        return cls(
            "gaussian",
            {
                "individual_probability": _validate_probability(
                    individual_probability, "mutation_individual_prob"
                ),
                "probability": _validate_probability(probability, "mutation_prob"),
                "sigma": _validate_sigma_fraction(sigma, "mutation_sigma"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def uniform(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
    ) -> MutationOperator:
        """Return uniform replacement mutation for numeric gene spaces."""
        return cls(
            "uniform",
            {
                "individual_probability": _validate_probability(
                    individual_probability, "mutation_individual_prob"
                ),
                "probability": _validate_probability(probability, "mutation_prob"),
            },
            NUMERIC_GENE_KINDS,
            "numeric",
        )

    @classmethod
    def bit_flip(
        cls,
        *,
        probability: float = DEFAULT_MUTATION_PROBABILITY,
        individual_probability: float = DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY,
    ) -> MutationOperator:
        """Return bit-flip mutation for binary gene spaces."""
        return cls(
            "bit_flip",
            {
                "individual_probability": _validate_probability(
                    individual_probability, "mutation_individual_prob"
                ),
                "probability": _validate_probability(probability, "mutation_prob"),
            },
            BINARY_GENE_KINDS,
            "binary",
        )

    def signature(self) -> dict[str, Any]:
        """Return the JSON-safe canonical operator signature."""
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class SelectionOperator:
    """Describe a parent-selection operator and its stable public signature."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    domain: OperatorDomain = "score"
    operator_type: Literal["selection"] = "selection"
    custom: bool = False
    implementation: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))

    @classmethod
    def tournament(cls, *, size: int = DEFAULT_TOURNAMENT_SIZE) -> SelectionOperator:
        """Return tournament selection with replacement."""
        tournament_size = int(size)
        if tournament_size <= 0:
            raise ConfigurationError("tournament_size must be >= 1.")
        return cls("tournament", {"tournament_size": tournament_size})

    @classmethod
    def roulette(cls) -> SelectionOperator:
        """Return roulette-wheel selection over comparison scores."""
        return cls("roulette", {})

    @classmethod
    def rank(cls) -> SelectionOperator:
        """Return rank-weighted selection over comparison scores."""
        return cls("rank", {})

    def signature(self) -> dict[str, Any]:
        """Return the JSON-safe canonical operator signature."""
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class BoundsPolicy:
    """Describe how reproduced values are repaired into a gene space."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    domain: OperatorDomain = "repair"
    operator_type: Literal["bounds"] = "bounds"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _immutable_parameters(self.parameters))

    @classmethod
    def clamp(cls) -> BoundsPolicy:
        """Return the clamp/round/threshold bounds policy."""
        return cls("clamp", {})

    def signature(self) -> dict[str, Any]:
        """Return the JSON-safe canonical bounds-policy signature."""
        return {
            "type": self.name,
            "operator_type": self.operator_type,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


def normalize_crossover_operator(
    value: str | CrossoverOperator,
    *,
    probability: float,
    eta: float,
    alpha: float,
) -> CrossoverOperator:
    """Normalize a crossover string or spec into a crossover operator."""
    if isinstance(value, CrossoverOperator):
        return value
    if value == "sbx":
        return CrossoverOperator.sbx(eta=eta, probability=probability)
    if value == "blx":
        return CrossoverOperator.blx(alpha=alpha, probability=probability)
    if value in ("uniform", "uniform_xo"):
        return CrossoverOperator.uniform(probability=probability)
    if value == "one_point":
        return CrossoverOperator.one_point(probability=probability)
    if value == "two_point":
        return CrossoverOperator.two_point(probability=probability)
    raise ConfigurationError(
        "Unknown crossover operator: "
        f"{value!r}. Valid: 'sbx', 'blx', 'uniform', 'one_point', 'two_point'."
    )


def normalize_mutation_operator(
    value: str | MutationOperator,
    *,
    probability: float,
    individual_probability: float,
    sigma: float,
) -> MutationOperator:
    """Normalize a mutation string or spec into a mutation operator."""
    if isinstance(value, MutationOperator):
        return value
    if value == "gaussian":
        return MutationOperator.gaussian(
            probability=probability,
            individual_probability=individual_probability,
            sigma=sigma,
        )
    if value == "uniform":
        return MutationOperator.uniform(
            probability=probability,
            individual_probability=individual_probability,
        )
    if value == "bit_flip":
        return MutationOperator.bit_flip(
            probability=probability,
            individual_probability=individual_probability,
        )
    raise ConfigurationError(
        f"Unknown mutation operator: {value!r}. Valid: 'gaussian', 'uniform', 'bit_flip'."
    )


def normalize_selection_operator(
    value: str | SelectionOperator, *, tournament_size: int
) -> SelectionOperator:
    """Normalize a selection string or spec into a selection operator."""
    if isinstance(value, SelectionOperator):
        return value
    if value == "tournament":
        return SelectionOperator.tournament(size=tournament_size)
    if value == "roulette":
        return SelectionOperator.roulette()
    if value == "rank":
        return SelectionOperator.rank()
    raise ConfigurationError(
        f"Unknown selection operator: {value!r}. Valid: 'tournament', 'roulette', 'rank'."
    )


def normalize_bounds_policy(value: str | BoundsPolicy | None) -> BoundsPolicy:
    """Normalize a bounds-policy string or spec into a bounds policy."""
    if value is None:
        return BoundsPolicy.clamp()
    if isinstance(value, BoundsPolicy):
        return value
    if value == "clamp":
        return BoundsPolicy.clamp()
    raise ConfigurationError(f"Unknown bounds policy: {value!r}. Valid: 'clamp'.")


def gene_space_profile(gene_space: GeneSpace) -> GeneSpaceProfile:
    """Return the flat GA profile implied by a gene space."""
    kinds = set(gene_space.kinds)
    if kinds == {"bool"}:
        return "binary"
    if "bool" in kinds:
        return "mixed"
    return "numeric"


def gene_space_domain(gene_space: GeneSpace) -> GeneSpaceProfile:
    """Return the GA operator domain implied by a gene space."""
    return gene_space_profile(gene_space)


def _supported_gene_kinds_for_profile(profile: GeneSpaceProfile) -> frozenset[GeneKind]:
    if profile == "numeric":
        return NUMERIC_GENE_KINDS
    if profile == "binary":
        return BINARY_GENE_KINDS
    return ALL_FLAT_GENE_KINDS


def resolve_operator_domain(
    operator: CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy,
    gene_space: GeneSpace,
) -> CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy:
    """Resolve profile-sensitive operators against a concrete gene space."""
    profile = gene_space_profile(gene_space)

    if isinstance(operator, CrossoverOperator) and operator.domain == "auto":
        return CrossoverOperator(
            operator.name,
            dict(operator.parameters),
            _supported_gene_kinds_for_profile(profile),
            profile,
            custom=operator.custom,
            implementation=operator.implementation,
        )

    if (
        isinstance(operator, MutationOperator)
        and profile == "mixed"
        and operator.name in ("gaussian", "uniform", "bit_flip")
        and not operator.custom
    ):
        return MutationOperator(
            operator.name,
            dict(operator.parameters),
            ALL_FLAT_GENE_KINDS,
            "mixed",
            custom=operator.custom,
            implementation=operator.implementation,
        )

    return operator


def validate_operator_compatibility(
    operator: CrossoverOperator | MutationOperator | SelectionOperator | BoundsPolicy,
    gene_space: GeneSpace,
) -> None:
    """Raise when an operator is incompatible with a gene space."""
    resolved = resolve_operator_domain(operator, gene_space)
    if getattr(resolved, "custom", False):
        implementation = getattr(resolved, "implementation", None)
        validate_compatibility = getattr(implementation, "validate_compatibility", None)
        if callable(validate_compatibility):
            validate_compatibility(gene_space)
    if isinstance(resolved, (SelectionOperator, BoundsPolicy)):
        return
    actual_kinds = frozenset(gene_space.kinds)
    if not actual_kinds.issubset(resolved.supported_gene_kinds):
        supported = ", ".join(sorted(resolved.supported_gene_kinds))
        actual = ", ".join(sorted(actual_kinds))
        raise ConfigurationError(
            f"{resolved.operator_type}='{resolved.name}' supports {resolved.domain} "
            f"GeneSpace kinds {{{supported}}}, got {{{actual}}}."
        )


def validate_operator_set(
    *,
    gene_space: GeneSpace,
    crossover: CrossoverOperator,
    mutation: MutationOperator,
    selection: SelectionOperator,
    bounds_policy: BoundsPolicy,
) -> None:
    """Validate a complete GA operator set against a gene space."""
    gene_space_domain(gene_space)
    validate_operator_compatibility(crossover, gene_space)
    validate_operator_compatibility(mutation, gene_space)
    validate_operator_compatibility(selection, gene_space)
    validate_operator_compatibility(bounds_policy, gene_space)


def apply_bounds_policy(
    values: Sequence[GeneValue | float | int],
    gene_space: GeneSpace,
    bounds_policy: BoundsPolicy,
) -> list[GeneValue]:
    """Apply a bounds policy to decoded gene values."""
    if bounds_policy.name != "clamp":
        raise ConfigurationError(f"Unsupported bounds policy: {bounds_policy.name!r}.")
    if len(values) != gene_space.length:
        raise ConfigurationError(
            f"Bounds policy expected {gene_space.length} genes, got {len(values)}."
        )

    bounded: list[GeneValue] = []
    for value, gene in zip(values, gene_space.genes, strict=False):
        if gene.kind == "bool":
            if type(value) is bool:
                bounded.append(value)
            elif isinstance(value, int | float):
                bounded.append(float(value) >= 0.5)
            else:
                raise ConfigurationError(
                    f"Gene {gene.name!r} expects bool-compatible value, got {type(value).__name__}."
                )
            continue

        low = float(gene.low)
        high = float(gene.high)
        if not isinstance(value, int | float) or type(value) is bool:
            raise ConfigurationError(
                f"Gene {gene.name!r} expects numeric value, got {type(value).__name__}."
            )
        clamped = min(max(float(value), low), high)
        if gene.kind == "int":
            bounded.append(int(round(clamped)))
        else:
            bounded.append(float(clamped))

    gene_space.validate_genes(bounded)
    return bounded


@dataclass(frozen=True)
class OperatorContext:
    """Base deterministic context passed to custom operators."""

    gene_space: GeneSpace
    generation: int
    seed: int
    individual_index: int | None
    pair_index: int | None
    bounds_policy: BoundsPolicy


@dataclass(frozen=True)
class MutationContext(OperatorContext):
    """Context passed to custom mutation operators."""

    probability: float
    mutation_sigma: float
    mutation_sigmas: tuple[float, ...]


@dataclass(frozen=True)
class CrossoverContext(OperatorContext):
    """Context passed to custom crossover operators."""

    probability: float


@dataclass(frozen=True)
class SelectionContext(OperatorContext):
    """Context passed to custom selection operators."""

    tournament_size: int | None = None


@runtime_checkable
class CustomMutationProtocol(Protocol):
    """Protocol implemented by custom mutation operators."""

    name: str
    operator_type: Literal["mutation"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None:
        """Raise when this operator is incompatible with a gene space."""
        ...

    def mutate(self, values: Sequence[GeneValue], context: MutationContext) -> Sequence[GeneValue]:
        """Return one mutated decoded genome."""
        ...


def _custom_signature_payload(operator: object) -> dict[str, Any]:
    config_signature = getattr(operator, "config_signature", None)
    if callable(config_signature):
        payload = config_signature()
        if not isinstance(payload, Mapping):
            raise ConfigurationError("custom operator config_signature() must return a mapping.")
        return dict(json_safe(payload))
    return {"identity": stable_object_identity(operator)}


def custom_mutation_operator(operator: CustomMutationProtocol) -> MutationOperator:
    """Wrap a custom mutation implementation as a public mutation operator."""
    if getattr(operator, "operator_type", None) != "mutation":
        raise ConfigurationError("custom mutation operator must declare operator_type='mutation'.")
    if not hasattr(operator, "mutate") or not callable(operator.mutate):
        raise ConfigurationError(
            "custom mutation operator must implement mutate(values, context)."
        )
    name = getattr(operator, "name", operator.__class__.__name__)
    supported = frozenset(getattr(operator, "supported_gene_kinds", ALL_FLAT_GENE_KINDS))
    return MutationOperator(
        str(name),
        _custom_signature_payload(operator),
        supported,
        "custom",
        custom=True,
        implementation=operator,
    )


@runtime_checkable
class CustomCrossoverProtocol(Protocol):
    """Protocol implemented by custom crossover operators."""

    name: str
    operator_type: Literal["crossover"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None:
        """Raise when this operator is incompatible with a gene space."""
        ...

    def crossover(
        self,
        left: Sequence[GeneValue],
        right: Sequence[GeneValue],
        context: CrossoverContext,
    ) -> tuple[Sequence[GeneValue], Sequence[GeneValue]]:
        """Return two decoded child genomes."""
        ...


@runtime_checkable
class CustomSelectionProtocol(Protocol):
    """Protocol implemented by custom selection operators."""

    name: str
    operator_type: Literal["selection"]

    def validate_compatibility(self, gene_space: GeneSpace) -> None:
        """Raise when this operator is incompatible with a gene space."""
        ...

    def select(
        self,
        scores: Sequence[float],
        count: int,
        context: SelectionContext,
    ) -> Sequence[int]:
        """Return selected parent indices."""
        ...


def custom_crossover_operator(operator: CustomCrossoverProtocol) -> CrossoverOperator:
    """Wrap a custom crossover implementation as a public crossover operator."""
    if getattr(operator, "operator_type", None) != "crossover":
        raise ConfigurationError(
            "custom crossover operator must declare operator_type='crossover'."
        )
    if not hasattr(operator, "crossover") or not callable(operator.crossover):
        raise ConfigurationError(
            "custom crossover operator must implement crossover(left, right, context)."
        )
    name = getattr(operator, "name", operator.__class__.__name__)
    supported = frozenset(getattr(operator, "supported_gene_kinds", ALL_FLAT_GENE_KINDS))
    return CrossoverOperator(
        str(name),
        _custom_signature_payload(operator),
        supported,
        "custom",
        custom=True,
        implementation=operator,
    )


def custom_selection_operator(operator: CustomSelectionProtocol) -> SelectionOperator:
    """Wrap a custom selection implementation as a public selection operator."""
    if getattr(operator, "operator_type", None) != "selection":
        raise ConfigurationError(
            "custom selection operator must declare operator_type='selection'."
        )
    if not hasattr(operator, "select") or not callable(operator.select):
        raise ConfigurationError(
            "custom selection operator must implement select(scores, count, context)."
        )
    name = getattr(operator, "name", operator.__class__.__name__)
    return SelectionOperator(
        str(name),
        _custom_signature_payload(operator),
        "custom",
        custom=True,
        implementation=operator,
    )


__all__ = [
    "ALL_FLAT_GENE_KINDS",
    "BINARY_GENE_KINDS",
    "DEFAULT_CROSSOVER_ALPHA",
    "DEFAULT_CROSSOVER_ETA",
    "DEFAULT_CROSSOVER_PROBABILITY",
    "DEFAULT_MUTATION_INDIVIDUAL_PROBABILITY",
    "DEFAULT_MUTATION_PROBABILITY",
    "DEFAULT_MUTATION_SIGMA",
    "DEFAULT_TOURNAMENT_SIZE",
    "NUMERIC_GENE_KINDS",
    "BoundsPolicy",
    "CrossoverContext",
    "CrossoverOperator",
    "CustomCrossoverProtocol",
    "CustomMutationProtocol",
    "CustomSelectionProtocol",
    "MutationContext",
    "MutationOperator",
    "SelectionContext",
    "SelectionOperator",
    "apply_bounds_policy",
    "custom_crossover_operator",
    "custom_mutation_operator",
    "custom_selection_operator",
    "gene_space_domain",
    "gene_space_profile",
    "normalize_bounds_policy",
    "normalize_crossover_operator",
    "normalize_mutation_operator",
    "normalize_selection_operator",
    "resolve_operator_domain",
    "validate_operator_compatibility",
    "validate_operator_set",
]
