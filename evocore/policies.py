"""vNext optimization policy objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evocore.evaluation import Rung
from evocore.exceptions import ConfigurationError


@dataclass(frozen=True)
class MultiFidelityPolicy:
    """Configure multi-fidelity scheduling for vNext engines."""

    rungs: list[Rung]
    max_evaluations: int
    batch_size: int | None = None
    exploration_fraction: float = 0.10
    audit_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ConfigurationError("MultiFidelityPolicy requires at least one rung.")
        if int(self.max_evaluations) <= 0:
            raise ConfigurationError("max_evaluations must be positive.")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ConfigurationError("batch_size must be positive when provided.")
        if not (0.0 <= float(self.exploration_fraction) < 1.0):
            raise ConfigurationError("exploration_fraction must be in [0, 1).")
        if not (0.0 <= float(self.audit_fraction) < 1.0):
            raise ConfigurationError("audit_fraction must be in [0, 1).")

        names = [rung.name for rung in self.rungs]
        if len(names) != len(set(names)):
            raise ConfigurationError("MultiFidelityPolicy contains duplicate rung names.")
        trusted_full_rungs = [rung for rung in self.rungs if rung.confidence == "trusted_full"]
        if not trusted_full_rungs:
            raise ConfigurationError("MultiFidelityPolicy requires a trusted_full rung.")
        if len(trusted_full_rungs) != 1:
            raise ConfigurationError("MultiFidelityPolicy requires exactly one trusted_full rung.")
        if self.rungs[-1].confidence != "trusted_full":
            raise ConfigurationError("MultiFidelityPolicy final rung must be trusted_full.")

    @property
    def rung_names(self) -> Sequence[str]:
        """Return rung names in execution order."""
        return tuple(rung.name for rung in self.rungs)

    @property
    def final_rung(self) -> Rung:
        """Return the last configured rung."""
        return self.rungs[-1]

    @classmethod
    def single_full(
        cls,
        *,
        max_evaluations: int | None = None,
        batch_size: int | None = None,
        **legacy_kwargs: object,
    ) -> MultiFidelityPolicy:
        """Create a one-rung full-evaluation vNext policy."""
        if "budget" in legacy_kwargs:
            raise ConfigurationError(
                "MultiFidelityPolicy.single_full() uses max_evaluations=..., not budget=...."
            )
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(
                f"MultiFidelityPolicy.single_full() got unexpected argument(s): {unknown}."
            )
        if max_evaluations is None:
            raise ConfigurationError("single_full() requires max_evaluations.")
        return cls(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            exploration_fraction=0.0,
            audit_fraction=0.0,
        )
