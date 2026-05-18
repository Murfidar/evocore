"""vNext optimization policy objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evocore.core.errors import ConfigurationError
from evocore.lifecycle.records import EvaluationStage


@dataclass(frozen=True)
class BudgetPolicy:
    """Configure multi-fidelity scheduling for vNext engines."""

    stages: list[EvaluationStage]
    max_evaluations: int
    batch_size: int | None = None
    exploration_fraction: float = 0.10
    audit_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ConfigurationError("BudgetPolicy requires at least one stage.")
        if int(self.max_evaluations) <= 0:
            raise ConfigurationError("max_evaluations must be positive.")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ConfigurationError("batch_size must be positive when provided.")
        if not (0.0 <= float(self.exploration_fraction) < 1.0):
            raise ConfigurationError("exploration_fraction must be in [0, 1).")
        if not (0.0 <= float(self.audit_fraction) < 1.0):
            raise ConfigurationError("audit_fraction must be in [0, 1).")

        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ConfigurationError("BudgetPolicy contains duplicate stage names.")
        trusted_full_stages = [
            stage for stage in self.stages if stage.confidence == "trusted_full"
        ]
        if not trusted_full_stages:
            raise ConfigurationError("BudgetPolicy requires a trusted_full stage.")
        if len(trusted_full_stages) != 1:
            raise ConfigurationError("BudgetPolicy requires exactly one trusted_full stage.")
        if self.stages[-1].confidence != "trusted_full":
            raise ConfigurationError("BudgetPolicy final stage must be trusted_full.")

    @property
    def stage_names(self) -> Sequence[str]:
        """Return stage names in execution order."""
        return tuple(stage.name for stage in self.stages)

    @property
    def final_stage(self) -> EvaluationStage:
        """Return the last configured stage."""
        return self.stages[-1]

    @classmethod
    def single_full(
        cls,
        *,
        max_evaluations: int | None = None,
        batch_size: int | None = None,
        **legacy_kwargs: object,
    ) -> BudgetPolicy:
        """Create a one-stage full-evaluation vNext policy."""
        if "budget" in legacy_kwargs:
            raise ConfigurationError(
                "BudgetPolicy.single_full() uses max_evaluations=..., not budget=...."
            )
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise ConfigurationError(
                f"BudgetPolicy.single_full() got unexpected argument(s): {unknown}."
            )
        if max_evaluations is None:
            raise ConfigurationError("single_full() requires max_evaluations.")
        return cls(
            stages=[
                EvaluationStage(
                    "full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                )
            ],
            max_evaluations=max_evaluations,
            batch_size=batch_size,
            exploration_fraction=0.0,
            audit_fraction=0.0,
        )
