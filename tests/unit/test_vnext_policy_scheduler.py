import pytest

from evocore.evaluation import Rung
from evocore.exceptions import ConfigurationError
from evocore.policies import MultiFidelityPolicy


def test_policy_requires_unique_rung_names_and_full_budget() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=32,
        batch_size=8,
        exploration_fraction=0.10,
        audit_fraction=0.05,
    )

    assert policy.rung_names == ("cheap", "full")
    assert policy.final_rung.name == "full"


def test_policy_rejects_duplicate_rung_names() -> None:
    with pytest.raises(ConfigurationError, match="duplicate rung"):
        MultiFidelityPolicy(
            rungs=[
                Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
                Rung("cheap", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
            ],
            full_evaluation_budget=16,
        )


def test_policy_rejects_missing_trusted_full_rung() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full"):
        MultiFidelityPolicy(
            rungs=[Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial")],
            full_evaluation_budget=16,
        )


def test_policy_rejects_invalid_budget_and_fractions() -> None:
    with pytest.raises(ConfigurationError, match="full_evaluation_budget"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=0,
        )

    with pytest.raises(ConfigurationError, match="exploration_fraction"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=1,
            exploration_fraction=1.5,
        )
