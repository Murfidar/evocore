from __future__ import annotations

import math

import pytest

from evocore import _core


BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0), (0.0, 1.0)]
KINDS = ["float", "float", "int", "bool"]
POPULATION = [
    [-4.0, -3.0, 1.0, 0.0],
    [-2.0, -1.0, 2.0, 1.0],
    [0.0, 1.0, 3.0, 0.0],
    [1.5, 2.0, 4.0, 1.0],
    [3.0, 4.0, 5.0, 0.0],
    [4.0, 5.0, 6.0, 1.0],
]
SCORES = [1.0, 2.0, 3.0, 4.0, 9.0, 5.0]


def _generate(strategy: str, *, direction: str = "maximize", target_slots=(0, 1, 2)):
    return _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        strategy,
        0.7,
        0.9,
        42,
        3,
        list(target_slots),
        direction,
    )


def _assert_valid_gene_vector(genes: list[float]) -> None:
    assert len(genes) == len(BOUNDS)
    for value, (low, high), kind in zip(genes, BOUNDS, KINDS, strict=True):
        assert low <= value <= high
        if kind == "int":
            assert value == round(value)
        if kind == "bool":
            assert value in (0.0, 1.0)


@pytest.mark.parametrize(
    ("strategy", "min_population", "donor_count"),
    [
        ("rand1bin", 4, 3),
        ("best1bin", 4, 3),
        ("rand2bin", 6, 5),
        ("current-to-best1bin", 4, 3),
    ],
)
def test_de_generate_trials_stateless_strategies_are_deterministic(
    strategy: str,
    min_population: int,
    donor_count: int,
) -> None:
    first = _generate(strategy)
    second = _generate(strategy)

    assert first == second
    assert len(first) == 3
    assert min_population <= len(POPULATION)

    for expected_slot, proposal in zip([0, 1, 2], first, strict=True):
        assert proposal["target_slot"] == expected_slot
        _assert_valid_gene_vector(proposal["genes"])
        metadata = proposal["metadata"]
        assert metadata["strategy"] == strategy
        assert metadata["target_slot"] == expected_slot
        assert len(metadata["donor_slots"]) == donor_count
        assert len(set(metadata["donor_slots"])) == donor_count
        sampled_difference_slots = {
            slot for pair in metadata["difference_pairs"] for slot in pair
        }
        assert expected_slot not in sampled_difference_slots


def test_de_generate_trials_best_strategy_reports_best_slot() -> None:
    proposals = _generate("best1bin")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 4
        assert metadata["base_slot"] == 4


def test_de_generate_trials_current_to_best_reports_target_base() -> None:
    proposals = _generate("current-to-best1bin")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 4
        assert metadata["base_slot"] == metadata["target_slot"]


def test_de_generate_trials_minimize_uses_lowest_score_as_best_slot() -> None:
    proposals = _generate("best1bin", direction="minimize")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 0
        assert metadata["base_slot"] == 0


def test_de_generate_trials_jde_returns_trial_parameters() -> None:
    first = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        3,
        [0, 1, 2],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )
    second = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        3,
        [0, 1, 2],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )

    assert first == second
    for proposal in first:
        _assert_valid_gene_vector(proposal["genes"])
        metadata = proposal["metadata"]
        assert metadata["strategy"] == "jde-rand1bin"
        assert metadata["adaptive_slot"] == metadata["target_slot"]
        assert 0.0 <= metadata["crossover_rate"] <= 1.0
        assert math.isfinite(metadata["mutation_factor"])
        assert metadata["mutation_factor"] >= 0.0


def test_de_generate_trials_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown DE strategy"):
        _generate("unknown")


def test_de_generate_trials_jde_can_refresh_parameters_from_seed() -> None:
    proposals = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        99,
        [0, 1, 2, 3, 4, 5],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )

    params = [
        (proposal["metadata"]["mutation_factor"], proposal["metadata"]["crossover_rate"])
        for proposal in proposals
    ]
    assert any(f != 0.5 or cr != 0.9 for f, cr in params)
    assert all(0.1 <= f <= 1.0 for f, _ in params)
    assert all(0.0 <= cr <= 1.0 for _, cr in params)
