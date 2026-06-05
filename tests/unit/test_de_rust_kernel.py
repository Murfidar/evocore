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
        sampled_difference_slots = {slot for pair in metadata["difference_pairs"] for slot in pair}
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


def test_de_generate_trials_forces_variable_gene_when_fixed_gene_exists() -> None:
    population = [
        [1.5, 0.0],
        [1.5, 1.0],
        [1.5, 3.0],
        [1.5, -2.0],
    ]

    proposal = _core.de_generate_trials(
        population,
        [0.0, 1.0, 2.0, 3.0],
        [(1.5, 1.5), (-10.0, 10.0)],
        ["float", "float"],
        "rand1bin",
        0.8,
        0.0,
        0,
        0,
        [0],
        "maximize",
    )[0]

    assert proposal["genes"][0] == pytest.approx(1.5)
    assert proposal["genes"][1] != pytest.approx(population[0][1])


def test_de_generate_trials_repairs_encoded_outputs_for_each_kind() -> None:
    population = [
        [-100.0, 1.0, 0.0],
        [100.0, 20.0, 1.0],
        [-50.0, 2.0, 0.0],
        [50.0, 19.0, 1.0],
    ]

    proposals = _core.de_generate_trials(
        population,
        [0.0, 1.0, 2.0, 3.0],
        [(-1.0, 1.0), (2.0, 20.0), (0.0, 1.0)],
        ["float", "int", "bool"],
        "rand1bin",
        2.0,
        1.0,
        123,
        0,
        [0, 1, 2, 3],
        "maximize",
    )

    for proposal in proposals:
        x, period, enabled = proposal["genes"]
        assert -1.0 <= x <= 1.0
        assert 2.0 <= period <= 20.0
        assert period == round(period)
        assert enabled in (0.0, 1.0)


def test_de_generate_trials_uses_python_integer_round_ties_for_repair() -> None:
    def generated_integer(value: float) -> float:
        proposal = _core.de_generate_trials(
            [[0.0], [value], [value], [value]],
            [0.0, 1.0, 2.0, 3.0],
            [(0.0, 10.0)],
            ["int"],
            "rand1bin",
            0.0,
            1.0,
            123,
            0,
            [0],
            "maximize",
        )[0]
        return proposal["genes"][0]

    assert generated_integer(2.5) == pytest.approx(2.0)
    assert generated_integer(3.5) == pytest.approx(4.0)
