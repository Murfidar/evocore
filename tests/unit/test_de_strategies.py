import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.optimizers.de.strategies import (
    TrialContext,
    repair_de_gene_value,
    strategy_spec_for,
    trial_proposal_for_strategy,
)


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _records(candidates, scores):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence="trusted_full",
            stage="full",
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def _trusted_population():
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    return engine


def _trusted_population_with_direction(direction: str):
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction=direction,
    )
    candidates = engine.ask()
    scores = [0, 1, 2, 3, 4, 5] if direction == "maximize" else [5, 4, 3, 2, 1, 0]
    engine.tell(_records(candidates, scores))
    return engine


def test_strategy_spec_for_returns_rand1bin_contract() -> None:
    spec = strategy_spec_for("rand1bin")

    assert spec.name == "rand1bin"
    assert spec.min_population_size == 4
    assert spec.is_adaptive is False
    assert spec.checkpoint_state_schema is None


def test_strategy_spec_for_rejects_unknown_strategy() -> None:
    with pytest.raises(
        ConfigurationError,
        match=(
            "strategy must be one of 'rand1bin', 'best1bin', 'rand2bin', 'current-to-best1bin'"
        ),
    ):
        strategy_spec_for("jade")


def test_repair_de_gene_value_preserves_mixed_gene_contract() -> None:
    space = _mixed_space()

    assert repair_de_gene_value(-7.0, space.genes[0]) == pytest.approx(-5.0)
    assert repair_de_gene_value(99.0, space.genes[1]) == 20
    assert repair_de_gene_value(0.49, space.genes[2]) is False
    assert repair_de_gene_value(0.50, space.genes[2]) is True
    assert repair_de_gene_value(2.0, space.genes[3]) == pytest.approx(1.5)


def test_rand1bin_strategy_proposal_matches_optimizer_fixture() -> None:
    engine = _trusted_population()
    population = [
        engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids
    ]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name="rand1bin",
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    assert proposal.genes == [
        pytest.approx(1.5811814881513984),
        2,
        True,
        pytest.approx(1.5),
    ]
    assert proposal.metadata["strategy"] == "rand1bin"
    assert proposal.metadata["target_slot"] == 0
    assert proposal.metadata["donor_slots"] == (4, 2, 1)


@pytest.mark.parametrize(
    "strategy",
    ["best1bin", "rand2bin", "current-to-best1bin"],
)
def test_stateless_strategy_proposals_have_required_metadata(strategy: str) -> None:
    engine = _trusted_population()
    population = [
        engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids
    ]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name=strategy,
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    assert proposal.metadata["strategy"] == strategy
    assert proposal.metadata["target_slot"] == 0
    assert proposal.metadata["donor_slots"]
    assert len(proposal.genes) == engine.gene_space.length
    engine.gene_space.validate_genes(proposal.genes)


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
@pytest.mark.parametrize("strategy", ["best1bin", "current-to-best1bin"])
def test_best_based_strategies_record_direction_aware_best_slot(
    direction: str,
    strategy: str,
) -> None:
    engine = _trusted_population_with_direction(direction)
    population = [
        engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids
    ]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name=strategy,
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
            direction=engine.direction,
        )
    )

    assert proposal.metadata["best_slot"] == 5


def test_rand2bin_records_five_distinct_donor_slots() -> None:
    engine = _trusted_population()
    population = [
        engine._candidates_by_id[candidate_id] for candidate_id in engine._target_candidate_ids
    ]

    proposal = trial_proposal_for_strategy(
        TrialContext(
            strategy_name="rand2bin",
            gene_space=engine.gene_space,
            population=population,
            target_slot=0,
            generation=engine.generation,
            seed=engine.seed,
            mutation_factor=engine.mutation_factor,
            crossover_rate=engine.crossover_rate,
        )
    )

    donor_slots = proposal.metadata["donor_slots"]
    assert len(donor_slots) == 5
    assert len(set(donor_slots)) == 5
    assert 0 not in donor_slots
