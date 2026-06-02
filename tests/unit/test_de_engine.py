import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    Gene,
    GeneSpace,
)
from evocore.callbacks import Callback
from evocore.core.errors import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def test_de_constructor_sets_public_configuration() -> None:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=8,
        max_generations=12,
        mutation_factor=0.7,
        crossover_rate=0.6,
        seed=123,
        direction="minimize",
    )

    assert engine.population_size == 8
    assert engine.max_generations == 12
    assert engine.mutation_factor == pytest.approx(0.7)
    assert engine.crossover_rate == pytest.approx(0.6)
    assert engine.strategy == "rand1bin"
    assert engine.seed == 123
    assert engine.direction == "minimize"
    assert engine.state_summary().trusted_count == 0


def test_de_config_signature_is_stable_and_hash_changes_with_parameters() -> None:
    left = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.5)
    right = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.9)

    assert left.config_signature()["optimizer_type"] == "DifferentialEvolutionOptimizer"
    assert left.config_hash() != right.config_hash()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gene_space": None}, "gene_space required"),
        ({"population_size": 3}, "population_size"),
        ({"mutation_factor": -0.1}, "mutation_factor"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"parallel": "gpu"}, "parallel"),
        ({"direction": "lowest"}, "direction"),
    ],
)
def test_de_rejects_invalid_configuration(kwargs, message) -> None:
    params = {"gene_space": _space(), **kwargs}
    with pytest.raises(ConfigurationError, match=message):
        DifferentialEvolutionOptimizer(**params)


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(
                    float(value) ** 2 for value in candidate.genes if type(value) is not bool
                ),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CountingCallback(Callback):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.completed = False

    def on_generation_start(self, generation, population) -> None:
        self.starts += 1

    def on_generation_end(self, generation, population, info) -> None:
        self.ends += 1

    def on_run_end(self, result) -> None:
        self.completed = True


def test_de_run_returns_optimization_result_with_events_and_generations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.best_candidate_id is not None
    assert result.n_evaluations >= 6
    assert len(result.generations) == 2
    assert len(result.events) > 0
    assert result.reproducibility is not None
    assert result.reproducibility.optimizer_type == "DifferentialEvolutionOptimizer"


def test_de_run_invokes_callbacks() -> None:
    callback = CountingCallback()
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
        callbacks=[callback],
    )

    engine.run(SphereEvaluator())

    assert callback.starts == 2
    assert callback.ends == 2
    assert callback.completed is True


def test_de_run_honors_max_evaluations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        max_evaluations=8,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations <= 12
