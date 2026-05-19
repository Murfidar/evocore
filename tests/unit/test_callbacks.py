import json
import pickle

from evocore import GeneSpace, GeneticAlgorithmOptimizer
from evocore.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    GenerationInfo,
    MetricsLogger,
    ProgressBar,
)
from evocore.search_space import Solution, SolutionSet


def test_generation_info_fields():
    info = GenerationInfo(generation=2, nan_score_count=1, cached_count=3)
    assert info.generation == 2
    assert info.nan_score_count == 1
    assert info.cached_count == 3


def test_early_stopping_sets_should_stop():
    cb = EarlyStopping(patience=2, min_delta=0.01)
    pop = SolutionSet([Solution([0.0], score=1.0)])
    info = GenerationInfo(0, 0, 0)
    cb.on_generation_end(0, pop, info)
    cb.on_generation_end(1, pop, info)
    cb.on_generation_end(2, pop, info)
    assert cb.should_stop is True


def test_checkpoint_callback_writes_legacy_pickle(tmp_path):
    cb = CheckpointCallback(path=str(tmp_path), every=1, format="legacy_pickle")
    cb.bind_context(seed=42)
    pop = SolutionSet([Solution([1.0], score=2.0)])
    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))
    payload = pickle.loads((tmp_path / "checkpoint_gen_3.pkl").read_bytes())
    assert payload["generation"] == 3
    assert payload["seed"] == 42


def test_checkpoint_callback_writes_stable_checkpoint(tmp_path):
    pop = SolutionSet([Solution([1.0], score=2.0, score_valid=True)])

    def factory(*, generation, population, metadata):
        assert generation == 3
        assert [solution.values for solution in population] == [[1.0]]
        assert metadata["callback"]["generation_info"] == {
            "generation": 3,
            "nan_score_count": 0,
            "cached_count": 0,
        }
        engine = GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 1),
            population_size=4,
            max_generations=4,
            seed=42,
        )
        return engine.checkpoint(
            generation=generation,
            population=population,
            metadata=metadata,
        )

    cb = CheckpointCallback(path=str(tmp_path), every=1, format="stable")
    cb.bind_context(seed=42, checkpoint_factory=factory)

    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))

    payload = json.loads(
        (tmp_path / "checkpoint_gen_3.evocore-checkpoint.json").read_text(encoding="utf-8")
    )
    assert payload["checkpoint_kind"] == "optimizer_state"
    assert payload["position"]["generation"] == 3


def test_metrics_logger_uses_utf8_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    cb = MetricsLogger(str(path))
    pop = SolutionSet([Solution([1.0], score=2.0)])
    cb.on_generation_end(0, pop, GenerationInfo(0, 1, 2))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["nan_score_count"] == 1
    assert record["cached_count"] == 2


def test_progress_bar_binds_max_generations():
    cb = ProgressBar()

    cb.bind_context(seed=42, max_generations=7)

    assert cb._total == 7
