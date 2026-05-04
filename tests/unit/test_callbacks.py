import json
import pickle

from evocore.callbacks import CheckpointCallback, EarlyStopping, GenerationInfo, MetricsLogger
from evocore.individual import Individual, Population


def test_generation_info_fields():
    info = GenerationInfo(generation=2, nan_fitness_count=1, cached_count=3)
    assert info.generation == 2
    assert info.nan_fitness_count == 1
    assert info.cached_count == 3


def test_early_stopping_sets_should_stop():
    cb = EarlyStopping(patience=2, min_delta=0.01)
    pop = Population([Individual([0.0], fitness=1.0)])
    info = GenerationInfo(0, 0, 0)
    cb.on_generation_end(0, pop, info)
    cb.on_generation_end(1, pop, info)
    cb.on_generation_end(2, pop, info)
    assert cb.should_stop is True


def test_checkpoint_callback_writes_pickle(tmp_path):
    cb = CheckpointCallback(path=str(tmp_path), every=1)
    cb.bind_context(seed=42)
    pop = Population([Individual([1.0], fitness=2.0)])
    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))
    payload = pickle.loads((tmp_path / "checkpoint_gen_3.pkl").read_bytes())
    assert payload["generation"] == 3
    assert payload["seed"] == 42


def test_metrics_logger_uses_utf8_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    cb = MetricsLogger(str(path))
    pop = Population([Individual([1.0], fitness=2.0)])
    cb.on_generation_end(0, pop, GenerationInfo(0, 1, 2))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["nan_fitness_count"] == 1
    assert record["cached_count"] == 2
