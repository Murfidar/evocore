import json

import pytest

from evocore import ConfigurationError, Gene, GeneSpace, GeneticAlgorithmOptimizer
from evocore.callbacks import Callback, EarlyStopping, MetricsLogger, ProgressBar
from evocore.optimizers.config import (
    OptimizerConfig,
    RuntimeHookSignature,
    callback_hook_signature,
    config_hash,
    reproducibility_from_hooks,
    stable_object_identity,
)


class CustomCallback(Callback):
    pass


def sample_initializer() -> None:
    return None


def test_optimizer_config_to_dict_and_hash_are_deterministic():
    config = OptimizerConfig(
        optimizer_type="ExampleOptimizer",
        parameters={"seed": 42, "population_size": 8},
        components={
            "mutation": {
                "type": "gaussian",
                "parameters": {"sigma": 0.2, "probability": 0.1},
            }
        },
    )

    assert config.to_dict() == {
        "schema_version": 1,
        "optimizer_type": "ExampleOptimizer",
        "parameters": {"population_size": 8, "seed": 42},
        "components": {
            "mutation": {
                "parameters": {"probability": 0.1, "sigma": 0.2},
                "type": "gaussian",
            }
        },
    }
    assert config_hash(config.to_dict()) == config_hash(config.to_dict())
    assert json.loads(config.to_json()) == config.to_dict()


def test_runtime_hook_signature_to_dict_is_json_safe():
    hook = RuntimeHookSignature(
        hook_type="artifact",
        identity="evocore.callbacks.MetricsLogger",
        config={"path": {"b", "a"}},
        reproducibility="configured",
        notes=("writes metrics",),
    )

    assert hook.to_dict() == {
        "hook_type": "artifact",
        "identity": "evocore.callbacks.MetricsLogger",
        "config": {"path": ["a", "b"]},
        "reproducibility": "configured",
        "notes": ["writes metrics"],
    }


def test_stable_object_identity_uses_module_and_qualname():
    assert stable_object_identity(sample_initializer).endswith(
        "test_optimizer_config.sample_initializer"
    )


def test_callback_hook_signature_classifies_known_callbacks(tmp_path):
    metrics = callback_hook_signature(MetricsLogger(str(tmp_path / "metrics.jsonl")))
    progress = callback_hook_signature(ProgressBar())
    early_stop = callback_hook_signature(EarlyStopping(patience=3, min_delta=0.5))

    assert metrics.hook_type == "artifact"
    assert metrics.identity == "evocore.callbacks.metrics.MetricsLogger"
    assert metrics.config == {"path": str(tmp_path / "metrics.jsonl")}
    assert metrics.reproducibility == "configured"

    assert progress.hook_type == "artifact"
    assert progress.identity == "evocore.callbacks.progress.ProgressBar"
    assert progress.config == {}
    assert progress.reproducibility == "configured"

    assert early_stop.hook_type == "termination"
    assert early_stop.identity == "evocore.callbacks.stopping.EarlyStopping"
    assert early_stop.config == {"patience": 3, "min_delta": 0.5}
    assert early_stop.reproducibility == "configured"


def test_unknown_callback_marks_reproducibility_partial():
    hook = callback_hook_signature(CustomCallback())

    assert hook.hook_type == "termination"
    assert hook.reproducibility == "partial"
    assert hook.notes == (
        "CustomCallback may affect termination or side effects without a stable hook signature.",
    )


def test_reproducibility_from_hooks_reports_partial_notes():
    hooks = (
        RuntimeHookSignature(
            hook_type="environment",
            identity="tests.sample_initializer",
            config={},
            reproducibility="partial",
            notes=("process_initializer is opaque.",),
        ),
    )

    status, notes = reproducibility_from_hooks(hooks)

    assert status == "partial"
    assert notes == ("process_initializer is opaque.",)


def test_ga_default_and_explicit_default_configs_match():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    default = GeneticAlgorithmOptimizer(space)
    explicit = GeneticAlgorithmOptimizer(
        space,
        population_size=100,
        max_generations=100,
        crossover="sbx",
        crossover_prob=0.9,
        crossover_eta=2.0,
        crossover_alpha=0.5,
        mutation="gaussian",
        mutation_prob=0.1,
        mutation_individual_prob=1.0,
        mutation_sigma=0.2,
        mutation_sigma_schedule="constant",
        mutation_sigma_end=0.02,
        selection="tournament",
        tournament_size=3,
        elitism=1,
        parallel="none",
        n_workers=None,
        seed=0,
        direction="maximize",
        max_evaluations=None,
        track_diversity=False,
    )

    assert default.config_signature() == explicit.config_signature()
    assert default.config_hash() == explicit.config_hash()
    assert default.config().hash() == default.config_hash()


def test_ga_config_signature_uses_nested_component_shape():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=8,
        max_generations=5,
        seed=42,
    )

    assert engine.config_signature() == {
        "schema_version": 1,
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "parameters": {
            "direction": "maximize",
            "elitism": 1,
            "max_evaluations": None,
            "max_generations": 5,
            "n_workers": None,
            "parallel": "none",
            "population_size": 8,
            "seed": 42,
            "track_diversity": False,
        },
        "components": {
            "crossover": {
                "type": "sbx",
                "parameters": {"alpha": 0.5, "eta": 2.0, "probability": 0.9},
            },
            "mutation": {
                "type": "gaussian",
                "parameters": {
                    "individual_probability": 1.0,
                    "probability": 0.1,
                    "sigma": 0.2,
                },
            },
            "mutation_schedule": {
                "type": "constant",
                "parameters": {"sigma_end": 0.02},
            },
            "selection": {
                "type": "tournament",
                "parameters": {"tournament_size": 3},
            },
        },
    }


def test_ga_algorithm_component_change_alters_hash():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    gaussian = GeneticAlgorithmOptimizer(space, mutation="gaussian")
    uniform = GeneticAlgorithmOptimizer(space, mutation="uniform")

    assert gaussian.config_hash() != uniform.config_hash()


def test_ga_artifact_hook_path_does_not_change_config_hash(tmp_path):
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    left = GeneticAlgorithmOptimizer(space, callbacks=[MetricsLogger(str(tmp_path / "a.jsonl"))])
    right = GeneticAlgorithmOptimizer(space, callbacks=[MetricsLogger(str(tmp_path / "b.jsonl"))])

    assert left.config_hash() == right.config_hash()


def test_ga_termination_hook_is_visible_in_reproducibility():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        callbacks=[EarlyStopping(patience=4, min_delta=0.25)],
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "full"
    assert payload["runtime_hooks"] == [
        {
            "hook_type": "termination",
            "identity": "evocore.callbacks.stopping.EarlyStopping",
            "config": {"patience": 4, "min_delta": 0.25},
            "reproducibility": "configured",
            "notes": [],
        }
    ]


def test_ga_process_initializer_marks_reproducibility_partial():
    def init_worker() -> None:
        return None

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        parallel="process",
        process_initializer=init_worker,
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "partial"
    assert payload["runtime_hooks"][0]["hook_type"] == "environment"
    assert payload["runtime_hooks"][0]["reproducibility"] == "partial"
    assert "process_initializer is opaque." in payload["reproducibility_notes"]


def test_ga_validate_compatibility_is_public():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace([Gene("a", "bool"), Gene("b", "bool")]),
        crossover="one_point",
        mutation="bit_flip",
    )

    assert engine.validate_compatibility() is None

    with pytest.raises(ConfigurationError, match="binary GeneSpace"):
        GeneticAlgorithmOptimizer(
            GeneSpace([Gene("a", "bool"), Gene("b", "bool")]),
            crossover="sbx",
            mutation="bit_flip",
        )
