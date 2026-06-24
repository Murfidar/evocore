import json

import pytest

from evocore import (
    CMAESOptimizer,
    ConfigurationError,
    CrossoverOperator,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
)
from evocore.callbacks import Callback, EarlyStopping, MetricsLogger, ProgressBar
from evocore.optimizers.config import (
    OptimizerConfig,
    RuntimeHookSignature,
    callback_hook_signature,
    config_hash,
    reproducibility_from_hooks,
    stable_object_identity,
)
from evocore.optimizers.operators import custom_mutation_operator


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
            "bounds_policy": {
                "type": "clamp",
                "operator_type": "bounds",
                "domain": "repair",
                "parameters": {},
            },
            "crossover": {
                "type": "sbx",
                "operator_type": "crossover",
                "domain": "numeric",
                "parameters": {"eta": 2.0, "probability": 0.9},
            },
            "mutation": {
                "type": "gaussian",
                "operator_type": "mutation",
                "domain": "numeric",
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
                "operator_type": "selection",
                "domain": "score",
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

    with pytest.raises(ConfigurationError, match="supports numeric"):
        GeneticAlgorithmOptimizer(
            GeneSpace([Gene("a", "bool"), Gene("b", "bool")]),
            crossover="sbx",
            mutation="bit_flip",
        )


def test_cmaes_default_and_explicit_default_configs_match():
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    default = CMAESOptimizer(space)
    explicit = CMAESOptimizer(
        space,
        population_size=50,
        initial_mean=None,
        initial_sigma=0.3,
        max_generations=300,
        parallel="none",
        n_workers=None,
        callbacks=None,
        seed=0,
        direction="maximize",
        track_diversity=False,
    )

    assert default.config_signature() == explicit.config_signature()
    assert default.config_hash() == explicit.config_hash()
    assert default.config().hash() == default.config_hash()


def test_cmaes_config_signature_uses_nested_component_shape():
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        initial_mean=[0.0, 0.1, 0.2],
        initial_sigma=0.4,
        max_generations=8,
        seed=42,
    )

    assert engine.config_signature() == {
        "schema_version": 1,
        "optimizer_type": "CMAESOptimizer",
        "parameters": {
            "direction": "maximize",
            "initial_mean": [0.0, 0.1, 0.2],
            "initial_sigma": 0.4,
            "max_generations": 8,
            "n_workers": None,
            "parallel": "none",
            "population_size": 6,
            "seed": 42,
            "track_diversity": False,
        },
        "components": {
            "distribution": {
                "type": "cma_es",
                "parameters": {"initial_sigma": 0.4},
            }
        },
    }


def test_cmaes_strategy_parameter_change_alters_hash():
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    small_sigma = CMAESOptimizer(space, initial_sigma=0.2)
    large_sigma = CMAESOptimizer(space, initial_sigma=0.4)

    assert small_sigma.config_hash() != large_sigma.config_hash()


def test_cmaes_margin_strategy_changes_config_hash() -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])

    assert (
        CMAESOptimizer(space, integer_strategy="round").config_hash()
        != CMAESOptimizer(
            space,
            integer_strategy="margin",
        ).config_hash()
    )


def test_cmaes_callback_hook_is_visible_in_reproducibility(tmp_path):
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        callbacks=[MetricsLogger(str(tmp_path / "metrics.jsonl"))],
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "full"
    assert payload["runtime_hooks"] == [
        {
            "hook_type": "artifact",
            "identity": "evocore.callbacks.metrics.MetricsLogger",
            "config": {"path": str(tmp_path / "metrics.jsonl")},
            "reproducibility": "configured",
            "notes": [],
        }
    ]


def test_ga_typed_operator_parameters_change_config_hash():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    left = GeneticAlgorithmOptimizer(space, crossover=CrossoverOperator.sbx(eta=2.0))
    right = GeneticAlgorithmOptimizer(space, crossover=CrossoverOperator.sbx(eta=3.0))

    assert left.config_hash() != right.config_hash()


def test_ga_custom_operator_is_visible_in_reproducibility_metadata():
    class IdentityMutation:
        name = "identity"
        operator_type = "mutation"
        supported_gene_kinds = frozenset({"float"})

        def validate_compatibility(self, gene_space):
            return None

        def mutate(self, values, context):
            return list(values)

    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        mutation=custom_mutation_operator(IdentityMutation()),
    )

    payload = engine._reproducibility_metadata().to_dict()

    assert payload["reproducibility_status"] == "partial"
    assert any(
        hook["config"].get("component") == "mutation" and hook["reproducibility"] == "partial"
        for hook in payload["runtime_hooks"]
    )


def _bool_space():
    return GeneSpace([Gene("a", "bool"), Gene("b", "bool")])


def _mixed_bool_space():
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_bool_only_default_ga_resolves_binary_operators():
    default = GeneticAlgorithmOptimizer(_bool_space(), population_size=4, max_generations=1)
    explicit = GeneticAlgorithmOptimizer(
        _bool_space(),
        population_size=4,
        max_generations=1,
        crossover="uniform",
        mutation="bit_flip",
    )

    assert default.crossover == "uniform"
    assert default.mutation == "bit_flip"
    assert default.config_signature() == explicit.config_signature()
    assert default.config_signature()["components"]["crossover"]["domain"] == "binary"
    assert default.config_signature()["components"]["mutation"]["domain"] == "binary"


def test_mixed_bool_default_ga_resolves_typed_defaults():
    default = GeneticAlgorithmOptimizer(_mixed_bool_space(), population_size=4, max_generations=1)
    explicit = GeneticAlgorithmOptimizer(
        _mixed_bool_space(),
        population_size=4,
        max_generations=1,
        crossover="uniform",
        mutation="gaussian",
    )

    assert default.crossover == "uniform"
    assert default.mutation == "gaussian"
    assert default.config_signature() == explicit.config_signature()
    assert default.config_signature()["components"]["crossover"]["domain"] == "mixed"
    assert default.config_signature()["components"]["mutation"]["domain"] == "mixed"


@pytest.mark.parametrize("crossover", ["sbx", "blx", "one_point", "two_point"])
def test_explicit_incompatible_crossovers_still_reject_mixed_bool_spaces(crossover):
    with pytest.raises(ConfigurationError, match=rf"crossover='{crossover}'.*bool"):
        GeneticAlgorithmOptimizer(_mixed_bool_space(), crossover=crossover)
