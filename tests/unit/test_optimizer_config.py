import json

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
