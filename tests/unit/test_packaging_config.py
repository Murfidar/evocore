from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyo3_extension_module_feature_is_maturin_only() -> None:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    pyo3_dependency = cargo["dependencies"]["pyo3"]
    cargo_features = (
        pyo3_dependency.get("features", []) if isinstance(pyo3_dependency, dict) else []
    )

    assert "extension-module" not in cargo_features
    assert "pyo3/extension-module" in pyproject["tool"]["maturin"]["features"]
