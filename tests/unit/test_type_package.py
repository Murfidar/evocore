from pathlib import Path

import evocore


def test_py_typed_marker_is_packaged_next_to_module():
    package_dir = Path(evocore.__file__).parent

    assert (package_dir / "py.typed").is_file()


def test_core_stub_is_packaged_next_to_module():
    package_dir = Path(evocore.__file__).parent

    assert (package_dir / "_core.pyi").is_file()


def test_core_stub_mentions_exported_symbols():
    stub = (Path(evocore.__file__).parent / "_core.pyi").read_text(encoding="utf-8")

    for symbol in [
        "class FloatIndividual",
        "class IntegerIndividual",
        "class BinaryIndividual",
        "class PyCMAESState",
        "def py_derive_seed",
        "def reproduce_population",
        "def evaluate_parallel_rayon",
    ]:
        assert symbol in stub
