import pytest

from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneDef, GeneSpace


def test_gene_def_float_requires_bounds():
    with pytest.raises(ConfigurationError, match="bounds required"):
        GeneDef("x", "float")


def test_gene_def_int_requires_integer_bounds():
    with pytest.raises(ConfigurationError, match="integer bounds"):
        GeneDef("period", "int", 1.5, 10)


def test_gene_def_bool_rejects_bounds():
    with pytest.raises(ConfigurationError, match="bool genes do not use bounds"):
        GeneDef("flag", "bool", 0, 1)


def test_gene_def_sigma_range():
    with pytest.raises(ConfigurationError, match="sigma"):
        GeneDef("x", "float", -1.0, 1.0, sigma=1.5)


def test_uniform_space_properties():
    space = GeneSpace.uniform(-5.0, 5.0, 3)
    assert space.length == 3
    assert space.kinds == ["float", "float", "float"]
    assert space.bounds == [(-5.0, 5.0)] * 3
    assert space.has_names is False
    assert space.params_for([1.0, 2.0, 3.0]) is None


def test_named_space_params():
    space = GeneSpace(
        [
            GeneDef("fast", "int", 5, 50),
            GeneDef("threshold", "float", 0.0, 1.0),
            GeneDef("enabled", "bool"),
        ]
    )
    assert space.has_names is True
    assert space.names == ["fast", "threshold", "enabled"]
    assert space.params_for([10, 0.25, True]) == {
        "fast": 10,
        "threshold": 0.25,
        "enabled": True,
    }


def test_duplicate_names_rejected():
    with pytest.raises(ConfigurationError, match="Duplicate gene name"):
        GeneSpace(
            [
                GeneDef("x", "float", 0.0, 1.0),
                GeneDef("x", "float", 0.0, 1.0),
            ]
        )


def test_rust_bounds_encode_bool_as_zero_one():
    space = GeneSpace([GeneDef("flag", "bool")])
    assert space.rust_bounds == [(0.0, 1.0)]
