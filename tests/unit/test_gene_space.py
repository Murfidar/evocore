import json

import pytest

from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneDef, GeneSpace
from evocore.stats import gene_space_hash, gene_space_signature


def test_gene_def_float_requires_bounds():
    with pytest.raises(ConfigurationError, match="bounds required"):
        GeneDef("x", "float")


def test_gene_def_float_rejects_non_finite_bounds():
    with pytest.raises(ConfigurationError, match="finite"):
        GeneDef("x", "float", float("nan"), 1.0)

    with pytest.raises(ConfigurationError, match="finite"):
        GeneDef("x", "float", 0.0, float("inf"))


def test_uniform_space_rejects_non_finite_bounds():
    with pytest.raises(ConfigurationError, match="finite"):
        GeneSpace.uniform(float("-inf"), 1.0, 3)


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


def test_fixed_numeric_genes_are_valid_and_report_fixed_metadata():
    fixed_float = GeneDef("threshold", "float", 0.5, 0.5)
    fixed_int = GeneDef("signal_mode", "int", 2, 2)
    variable = GeneDef("period", "int", 5, 20)

    space = GeneSpace([fixed_float, variable, fixed_int])

    assert fixed_float.is_fixed is True
    assert fixed_int.is_fixed is True
    assert variable.is_fixed is False
    assert space.fixed_indices == [0, 2]
    assert space.variable_indices == [1]
    assert space.fixed_count == 2
    assert space.variable_count == 1
    assert space.bounds == [(0.5, 0.5), (5, 20), (2, 2)]
    assert space.rust_bounds == [(0.5, 0.5), (5.0, 20.0), (2.0, 2.0)]


def test_reversed_numeric_bounds_are_still_rejected():
    with pytest.raises(ConfigurationError, match="requires low <= high"):
        GeneDef("threshold", "float", 1.0, 0.5)


def test_fixed_int_genes_still_require_integer_bounds():
    with pytest.raises(ConfigurationError, match="integer bounds"):
        GeneDef("signal_mode", "int", 2.0, 2.0)


def test_bool_genes_are_not_fixed_in_this_iteration():
    gene = GeneDef("flag", "bool")

    assert gene.is_fixed is False


def test_gene_space_signature_to_dict_hash_and_json_are_canonical():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0, sigma=0.2),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    expected = {
        "schema_version": 1,
        "genes": [
            {
                "name": "x",
                "kind": "float",
                "low": -1.0,
                "high": 1.0,
                "sigma": 0.2,
                "is_fixed": False,
            },
            {
                "name": "period",
                "kind": "int",
                "low": 2,
                "high": 20,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "enabled",
                "kind": "bool",
                "low": None,
                "high": None,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "fixed_threshold",
                "kind": "float",
                "low": 0.5,
                "high": 0.5,
                "sigma": None,
                "is_fixed": True,
            },
        ],
        "has_names": True,
        "length": 4,
    }

    assert space.signature() == expected
    assert space.to_dict() == expected
    assert gene_space_signature(space) == expected
    assert space.hash() == gene_space_hash(expected)
    assert json.loads(space.to_json()) == expected
    assert space.to_json() == space.to_json()


def test_uniform_gene_space_signature_preserves_unnamed_contract():
    space = GeneSpace.uniform(-5.0, 5.0, 2)

    assert space.signature() == {
        "schema_version": 1,
        "genes": [
            {
                "name": "gene_0",
                "kind": "float",
                "low": -5.0,
                "high": 5.0,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "gene_1",
                "kind": "float",
                "low": -5.0,
                "high": 5.0,
                "sigma": None,
                "is_fixed": False,
            },
        ],
        "has_names": False,
        "length": 2,
    }


def test_validate_genes_accepts_valid_decoded_values():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    assert space.validate_genes([0.25, 10, True, 0.5]) is None
    assert space.validate_genes([0, 2, False, 0.5]) is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.25], "GeneSpace expected 4 genes, got 1."),
        ([True, 10, True, 0.5], "Gene 'x' at index 0 expects float"),
        ([float("nan"), 10, True, 0.5], "Gene 'x' at index 0 must be finite"),
        ([2.0, 10, True, 0.5], "Gene 'x' at index 0 must be within"),
        ([0.25, True, True, 0.5], "Gene 'period' at index 1 expects int"),
        ([0.25, 10.0, True, 0.5], "Gene 'period' at index 1 expects int"),
        ([0.25, 21, True, 0.5], "Gene 'period' at index 1 must be within"),
        ([0.25, 10, 1, 0.5], "Gene 'enabled' at index 2 expects bool"),
        ([0.25, 10, True, 0.6], "Gene 'fixed_threshold' at index 3 must be within"),
    ],
)
def test_validate_genes_rejects_invalid_decoded_values(values, message):
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    with pytest.raises(ConfigurationError, match=message):
        space.validate_genes(values)
