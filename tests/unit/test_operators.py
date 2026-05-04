import pytest

from evocore.exceptions import ConfigurationError
from evocore.gene_space import GeneDef, GeneSpace
from evocore.operators import OperatorSet


def test_numeric_space_accepts_sbx_gaussian():
    ops = OperatorSet(GeneSpace.uniform(-1.0, 1.0, 2), "sbx", "gaussian")
    assert ops.gene_kinds == ["float", "float"]


def test_binary_space_rejects_sbx():
    space = GeneSpace([GeneDef("a", "bool"), GeneDef("b", "bool")])
    with pytest.raises(ConfigurationError, match="binary"):
        OperatorSet(space, "sbx", "bit_flip")


def test_mixed_bool_numeric_rejected():
    space = GeneSpace([GeneDef("x", "float", 0.0, 1.0), GeneDef("flag", "bool")])
    with pytest.raises(ConfigurationError, match="bool genes alongside"):
        OperatorSet(space, "sbx", "gaussian")


def test_encode_decode_roundtrip_named_mixed_numeric():
    space = GeneSpace(
        [
            GeneDef("period", "int", 5, 20),
            GeneDef("x", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")
    encoded = ops.encode_genes([10, 0.25])
    assert encoded == [10.0, 0.25]
    decoded = ops.decode_genes([10.2, 0.25])
    assert decoded == [10, 0.25]


def test_decode_individual_adds_params_metadata():
    space = GeneSpace([GeneDef("period", "int", 5, 20)])
    ops = OperatorSet(space, "sbx", "gaussian")
    ind = ops.decode_individual([12.0], fitness=3.0, fitness_valid=True)
    assert ind.genes == [12]
    assert ind.fitness == 3.0
    assert ind.fitness_valid is True
    assert ind.params == {"period": 12}


def test_sigma_override_takes_precedence():
    space = GeneSpace(
        [
            GeneDef("wide", "int", 0, 1000, sigma=0.01),
            GeneDef("x", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")
    assert ops.sigma_abs_list(0.2) == [10.0, 0.4]
