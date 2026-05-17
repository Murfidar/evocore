import pytest

from evocore.core.errors import ConfigurationError
from evocore.search_space import Gene, GeneSpace, OperatorCodec


def test_numeric_space_accepts_sbx_gaussian():
    ops = OperatorCodec(GeneSpace.uniform(-1.0, 1.0, 2), "sbx", "gaussian")
    assert ops.gene_kinds == ["float", "float"]


def test_numeric_space_accepts_uniform_crossover_for_deap_parity():
    space = GeneSpace(
        [
            Gene("signal_mode", "int", 0, 4),
            Gene("threshold", "float", -1.0, 1.0),
        ]
    )

    ops = OperatorCodec(space, "uniform", "gaussian")

    assert ops.crossover == "uniform"
    assert ops.gene_kinds == ["int", "float"]


def test_binary_space_rejects_sbx():
    space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])
    with pytest.raises(ConfigurationError, match="binary"):
        OperatorCodec(space, "sbx", "bit_flip")


def test_mixed_bool_numeric_rejected():
    space = GeneSpace([Gene("x", "float", 0.0, 1.0), Gene("flag", "bool")])
    with pytest.raises(ConfigurationError, match="bool genes alongside"):
        OperatorCodec(space, "sbx", "gaussian")


def test_encode_decode_roundtrip_named_mixed_numeric():
    space = GeneSpace(
        [
            Gene("period", "int", 5, 20),
            Gene("x", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")
    encoded = ops.encode_genes([10, 0.25])
    assert encoded == [10.0, 0.25]
    decoded = ops.decode_genes([10.2, 0.25])
    assert decoded == [10, 0.25]


def test_decode_individual_adds_params_metadata():
    space = GeneSpace([Gene("period", "int", 5, 20)])
    ops = OperatorCodec(space, "sbx", "gaussian")
    ind = ops.decode_individual([12.0], fitness=3.0, fitness_valid=True)
    assert ind.genes == [12]
    assert ind.fitness == 3.0
    assert ind.fitness_valid is True
    assert ind.params == {"period": 12}


def test_sigma_override_takes_precedence():
    space = GeneSpace(
        [
            Gene("wide", "int", 0, 1000, sigma=0.01),
            Gene("x", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")
    assert ops.sigma_abs_list(0.2) == [10.0, 0.4]


def test_decode_individual_preserves_fixed_numeric_params():
    space = GeneSpace(
        [
            Gene("signal_mode", "int", 2, 2),
            Gene("threshold", "float", 0.5, 0.5),
            Gene("period", "int", 5, 20),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")

    solution = ops.decode_individual([2.0, 0.5, 12.0])

    assert solution.genes == [2, 0.5, 12]
    assert solution.params == {
        "signal_mode": 2,
        "threshold": 0.5,
        "period": 12,
    }


def test_encode_genes_uses_gene_space_validator_for_invalid_decoded_values():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")

    with pytest.raises(ConfigurationError, match="Gene 'x' at index 0 expects float"):
        ops.encode_genes([True, 10])

    with pytest.raises(ConfigurationError, match="Gene 'period' at index 1 expects int"):
        ops.encode_genes([0.5, 10.0])
