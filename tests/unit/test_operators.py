import pytest

from evocore import (
    ConfigurationError,
    CrossoverOperator,
    Gene,
    GeneSpace,
    MutationOperator,
)
from evocore.search_space import OperatorCodec


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
    with pytest.raises(ConfigurationError, match=r"sbx.*bool"):
        OperatorCodec(space, "sbx", "bit_flip")


def test_mixed_bool_numeric_accepts_uniform_gaussian_and_decodes_bool():
    space = GeneSpace(
        [
            Gene("x", "float", 0.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
        ]
    )

    ops = OperatorCodec(space, "uniform", "gaussian")

    assert ops.crossover == "uniform"
    assert ops.mutation == "gaussian"
    assert ops.crossover_operator.domain == "mixed"
    assert ops.mutation_operator.domain == "mixed"
    assert ops.encode_values([0.25, 10, True]) == [0.25, 10.0, 1.0]
    assert ops.decode_values([0.25, 10.2, 0.8]) == [0.25, 10, True]


def test_mixed_bool_numeric_rejects_numeric_only_crossover():
    space = GeneSpace([Gene("x", "float", 0.0, 1.0), Gene("flag", "bool")])

    with pytest.raises(ConfigurationError, match=r"crossover='sbx'.*bool.*float"):
        OperatorCodec(space, "sbx", "gaussian")


def test_encode_decode_roundtrip_named_mixed_numeric():
    space = GeneSpace(
        [
            Gene("period", "int", 5, 20),
            Gene("x", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")
    encoded = ops.encode_values([10, 0.25])
    assert encoded == [10.0, 0.25]
    decoded = ops.decode_values([10.2, 0.25])
    assert decoded == [10, 0.25]


def test_decode_solution_adds_params_metadata():
    space = GeneSpace([Gene("period", "int", 5, 20)])
    ops = OperatorCodec(space, "sbx", "gaussian")
    ind = ops.decode_solution([12.0], score=3.0, score_valid=True)
    assert ind.values == [12]
    assert ind.score == 3.0
    assert ind.score_valid is True
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


def test_decode_solution_preserves_fixed_numeric_params():
    space = GeneSpace(
        [
            Gene("signal_mode", "int", 2, 2),
            Gene("threshold", "float", 0.5, 0.5),
            Gene("period", "int", 5, 20),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")

    solution = ops.decode_solution([2.0, 0.5, 12.0])

    assert solution.values == [2, 0.5, 12]
    assert solution.params == {
        "signal_mode": 2,
        "threshold": 0.5,
        "period": 12,
    }


def test_encode_values_uses_gene_space_validator_for_invalid_decoded_values():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
        ]
    )
    ops = OperatorCodec(space, "sbx", "gaussian")

    with pytest.raises(ConfigurationError, match="Gene 'x' at index 0 expects float"):
        ops.encode_values([True, 10])

    with pytest.raises(ConfigurationError, match="Gene 'period' at index 1 expects int"):
        ops.encode_values([0.5, 10.0])


def test_operator_codec_accepts_normalized_typed_operators():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    ops = OperatorCodec(space, CrossoverOperator.sbx(), MutationOperator.gaussian())

    assert ops.crossover == "sbx"
    assert ops.mutation == "gaussian"
