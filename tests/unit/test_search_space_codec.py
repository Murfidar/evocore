from __future__ import annotations

import pytest

from evocore import ConfigurationError, Gene, GeneSpace
from evocore.search_space import (
    decode_gene_values,
    encode_gene_values,
    repair_gene_value,
    repair_gene_values,
)


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 0.5, 0.5),
        ]
    )


def test_repair_gene_value_clamps_rounds_thresholds_and_preserves_types() -> None:
    space = _mixed_space()

    assert repair_gene_value(99.0, space.genes[0]) == pytest.approx(1.0)
    assert repair_gene_value(20.8, space.genes[1]) == 20
    assert repair_gene_value(0.49, space.genes[2]) is False
    assert repair_gene_value(0.5, space.genes[2]) is True
    disabled = False
    assert repair_gene_value(disabled, space.genes[2]) is False
    assert repair_gene_value(99.0, space.genes[3]) == pytest.approx(0.5)


def test_repair_gene_values_validates_length_and_repaired_values() -> None:
    space = _mixed_space()

    assert repair_gene_values(space, [99.0, 20.8, 0.2, -9.0]) == [
        1.0,
        20,
        False,
        0.5,
    ]

    with pytest.raises(ConfigurationError, match="expected 4 genes, got 3"):
        repair_gene_values(space, [0.0, 3, True])


def test_decode_gene_values_repairs_encoded_numeric_vectors() -> None:
    space = _mixed_space()

    assert decode_gene_values(space, [-9.0, 1.2, 0.8, 99.0]) == [
        -1.0,
        2,
        True,
        0.5,
    ]


def test_encode_gene_values_validates_decoded_values_before_encoding() -> None:
    space = _mixed_space()

    assert encode_gene_values(space, [0.25, 7, True, 0.5]) == [0.25, 7.0, 1.0, 0.5]

    with pytest.raises(ConfigurationError, match="Gene 'period' at index 1 expects int"):
        encode_gene_values(space, [0.25, 7.0, True, 0.5])


def test_repair_gene_value_rejects_incompatible_inputs() -> None:
    space = _mixed_space()

    with pytest.raises(ConfigurationError, match="expects numeric-compatible value"):
        repair_gene_value("bad", space.genes[0])

    with pytest.raises(ConfigurationError, match="expects bool-compatible value"):
        repair_gene_value("bad", space.genes[2])
