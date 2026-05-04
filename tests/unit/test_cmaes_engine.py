import pytest

from evocore import ConfigurationError, GeneDef, GeneSpace
from evocore.cmaes import CMAESEngine


def test_cmaes_requires_gene_space():
    with pytest.raises(ConfigurationError, match="gene_space required"):
        CMAESEngine(gene_space=None)


def test_cmaes_rejects_bool_genes():
    with pytest.raises(ConfigurationError, match="bool"):
        CMAESEngine(GeneSpace([GeneDef("flag", "bool")]))


def test_cmaes_process_parallel_raises_at_construction():
    with pytest.raises(ConfigurationError) as exc:
        CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), parallel="process")

    message = str(exc.value)
    assert "parallel='process'" in message
    assert "not picklable" in message
    assert "parallel='thread'" in message


def test_apply_bounds_and_round_for_int_genes():
    space = GeneSpace(
        [GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)]
    )
    engine = CMAESEngine(space, population_size=6, generations=1, seed=42)

    assert engine._apply_bounds_and_round([20.8, 1.5]) == [20.0, 1.0]
    assert engine._decode_individual([10.2, 0.25]).genes == [10, 0.25]
