import json

from hypothesis import given
from hypothesis import strategies as st

from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual
from evocore.operators import OperatorSet

GENE_NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@st.composite
def valid_numeric_gene_specs(draw):
    kind = draw(st.sampled_from(["float", "int"]))
    name = draw(st.text(alphabet=GENE_NAME_CHARS, min_size=1, max_size=12))
    if kind == "int":
        low = draw(st.integers(min_value=-1000, max_value=999))
        high = draw(st.integers(min_value=low + 1, max_value=low + 1000))
    else:
        low = draw(
            st.floats(
                min_value=-1000.0,
                max_value=999.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        span = draw(
            st.floats(
                min_value=1e-6,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        high = low + span
    sigma = draw(
        st.none()
        | st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return GeneDef(name, kind, low, high, sigma=sigma)


@given(valid_numeric_gene_specs())
def test_numeric_gene_def_preserves_bounds_and_kind(gene):
    assert gene.kind in {"float", "int"}
    assert gene.low < gene.high
    if gene.sigma is not None:
        assert 0.0 < gene.sigma <= 1.0


@given(st.integers(min_value=1, max_value=25))
def test_uniform_space_has_requested_length(length):
    space = GeneSpace.uniform(-5.0, 5.0, length)

    assert space.length == length
    assert space.has_names is False
    assert space.params_for([0.0] * length) is None
    assert space.rust_bounds == [(-5.0, 5.0)] * length


@given(st.lists(st.sampled_from(["float", "int", "bool"]), min_size=1, max_size=10))
def test_named_params_match_gene_order(kinds):
    genes = []
    values = []
    for index, kind in enumerate(kinds):
        name = f"gene_{index}"
        if kind == "float":
            genes.append(GeneDef(name, "float", -10.0, 10.0))
            values.append(float(index) / 10.0)
        elif kind == "int":
            genes.append(GeneDef(name, "int", -10, 10))
            values.append(index - 5)
        else:
            genes.append(GeneDef(name, "bool"))
            values.append(index % 2 == 0)

    space = GeneSpace(genes)

    assert space.params_for(values) == dict(zip(space.names, values, strict=False))


@given(st.lists(st.integers(min_value=-20, max_value=20), min_size=1, max_size=10))
def test_individual_clone_preserves_genes_and_metadata(values):
    ind = Individual(
        list(values),
        fitness=1.25,
        fitness_valid=True,
        metadata={"params": {"x": 1}},
    )

    cloned = ind.clone()

    assert cloned.genes == ind.genes
    assert cloned.fitness == ind.fitness
    assert cloned.fitness_valid is True
    assert cloned.metadata == ind.metadata
    assert cloned is not ind


@given(
    st.integers(min_value=0, max_value=100),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_operator_decode_restores_named_params(period, threshold):
    space = GeneSpace(
        [
            GeneDef("period", "int", 0, 100),
            GeneDef("threshold", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")

    ind = ops.decode_individual([float(period), threshold])

    assert ind.genes == [period, threshold]
    assert ind.params == {"period": period, "threshold": threshold}


@st.composite
def valid_flat_gene_spaces(draw):
    kinds = draw(st.lists(st.sampled_from(["float", "int", "bool"]), min_size=1, max_size=8))
    genes = []
    for index, kind in enumerate(kinds):
        name = f"gene_{index}"
        if kind == "float":
            low = draw(
                st.floats(
                    min_value=-1000.0,
                    max_value=999.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            fixed = draw(st.booleans())
            if fixed:
                high = low
            else:
                span = draw(
                    st.floats(
                        min_value=1e-6,
                        max_value=1000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                high = low + span
            genes.append(GeneDef(name, "float", low, high))
        elif kind == "int":
            low = draw(st.integers(min_value=-1000, max_value=999))
            fixed = draw(st.booleans())
            high = low if fixed else draw(st.integers(min_value=low + 1, max_value=low + 1000))
            genes.append(GeneDef(name, "int", low, high))
        else:
            genes.append(GeneDef(name, "bool"))
    return GeneSpace(genes)


@given(valid_flat_gene_spaces())
def test_gene_space_signature_json_round_trips(space):
    signature = space.signature()

    assert json.loads(space.to_json()) == signature
    assert space.to_dict() == signature


@given(valid_flat_gene_spaces())
def test_gene_space_hash_is_stable_for_equivalent_flat_spaces(space):
    equivalent = GeneSpace(list(space.genes), has_names=space.has_names)

    assert equivalent.signature() == space.signature()
    assert equivalent.hash() == space.hash()
