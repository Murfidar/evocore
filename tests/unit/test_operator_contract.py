import pytest

from evocore import (
    BoundsPolicy,
    ConfigurationError,
    CrossoverOperator,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    MutationOperator,
    SelectionOperator,
)
from evocore.optimizers.operators import (
    apply_bounds_policy,
    custom_crossover_operator,
    custom_mutation_operator,
    custom_selection_operator,
    gene_space_profile,
    normalize_bounds_policy,
    normalize_crossover_operator,
    normalize_mutation_operator,
    normalize_selection_operator,
    resolve_operator_domain,
    validate_operator_compatibility,
)


class ShiftMutation:
    name = "shift"
    operator_type = "mutation"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": "shift", "amount": 0.25}

    def validate_compatibility(self, gene_space):
        return None

    def mutate(self, values, context):
        return [float(value) + 0.25 for value in values]


class SwapCrossover:
    name = "swap"
    operator_type = "crossover"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": "swap"}

    def validate_compatibility(self, gene_space):
        return None

    def crossover(self, left, right, context):
        return right, left


class FirstParentSelection:
    name = "first_parent"
    operator_type = "selection"

    def config_signature(self):
        return {"name": "first_parent"}

    def validate_compatibility(self, gene_space):
        return None

    def select(self, scores, count, context):
        return [0 for _ in range(count)]


def test_builtin_operator_factories_have_canonical_signatures():
    assert CrossoverOperator.sbx(eta=3.0, probability=0.8).signature() == {
        "type": "sbx",
        "operator_type": "crossover",
        "domain": "numeric",
        "parameters": {"eta": 3.0, "probability": 0.8},
    }
    assert MutationOperator.gaussian(
        probability=0.25,
        individual_probability=0.75,
        sigma=0.15,
    ).signature() == {
        "type": "gaussian",
        "operator_type": "mutation",
        "domain": "numeric",
        "parameters": {
            "individual_probability": 0.75,
            "probability": 0.25,
            "sigma": 0.15,
        },
    }
    assert SelectionOperator.tournament(size=5).signature() == {
        "type": "tournament",
        "operator_type": "selection",
        "domain": "score",
        "parameters": {"tournament_size": 5},
    }
    assert BoundsPolicy.clamp().signature() == {
        "type": "clamp",
        "operator_type": "bounds",
        "domain": "repair",
        "parameters": {},
    }


def test_legacy_strings_normalize_to_builtin_operator_specs():
    assert normalize_crossover_operator(
        "sbx",
        probability=0.9,
        eta=2.0,
        alpha=0.5,
    ) == CrossoverOperator.sbx(eta=2.0, probability=0.9)
    assert normalize_mutation_operator(
        "gaussian",
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    ) == MutationOperator.gaussian(
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    )
    assert normalize_selection_operator("tournament", tournament_size=3) == (
        SelectionOperator.tournament(size=3)
    )
    assert normalize_bounds_policy(None) == BoundsPolicy.clamp()


def test_uniform_crossover_resolves_domain_from_gene_space():
    numeric_space = GeneSpace([Gene("period", "int", 2, 20), Gene("x", "float", -1.0, 1.0)])
    binary_space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    assert resolve_operator_domain(CrossoverOperator.uniform(), numeric_space).signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "numeric",
        "parameters": {"probability": 0.9},
    }
    assert resolve_operator_domain(CrossoverOperator.uniform(), binary_space).signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "binary",
        "parameters": {"probability": 0.9},
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CrossoverOperator.sbx(probability=-0.1),
        lambda: CrossoverOperator.sbx(eta=0.0),
        lambda: CrossoverOperator.blx(alpha=-0.1),
        lambda: MutationOperator.gaussian(probability=1.2),
        lambda: MutationOperator.gaussian(individual_probability=-0.1),
        lambda: MutationOperator.gaussian(sigma=1.5),
        lambda: SelectionOperator.tournament(size=0),
    ],
)
def test_operator_factories_validate_parameters(factory):
    with pytest.raises(ConfigurationError):
        factory()


def test_operator_contract_names_are_public_imports():
    assert CrossoverOperator.sbx().name == "sbx"
    assert MutationOperator.gaussian().name == "gaussian"
    assert SelectionOperator.tournament().name == "tournament"
    assert BoundsPolicy.clamp().name == "clamp"


def test_ga_constructor_accepts_typed_operator_specs():
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    engine = GeneticAlgorithmOptimizer(
        space,
        crossover=CrossoverOperator.sbx(eta=3.0, probability=0.75),
        mutation=MutationOperator.gaussian(
            probability=0.25,
            individual_probability=0.5,
            sigma=0.15,
        ),
        selection=SelectionOperator.tournament(size=5),
        bounds_policy=BoundsPolicy.clamp(),
    )

    assert engine.crossover == "sbx"
    assert engine.crossover_prob == 0.75
    assert engine.crossover_eta == 3.0
    assert engine.mutation == "gaussian"
    assert engine.mutation_prob == 0.25
    assert engine.mutation_individual_prob == 0.5
    assert engine.mutation_sigma == 0.15
    assert engine.selection == "tournament"
    assert engine.tournament_size == 5
    assert engine.bounds_policy == BoundsPolicy.clamp()


def test_ga_constructor_preserves_positional_crossover_argument():
    engine = GeneticAlgorithmOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), 4, 1, "blx")

    assert engine.crossover == "blx"
    assert engine.bounds_policy == BoundsPolicy.clamp()


def test_typed_crossover_rejects_conflicting_legacy_scalar():
    with pytest.raises(ConfigurationError, match="crossover_eta conflicts"):
        GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 2),
            crossover=CrossoverOperator.sbx(eta=3.0),
            crossover_eta=4.0,
        )


def test_typed_mutation_rejects_conflicting_legacy_scalar():
    with pytest.raises(ConfigurationError, match="mutation_prob conflicts"):
        GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 2),
            mutation=MutationOperator.gaussian(probability=0.25),
            mutation_prob=0.3,
        )


def test_numeric_operator_matrix_accepts_mixed_float_int_space():
    space = GeneSpace([Gene("period", "int", 2, 20), Gene("x", "float", -1.0, 1.0)])

    validate_operator_compatibility(CrossoverOperator.sbx(), space)
    validate_operator_compatibility(CrossoverOperator.blx(), space)
    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.gaussian(), space)
    validate_operator_compatibility(MutationOperator.uniform(), space)


def test_binary_operator_matrix_accepts_bool_space():
    space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    validate_operator_compatibility(CrossoverOperator.one_point(), space)
    validate_operator_compatibility(CrossoverOperator.two_point(), space)
    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.bit_flip(), space)


def test_incompatible_operator_errors_name_domain_and_actual_kinds():
    space = GeneSpace([Gene("a", "bool"), Gene("b", "bool")])

    with pytest.raises(ConfigurationError, match=r"crossover='sbx'.*numeric.*bool"):
        validate_operator_compatibility(CrossoverOperator.sbx(), space)


def _mixed_bool_numeric_space():
    return GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_gene_space_profile_distinguishes_numeric_binary_and_mixed():
    assert gene_space_profile(GeneSpace.uniform(-1.0, 1.0, 2)) == "numeric"
    assert gene_space_profile(GeneSpace([Gene("a", "bool"), Gene("b", "bool")])) == "binary"
    assert gene_space_profile(_mixed_bool_numeric_space()) == "mixed"


def test_mixed_bool_numeric_operator_matrix_accepts_typed_ga_defaults():
    space = _mixed_bool_numeric_space()

    crossover = resolve_operator_domain(CrossoverOperator.uniform(), space)
    gaussian = resolve_operator_domain(MutationOperator.gaussian(), space)
    uniform = resolve_operator_domain(MutationOperator.uniform(), space)
    bit_flip = resolve_operator_domain(MutationOperator.bit_flip(), space)

    assert crossover.signature() == {
        "type": "uniform",
        "operator_type": "crossover",
        "domain": "mixed",
        "parameters": {"probability": 0.9},
    }
    assert gaussian.signature() == {
        "type": "gaussian",
        "operator_type": "mutation",
        "domain": "mixed",
        "parameters": {
            "individual_probability": 1.0,
            "probability": 0.1,
            "sigma": 0.2,
        },
    }
    assert uniform.signature()["domain"] == "mixed"
    assert bit_flip.signature()["domain"] == "mixed"

    validate_operator_compatibility(CrossoverOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.gaussian(), space)
    validate_operator_compatibility(MutationOperator.uniform(), space)
    validate_operator_compatibility(MutationOperator.bit_flip(), space)


@pytest.mark.parametrize(
    ("operator", "pattern"),
    [
        (CrossoverOperator.sbx(), r"crossover='sbx'.*bool.*float.*int"),
        (CrossoverOperator.blx(), r"crossover='blx'.*bool.*float.*int"),
        (CrossoverOperator.one_point(), r"crossover='one_point'.*bool.*float.*int"),
        (CrossoverOperator.two_point(), r"crossover='two_point'.*bool.*float.*int"),
    ],
)
def test_mixed_bool_numeric_operator_matrix_rejects_incompatible_crossovers(
    operator,
    pattern,
):
    with pytest.raises(ConfigurationError, match=pattern):
        validate_operator_compatibility(operator, _mixed_bool_numeric_space())


def test_custom_mutation_operator_must_cover_mixed_bool_gene_kind():
    operator = custom_mutation_operator(ShiftMutation())

    with pytest.raises(ConfigurationError, match=r"mutation='shift'.*bool.*float.*int"):
        validate_operator_compatibility(operator, _mixed_bool_numeric_space())


def test_bounds_policy_clamps_rounds_thresholds_and_preserves_fixed_values():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 0.5, 0.5),
        ]
    )

    assert apply_bounds_policy([5.0, 20.8, 0.2, 99.0], space, BoundsPolicy.clamp()) == [
        1.0,
        20,
        False,
        0.5,
    ]
    assert apply_bounds_policy([-5.0, 1.2, 0.8, -99.0], space, BoundsPolicy.clamp()) == [
        -1.0,
        2,
        True,
        0.5,
    ]


def test_per_gene_sigma_override_does_not_decay_with_global_schedule():
    space = GeneSpace(
        [
            Gene("override", "float", 0.0, 10.0, sigma=0.5),
            Gene("scheduled", "float", 0.0, 10.0),
        ]
    )
    engine = GeneticAlgorithmOptimizer(
        space,
        mutation_sigma=0.4,
        mutation_sigma_schedule="linear_decay",
        mutation_sigma_end=0.1,
        max_generations=3,
    )

    assert engine.operators.sigma_abs_list(engine._compute_sigma_fraction(2)) == pytest.approx(
        [5.0, 1.0]
    )


def test_custom_mutation_operator_uses_stable_signature_when_available():
    operator = custom_mutation_operator(ShiftMutation())

    assert operator.custom is True
    assert operator.signature() == {
        "type": "shift",
        "operator_type": "mutation",
        "domain": "custom",
        "parameters": {"name": "shift", "amount": 0.25},
    }


def test_custom_mutation_operator_without_signature_uses_identity_and_partial_note():
    class IdentityMutation:
        name = "identity"
        operator_type = "mutation"
        supported_gene_kinds = frozenset({"float"})

        def validate_compatibility(self, gene_space):
            return None

        def mutate(self, values, context):
            return list(values)

    operator = custom_mutation_operator(IdentityMutation())

    assert operator.signature()["type"] == "identity"
    assert operator.signature()["parameters"]["identity"].endswith("IdentityMutation")


def test_custom_crossover_and_selection_operators_have_signatures():
    crossover = custom_crossover_operator(SwapCrossover())
    selection = custom_selection_operator(FirstParentSelection())

    assert crossover.signature() == {
        "type": "swap",
        "operator_type": "crossover",
        "domain": "custom",
        "parameters": {"name": "swap"},
    }
    assert selection.signature() == {
        "type": "first_parent",
        "operator_type": "selection",
        "domain": "custom",
        "parameters": {"name": "first_parent"},
    }
