from collections.abc import Callable, Sequence

OP_INIT: int
OP_CROSSOVER: int
OP_MUTATION: int
OP_SELECTION: int
OP_CMAES_ASK: int
OP_MULTI_RUN: int
OP_CROSSOVER_PROB: int


class FloatIndividual:
    genes: list[float]
    fitness: float | None

    def __init__(self, genes: Sequence[float], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class IntegerIndividual:
    genes: list[int]
    fitness: float | None

    def __init__(self, genes: Sequence[int], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class BinaryIndividual:
    genes: list[bool]
    fitness: float | None

    def __init__(self, genes: Sequence[bool], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class PyCMAESState:
    generation: int
    sigma: float
    mean: list[float]
    eigendecomp_interval: int

    def __init__(
        self,
        mean: Sequence[float],
        sigma: float,
        lambda_: int,
        bounds: Sequence[tuple[float, float]],
    ) -> None: ...
    def ask(self, master_seed: int, generation: int) -> list[list[float]]: ...
    def tell(self, samples: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None: ...
    def __repr__(self) -> str: ...


def py_derive_seed(master_seed: int, generation: int, individual_idx: int, op: int) -> int: ...
def blend_crossover(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def simulated_binary_crossover(
    a: Sequence[float],
    b: Sequence[float],
    eta: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def gaussian_mutation(
    genes: Sequence[float],
    sigma: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def uniform_mutation(
    genes: Sequence[float],
    low: float,
    high: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def int_simulated_binary_crossover(
    a: Sequence[float],
    b: Sequence[float],
    eta: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def int_gaussian_mutation(
    genes: Sequence[float],
    sigma: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def int_uniform_mutation(
    genes: Sequence[float],
    low: float,
    high: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def one_point_crossover(
    a: Sequence[float],
    b: Sequence[float],
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def two_point_crossover(
    a: Sequence[float],
    b: Sequence[float],
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def uniform_crossover(
    a: Sequence[float],
    b: Sequence[float],
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def bit_flip_mutation(
    genes: Sequence[float],
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def tournament_selection(
    fitnesses: Sequence[float],
    k: int,
    tournament_size: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def roulette_selection(
    fitnesses: Sequence[float],
    k: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def rank_selection(
    fitnesses: Sequence[float],
    k: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def init_population(
    gene_bounds: Sequence[tuple[float, float]],
    gene_kinds_str: Sequence[str],
    population_size: int,
    master_seed: int,
) -> list[list[float]]: ...
def reproduce_population(
    population: Sequence[Sequence[float]],
    fitnesses: Sequence[float],
    crossover_type: str,
    crossover_prob: float,
    crossover_eta: float,
    crossover_alpha: float,
    mutation_type: str,
    mutation_prob: float,
    mutation_sigmas: Sequence[float],
    gene_bounds: Sequence[tuple[float, float]],
    gene_kinds: Sequence[str],
    selection_type: str,
    tournament_size: int,
    population_size: int,
    master_seed: int,
    generation: int,
) -> list[list[float]]: ...
def evaluate_sequential(
    genes_list: Sequence[Sequence[float]],
    fitness_fn: Callable[[list[float]], float],
) -> list[float]: ...
def evaluate_parallel_rayon(
    genes_list: Sequence[Sequence[float]],
    fitness_fn: Callable[[list[float]], float],
    n_threads: int,
) -> list[float]: ...
