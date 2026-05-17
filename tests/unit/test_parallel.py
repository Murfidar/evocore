import pickle

import pytest

from evocore.core.errors import ConfigurationError
from evocore.core.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.optimizers.ga import run_child_optimizer
from evocore.search_space import Solution


def module_level_fitness(ind):
    return sum(ind.genes)


def test_ensure_picklable_rejects_lambda():
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        ensure_picklable(lambda _ind: 1.0, context="parallel='process'")


def testrun_child_optimizer_is_picklable():
    pickle.dumps(run_child_optimizer)
    assert ".<locals>." not in run_child_optimizer.__qualname__
    assert "." not in run_child_optimizer.__qualname__


def test_process_parallel_forces_spawn_context():
    pp = ProcessParallel(n_workers=2)
    assert pp._ctx.get_start_method() == "spawn"


def test_thread_parallel_evaluates_population():
    tp = ThreadParallel(n_workers=2)
    pop = [Solution([1.0]), Solution([2.0])]
    assert tp.evaluate(pop, module_level_fitness) == [1.0, 2.0]


def test_process_parallel_evaluates_population():
    pp = ProcessParallel(n_workers=2)
    pop = [Solution([1.0]), Solution([2.0])]
    assert pp.evaluate(pop, module_level_fitness) == [1.0, 2.0]


def test_process_parallel_reuses_executor_until_closed(monkeypatch):
    created = []
    shutdowns = []

    class FakeProcessPoolExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def map(self, fitness_fn, solutions):
            return [fitness_fn(solution) for solution in solutions]

        def shutdown(self, *, cancel_futures=False, wait=True):
            shutdowns.append((cancel_futures, wait))

    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )

    pp = ProcessParallel(n_workers=2)
    pop = [Solution([1.0]), Solution([2.0])]

    assert pp.evaluate(pop, module_level_fitness) == [1.0, 2.0]
    assert pp.evaluate(pop, module_level_fitness) == [1.0, 2.0]
    assert len(created) == 1

    pp.close()

    assert shutdowns == [(True, True)]


def test_process_parallel_context_manager_closes_executor(monkeypatch):
    shutdowns = []

    class FakeProcessPoolExecutor:
        def __init__(self, **_kwargs):
            pass

        def map(self, fitness_fn, solutions):
            return [fitness_fn(solution) for solution in solutions]

        def shutdown(self, *, cancel_futures=False, wait=True):
            shutdowns.append((cancel_futures, wait))

    monkeypatch.setattr(
        "concurrent.futures.ProcessPoolExecutor",
        FakeProcessPoolExecutor,
    )

    with ProcessParallel(n_workers=2) as pp:
        assert pp.evaluate([Solution([1.0])], module_level_fitness) == [1.0]

    assert shutdowns == [(True, True)]
