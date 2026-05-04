import pickle

import pytest

from evocore.exceptions import ConfigurationError
from evocore.ga import _run_child_engine
from evocore.individual import Individual
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable


def module_level_fitness(ind):
    return sum(ind.genes)


def test_ensure_picklable_rejects_lambda():
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        ensure_picklable(lambda ind: 1.0, context="parallel='process'")


def test_run_child_engine_is_picklable():
    pickle.dumps(_run_child_engine)
    assert ".<locals>." not in _run_child_engine.__qualname__
    assert "." not in _run_child_engine.__qualname__


def test_process_parallel_forces_spawn_context():
    pp = ProcessParallel(n_workers=2)
    assert pp._ctx.get_start_method() == "spawn"


def test_thread_parallel_evaluates_population():
    tp = ThreadParallel(n_workers=2)
    pop = [Individual([1.0]), Individual([2.0])]
    assert tp.evaluate(pop, module_level_fitness) == [1.0, 2.0]


def test_process_parallel_evaluates_population():
    pp = ProcessParallel(n_workers=2)
    pop = [Individual([1.0]), Individual([2.0])]
    assert pp.evaluate(pop, module_level_fitness) == [1.0, 2.0]
