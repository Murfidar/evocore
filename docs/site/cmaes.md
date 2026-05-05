# CMA-ES

`CMAESEngine` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"`
because the Rust covariance state is not picklable.

::: evocore.cmaes.CMAESEngine
    options:
      members:
        - run
