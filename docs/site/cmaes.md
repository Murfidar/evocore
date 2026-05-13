# CMA-ES

`CMAESEngine` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"`
because the Rust covariance state is not picklable.

`direction="maximize"` and `direction="minimize"` preserve raw user fitness values in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state.

::: evocore.cmaes.CMAESEngine
    options:
      members:
        - run
