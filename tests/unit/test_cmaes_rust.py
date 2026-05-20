"""
Smoke tests for PyCMAESState exposed via PyO3.

Focus:
  - Correct shapes (lambda x n) from ask()
  - Determinism: same (master_seed, generation) -> identical samples
  - Boundary compliance: all samples within bounds after mirror-folding
  - tell() correctness: generation increments, sigma stays positive
  - RNG independence: different seeds/generations diverge
  - Idempotency: ask() is a pure function for a fixed CMA-ES state
  - Convergence signal: mean norm decreases on sphere over 30 generations
  - Integer-gene workflow: tell() receives continuous samples
"""

import math

import pytest

from evocore._core import PyCMAESState


def make_state(
    n: int = 5,
    lambda_: int = 10,
    sigma: float = 0.5,
    bounds=None,
) -> PyCMAESState:
    if bounds is None:
        bounds = [(-5.0, 5.0)] * n
    mean = [(lo + hi) / 2.0 for lo, hi in bounds]
    return PyCMAESState(mean, sigma, lambda_, bounds)


def neg_sphere(genes: list[float]) -> float:
    return -sum(x * x for x in genes)


class TestPyCMAESStateConstruction:
    def test_generation_starts_at_zero(self):
        assert make_state().generation == 0

    def test_sigma_preserved(self):
        s = make_state(sigma=0.3)
        assert abs(s.sigma - 0.3) < 1e-12

    def test_mean_matches_input(self):
        mean_in = [1.0, 2.0, 3.0]
        s = PyCMAESState(mean_in, 0.5, 6, [(-5.0, 5.0)] * 3)
        assert s.mean == mean_in

    def test_eigendecomp_interval_at_least_one(self):
        for n in [3, 5, 10, 20]:
            s = make_state(n=n, lambda_=4 + int(3 * math.log(n)))
            assert s.eigendecomp_interval >= 1, f"n={n}"

    def test_invalid_bounds_length_raises(self):
        with pytest.raises(ValueError, match="bounds length must match mean length"):
            PyCMAESState([0.0] * 5, 0.5, 10, [(-1.0, 1.0)] * 3)


class TestAskShape:
    def test_returns_correct_lambda_samples(self):
        s = make_state(lambda_=15)
        samples = s.ask(42, 0)
        assert len(samples) == 15

    def test_returns_correct_n_genes_per_sample(self):
        s = make_state(n=7, lambda_=10)
        samples = s.ask(42, 0)
        assert all(len(samp) == 7 for samp in samples)

    def test_all_samples_within_bounds(self):
        bounds = [(-2.0, 2.0), (-1.0, 1.0), (0.0, 5.0), (-10.0, 0.0)]
        s = PyCMAESState([0.0, 0.0, 2.5, -5.0], 5.0, 20, bounds)
        samples = s.ask(42, 0)
        for i, sample in enumerate(samples):
            for j, (gene, (low, high)) in enumerate(zip(sample, bounds)):
                assert low - 1e-9 <= gene <= high + 1e-9, (
                    f"sample[{i}][{j}]={gene} outside [{low}, {high}]"
                )

    def test_samples_are_floats(self):
        samples = make_state().ask(42, 0)
        assert all(isinstance(gene, float) for sample in samples for gene in sample)


class TestAskDeterminism:
    def test_same_args_same_result(self):
        s = make_state()
        a = s.ask(42, 5)
        b = s.ask(42, 5)
        assert a == b

    def test_different_generation_diverges(self):
        s = make_state()
        a = s.ask(42, 0)
        b = s.ask(42, 1)
        assert a != b

    def test_different_master_seed_diverges(self):
        s = make_state()
        a = s.ask(1, 0)
        b = s.ask(2, 0)
        assert a != b

    def test_ask_uses_updated_distribution_after_tell(self):
        s = make_state(n=3, lambda_=6)

        samples_before = s.ask(42, 0)
        fitnesses = [neg_sphere(sample) for sample in samples_before]
        s.tell(samples_before, fitnesses)

        samples_after = s.ask(42, 0)
        assert samples_before != samples_after


class TestTell:
    def test_increments_generation(self):
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(sample) for sample in samples])
        assert s.generation == 1

    def test_generation_tracks_call_count(self):
        s = make_state(n=3, lambda_=6)
        for generation in range(5):
            samples = s.ask(42, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])
        assert s.generation == 5

    def test_sigma_stays_positive(self):
        s = make_state(n=4, lambda_=8)
        for generation in range(20):
            samples = s.ask(42, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])
            assert s.sigma > 0.0, f"sigma must stay positive at gen {generation + 1}"

    def test_sigma_is_finite(self):
        s = make_state(n=4, lambda_=8)
        for generation in range(20):
            samples = s.ask(42, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])
            assert math.isfinite(s.sigma), f"sigma became non-finite at gen {generation + 1}"

    def test_mean_is_a_list_of_floats(self):
        s = make_state(n=5, lambda_=10)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(sample) for sample in samples])
        assert all(isinstance(value, float) for value in s.mean)
        assert len(s.mean) == 5

    def test_nan_fitness_handled_gracefully(self):
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        fitnesses = [float("nan")] * 3 + [neg_sphere(sample) for sample in samples[3:]]
        s.tell(samples, fitnesses)
        assert s.generation == 1

    def test_mismatched_lengths_raise(self):
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        with pytest.raises(ValueError, match="same length"):
            s.tell(samples, [1.0, 2.0])


class TestConvergence:
    def test_mean_norm_decreases_on_sphere_30_gens(self):
        n = 5
        bounds = [(-10.0, 10.0)] * n
        mean_start = [3.0] * n
        initial_norm = math.sqrt(sum(x * x for x in mean_start))

        s = PyCMAESState(mean_start, 1.0, 20, bounds)
        for generation in range(30):
            samples = s.ask(42, generation)
            fitnesses = [neg_sphere(sample) for sample in samples]
            s.tell(samples, fitnesses)

        final_norm = math.sqrt(sum(x * x for x in s.mean))
        assert final_norm < initial_norm

    def test_run_twice_same_engine_identical_trajectory(self):
        n = 4
        bounds = [(-5.0, 5.0)] * n
        mean = [2.0] * n

        def run_trajectory(seed: int, n_gens: int) -> list[float]:
            state = PyCMAESState(mean, 0.5, 10, bounds)
            trajectory = []
            for generation in range(n_gens):
                samples = state.ask(seed, generation)
                trajectory.append(state.mean[0])
                state.tell(samples, [neg_sphere(sample) for sample in samples])
            return trajectory

        assert run_trajectory(42, 10) == run_trajectory(42, 10)


class TestIntegerGeneWorkflow:
    def test_continuous_samples_are_not_all_integers(self):
        s = make_state(n=3, lambda_=20, sigma=0.5)
        samples = s.ask(42, 0)
        assert any(gene != round(gene) for sample in samples for gene in sample)

    def test_tell_with_continuous_samples_succeeds(self):
        s = make_state(n=3, lambda_=10, sigma=0.5)
        samples_continuous = s.ask(42, 0)
        fitnesses = [neg_sphere(sample) for sample in samples_continuous]
        s.tell(samples_continuous, fitnesses)
        assert s.generation == 1

    def test_rounded_samples_are_integers(self):
        s = make_state(n=3, lambda_=10, sigma=0.5)
        samples_continuous = s.ask(42, 0)
        samples_rounded = [[round(gene) for gene in sample] for sample in samples_continuous]
        for sample in samples_rounded:
            for gene in sample:
                assert gene == int(gene)


class TestStateSnapshots:
    def test_to_dict_returns_json_safe_schema_v1_payload(self):
        s = make_state(n=3, lambda_=6, sigma=0.4)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(sample) for sample in samples])

        payload = s.to_dict()

        assert payload["schema_version"] == 1
        assert payload["optimizer_type"] == "cmaes"
        assert set(payload["state"]) == {
            "n",
            "lambda",
            "generation",
            "mean",
            "sigma",
            "cov",
            "pc",
            "ps",
            "bounds",
            "eigendecomp_interval",
            "pending_eigen_updates",
            "eigen_cache",
        }
        assert payload["state"]["n"] == 3
        assert payload["state"]["lambda"] == 6
        assert payload["state"]["generation"] == 1
        assert len(payload["state"]["mean"]) == 3
        assert len(payload["state"]["cov"]) == 3
        assert len(payload["state"]["cov"][0]) == 3
        assert payload["state"]["eigen_cache"]["valid"] in {True, False}
        assert len(payload["state"]["eigen_cache"]["eigenvectors"]) == 3
        assert len(payload["state"]["eigen_cache"]["eigenvalues_sqrt"]) == 3

        import json

        json.dumps(payload, sort_keys=True)

    def test_from_dict_restores_next_ask_after_stale_eigen_cache(self):
        s = make_state(n=4, lambda_=8, sigma=0.5)
        for generation in range(3):
            samples = s.ask(99, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])

        payload = s.to_dict()
        restored = PyCMAESState.from_dict(payload)

        assert restored.generation == s.generation
        assert restored.sigma == pytest.approx(s.sigma)
        assert restored.mean == pytest.approx(s.mean)
        assert restored.ask(123, restored.generation) == s.ask(123, s.generation)

    def test_restored_state_matches_uninterrupted_state_after_same_tell(self):
        s = make_state(n=4, lambda_=8, sigma=0.5)
        for generation in range(3):
            samples = s.ask(99, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])

        restored = PyCMAESState.from_dict(s.to_dict())
        samples = s.ask(123, s.generation)
        assert samples == restored.ask(123, restored.generation)
        fitnesses = [neg_sphere(sample) for sample in samples]

        s.tell(samples, fitnesses)
        restored.tell(samples, fitnesses)

        assert restored.to_dict() == s.to_dict()

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda payload: payload.update({"schema_version": 999}), "schema_version"),
            (lambda payload: payload.update({"optimizer_type": "ga"}), "optimizer_type"),
            (lambda payload: payload["state"].update({"lambda": 1}), "lambda"),
            (lambda payload: payload["state"].update({"sigma": 0.0}), "sigma"),
            (lambda payload: payload["state"].update({"generation": -1}), "invalid|generation"),
            (lambda payload: payload["state"].update({"mean": [0.0]}), "mean"),
            (lambda payload: payload["state"].update({"cov": [[1.0, 0.0]]}), "cov"),
            (lambda payload: payload["state"].update({"bounds": [[1.0, -1.0]] * 3}), "bound"),
            (
                lambda payload: payload["state"]["eigen_cache"].update(
                    {"eigenvalues_sqrt": [1.0]}
                ),
                "eigen_cache",
            ),
        ],
    )
    def test_from_dict_rejects_malformed_snapshots(self, mutate, match):
        payload = make_state(n=3, lambda_=6).to_dict()
        mutate(payload)

        with pytest.raises(ValueError, match=match):
            PyCMAESState.from_dict(payload)

    def test_from_dict_rejects_nonsymmetric_covariance(self):
        payload = make_state(n=3, lambda_=6).to_dict()
        payload["state"]["cov"][0][1] = 0.25

        with pytest.raises(ValueError, match="cov"):
            PyCMAESState.from_dict(payload)

    def test_from_dict_rejects_negative_covariance_eigenvalue(self):
        payload = make_state(n=3, lambda_=6).to_dict()
        payload["state"]["cov"][0][0] = -1.0

        with pytest.raises(ValueError, match="cov"):
            PyCMAESState.from_dict(payload)
