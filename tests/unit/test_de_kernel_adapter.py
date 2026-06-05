from __future__ import annotations

import pytest

from evocore import EvaluationRecord, _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.optimizers.de.kernel import DERustKernelAdapter
from evocore.search_space import Gene, GeneSpace


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _candidate(candidate_id: str, genes) -> Candidate:
    candidate = Candidate(candidate_id=candidate_id, genes=list(genes), batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id=candidate_id,
            batch_id="b-1",
            score=float(candidate_id.rsplit("-", 1)[-1]),
            confidence="trusted_full",
            stage="full",
        )
    )
    return candidate


def test_adapter_passes_encoded_population_and_kernel_arguments(monkeypatch) -> None:
    calls = []

    def fake_de_generate_trials(
        population,
        scores,
        gene_bounds,
        gene_kinds,
        strategy,
        mutation_factor,
        crossover_rate,
        seed,
        generation,
        target_slots,
        direction,
        jde_state=None,
    ):
        calls.append(
            {
                "population": population,
                "scores": scores,
                "gene_bounds": gene_bounds,
                "gene_kinds": gene_kinds,
                "strategy": strategy,
                "mutation_factor": mutation_factor,
                "crossover_rate": crossover_rate,
                "seed": seed,
                "generation": generation,
                "target_slots": target_slots,
                "direction": direction,
                "jde_state": jde_state,
            }
        )
        return [
            {
                "target_slot": 0,
                "genes": [99.0, 20.8, 0.2, -9.0],
                "metadata": {
                    "strategy": strategy,
                    "target_slot": 0,
                    "base_slot": 1,
                    "donor_slots": [1, 2, 3],
                    "difference_pairs": [[2, 3]],
                },
            }
        ]

    monkeypatch.setattr(_core, "de_generate_trials", fake_de_generate_trials)

    proposals = DERustKernelAdapter().generate_trials(
        target_population=[
            _candidate("candidate-0", [0.25, 7, True, 1.5]),
            _candidate("candidate-1", [0.5, 8, False, 1.5]),
            _candidate("candidate-2", [0.75, 9, True, 1.5]),
            _candidate("candidate-3", [1.0, 10, False, 1.5]),
        ],
        scores=[0.0, 1.0, 2.0, 3.0],
        gene_space=_space(),
        strategy="rand1bin",
        mutation_factor=0.7,
        crossover_rate=0.9,
        seed=42,
        generation=3,
        target_slots=[0],
        direction="maximize",
        jde_state={"f_by_slot": [0.5] * 4, "cr_by_slot": [0.9] * 4},
    )

    assert calls == [
        {
            "population": [
                [0.25, 7.0, 1.0, 1.5],
                [0.5, 8.0, 0.0, 1.5],
                [0.75, 9.0, 1.0, 1.5],
                [1.0, 10.0, 0.0, 1.5],
            ],
            "scores": [0.0, 1.0, 2.0, 3.0],
            "gene_bounds": [(-5.0, 5.0), (2.0, 20.0), (0.0, 1.0), (1.5, 1.5)],
            "gene_kinds": ["float", "int", "bool", "float"],
            "strategy": "rand1bin",
            "mutation_factor": 0.7,
            "crossover_rate": 0.9,
            "seed": 42,
            "generation": 3,
            "target_slots": [0],
            "direction": "maximize",
            "jde_state": {"f_by_slot": [0.5] * 4, "cr_by_slot": [0.9] * 4},
        }
    ]
    assert len(proposals) == 1
    assert proposals[0].genes == [5.0, 20, False, pytest.approx(1.5)]
    assert proposals[0].metadata["strategy"] == "rand1bin"


def test_adapter_rejects_score_population_length_mismatch() -> None:
    with pytest.raises(ConfigurationError, match="scores length"):
        DERustKernelAdapter().generate_trials(
            target_population=[_candidate("candidate-0", [0.25, 7, True, 1.5])],
            scores=[],
            gene_space=_space(),
            strategy="rand1bin",
            mutation_factor=0.7,
            crossover_rate=0.9,
            seed=42,
            generation=3,
            target_slots=[0],
            direction="maximize",
            jde_state=None,
        )


def test_adapter_rejects_malformed_rust_payload(monkeypatch) -> None:
    def fake_de_generate_trials(*_args, **_kwargs):
        return [{"genes": [0.0]}]

    monkeypatch.setattr(_core, "de_generate_trials", fake_de_generate_trials)

    with pytest.raises(ConfigurationError, match="metadata"):
        DERustKernelAdapter().generate_trials(
            target_population=[
                _candidate("candidate-0", [0.25, 7, True, 1.5]),
                _candidate("candidate-1", [0.5, 8, False, 1.5]),
                _candidate("candidate-2", [0.75, 9, True, 1.5]),
                _candidate("candidate-3", [1.0, 10, False, 1.5]),
            ],
            scores=[0.0, 1.0, 2.0, 3.0],
            gene_space=_space(),
            strategy="rand1bin",
            mutation_factor=0.7,
            crossover_rate=0.9,
            seed=42,
            generation=3,
            target_slots=[0],
            direction="maximize",
            jde_state=None,
        )
