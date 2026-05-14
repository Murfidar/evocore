import pytest

from evocore import GeneDef, GeneSpace
from evocore.exceptions import ConfigurationError
from evocore.stats import (
    EventHistory,
    EventRecord,
    Logbook,
    LogEntry,
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)


def test_logbook_append_len_iter_getitem():
    book = Logbook()
    entry = LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 2, [], {"sharpe": 1.2})
    book.append(entry)
    assert len(book) == 1
    assert list(book) == [entry]
    assert book[0].custom["sharpe"] == 1.2


def test_logbook_fitness_lists():
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {}))
    book.append(LogEntry(1, 2.0, 1.0, 0.2, 15.0, 10, 1, 0, [], {}))
    assert book.best_fitnesses() == [1.0, 2.0]
    assert book.nan_counts() == [0, 1]


def test_to_dataframe_missing_pandas_message(monkeypatch):
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {}))
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="pip install pandas"):
        book.to_dataframe()


def test_log_entry_to_dict_is_json_safe_and_preserves_custom_metrics():
    entry = LogEntry(
        gen=2,
        best_fitness=1.5,
        mean_fitness=1.0,
        std_fitness=0.25,
        wall_time_ms=12.0,
        n_evaluations=8,
        nan_fitness_count=0,
        cached_count=1,
        diversity=[0.1, 0.2],
        custom={"loss": 0.4, "tags": {"b", "a"}},
    )

    assert entry.to_dict() == {
        "gen": 2,
        "best_fitness": 1.5,
        "mean_fitness": 1.0,
        "std_fitness": 0.25,
        "wall_time_ms": 12.0,
        "n_evaluations": 8,
        "nan_fitness_count": 0,
        "cached_count": 1,
        "diversity": [0.1, 0.2],
        "loss": 0.4,
        "tags": ["a", "b"],
    }


def test_logbook_to_dict_and_json_are_stable():
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {"z": 2, "a": 1}))

    assert book.to_dict() == [
        {
            "gen": 0,
            "best_fitness": 1.0,
            "mean_fitness": 0.5,
            "std_fitness": 0.1,
            "wall_time_ms": 12.0,
            "n_evaluations": 10,
            "nan_fitness_count": 0,
            "cached_count": 0,
            "diversity": [],
            "a": 1,
            "z": 2,
        }
    ]
    assert book.to_json() == book.to_json()


def test_event_history_to_rows_preserves_append_order():
    history = EventHistory()
    history.append(
        EventRecord(
            event_index=0,
            event_type="ask",
            batch_id="b-1",
            candidate_id="c-1",
            candidate_hash="hash-1",
            origin="random",
            genes=(1.0, 2),
            params={"x": 1.0, "period": 2},
        )
    )
    history.append(
        EventRecord(
            event_index=1,
            event_type="tell",
            batch_id="b-1",
            candidate_id="c-1",
            candidate_hash="hash-1",
            confidence="trusted_full",
            raw_score=4.0,
            comparison_score=4.0,
            cost=1.0,
            status="trusted",
            origin="random",
            genes=(1.0, 2),
            params={"x": 1.0, "period": 2},
            metrics={"loss": 0.2},
            metadata={"source": "unit"},
        )
    )

    assert len(history) == 2
    assert history[0].event_type == "ask"
    assert [event.event_type for event in history] == ["ask", "tell"]
    assert history.to_rows() == [
        {
            "event_index": 0,
            "event_type": "ask",
            "batch_id": "b-1",
            "candidate_id": "c-1",
            "candidate_hash": "hash-1",
            "generation": None,
            "rung": None,
            "confidence": None,
            "raw_score": None,
            "comparison_score": None,
            "cost": 0.0,
            "status": None,
            "origin": "random",
            "parents": [],
            "genes": [1.0, 2],
            "params": {"period": 2, "x": 1.0},
            "metrics": {},
            "metadata": {},
        },
        {
            "event_index": 1,
            "event_type": "tell",
            "batch_id": "b-1",
            "candidate_id": "c-1",
            "candidate_hash": "hash-1",
            "generation": None,
            "rung": None,
            "confidence": "trusted_full",
            "raw_score": 4.0,
            "comparison_score": 4.0,
            "cost": 1.0,
            "status": "trusted",
            "origin": "random",
            "parents": [],
            "genes": [1.0, 2],
            "params": {"period": 2, "x": 1.0},
            "metrics": {"loss": 0.2},
            "metadata": {"source": "unit"},
        },
    ]


def test_event_history_rejects_non_append_event_index():
    history = EventHistory()

    with pytest.raises(ConfigurationError, match="append-only"):
        history.append(EventRecord(event_index=2, event_type="ask"))


def test_event_history_to_dataframe_missing_pandas_message(monkeypatch):
    history = EventHistory()
    history.append(EventRecord(event_index=0, event_type="generation", generation=0))
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="pip install pandas"):
        history.to_dataframe()


def test_gene_space_signature_preserves_gene_order_and_fields():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0, sigma=0.2),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
        ]
    )

    expected = {
        "schema_version": 1,
        "genes": [
            {
                "name": "x",
                "kind": "float",
                "low": -1.0,
                "high": 1.0,
                "sigma": 0.2,
                "is_fixed": False,
            },
            {
                "name": "period",
                "kind": "int",
                "low": 2,
                "high": 20,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "enabled",
                "kind": "bool",
                "low": None,
                "high": None,
                "sigma": None,
                "is_fixed": False,
            },
        ],
        "has_names": True,
        "length": 3,
    }

    assert gene_space_signature(space) == expected
    assert gene_space_signature(space) == space.signature()


def test_gene_space_hash_is_stable_for_equivalent_spaces():
    left = GeneSpace([GeneDef("x", "float", -1.0, 1.0)])
    right = GeneSpace([GeneDef("x", "float", -1.0, 1.0)])

    assert gene_space_hash(gene_space_signature(left)) == gene_space_hash(
        gene_space_signature(right)
    )


def test_reproducibility_metadata_to_dict_is_json_safe():
    metadata = ReproducibilityMetadata(
        evocore_version="0.7.0",
        engine_type="GAEngine",
        seed=42,
        direction="maximize",
        gene_space_signature={"genes": [{"name": "x", "kind": "float"}]},
        gene_space_hash="abc123",
        optimizer_config={"population_size": 8, "callbacks": {"not", "serialized"}},
    )

    assert metadata.to_dict() == {
        "evocore_version": "0.7.0",
        "engine_type": "GAEngine",
        "seed": 42,
        "direction": "maximize",
        "gene_space_signature": {"genes": [{"kind": "float", "name": "x"}]},
        "gene_space_hash": "abc123",
        "optimizer_config": {"callbacks": ["not", "serialized"], "population_size": 8},
        "extension": {},
    }
