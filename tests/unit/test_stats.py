import pytest

from evocore.stats import Logbook, LogEntry


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
