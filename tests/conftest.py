"""Shared fixtures: the committed datasets (session-scoped, read once), the
backtest candidate frame for a target, and a backtest-results row factory."""
import pytest

from banzuke.harness import METRICS, split
from banzuke.paths import load_bouts, load_tidy, load_transitions

SMALL = {"n_seeds": 1, "base": {"n_estimators": 20}}  # the GBMs default to a 5-seed bag
FAST_RERANK = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}


@pytest.fixture(scope="session")
def tidy():
    return load_tidy()


@pytest.fixture(scope="session")
def trans():
    return load_transitions()


@pytest.fixture(scope="session")
def bouts():
    return load_bouts()


@pytest.fixture(scope="session")
def small_train(trans):
    return trans[(trans["next_basho"] < 200401) & (trans["basho"] >= 199501)
                 & trans["position_next"].notna()]


@pytest.fixture(scope="session")
def cands_for(trans, tidy):
    """Backtest candidates for a target: previous-basho rows still on the sheet."""
    return lambda target: split(trans, tidy, target)[2]


def fake_results(cells, **columns):
    """Backtest result rows for a (model, basho) -> exact_n mapping, every
    other metric 1.0; keyword columns override per row via callables or values."""
    rows = []
    for (model, basho), exact_n in cells.items():
        row = {"config": "base", "model": model, "seed": 0, "basho": basho,
               **dict.fromkeys(METRICS, 1.0), "exact": exact_n / 42, "exact_n": exact_n}
        row.update({k: v(row) if callable(v) else v for k, v in columns.items()})
        rows.append(row)
    return rows
