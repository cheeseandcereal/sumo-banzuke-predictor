"""Shared fixtures: the committed processed data, loaded once per session.

The project is not installed into the venv, so the repo root is put on
sys.path here (pytest only inserts the tests/ directory itself).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def tidy() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "tidy.parquet")


@pytest.fixture(scope="session")
def trans() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "transitions.parquet")


@pytest.fixture(scope="session")
def bouts() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "bouts.parquet")


@pytest.fixture(scope="session")
def bashos(tidy) -> list[int]:
    """All basho ids in the data, ascending."""
    return [int(b) for b in sorted(tidy["basho"].unique())]


@pytest.fixture(scope="session")
def make_cands(trans, bashos):
    """Backtest-style candidate rows for a target basho: transitions at the
    previous basho, excluding rikishi absent from the target banzuke."""
    def _make(target: int) -> pd.DataFrame:
        prev = bashos[bashos.index(target) - 1]
        return trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
    return _make


@pytest.fixture(scope="session")
def resolve_true_order(make_cands, tidy):
    """Feed the resolver the *actual* next order (score = position_next) for a
    target and return (cands, actual tidy rows, resolver output), memoized."""
    from banzuke.resolver import resolve

    cache: dict[int, tuple] = {}

    def _run(target: int):
        if target not in cache:
            cands = make_cands(target)
            actual = tidy[tidy["basho"] == target]
            mak_size = int((actual["division"] == 0).sum())
            scores = cands["position_next"].to_numpy(dtype=float)
            assert not pd.isna(scores).any(), "candidates all appear on the next banzuke"
            cache[target] = (cands, actual, resolve(cands, scores, mak_size))
        return cache[target]
    return _run


def slot_agreement(pred: pd.DataFrame, actual: pd.DataFrame) -> tuple[int, int]:
    """(exact makuuchi slots, makuuchi slots) of a resolver output against the
    actual banzuke; rikishi missing from pred count as misses."""
    mak = actual[actual["division"] == 0].merge(pred, on="rikishi_id", how="left")
    exact = ((mak["pred_class"] == mak["rank_class"])
             & (mak["pred_number"] == mak["rank_number"])
             & (mak["pred_side"] == mak["side"]))
    return int(exact.sum()), len(mak)


@pytest.fixture(scope="session")
def small_train(trans) -> pd.DataFrame:
    """A small labeled training set: 1995-2003 transitions, targets < 200401."""
    return trans[(trans["next_basho"] < 200401) & (trans["basho"] >= 199501)
                 & trans["position_next"].notna()]


@pytest.fixture(scope="session")
def cands_200401(make_cands) -> pd.DataFrame:
    return make_cands(200401)
