"""banzuke/features.py: chronological feature construction (no look-ahead)
and lag semantics, checked against the committed transitions.parquet."""
import numpy as np
import pandas as pd
import pytest

from banzuke.features import FEATURES, build_transitions

CUTOFF = 201911
TARGET_COLS = ["next_basho", "position_next", "division_next", "class_next",
               "number_next", "side_next", "dropped", "delta"]


def _aligned(old: pd.DataFrame, new: pd.DataFrame):
    key = ["basho", "rikishi_id"]
    a = old.set_index(key).sort_index()
    b = new.set_index(key).sort_index()
    assert a.index.equals(b.index)
    return a, b


def _same_values(x: pd.Series, y: pd.Series) -> bool:
    """Equality allowing dtype differences and NaN == NaN."""
    xa, ya = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    return np.array_equal(xa, ya, equal_nan=True)


@pytest.fixture(scope="module")
def prefix_rebuild(tidy, bouts, trans):
    tidy_p = tidy[tidy["basho"] <= CUTOFF]
    bouts_p = bouts[bouts["basho"] <= CUTOFF]
    new = build_transitions(tidy_p, bouts_p)
    old = trans[trans["basho"] <= CUTOFF]
    return _aligned(old, new)


def test_prefix_rebuild_has_same_rows_and_columns(prefix_rebuild, trans):
    old, new = prefix_rebuild
    assert len(new) == int((trans["basho"] <= CUTOFF).sum())
    assert set(new.columns) == set(old.columns)
    assert set(FEATURES) <= set(new.columns)
    assert len(FEATURES) == 35


@pytest.mark.parametrize("col", FEATURES)
def test_feature_prefix_reproduction_no_lookahead(prefix_rebuild, col):
    """Every feature at basho b computed from data truncated at CUTOFF equals
    the committed value computed from the full history: features use only
    information available at b."""
    old, new = prefix_rebuild
    assert _same_values(old[col], new[col]), col


def test_target_columns_reproduce_before_cutoff(prefix_rebuild):
    # the targets (next-basho labels) also agree wherever the prefix knows
    # the next banzuke; the last prefix basho has no labels by construction
    old, new = prefix_rebuild
    before = old.index.get_level_values("basho") < CUTOFF
    for col in TARGET_COLS:
        assert _same_values(old[col][before], new[col][before]), col
    last = new.loc[CUTOFF]
    assert last["next_basho"].isna().all()
    assert last["position_next"].isna().all()
    assert not last["dropped"].any()


def test_lag_features_nan_when_absent_previous_basho_real_example(trans, bashos):
    # Asanoyama sat out his suspension on the banzuke down to J4 at 202201,
    # fell to makushita (no makuuchi/juryo row 202203-202211) and returned
    # at J12 for 202301: his lag features there are NaN
    a = trans[trans["shikona"] == "Asanoyama"].sort_values("basho")
    row = a[a["basho"] == 202301].iloc[0]
    prev_rows = a[a["basho"] < 202301]
    assert int(prev_rows["basho"].max()) == 202201
    assert bashos[bashos.index(202301) - 1] == 202211  # gap of five basho
    assert row["division"] == 1 and row["wins"] == 14
    for col in ("w1", "pos1", "w2", "yusho1", "junyusho1", "roll3", "traj3", "ozeki_run3"):
        assert np.isnan(row[col]), col
    # the row right after a contiguous appearance carries the lag
    contig = a[a["basho"] == 202303].iloc[0]
    assert contig["w1"] == row["wins"] == 14
    assert contig["pos1"] == row["position"] == 65
    assert np.isnan(contig["w2"])  # two back (202211) is still absent


def test_lag_features_nan_iff_previous_basho_absent(trans, bashos):
    bidx = {b: i for i, b in enumerate(bashos)}
    d = trans.sort_values(["rikishi_id", "basho"], kind="stable")
    step = d.groupby("rikishi_id")["basho"].transform(lambda s: s.map(bidx).diff())
    absent = step != 1  # first appearance or gap
    assert absent.sum() > 500  # plenty of real gaps
    for col in ("w1", "pos1", "yusho1", "junyusho1"):
        assert d.loc[absent, col].isna().all(), col
        assert d.loc[~absent, col].notna().all(), col
    # and the lag is the previous row's value when contiguous
    prev_w = d.groupby("rikishi_id")["wins"].shift(1)
    prev_p = d.groupby("rikishi_id")["position"].shift(1)
    assert (d.loc[~absent, "w1"] == prev_w[~absent]).all()
    assert (d.loc[~absent, "pos1"] == prev_p[~absent]).all()


def test_simple_feature_invariants(trans):
    assert (trans["kk"] == (trans["wins"] >= 8).astype(int)).all()
    assert (trans["win8"] == trans["wins"] - 8).all()
    assert (trans["boundary_dist"] == trans["position"] - trans["mak_size"]).all()
    assert (trans["year"] == trans["basho"] // 100).all()
    assert (trans["kosho"] == trans["basho"].between(197201, 200311).astype(int)).all()
    ok = trans["roll3"].notna()
    assert (trans.loc[ok, "roll3"] == trans.loc[ok, "wins"] + trans.loc[ok, "w1"]
            + trans.loc[ok, "w2"]).all()
    # n_basho counts appearances per rikishi in chronological order
    d = trans.sort_values(["rikishi_id", "basho"], kind="stable")
    assert (d["n_basho"] == d.groupby("rikishi_id").cumcount() + 1).all()


def test_targets_are_next_calendar_basho(trans, bashos):
    nxt = dict(zip(bashos, bashos[1:]))
    has = trans["next_basho"].notna()
    assert (trans.loc[has, "next_basho"] == trans.loc[has, "basho"].map(nxt)).all()
    assert trans.loc[~has, "basho"].eq(bashos[-1]).all()
    # dropped = has a next basho but no row on it
    assert (trans["dropped"] == (has & trans["position_next"].isna())).all()
    ok = trans["position_next"].notna()
    assert (trans.loc[ok, "delta"] == trans.loc[ok, "position_next"] - trans.loc[ok, "position"]).all()
