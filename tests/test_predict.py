"""predict.py: the training cutoff logic (main() line `train = trans[...]`)
and the predict() assembly of point forecast, per-seed spread and overrides."""
import numpy as np
import pandas as pd
import pytest

import predict
from banzuke.build import JURYO
from banzuke.overrides import parse
from banzuke.resolver import resolve

LATEST = 202609


def _train_cutoff(trans, latest):
    # verbatim filter from predict.main()
    return trans[trans["position_next"].notna() & (trans["next_basho"] <= latest)]


def test_latest_is_last_played_basho(trans):
    latest = int(trans.loc[trans["yusho"].eq(1), "basho"].max())
    assert latest == LATEST == int(trans["basho"].max())
    assert predict.next_basho_id(latest) == 202611


def test_training_cutoff_excludes_labels_after_latest(trans):
    train = _train_cutoff(trans, LATEST)
    assert not (train["next_basho"] > LATEST).any()
    assert train["position_next"].notna().all()
    assert int(train["next_basho"].max()) == LATEST
    # the candidate rows themselves (basho == latest) carry no labels yet
    assert trans.loc[trans["basho"] == LATEST, "position_next"].isna().all()


def test_training_cutoff_drops_fetched_but_unplayed_next_banzuke(trans):
    # simulate having fetched the 202611 banzuke before it is played: the
    # 202609 -> 202611 transition gains a label that must not train the model
    fake = trans[trans["basho"] == LATEST].head(1).copy()
    fake["next_basho"], fake["position_next"], fake["delta"] = 202611.0, 5.0, 5.0 - fake["position"]
    fake["rikishi_id"] = -1
    with_future = pd.concat([trans, fake], ignore_index=True)
    naive = with_future[with_future["position_next"].notna()]
    assert (naive["rikishi_id"] == -1).any()
    assert int(naive["next_basho"].max()) == 202611
    cut = _train_cutoff(with_future, LATEST)
    assert not (cut["rikishi_id"] == -1).any()
    assert len(cut) == len(naive) - 1
    pd.testing.assert_frame_equal(cut, _train_cutoff(trans, LATEST))


def test_next_basho_id_and_label():
    assert predict.next_basho_id(202601) == 202603
    assert predict.next_basho_id(202611) == 202701
    assert [predict.next_basho_id(202600 + m) for m in predict.BASHO_MONTHS] == [
        202603, 202605, 202607, 202609, 202611, 202701]
    assert predict.label(4, 2, 1) == "M2W"
    assert predict.label(0, 1, 0) == "Y1E"


@pytest.fixture(scope="module")
def latest_cands(trans):
    return trans[trans["basho"] == LATEST].reset_index(drop=True)


def test_predict_point_and_spread(latest_cands):
    cands = latest_cands
    ov = parse(cands)
    assert ov == {"relative": [], "class": {}, "count": {}, "pins": {}}
    point = cands["position"].to_numpy().astype(float)
    mak_size = int(cands["mak_size"].iloc[0])
    pred, warnings = predict.predict(cands, point, [point, point + 0.1], ov, mak_size)
    assert warnings == []
    assert isinstance(pred, pd.DataFrame) and len(pred) == len(cands)
    assert "spread" in pred.columns
    assert (pred["spread"] == 0).all()  # a constant shift keeps the order
    assert sorted(pred["pred_pos"]) == list(range(len(cands)))
    assert int((pred["pred_class"] < JURYO).sum()) == mak_size
    # the point prediction is the resolver applied to that order
    base = resolve(cands, point, mak_size)
    merged = pred.merge(base, on="rikishi_id", suffixes=("", "_r"))
    for c in ("pred_class", "pred_number", "pred_side", "pred_pos"):
        assert (merged[c] == merged[c + "_r"]).all()
    # candidate columns are carried through
    assert {"shikona", "rank_class", "wins", "position"} <= set(pred.columns)


def test_predict_spread_reflects_seed_disagreement(latest_cands):
    cands = latest_cands
    ov = parse(cands)
    point = cands["position"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    noisy = point + rng.normal(0, 3, len(point))
    pred, _ = predict.predict(cands, point, [point, noisy], ov, 42)
    assert (pred["spread"] >= 0).all() and (pred["spread"] > 0).any()
    a = resolve(cands, point, 42).set_index("rikishi_id")["pred_pos"]
    b = resolve(cands, noisy, 42).set_index("rikishi_id")["pred_pos"]
    expect = (a - b).abs()
    got = pred.set_index("rikishi_id")["spread"].reindex(expect.index)
    assert (got == expect).all()
    # no per-seed orders: spread degenerates to zero
    pred0, _ = predict.predict(cands, point, [], ov, 42)
    assert (pred0["spread"] == 0).all()


def test_predict_applies_relative_override(latest_cands):
    cands = latest_cands
    point = cands["position"].to_numpy().astype(float)
    # lift the lowest kachi-koshi mid-maegashira to immediately above the
    # highest one (both unbounded by the make-koshi ceiling and below the
    # komusubi-claim zone, so the structural rules leave the splice intact)
    mak = cands[(cands["rank_class"] == 4) & (cands["kk"] == 1)
                & (cands["rank_number"] >= 6)].sort_values("position")
    upper, lower = mak.iloc[0], mak.iloc[-1]
    ov = parse(cands, above=[f"{lower['shikona']} > {upper['shikona']}"])
    assert len(ov["relative"]) == 1
    pred, warnings = predict.predict(cands, point, [], ov, 42)
    pos = pred.set_index("rikishi_id")["pred_pos"]
    assert pos[lower["rikishi_id"]] < pos[upper["rikishi_id"]]
    assert pos[lower["rikishi_id"]] < lower["position"]  # actually moved up
    assert sorted(pred["pred_pos"]) == list(range(len(cands)))
    # without the override the prior order stands
    plain, _ = predict.predict(cands, point, [], parse(cands), 42)
    ppos = plain.set_index("rikishi_id")["pred_pos"]
    assert ppos[lower["rikishi_id"]] > ppos[upper["rikishi_id"]]
