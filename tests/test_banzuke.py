"""Invariant suite on the committed processed data: chronological features,
resolver conventions, model seed/option semantics, bagging, reranking, OOF
scores, backtest determinism, summarize() statistics, predict() assembly."""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent  # the package is not installed
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predict  # noqa: E402
from banzuke.build import KOMUSUBI, MAEGASHIRA, OZEKI, SEKIWAKE  # noqa: E402
from banzuke.features import FEATURES, build_transitions  # noqa: E402
from banzuke.harness import METRICS, parse_seeds, parse_sets, run_backtest, summarize  # noqa: E402
from banzuke.metrics import evaluate  # noqa: E402
from banzuke.models import (MODELS, GBMMedian, GBMRanker, GBMRerank, RulesBaseline,  # noqa: E402
                            _gap_pairs, _seeds, _window_pairs, oof_base_scores)
from banzuke.overrides import parse  # noqa: E402
from banzuke.resolver import block_slots, resolve  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SMALL = {"n_seeds": 1, "base": {"n_estimators": 20}}  # the GBMs default to a 5-seed bag
CELL = ["pred_class", "pred_number", "pred_side"]


@pytest.fixture(scope="session")
def tidy():
    return pd.read_parquet(PROCESSED / "tidy.parquet")


@pytest.fixture(scope="session")
def trans():
    return pd.read_parquet(PROCESSED / "transitions.parquet")


@pytest.fixture(scope="session")
def bouts():
    return pd.read_parquet(PROCESSED / "bouts.parquet")


@pytest.fixture(scope="session")
def small_train(trans):
    return trans[(trans["next_basho"] < 200401) & (trans["basho"] >= 199501)
                 & trans["position_next"].notna()]


@pytest.fixture(scope="session")
def cands_for(trans, tidy):
    """Backtest candidates for a target: previous-basho rows still on the sheet."""
    bashos = sorted(tidy["basho"].unique())

    def _cands(target):
        prev = bashos[bashos.index(target) - 1]
        return trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
    return _cands


def _same(x, y):  # NaN == NaN, dtype-insensitive
    return np.array_equal(x.to_numpy(dtype=float), y.to_numpy(dtype=float), equal_nan=True)


def _true_order(cands_for, target, sign=1.0):
    cands = cands_for(target)
    return cands, resolve(cands, sign * cands["position_next"].to_numpy(dtype=float), 42)


def test_features_are_chronological(tidy, bouts, trans):
    """Features rebuilt from history truncated at `cut` equal the committed values
    from the full history (no look-ahead); targets agree wherever the truncated
    data knows the next banzuke, i.e. strictly before the cut."""
    cut, key = 201911, ["basho", "rikishi_id"]
    new = build_transitions(tidy[tidy["basho"] <= cut], bouts[bouts["basho"] <= cut])
    new = new.set_index(key).sort_index()
    old = trans[trans["basho"] <= cut].set_index(key).sort_index()
    assert old.index.equals(new.index)
    for col in FEATURES:
        assert _same(old[col], new[col]), col
    before = old.index.get_level_values("basho") < cut
    for col in ("position_next", "delta"):
        assert _same(old[col][before], new[col][before]), col
    # Asanoyama returned at 202301 after five basho below juryo (last seen 202201): no lags
    a = trans[trans["shikona"] == "Asanoyama"].set_index("basho")
    assert a.index[a.index < 202301].max() == 202201
    assert np.isnan(a.loc[202301, "w1"]) and np.isnan(a.loc[202301, "pos1"])


def test_block_slots_layout_conventions():
    # maegashira: strict E,W alternation, numbers per side, column balance ignored
    assert block_slots(MAEGASHIRA, 5, 0, 10) == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    for c in (OZEKI, SEKIWAKE, KOMUSUBI):
        # odd block: last slot to the lighter column (East when balanced); E,W,W -> 1E,1W,2W
        assert block_slots(c, 3, 1, 0) == [(1, 0), (1, 1), (2, 1)]
        assert block_slots(c, 3, 0, 1) == [(1, 0), (1, 1), (2, 0)]
        assert block_slots(c, 1, 1, 0) == [(1, 1)] and block_slots(c, 1, 0, 0) == [(1, 0)]
        # even block: alternation regardless of balance
        assert block_slots(c, 2, 5, 0) == [(1, 0), (1, 1)]
        assert block_slots(c, 4, 0, 5) == [(1, 0), (1, 1), (2, 0), (2, 1)]


@pytest.mark.parametrize("target", [200401, 202309, 202401, 202609])
def test_resolver_reproduces_banzuke_from_true_order(cands_for, tidy, target):
    cands, pred = _true_order(cands_for, target)
    actual = tidy[tidy["basho"] == target]
    mak = actual[actual["division"] == 0].merge(pred, on="rikishi_id", how="left")
    assert len(mak) == 42
    assert (mak[CELL].to_numpy() == mak[["rank_class", "rank_number", "side"]].to_numpy()).all()
    assert evaluate(pred, cands, actual)["exact"] == 1.0


def test_resolver_make_koshi_ceiling_and_block_order(cands_for):
    # E15a: cells inside an S/K block follow score order, not claim status. 200607:
    # Asasekiryu (M2E 10-5, a fill) ranks above Kisenosato (M1E 8-7, forced claim)
    cands, pred = _true_order(cands_for, 200607)
    cells = pred.merge(cands[["rikishi_id", "shikona"]], on="rikishi_id").set_index("shikona")
    assert cells.loc["Asasekiryu", CELL].tolist() == [KOMUSUBI, 1, 0]
    assert cells.loc["Kisenosato", CELL].tolist() == [KOMUSUBI, 1, 1]
    # E10: a make-koshi S/K/M never lands above his prior cell, whatever order is fed
    # in (the reversed order asks for exactly such promotions)
    for target, sign in product((200607, 202309), (1.0, -1.0)):
        cands, pred = _true_order(cands_for, target, sign)
        m = cands.merge(pred, on="rikishi_id")
        mk = m[m["rank_class"].isin([SEKIWAKE, KOMUSUBI, MAEGASHIRA]) & (m["kk"] == 0)]
        prior = zip(mk["rank_class"], mk["rank_number"], mk["side"])
        assert len(mk) > 10 and all(q >= p for p, q in zip(prior, zip(*[mk[c] for c in CELL])))


def test_seed_and_option_semantics():
    assert _seeds(0, 1) == [0] and _seeds(1, 5) == [5, 6, 7, 8, 9] and _seeds(None, 1) == [None]
    with pytest.raises(ValueError):
        _seeds(None, 2)  # library defaults cannot be bagged
    with pytest.raises(ValueError):
        _seeds(0, 0)
    assert GBMRerank(seed=0).seeds == [0, 1, 2, 3, 4] and RulesBaseline().seeds == [0]
    assert GBMRanker().truncation == 60
    with pytest.raises(TypeError, match="unknown options"):
        GBMRerank(foo=1)
    for cls in MODELS.values():  # a subclass keeps every option an ancestor's fit() reads
        for parent in cls.__mro__[1:]:
            assert set(getattr(parent, "OPTIONS", {})) <= set(cls.OPTIONS), (cls.name, parent)


def test_bag_is_mean_of_members(small_train, cands_for):
    cands = cands_for(200401)
    bag = GBMMedian(seed=0, n_seeds=2, base={"n_estimators": 20})
    bag.fit(small_train)
    assert [m.get_params()["random_state"] for m in bag.ms] == [0, 1]
    s0, s1 = bag.score(cands, 0), bag.score(cands, 1)
    assert not np.array_equal(s0, s1) and bag.score(cands) == pytest.approx((s0 + s1) / 2)


def test_rerank_is_deterministic_permutation(small_train, cands_for):
    cands = cands_for(200401)
    kw = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}
    scores = []
    for _ in range(2):
        m = GBMRerank(**kw)
        m.fit(small_train)
        scores.append(m.score(cands))
    assert sorted(scores[0].tolist()) == list(range(len(cands)))
    assert np.array_equal(scores[0], scores[1])
    with pytest.raises(ValueError, match="OOF|prepare"):  # default near_ties needs prepare()
        GBMRerank(seed=0, n_seeds=1, base={"n_estimators": 1}).fit(small_train)


def test_pair_helpers():
    i, j = _window_pairs(5, 2)
    assert list(zip(i.tolist(), j.tolist())) == [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    scores = np.array([0.0, 0.3, 1.0, 1.2, 5.0, np.nan])
    pos = np.array([4, 3, 2, 1, 0, 5])  # current order is the reverse of the scores
    i, j = _gap_pairs(scores, pos, 0.5)
    assert set(zip(i.tolist(), j.tolist())) == {(1, 0), (3, 2)}
    i, j = _gap_pairs(scores, pos, 1.5)
    pairs = set(zip(i.tolist(), j.tolist()))
    assert pairs == {(b, a) for a in range(5) for b in range(a + 1, 5)
                     if scores[b] - scores[a] <= 1.5}
    assert all(pos[a] <= pos[b] for a, b in pairs) and 5 not in {*i, *j}  # NaN pairs with nothing


def test_oof_scores_are_leak_free_and_cached(trans, tmp_path, monkeypatch):
    lab = trans[trans["position_next"].notna()]
    labeled = sorted(lab["basho"].unique())
    oof = oof_base_scores(trans, GBMMedian, SMALL, min_history=396, workers=1, cache_dir=tmp_path)
    assert sorted(oof["basho"].unique()) == labeled[396:]
    assert list(oof.columns) == ["basho", "rikishi_id", "oof"]
    b = labeled[-1]  # source basho b is scored by the base stage fitted on labels known at b
    m = GBMMedian(seed=0, **SMALL)
    m.fit(lab[lab["next_basho"] <= b])
    rows = lab[lab["basho"] == b]
    got = oof[oof["basho"] == b].set_index("rikishi_id")["oof"].reindex(rows["rikishi_id"])
    assert got.to_numpy() == pytest.approx(m.score(rows))
    assert len(list(tmp_path.glob("*.parquet"))) == 1
    again = oof_base_scores(trans, GBMMedian, SMALL, min_history=396, workers=1, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(again, oof)
    # prepare() fills `oof` only when near_ties is on
    import banzuke.models as models
    stub, calls = pd.DataFrame({"basho": [200311], "rikishi_id": [1], "oof": [1.0]}), []
    monkeypatch.setattr(models, "oof_base_scores", lambda *a, **k: calls.append(a) or stub)
    assert GBMRerank.prepare(SMALL, trans)["oof"] is stub and len(calls) == 1
    off = {**SMALL, "near_ties": False}
    assert GBMRerank.prepare(off, trans) == off and len(calls) == 1


def test_backtest_deterministic_and_parallel(trans, tidy):
    targets = [202309, 202311]
    r = run_backtest(["Aq"], targets, trans, tidy, progress=False, model_kwargs=SMALL)
    assert len(r) == 2 and r["basho"].tolist() == targets
    assert {"config", "model", "seed", "basho", *METRICS} <= set(r.columns)
    assert (r["model"] == "Aq").all() and (r["seed"] == 0).all() and (r["config"] == "base").all()
    assert r["exact_n"].between(1, 42).all()
    again = run_backtest(["Aq"], targets, trans, tidy, progress=False, model_kwargs=SMALL)
    pd.testing.assert_frame_equal(again, r)
    par = run_backtest(["Aq"], targets, trans, tidy, progress=False, model_kwargs=SMALL, workers=2)
    pd.testing.assert_frame_equal(par, r)


def test_backtest_runs_every_model_together(trans, tidy):
    # each model takes only the options it declares; near_ties off: no OOF pass
    kw = {**SMALL, "pair": {"n_estimators": 10}, "near_ties": False}
    names = ["R", "L", "A", "Aq", "Aw", "B", "C", "Ar"]
    r = run_backtest(names, [202609], trans, tidy, progress=False, model_kwargs=kw)
    assert sorted(r["model"]) == sorted(names) and (r["basho"] == 202609).all()


def test_summarize_pairs_against_baseline_and_averages_seeds():
    # config x: +2 exact slots on seed 1, +0 on seed 0 (+1 seed-averaged), mae -0.05
    rows = [{"config": cfg, "model": "Aq", "seed": seed, "basho": 200401 + b,
             **dict.fromkeys(METRICS, 1.0), "exact": e / 42, "exact_n": e,
             "mae": 2.0 - 0.05 * (cfg == "x")}
            for b in range(20) for cfg in ("base", "x") for seed in (0, 1)
            for e in [15 + b % 5 + 2 * (cfg == "x" and seed == 1)]]
    s = summarize(pd.DataFrame(rows), baseline="Aq")
    assert s.attrs["baseline"] == "Aq" and list(s.index) == ["Aq:x", "Aq"] and (s["n"] == 20).all()
    x = s.loc["Aq:x"]
    assert x["d_exact"] == pytest.approx(1.0) and x["wl"] == "20-0" and x["p_exact"] < 0.01
    lo, hi = map(float, x["ci_exact"].split(","))
    assert 0 < lo <= hi and x["d_mae"] == pytest.approx(-0.05)
    base = s.loc["Aq"]
    assert base["d_exact"] == 0 and base["ci_exact"] == "" and np.isnan(base["p_exact"])
    assert parse_sets(["base.n_estimators=200", "gap=0.25", "context=false"]) == {
        "base": {"n_estimators": 200}, "gap": 0.25, "context": False}
    assert parse_seeds("0-2") == (0, 1, 2) and parse_seeds("0,3") == (0, 3)


def test_predict_spread(trans):
    cands = trans[(trans["basho"] == 202609) & ~trans["dropped"]].reset_index(drop=True)
    ov = parse(cands)
    assert ov == {"relative": [], "class": {}, "count": {}, "pins": {}}
    point = cands["position"].to_numpy(dtype=float)
    pred, warnings = predict.predict(cands, point, [point, point + 0.1], ov, 42)
    assert warnings == [] and sorted(pred["pred_pos"]) == list(range(len(cands)))
    assert (pred["spread"] == 0).all()  # a constant shift keeps the order
    pred, _ = predict.predict(cands, point, [point, point[::-1]], ov, 42)
    assert (pred["spread"] > 0).any()
