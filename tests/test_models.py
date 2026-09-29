"""banzuke/models.py: seed semantics, option validation, the pairwise
helpers, small fits of the GBM models, and the rolling OOF cache."""
import numpy as np
import pandas as pd
import pytest

from banzuke.features import FEATURES
from banzuke.models import (
    BASE_PARAMS, CONTEXT_ENDS, CONTEXT_MEAN, CONTEXT_SHARED, MODELS, PAIR_PARAMS,
    GBMMedian, GBMRanker, GBMRerank, LinearModel, RulesBaseline, _gap_pairs,
    _pair_matrix, _PairStage, _seeded, _seeds, _window_pairs, oof_base_scores,
)

SMALL_BASE = {"n_estimators": 30}
SMALL_PAIR = {"n_estimators": 20}
# the GBM classes default to a 5-seed bag; fits below use one seed unless the
# test is about bagging, and Ar's default pairs='mixed' needs an OOF table, so
# reranker fits select 'window' pairs unless the test is about the mixed path
ONE = {"n_seeds": 1}


# --- seeds and options --------------------------------------------------------

def test_seeds_semantics():
    assert _seeds(0, 1) == [0]
    assert _seeds(1, 5) == [5, 6, 7, 8, 9]
    assert _seeds(0, 5) == [0, 1, 2, 3, 4]  # replicate bags are disjoint
    assert _seeds(None, 1) == [None]
    with pytest.raises(ValueError):
        _seeds(None, 2)


def test_seeded_params():
    p = {"n_estimators": 3, "random_state": 7}
    assert _seeded(p, 5)["random_state"] == 5
    assert "random_state" not in _seeded(p, None)
    assert p == {"n_estimators": 3, "random_state": 7}  # input untouched


def test_unknown_option_raises_type_error():
    with pytest.raises(TypeError, match="unknown options"):
        GBMRerank(foo=1)
    with pytest.raises(TypeError):
        GBMMedian(pair_window=3)  # valid for Ar only


def test_constructor_merges_params_without_mutating_module_state():
    m = GBMRerank(seed=2, n_seeds=3, base={"n_estimators": 5}, pair={"n_estimators": 7}, gap=0.25)
    assert m.seeds == [6, 7, 8]
    assert m.base_params["n_estimators"] == 5
    assert m.pair_params["n_estimators"] == 7
    assert m.base_params["learning_rate"] == BASE_PARAMS["learning_rate"]
    # override vs default
    assert m.gap == 0.25 and m.cluster_max == GBMRerank.OPTIONS["cluster_max"] == 6
    assert BASE_PARAMS["n_estimators"] == 300 and PAIR_PARAMS["n_estimators"] == 150
    assert GBMRerank.OPTIONS["gap"] == 1.0  # the instance override did not leak


def test_e11_e14_defaults():
    # the 2026-09 selection (docs/EXPERIMENTS.md): rounds, reranker options,
    # LambdaRank truncation
    assert BASE_PARAMS["n_estimators"] == 300 and PAIR_PARAMS["n_estimators"] == 150
    assert PAIR_PARAMS["learning_rate"] == BASE_PARAMS["learning_rate"]  # shares the rest
    opts = GBMRerank.OPTIONS
    assert opts["gap"] == 1.0 and opts["cluster_max"] == 6 and opts["context"] is True
    assert opts["pairs"] == "mixed" and opts["oof_gap"] == 2.0
    assert opts["pair_window"] == 6 and opts["oof"] is None and opts["oof_min_history"] == 60
    ar = GBMRerank(seed=0)
    assert (ar.gap, ar.cluster_max, ar.context) == (1.0, 6, True)
    assert (ar.pairs, ar.oof_gap) == ("mixed", 2.0)
    assert GBMRanker.OPTIONS["truncation"] == 60
    assert GBMRanker().truncation == 60


def test_default_bag_size_per_class():
    # n_seeds=None -> the class N_SEEDS: a 5-bag for the GBMs, one seed for the
    # deterministic baselines; explicit n_seeds still wins
    assert GBMRerank(seed=0).seeds == [0, 1, 2, 3, 4]
    assert RulesBaseline(seed=0).seeds == [0]
    assert LinearModel(seed=0).seeds == [0]
    assert GBMRerank.N_SEEDS == 5 and RulesBaseline.N_SEEDS == LinearModel.N_SEEDS == 1
    for name, cls in MODELS.items():
        expect = [0] if name in ("R", "L") else [0, 1, 2, 3, 4]
        assert cls(seed=0).seeds == expect, name
        assert cls(seed=0, n_seeds=1).seeds == [0], name
    assert GBMRerank(seed=1).seeds == [5, 6, 7, 8, 9]  # replicate 1 of the default bag
    assert GBMRerank(seed=0, n_seeds=2).seeds == [0, 1]


def test_seed_none_cannot_use_default_bag():
    # library-default seeds cannot be bagged, so seed=None needs n_seeds=1 on
    # the GBMs (whose default is a 5-bag) but works as-is on the baselines
    with pytest.raises(ValueError, match="cannot be bagged"):
        GBMRerank(seed=None)
    with pytest.raises(ValueError, match="cannot be bagged"):
        GBMMedian(seed=None)
    assert GBMRerank(seed=None, n_seeds=1).seeds == [None]
    assert RulesBaseline(seed=None).seeds == [None]


def test_rerank_mixed_pairs_need_oof_at_fit(small_train):
    # the default pairs='mixed' selects training pairs by rolling OOF base
    # near-ties, so fit() without an `oof` table (from GBMRerank.prepare /
    # oof_base_scores) must fail loudly rather than silently train on window
    # pairs only; the base stage is fitted first, so keep it tiny
    m = GBMRerank(seed=0, n_seeds=1, base={"n_estimators": 1}, pair=SMALL_PAIR)
    assert m.pairs == "mixed" and m.oof is None
    with pytest.raises(ValueError, match="OOF"):
        m.fit(small_train)
    assert not hasattr(m, "pair")  # the pair stage was never built


def test_model_registry():
    assert set(MODELS) == {"R", "L", "A", "Aq", "Aw", "B", "C", "Ar", "Ah"}
    assert all(cls.name == k for k, cls in MODELS.items())
    assert MODELS["Ah"](seed=0).h2h is True and MODELS["Ar"](seed=0).h2h is False
    assert MODELS["Ah"].OPTIONS == {**GBMRerank.OPTIONS, "h2h": True}  # Ah = Ar + h2h


# --- pairwise helpers ----------------------------------------------------------

def test_window_pairs():
    i, j = _window_pairs(5, 2)
    assert list(zip(i.tolist(), j.tolist())) == [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    i, j = _window_pairs(3, 10)  # window larger than n
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1), (0, 2), (1, 2)]
    i, j = _window_pairs(1, 3)
    assert len(i) == len(j) == 0


def test_gap_pairs_orientation_and_coverage():
    # base scores (lower = ranked higher) and current positions
    scores = np.array([0.0, 0.3, 1.0, 1.2, 5.0])
    pos = np.array([4, 3, 2, 1, 0])  # reverse of the score order
    i, j = _gap_pairs(scores, pos, 0.5)
    pairs = set(zip(i.tolist(), j.tolist()))
    # within-gap pairs: (0,1) and (2,3); oriented so pos[i] <= pos[j]
    assert pairs == {(1, 0), (3, 2)}
    assert all(pos[a] <= pos[b] for a, b in pairs)
    # a bigger gap: every pair with |score diff| <= gap, still oriented
    i, j = _gap_pairs(scores, pos, 1.5)
    pairs = set(zip(i.tolist(), j.tolist()))
    expected = {(b, a) for a in range(5) for b in range(a + 1, 5)
                if scores[b] - scores[a] <= 1.5}  # (b, a) since pos is reversed
    assert pairs == expected == {(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)}
    # already-oriented input is unchanged
    i, j = _gap_pairs(scores, np.arange(5), 0.5)
    assert set(zip(i.tolist(), j.tolist())) == {(0, 1), (2, 3)}


def test_pair_matrix_shapes(cands_200401):
    df = cands_200401
    i, j = _window_pairs(len(df), 3)
    n_ctx = len(CONTEXT_SHARED) + len(CONTEXT_MEAN) + 2 * len(CONTEXT_ENDS)
    assert n_ctx == 12
    assert _pair_matrix(df, i, j).shape == (len(i), len(FEATURES)) == (len(i), 35)
    assert _pair_matrix(df, i, j, context=True).shape == (len(i), 35 + n_ctx)
    base = np.arange(len(df), dtype=float)
    assert _pair_matrix(df, i, j, base=base).shape == (len(i), 36)
    assert _pair_matrix(df, i, j, h2h=True).shape == (len(i), 36)
    full = _pair_matrix(df, i, j, h2h=True, context=True, base=base)
    assert full.shape == (len(i), 35 + 1 + n_ctx + 1)
    # first block is the feature difference i - j; last column the base gap
    X = df[FEATURES].to_numpy(dtype=float)
    assert np.array_equal(full[:, :35], X[i] - X[j], equal_nan=True)
    assert np.array_equal(full[:, -1], base[i] - base[j])
    # h2h column is in {-1, 0, 1}
    assert set(np.unique(full[:, 35])) <= {-1.0, 0.0, 1.0}


def test_pair_matrix_h2h_matches_bouts(cands_200401, bouts):
    df = cands_200401
    i, j = _window_pairs(len(df), 6)
    col = _pair_matrix(df, i, j, h2h=True)[:, -1]
    b = bouts[bouts["basho"] == int(df["basho"].iloc[0])]
    wins = set(zip(b["winner"], b["loser"]))
    rid = df["rikishi_id"].to_numpy()
    expect = np.array([float((rid[a], rid[c]) in wins) - float((rid[c], rid[a]) in wins)
                       for a, c in zip(i, j)])
    assert np.array_equal(col, expect)
    assert (col != 0).any()  # some neighbours did meet


def test_pair_stage_oof_requires_oof_table():
    with pytest.raises(ValueError, match="OOF"):
        _PairStage([0], PAIR_PARAMS, 6, pairs="oof", oof=None)
    with pytest.raises(ValueError, match="OOF"):
        _PairStage([0], PAIR_PARAMS, 6, pairs="mixed", oof=None)
    with pytest.raises(ValueError, match="window|oof|mixed"):
        _PairStage([0], PAIR_PARAMS, 6, pairs="bogus")
    # window mode needs nothing
    assert _PairStage([0], PAIR_PARAMS, 6).use_base is False


# --- fits ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rerank_fit(small_train):
    m = GBMRerank(seed=0, pairs="window", base=SMALL_BASE, pair=SMALL_PAIR, **ONE)
    m.fit(small_train)
    return m


def test_rerank_score_is_permutation_and_deterministic(rerank_fit, small_train, cands_200401):
    s = rerank_fit.score(cands_200401)
    n = len(cands_200401)
    assert s.shape == (n,)
    assert sorted(s.tolist()) == list(range(n))
    # single-seed instance: k=0 is the whole bag
    assert rerank_fit.seeds == [0] and len(rerank_fit.ms) == len(rerank_fit.pair.ms) == 1
    assert np.array_equal(rerank_fit.score(cands_200401, k=0), s)
    # identical refit gives identical scores
    m2 = GBMRerank(seed=0, pairs="window", base=SMALL_BASE, pair=SMALL_PAIR, **ONE)
    m2.fit(small_train)
    assert np.array_equal(m2.score(cands_200401), s)
    # diagnostics of the adjudicated near-tie pairs
    i, j, p, in_window = rerank_fit.last_pairs
    assert len(i) == len(j) == len(p) == len(in_window) > 0
    assert ((p >= 0) & (p <= 1)).all()
    pos = cands_200401["position"].to_numpy()
    assert (pos[i] <= pos[j]).all()  # pairs oriented by current position


def test_rerank_only_reorders_within_clusters(rerank_fit, cands_200401):
    # the final order is Aq's base order with reordering confined to
    # near-tie clusters: cluster blocks keep their relative placement
    assert rerank_fit.cluster_max == GBMRerank.OPTIONS["cluster_max"]
    base = GBMMedian.score(rerank_fit, cands_200401)
    final = rerank_fit.score(cands_200401)
    clusters = rerank_fit._clusters(base, cands_200401["position"].to_numpy())
    assert sorted(i for cl in clusters for i in cl) == list(range(len(cands_200401)))
    assert all(len(cl) <= rerank_fit.cluster_max for cl in clusters)
    start = 0
    for cl in clusters:
        assert set(final[cl].astype(int)) == set(range(start, start + len(cl)))
        start += len(cl)


def test_rerank_window_pairs_use_context_by_default(rerank_fit):
    # context=True is now the default: pair rows are the 35 feature
    # differences plus 12 context columns (no base gap with window pairs)
    assert rerank_fit.context is True and rerank_fit.pair.use_base is False
    n_ctx = len(CONTEXT_SHARED) + len(CONTEXT_MEAN) + 2 * len(CONTEXT_ENDS)
    assert rerank_fit.pair.ms[0].n_features_in_ == len(FEATURES) + n_ctx == 47


def test_rerank_mixed_pairs_fit_with_oof(trans, bashos, make_cands, tmp_path):
    # the default configuration end to end on a short history: rolling OOF
    # scores for the last 6 labeled basho, training on the first 5 of them
    # (labels at or before the 6th), scoring the transition off the 6th
    lab = trans[trans["position_next"].notna()]
    labeled = sorted(lab["basho"].unique())
    covered = labeled[-6:]
    params = {**BASE_PARAMS, **SMALL_BASE}
    oof = oof_base_scores(trans, params, GBMRerank.objective, [0], min_history=len(labeled) - 6,
                          workers=1, cache_dir=tmp_path)
    assert sorted(oof["basho"].unique()) == covered
    train = lab[lab["basho"].isin(covered[:-1])]
    cands = make_cands(bashos[-1])
    assert int(cands["basho"].iloc[0]) == covered[-1] == bashos[-2]
    kw = dict(seed=0, base=SMALL_BASE, pair=SMALL_PAIR, **ONE)
    m = GBMRerank(oof=oof, **kw)  # pairs='mixed' by default
    assert m.pairs == "mixed" and m.oof_gap == 2.0
    m.fit(train)
    w = GBMRerank(pairs="window", **kw)
    w.fit(train)
    # mixed = window pairs plus the OOF near-tie pairs, with the base gap as
    # an extra pair feature; the reranked order is still a permutation
    assert m.pair.use_base and m.pair.n_train_pairs > w.pair.n_train_pairs
    assert m.pair.ms[0].n_features_in_ == w.pair.ms[0].n_features_in_ + 1
    s = m.score(cands)
    assert sorted(s.tolist()) == list(range(len(cands)))
    i, j, p, _ = m.last_pairs
    assert len(i) > 0 and ((p >= 0) & (p <= 1)).all()
    # 'oof' alone needs the table to cover the training basho
    before = lab[lab["basho"] == labeled[-8]]
    with pytest.raises(ValueError, match="no training pairs"):
        GBMRerank(pairs="oof", oof=oof, **kw).fit(before)


def test_seed_changes_scores(small_train, cands_200401):
    a = GBMMedian(seed=0, base=SMALL_BASE, **ONE)
    b = GBMMedian(seed=1, base=SMALL_BASE, **ONE)
    a.fit(small_train)
    b.fit(small_train)
    assert not np.array_equal(a.delta(cands_200401), b.delta(cands_200401))
    assert a.ms[0].get_params()["random_state"] == 0
    assert b.ms[0].get_params()["random_state"] == 1


def test_bag_delta_is_mean_of_members(small_train, cands_200401):
    bag = GBMMedian(seed=0, n_seeds=2, base=SMALL_BASE)
    bag.fit(small_train)
    assert bag.seeds == [0, 1]
    assert [m.get_params()["random_state"] for m in bag.ms] == [0, 1]
    d0, d1 = bag.delta(cands_200401, k=0), bag.delta(cands_200401, k=1)
    assert not np.array_equal(d0, d1)
    assert bag.delta(cands_200401) == pytest.approx((d0 + d1) / 2)
    pos = cands_200401["position"].to_numpy()
    assert bag.score(cands_200401) == pytest.approx(pos + (d0 + d1) / 2)
    # bag members equal the corresponding single-seed fits
    single = GBMMedian(seed=1, base=SMALL_BASE, **ONE)  # seed 1 == member 1 of bag 0
    single.fit(small_train)
    assert single.delta(cands_200401) == pytest.approx(d1)


def test_default_bag_fits_five_members(small_train, cands_200401):
    # the class default: seed=0 alone is the 5-seed bag 0..4
    bag = GBMMedian(seed=0, base={"n_estimators": 5})
    bag.fit(small_train)
    assert bag.seeds == [0, 1, 2, 3, 4]
    assert [m.get_params()["random_state"] for m in bag.ms] == [0, 1, 2, 3, 4]
    members = [bag.delta(cands_200401, k=k) for k in range(5)]
    assert bag.delta(cands_200401) == pytest.approx(np.mean(members, axis=0))


def test_seed_none_uses_library_defaults(small_train, cands_200401):
    m = GBMMedian(seed=None, base=SMALL_BASE, **ONE)
    m.fit(small_train)
    assert m.seeds == [None]
    assert m.ms[0].get_params()["random_state"] is None
    assert m.score(cands_200401).shape == (len(cands_200401),)
    with pytest.raises(ValueError):
        GBMMedian(seed=None, n_seeds=2)


def test_linear_and_rules_baselines_score(small_train, cands_200401):
    for m in (LinearModel(), RulesBaseline(recent=20)):
        m.fit(small_train)
        s = m.score(cands_200401)
        assert s.shape == (len(cands_200401),)
        assert np.isfinite(s).all()


# --- rolling OOF base scores -----------------------------------------------------

def test_oof_base_scores_floor_and_cache(trans, tmp_path):
    lab = trans[trans["position_next"].notna()]
    labeled = sorted(lab["basho"].unique())
    min_history = len(labeled) - 4
    params = {**BASE_PARAMS, "n_estimators": 20}
    out = oof_base_scores(trans, params, "regression_l1", [0], min_history=min_history,
                          workers=1, cache_dir=tmp_path)
    # rows only for basho after the history floor: the last 4 labeled basho
    assert sorted(out["basho"].unique()) == labeled[min_history:]
    assert list(out.columns) == ["basho", "rikishi_id", "oof"]
    for b in labeled[min_history:]:
        rows = lab[lab["basho"] == b]
        assert set(out.loc[out["basho"] == b, "rikishi_id"]) == set(rows["rikishi_id"])
    assert np.isfinite(out["oof"]).all()
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1 and not files[0].name.endswith(".tmp.parquet")
    mtime = files[0].stat().st_mtime_ns
    again = oof_base_scores(trans, params, "regression_l1", [0], min_history=min_history,
                            workers=1, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(again, out)
    assert files[0].stat().st_mtime_ns == mtime  # reused, not recomputed
    # a different configuration gets its own cache key
    other = oof_base_scores(trans, params, "regression_l1", [0], min_history=min_history + 1,
                            workers=1, cache_dir=tmp_path)
    assert sorted(other["basho"].unique()) == labeled[min_history + 1:]
    assert len(list(tmp_path.glob("*.parquet"))) == 2


def test_oof_scores_are_base_prediction_without_label(trans, tmp_path):
    # the OOF score of source basho b = position + delta predicted by a bag
    # trained on transitions labeled at or before b (next_basho <= b)
    from lightgbm import LGBMRegressor

    lab = trans[trans["position_next"].notna()]
    labeled = sorted(lab["basho"].unique())
    b = labeled[-1]
    params = {**BASE_PARAMS, "n_estimators": 20}
    out = oof_base_scores(trans, params, "regression_l1", [0], min_history=len(labeled) - 1,
                          workers=1, cache_dir=tmp_path)
    assert sorted(out["basho"].unique()) == [b]
    train = lab[lab["next_basho"] <= b]
    assert (train["basho"] < b).all()  # the scored basho's own labels are excluded
    rows = lab[lab["basho"] == b]
    pred = LGBMRegressor(objective="regression_l1", **{**params, "random_state": 0}) \
        .fit(train[FEATURES], train["delta"]).predict(rows[FEATURES])
    expect = pd.Series(rows["position"].to_numpy() + pred, index=rows["rikishi_id"].to_numpy())
    got = out.set_index("rikishi_id")["oof"]
    assert got.reindex(expect.index).to_numpy() == pytest.approx(expect.to_numpy())


def _fake_oof(calls):
    def fake_oof(trans_, base_params, objective, seeds, min_history, workers, **model_opts):
        calls.append((base_params["n_estimators"], objective, seeds, min_history, workers))
        return pd.DataFrame({"basho": [200311], "rikishi_id": [1], "oof": [1.0]})
    return fake_oof


def test_rerank_prepare_supplies_oof(trans, monkeypatch):
    # prepare() is a no-op for window pairs and fills `oof` for oof/mixed
    assert GBMRerank.prepare({"pairs": "window", "gap": 0.3}, trans) == {
        "pairs": "window", "gap": 0.3}
    import banzuke.models as models_mod
    calls = []
    monkeypatch.setattr(models_mod, "oof_base_scores", _fake_oof(calls))
    kw = GBMRerank.prepare({"pairs": "oof", "seed": 3, "base": SMALL_BASE,
                            "oof_min_history": 12}, trans, workers=2)
    assert kw["oof"].shape == (1, 3) and kw["pairs"] == "oof"
    # the configuration's seed-0 bag (the class default of 5), seed dropped
    assert calls == [(30, "regression_l1", [0, 1, 2, 3, 4], 12, 2)]
    given = GBMRerank.prepare({"pairs": "oof", "oof": kw["oof"]}, trans)
    assert len(calls) == 1 and given["oof"] is kw["oof"]  # supplied table reused
    # n_seeds in the configuration sizes the OOF bag too
    kw1 = GBMRerank.prepare({"pairs": "mixed", "n_seeds": 1, "seed": 2}, trans)
    assert calls[-1] == (BASE_PARAMS["n_estimators"], "regression_l1", [0], 60, 1)
    assert kw1["oof"] is not None and kw1["n_seeds"] == 1 and kw1["seed"] == 2
    # the prepared kwargs build a model that fits the mixed pair stage
    m = GBMRerank(**{k: v for k, v in kw1.items() if k != "seed"}, seed=kw1["seed"])
    assert m.pairs == "mixed" and m.oof is not None and m.seeds == [2]


def test_rerank_prepare_honours_default_pairs(trans, monkeypatch):
    """prepare() must read the class default for `pairs`, not assume 'window':
    predict.py / backtest.py pass no explicit pairs option."""
    import banzuke.models as models_mod
    calls = []
    monkeypatch.setattr(models_mod, "oof_base_scores", _fake_oof(calls))
    kw = GBMRerank.prepare({"n_seeds": 1, "base": SMALL_BASE}, trans)
    assert GBMRerank(**kw).pairs == "mixed"
    assert calls and kw.get("oof") is not None


def test_subclass_options_extend_parent_options():
    """A subclass must not drop the options its parent's fit() reads."""
    for cls in MODELS.values():
        for parent in cls.__mro__[1:]:
            if hasattr(parent, "OPTIONS") and parent.OPTIONS:
                assert set(parent.OPTIONS) <= set(cls.OPTIONS), (cls.name, parent.__name__)
