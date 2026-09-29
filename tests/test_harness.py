"""banzuke/harness.py: backtest rows, determinism across workers and
mak-size policies, summarize() statistics, CLI option parsing, fingerprint."""
import numpy as np
import pandas as pd
import pytest

from banzuke.harness import (
    METRICS, block_bootstrap_ci, fingerprint, label_of, parse_sets, run_backtest,
    summarize,
)

TARGETS = [202309, 202311]
KW = {"base": {"n_estimators": 30}}


@pytest.fixture(scope="module")
def aq_rows(trans, tidy):
    return run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                        model_kwargs=KW)


def test_backtest_rows_and_columns(aq_rows):
    r = aq_rows
    assert len(r) == 2
    assert {"config", "model", "seed", "basho", "exact_n", "mae"} <= set(r.columns)
    assert set(METRICS) <= set(r.columns)
    assert r["basho"].tolist() == TARGETS
    assert (r["model"] == "Aq").all() and (r["seed"] == 0).all()
    assert (r["config"] == "base").all()
    assert (r["n_slots"] == 42).all()
    assert r["exact_n"].between(0, 42).all()
    assert (r["exact"] == r["exact_n"] / 42).all()
    assert (r["mae"] >= 0).all() and r["tau"].between(-1, 1).all()
    assert r["exact_n"].sum() > 10  # a real model, not noise
    # Aq has no reranker, so no pair diagnostics columns
    assert "pair_n" not in r.columns


def test_backtest_is_deterministic(aq_rows, trans, tidy):
    again = run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                         model_kwargs=KW)
    pd.testing.assert_frame_equal(again, aq_rows)


def test_backtest_workers_match_serial(aq_rows, trans, tidy):
    par = run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                       model_kwargs=KW, workers=2)
    pd.testing.assert_frame_equal(par, aq_rows)


def test_mak_size_policy_equal_when_sizes_agree(aq_rows, trans, tidy):
    # all 2004+ targets have a 42-man makuuchi: 'prior' and 'actual' agree
    for t in TARGETS:
        assert int((tidy.loc[tidy["basho"] == t, "division"] == 0).sum()) == 42
    act = run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                       model_kwargs=KW, mak_size_policy="actual")
    pd.testing.assert_frame_equal(act, aq_rows)


def test_backtest_return_preds_skip_and_config(trans, tidy):
    res, preds = run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                              model_kwargs=KW, return_preds=True, config="x",
                              skip={("Aq", 0, 202311)})
    assert res["basho"].tolist() == [202309]
    assert (res["config"] == "x").all()
    assert set(preds["target"]) == {202309}
    assert {"rikishi_id", "pred_class", "pred_number", "pred_side", "pred_pos",
            "config", "model", "seed", "target"} <= set(preds.columns)
    prev_rows = trans[(trans["basho"] == 202307) & ~trans["dropped"]]
    assert len(preds) == len(prev_rows)
    assert sorted(preds["pred_pos"]) == list(range(len(prev_rows)))
    # skipping everything yields an empty frame
    empty = run_backtest(["Aq"], TARGETS, trans, tidy, progress=False, seeds=(0,),
                         model_kwargs=KW, skip={("Aq", 0, t) for t in TARGETS})
    assert len(empty) == 0


# --- summarize -------------------------------------------------------------------

def _synthetic(n_basho=20, seeds=(0, 1), bonus=None):
    """Two configs of model Aq; config x gains `bonus(seed)` exact slots on
    every basho (default: +1 for every seed)."""
    bonus = bonus or (lambda seed: 1)
    rows = []
    for b in range(n_basho):
        base_exact = 15 + (b % 5)
        for cfg in ("base", "x"):
            for seed in seeds:
                e = base_exact + (bonus(seed) if cfg == "x" else 0)
                rows.append({
                    "config": cfg, "model": "Aq", "seed": seed, "basho": 200401 + b,
                    "n_slots": 42, "exact_n": e, "exact": e / 42, "gtb_points": 2 * e,
                    "within1": 0.6, "mae": 2.0 - (0.1 if cfg == "x" else 0.0), "tau": 0.9,
                    "promo_f1": 1.0, "demo_f1": 1.0, "sanyaku_acc": 1.0, "sanyaku_exact": 1.0,
                })
    return pd.DataFrame(rows)


def _ci(text):
    lo, hi = text.split(",")
    return float(lo), float(hi)


def test_summarize_paired_comparison():
    s = summarize(_synthetic(), baseline="Aq")
    assert s.attrs["baseline"] == "Aq"
    assert list(s.index) == ["Aq:x", "Aq"]  # sorted by exact, leader first
    assert (s["n"] == 20).all()
    x = s.loc["Aq:x"]
    assert x["d_exact"] == pytest.approx(1.0)
    assert x["wl"] == "20-0"
    assert x["p_exact"] < 0.01
    lo, hi = _ci(x["ci_exact"])
    assert lo > 0 and hi >= lo
    assert x["d_gtb"] == pytest.approx(2.0)
    assert x["d_mae"] == pytest.approx(-0.1)
    assert x["p_mae"] < 0.01
    base = s.loc["Aq"]
    assert base["d_exact"] == 0 and base["d_mae"] == 0 and base["wl"] == "0-0"
    assert np.isnan(base["p_exact"]) and base["ci_exact"] == ""
    # means: seed-averaged per basho then averaged over basho
    assert base["exact_n"] == pytest.approx(17.0) and x["exact_n"] == pytest.approx(18.0)


def test_summarize_averages_seeds_within_basho():
    s = summarize(_synthetic(bonus=lambda seed: 2 if seed == 1 else 0), baseline="Aq")
    assert s.loc["Aq:x", "d_exact"] == pytest.approx(1.0)
    assert s.loc["Aq:x", "wl"] == "20-0"  # every basho's seed-mean is +1


def test_summarize_default_baseline_and_windows():
    r = _synthetic()
    s = summarize(r)  # default baseline: exact leader (config x)
    assert s.attrs["baseline"] == "Aq:x"
    assert s.loc["Aq", "d_exact"] == pytest.approx(-1.0)
    assert s.loc["Aq", "wl"] == "0-20"
    s2 = summarize(r, baseline="Aq", since=200411, until=200501)
    assert (s2["n"] == 10).all()
    s3 = summarize(r, baseline="nope")  # unknown baseline falls back to leader
    assert s3.attrs["baseline"] == "Aq:x"


def test_label_of():
    r = pd.DataFrame({"model": ["Aq", "Aq", "B"], "config": ["base", "x", "base"]})
    assert label_of(r).tolist() == ["Aq", "Aq:x", "B"]
    assert label_of(r.drop(columns="config")).tolist() == ["Aq", "Aq", "B"]


# --- helpers ----------------------------------------------------------------------

def test_block_bootstrap_ci():
    assert block_bootstrap_ci(np.full(12, 3.0)) == (3.0, 3.0)
    x = np.arange(20.0)
    lo, hi = block_bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert block_bootstrap_ci(x) == block_bootstrap_ci(x)  # seeded
    assert block_bootstrap_ci(x, seed=1) != block_bootstrap_ci(x, seed=2)
    rng = np.random.default_rng(0)
    y = rng.normal(0.5, 1.0, size=120)
    lo, hi = block_bootstrap_ci(y)
    assert lo < y.mean() < hi and hi - lo < 1.0
    assert all(np.isnan(v) for v in block_bootstrap_ci([1.0]))


def test_parse_sets():
    assert parse_sets(["base.n_estimators=200", "gap=0.25", "pairs=oof", "context=true"]) == {
        "base": {"n_estimators": 200}, "gap": 0.25, "pairs": "oof", "context": True}
    assert parse_sets([]) == {}
    assert parse_sets(["base.n_estimators=200", "base.num_leaves=31"]) == {
        "base": {"n_estimators": 200, "num_leaves": 31}}
    assert parse_sets(["half_life=null"]) == {"half_life": None}
    with pytest.raises(ValueError):
        parse_sets(["gap"])


def test_fingerprint():
    a, b, a2 = fingerprint("a"), fingerprint("b"), fingerprint("a")
    assert a != b and a == a2
    assert len(a) == 16 and int(a, 16) >= 0
    assert fingerprint() == fingerprint("")
