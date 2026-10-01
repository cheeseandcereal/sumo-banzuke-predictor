"""Invariant suite on the committed processed data: chronological features,
resolver conventions, model seed/option semantics, bagging, reranking, OOF
scores, backtest determinism, summarize() statistics, predict() assembly,
forced S/K claims, confidence signals and review items."""
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
from banzuke import confidence  # noqa: E402
from banzuke.build import JURYO, KOMUSUBI, MAEGASHIRA, OZEKI, SEKIWAKE, YOKOZUNA  # noqa: E402
from banzuke.features import FEATURES, build_transitions  # noqa: E402
from banzuke.harness import METRICS, parse_seeds, parse_sets, run_backtest, summarize  # noqa: E402
from banzuke.metrics import evaluate  # noqa: E402
from banzuke.models import (MODELS, GBMMedian, GBMRanker, GBMRerank, RulesBaseline,  # noqa: E402
                            _gap_pairs, _seeds, _twin_units, _window_pairs, oof_base_scores)
from banzuke.overrides import parse  # noqa: E402
from banzuke.resolver import block_slots, forced_claims, resolve  # noqa: E402

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
    # lagged class/number are resolver inputs (not FEATURES), kept with the same contiguity
    for col in ("class1", "class2", "num1", "num2"):
        assert col not in FEATURES and _same(old[col], new[col]), col
    assert np.isnan(a.loc[202301, "class1"]) and a.loc[202305, ["class1", "num1"]].tolist() == [JURYO, 1]


def test_raw_corrections_and_blank_result_check():
    """202507 juryo: the day-15 bout Nishikigi (J1E) vs Fujiseiun (J8W) has a blank
    result in the API record, so its wins/losses were short by one (the torikumi
    has Nishikigi by kotenage). The correction is applied at build time, and a
    scheduled bout without a result anywhere else fails the build loudly."""
    from banzuke.build import CORRECTIONS, _correct
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    rows = tidy[(tidy["basho"] == 202507) & tidy["rikishi_id"].isin([16, 82])].set_index("rikishi_id")
    assert rows.loc[16, ["wins", "losses", "absences"]].tolist() == [8, 7, 0]
    assert rows.loc[82, ["wins", "losses", "absences"]].tolist() == [9, 6, 0]
    assert (tidy[tidy["basho"] >= 200401].eval("wins + losses + absences") == 15).mean() > 0.99
    record = [{"result": "win", "opponentID": 1}, {"result": "", "opponentID": 2}]
    with pytest.raises(ValueError, match="without a result"):
        _correct(209901, {"rikishiID": 999, "shikonaEn": "X"}, record)
    r = {"rikishiID": 16, "shikonaEn": "Nishikigi", "wins": 7, "losses": 7}
    record = [{"result": "loss", "opponentID": 1}] * 14 + [{"result": "", "opponentID": 82}]
    _correct(202507, r, [dict(b) for b in record])
    assert r["wins"] == 8 and set(CORRECTIONS) == {(202507, 16), (202507, 82)}


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


def test_resolver_keeps_identical_record_twins_in_order(cands_for):
    # E16: E and W of one M/J rank with the same record never swap. 202609:
    # Gonoyama (M2E) / Churanoumi (M2W), both 7-8, landed M2E/M2W; with their
    # scores exchanged the resolver still emits them in prior order, and the
    # rest of the sheet is untouched
    cands, pred = _true_order(cands_for, 202609)
    scores = cands["position_next"].to_numpy(dtype=float)
    names = cands["shikona"]
    e, w = names.eq("Gonoyama").idxmax(), names.eq("Churanoumi").idxmax()
    assert cands.loc[[e, w], ["rank_class", "rank_number", "wins", "losses"]].nunique().eq(1).all()
    swapped = scores.copy()
    swapped[[e, w]] = swapped[[w, e]]
    assert (resolve(cands, swapped, 42).to_numpy() == pred.to_numpy()).all()
    # a pair with different records is not protected: swapping does swap
    others = names.eq("Kotoeiho").idxmax(), names.eq("Gonoyama").idxmax()
    swapped = scores.copy()
    swapped[list(others)] = swapped[list(others)[::-1]]
    assert not (resolve(cands, swapped, 42).to_numpy() == pred.to_numpy()).all()


def _cells(cands, pred):
    return pred.merge(cands[["rikishi_id", "shikona"]], on="rikishi_id").set_index("shikona")


@pytest.mark.parametrize("target", [200607, 201507, 201805, 201807, 202107, 202305, 202601, 202605])
def test_resolver_rules_reproduce_committee_frames(cands_for, tidy, target):
    """E22: frames the committee rules decide. From the true order the sheet
    is reproduced cell for cell; before the rules were ported these eight were
    not (Y/O membership or a make-koshi sekiwake's exit was wrong)."""
    cands, pred = _true_order(cands_for, target)
    actual = tidy[tidy["basho"] == target]
    assert evaluate(pred, cands, actual)["exact"] == 1.0


def test_resolver_yo_order_and_demoted_ozeki(cands_for):
    # R14, 201901: all three yokozuna have 0 wins; Kisenosato (Y2E 0-5-10) fought,
    # Hakuho (Y1E) and Kakuryu (Y1W) did not, so Kisenosato / Hakuho / Kakuryu
    # whatever order the model gives; ozeki by wins (Takayasu 12-3 over Goeido 8-4-3)
    for sign in (1.0, -1.0):
        cands, pred = _true_order(cands_for, 201901, sign)
        cells = _cells(cands, pred)
        assert cells.loc[["Kisenosato", "Hakuho", "Kakuryu"], CELL].to_numpy().tolist() == [
            [YOKOZUNA, 1, 0], [YOKOZUNA, 1, 1], [YOKOZUNA, 2, 0]]
        assert cells.loc[["Takayasu", "Goeido", "Tochinoshin"], CELL].to_numpy().tolist() == [
            [OZEKI, 1, 0], [OZEKI, 1, 1], [OZEKI, 2, 1]]
    # R12, 202009: Okinoumi (K1W 9-6) stays above the newcomer Endo (M1E 8-7)
    # whatever order the model gives
    for sign in (1.0, -1.0):
        cands, pred = _true_order(cands_for, 202009, sign)
        cells = _cells(cands, pred)
        assert cells.loc[["Okinoumi", "Endo"], CELL].to_numpy().tolist() == [[KOMUSUBI, 1, 0], [KOMUSUBI, 1, 1]]
    # R4, 202607: Aonishiki (O 0-0-15, kadoban) is the bottom sekiwake even when
    # the model ranks him first among the sekiwake candidates
    cands, pred = _true_order(cands_for, 202607, -1.0)
    s = pred[pred["pred_class"] == SEKIWAKE].merge(cands[["rikishi_id", "shikona", "kadoban"]], on="rikishi_id")
    assert s["shikona"].iloc[-1] == "Aonishiki" and len(s) >= 2
    assert (s["kadoban"].iloc[:-1] == 0).all()


def test_resolver_make_koshi_exits(cands_for):
    """R2/R3: a 7-win sekiwake takes a komusubi slot, weak M1 claims do not grow
    the block for him (201805: Tamawashi M1W 9-6 back to M1E); S/K with 6 or
    fewer wins, or a komusubi with 7 wins below K1E, never stay in sanyaku."""
    for sign in (1.0, -1.0):
        cands, pred = _true_order(cands_for, 201805, sign)
        cells = _cells(cands, pred)
        assert cells.loc["Mitakeumi", "pred_class"] == KOMUSUBI
        assert (pred["pred_class"] == KOMUSUBI).sum() == 2
        assert cells.loc["Tamawashi", "pred_class"] >= MAEGASHIRA
        if sign > 0:  # true order: Endo K1W, Tamawashi M1E
            assert cells.loc[["Endo", "Tamawashi"], CELL].to_numpy().tolist() == [[KOMUSUBI, 1, 1], [MAEGASHIRA, 1, 0]]
    n = 0
    for target, sign in product((202105, 202111, 202403, 202603), (1.0, -1.0)):
        cands, pred = _true_order(cands_for, target, sign)
        m = cands.merge(pred, on="rikishi_id")
        out = m[m["rank_class"].isin([SEKIWAKE, KOMUSUBI]) & (m["wins"] <= 6)
                | (m["rank_class"] == KOMUSUBI) & (m["wins"] == 7) & ~((m["rank_number"] == 1) & (m["side"] == 0))]
        n += len(out)
        assert (out["pred_class"] >= MAEGASHIRA).all(), target
    assert n >= 8
    # the override path still wins, and says so (202405: Nishikigi K1W 3-12)
    cands = cands_for(202405)
    warnings = []
    rid = cands.loc[cands["shikona"] == "Nishikigi", "rikishi_id"].iloc[0]
    pred = resolve(cands, cands["position_next"].to_numpy(dtype=float), 42,
                   overrides={"class": {rid: KOMUSUBI}}, warnings=warnings)
    assert _cells(cands, pred).loc["Nishikigi", "pred_class"] == KOMUSUBI
    assert any("kept at komusubi by override" in w for w in warnings)


def test_rule_masks_yo_promotions(trans):
    """R7: a yusho / jun-yusho fought at sekiwake does not count towards the
    yokozuna rule; R8: a 33-win run with 12+ wins now and one M1-M3 basho
    counts towards the ozeki rule, two maegashira basho do not."""
    from banzuke.resolver import rule_masks
    rm = rule_masks(trans)
    index = pd.MultiIndex.from_arrays([trans["basho"], trans["shikona"]])
    y = pd.Series(rm["y_promo"], index=index)
    assert not y[(202105, "Terunofuji")] and not y[(200605, "Hakuho")] and not y[(202601, "Aonishiki")]
    assert y[(200705, "Hakuho")] and y[(202505, "Onosato")]
    o = pd.Series(rm["o_promo"], index=index)
    assert all(o[k] for k in [(201505, "Terunofuji"), (201805, "Tochinoshin"), (202511, "Aonishiki"), (202603, "Kirishima")])
    assert not any(o[k] for k in [(202011, "Terunofuji"), (202405, "Onosato"), (200509, "Kotooshu")])
    assert not o[(201811, "Takakeisho")] and o[(201903, "Takakeisho")]  # 33 wins over K/S/S: the classic rule
    # kadoban survives an exempted second make-koshi (Mitakeumi 202207 -> 202209), so
    # his third one demotes him in the resolver
    k = trans.set_index(["basho", "shikona"])["kadoban"]
    assert k[(202207, "Mitakeumi")] == 1 and k[(202209, "Mitakeumi")] == 1
    assert pd.Series(rm["kadoban_out"], index=index)[(202209, "Mitakeumi")]


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


def test_twin_unit_keeps_identical_record_twins_together(small_train, cands_for):
    # E17: with twin_unit, E/W twins with the same record leave the reranker
    # adjacent and in prior order; a sheet without twins is scored exactly as
    # without the option
    kw = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}
    plain, unit = GBMRerank(**kw, twin_unit=""), GBMRerank(**kw, twin_unit="all")
    plain.fit(small_train)
    unit.fit(small_train)
    changed = 0
    for target in (200401, 200607, 202309, 202609):
        cands = cands_for(target)
        twins = _twin_units(cands, "all")
        assert twins and all(cands.loc[e, "side"] == 0 and cands.loc[w, "side"] == 1
                             and cands.loc[e, "wins"] == cands.loc[w, "wins"] for e, w in twins)
        assert len(_twin_units(cands, "sk")) <= len(twins)
        s_plain, s_unit = plain.score(cands), unit.score(cands)
        assert sorted(s_unit.tolist()) == list(range(len(cands)))
        assert all(s_unit[w] == s_unit[e] + 1 for e, w in twins)
        changed += not np.array_equal(s_plain, s_unit)
        solo = cands.drop(index=[w for _, w in twins]).reset_index(drop=True)
        assert not _twin_units(solo, "all")
        assert np.array_equal(plain.score(solo), unit.score(solo))
    assert changed  # the option does something on at least one of these sheets
    with pytest.raises(ValueError, match="twin_unit"):
        _twin_units(cands, "yo")


def test_protected_and_supply_columns(trans):
    """E23/E25 columns. rank_protected (a feature) marks exactly the full-kyujo
    rows whose rank was frozen (kosho granted, or a listed modern exemption);
    the supply columns (opt-in) describe the source basho's makuuchi field."""
    from banzuke.features import PROTECTED, SUPPLY_FEATURES
    assert "rank_protected" in FEATURES and set(SUPPLY_FEATURES).isdisjoint(FEATURES)
    full = trans[(trans["wins"] == 0) & (trans["absences"] >= 8) & (trans["rank_class"] >= SEKIWAKE)
                 & trans["delta"].notna()]
    kept, dropped = full[full["rank_protected"] == 1], full[full["rank_protected"] == 0]
    assert kept["delta"].max() <= 3 and dropped["delta"].min() >= 10
    modern = trans[(trans["basho"] >= 200401) & (trans["rank_protected"] == 1)]
    assert set(zip(modern["basho"], modern["rikishi_id"])) == PROTECTED and len(PROTECTED) == 26
    assert (trans.loc[trans["rank_class"] <= OZEKI, "rank_protected"] == 0).all()
    assert trans.loc[(trans["basho"] == 202201) & (trans["shikona"] == "Takayasu"), "rank_protected"].item() == 1
    assert trans.loc[(trans["basho"] == 202207) & (trans["shikona"] == "Takanosho"), "rank_protected"].item() == 0
    b = trans[trans["basho"] == 202207].iloc[0]
    assert [b[c] for c in SUPPLY_FEATURES] == [9, 4, 3, 3, 14]  # Nagoya 2022: 14 men with an absence
    assert trans.groupby("basho")[SUPPLY_FEATURES].nunique().eq(1).all().all()


def test_extra_feature_options(small_train, cands_for):
    """`extra` / `extra_shared` append dataset columns to the GBM inputs (base
    model, pair differences, pair context) and the OOF cache key."""
    cands = cands_for(200401)
    kw = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}
    plain = GBMRerank(**kw)
    more = GBMRerank(**kw, extra="strong_below", extra_shared=["mk_joi", "n_absent"])
    assert more.features == FEATURES + ["strong_below", "mk_joi", "n_absent"] and plain.features == FEATURES
    plain.fit(small_train)
    more.fit(small_train)
    assert more.ms[0].n_features_ == len(FEATURES) + 3
    assert more.pair.ms[0].n_features_ == plain.pair.ms[0].n_features_ + 1 + 2
    assert sorted(more.score(cands).tolist()) == list(range(len(cands)))
    assert not np.array_equal(plain.base_score(cands), more.base_score(cands))
    import tempfile
    with tempfile.TemporaryDirectory() as d:  # the option is part of the OOF cache key
        kw_oof = {"n_seeds": 1, "base": {"n_estimators": 5}}
        n = len(small_train["basho"].unique()) - 1
        oof_base_scores(small_train, GBMRerank, kw_oof, min_history=n, cache_dir=d)
        oof_base_scores(small_train, GBMRerank, {**kw_oof, "extra": "strong_below"}, min_history=n, cache_dir=d)
        assert len(list(Path(d).glob("*.parquet"))) == 2


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
    pred, warnings, _ = predict.predict(cands, point, [point, point + 0.1], ov, 42)
    assert warnings == [] and sorted(pred["pred_pos"]) == list(range(len(cands)))
    assert (pred["spread"] == 0).all()  # a constant shift keeps the order
    pred, _, _ = predict.predict(cands, point, [point, point[::-1]], ov, 42)
    assert (pred["spread"] > 0).any()


def test_predict_notes_name_every_yo_promotion(cands_for):
    """Notes report Y/O promotions whatever decided them (a rule or an
    override), kadoban and juryo demotions; nothing else about the rules."""
    def notes_for(target, **ov):
        cands = cands_for(target).assign(ozeki_run2=0)
        pred, _, _ = predict.predict(cands, cands["position_next"].to_numpy(dtype=float), [],
                                     parse(cands, **ov), 42)
        return predict.notes(pred, target)

    n = notes_for(202609)  # Aonishiki S2W 12-3, a demoted ozeki returning
    assert "Aonishiki: promoted to ozeki" in n and not any("yokozuna" in x for x in n)
    n = notes_for(202601)  # Aonishiki S1E 12-3, 34 wins over K/S with one M1 basho
    assert "Aonishiki: promoted to ozeki" in n
    n = notes_for(202507)  # Onosato O1E 14-1 after a 12-3 yusho
    assert "Onosato: promoted to yokozuna" in n
    n = notes_for(202511, classes=["Kotozakura=Y"])  # an override promotes too (Kotozakura O 9-5)
    assert "Kotozakura: promoted to yokozuna" in n
    n = notes_for(202511)
    assert not any("promoted" in x for x in n) and any(x.startswith("demoted to juryo") for x in n)
    assert all(x.split(": ")[1].startswith(("promoted", "kadoban", "ozeki run")) or x.startswith("demoted")
               for x in notes_for(202609) + notes_for(202507) + notes_for(202511))


def test_forced_claims_matches_resolver(trans):
    cands = trans[(trans["basho"] == 202609) & ~trans["dropped"]].reset_index(drop=True)
    claims = forced_claims(cands)
    assert all(v.dtype == bool and v.shape == (len(cands),) for v in claims.values())
    kk_s = (cands["rank_class"] == SEKIWAKE) & (cands["kk"] == 1)
    assert kk_s.any() and (claims["s"] | ~kk_s).all()
    # claims are honoured, and a block only grows past 2 when claims force it
    pred = resolve(cands, cands["position"].to_numpy(dtype=float), 42)
    m = cands.merge(pred, on="rikishi_id")
    for c, mask in ((SEKIWAKE, claims["s"]), (KOMUSUBI, claims["k"])):
        holders = set(cands.loc[mask, "rikishi_id"])
        assert (m.loc[m["rikishi_id"].isin(holders), "pred_class"] <= c).all()
        block = set(m.loc[m["pred_class"] == c, "rikishi_id"])
        assert len(block) == max(2, len(block & holders))


def _synthetic(invert=False):
    """Eight-row makuuchi: Y, O, then six maegashira in pred order. Base scores
    put rows 2-3 0.1 apart (a tight call) and the rest 2 apart; `invert` has
    the base order of that pair disagree with the resolved order."""
    n = 8
    pred = pd.DataFrame({
        "rikishi_id": np.arange(n), "shikona": [f"R{i}" for i in range(n)],
        "pred_pos": np.arange(n), "pred_class": [YOKOZUNA, OZEKI] + [MAEGASHIRA] * 6,
        "pred_number": [1, 1, 1, 1, 2, 2, 3, 3], "pred_side": [0, 0, 0, 1, 0, 1, 0, 1],
        "position": np.arange(n), "division": 0, "rank_class": MAEGASHIRA,
        "rank_number": 1, "side": 0, "wins": 8, "losses": 7, "absences": 0,
    })
    base = pd.Series([0.0, 1.0, 2.0, 2.1, 4.1, 6.1, 8.1, 10.1], index=pred["rikishi_id"])
    if invert:
        base[2], base[3] = 2.1, 2.0
    final = pd.Series(pred["pred_pos"].to_numpy(), index=pred["rikishi_id"])
    return pred, base, final


def test_confidence_signals_tiers():
    pred, base, final = _synthetic()
    pred.loc[6, "position"] = 16  # incumbent climbing 10 half-ranks
    sig = confidence.signals(pred, base, final)
    assert sig.index.equals(pred.index)
    assert (sig.loc[[0, 1], "gap"] == np.inf).all() and (sig.loc[[0, 1], "tier"] == "").all()
    assert sig.loc[[2, 3], "tight"].all() and (sig.loc[[2, 3], "tier"] == "?").all()
    assert (sig.loc[[4, 5, 6, 7], "tier"] == "").all() and (sig["spread"] == 0).all()
    assert sig["big_move"].tolist() == [False] * 6 + [True, False]
    assert sig.loc[6, "marker"] == "~" and sig.loc[6, "move"] == -10
    # two seed orders that swap rows 5 and 7: spread 2 for both, "??" without tightness
    s1 = pred[["rikishi_id", "pred_pos"]]
    s2 = s1.assign(pred_pos=s1["pred_pos"].replace({5: 7, 7: 5}))
    sig = confidence.signals(pred, base, final, [s1, s2])
    assert sig.loc[5, "spread"] == 2 and sig.loc[5, "tier"] == "??" and not sig.loc[5, "tight"]
    assert sig.loc[7, "tier"] == "??" and (sig.loc[[2, 3], "tier"] == "?").all()
    assert sig["marker"].str.len().max() <= 3


def test_confidence_review_hint_restores_base_order():
    pred, base, final = _synthetic(invert=True)
    sig = confidence.signals(pred, base, final)
    assert sig.loc[[2, 3], "inverted"].all() and sig.loc[2, "gap"] == pytest.approx(-0.1)
    items = confidence.review(pred, sig, base, final)
    assert len(items) == 1
    it = items[0]
    assert it["marker"] == "?" and it["range"] == "M1E-M1W" and it["members"] == ["R2", "R3"]
    assert ("above", "R3 > R2") in it["hints"]  # base-higher R3 named first
    assert "reranker reversed the base order (R3 below R2)" in it["text"]
    # a skipped (overridden) pair is not a model decision: no item
    assert confidence.review(pred, sig, base, final, skip={2}) == []


def test_claim_rates_reference(trans):
    rates = confidence.claim_rates(trans)
    assert rates.attrs["since"] == 199001
    s = rates[rates["claimed"] == SEKIWAKE]
    assert int(s["n"].sum()) == 18 and int(s["honoured"].sum()) == 18
    cell = rates.set_index(["claimed", "rank_class", "rank_number", "wins"]).loc[
        (KOMUSUBI, MAEGASHIRA, 1, 8)]
    assert int(cell["n"]) == 17 and int(cell["honoured"]) == 10
    txt = confidence.precedent(rates, KOMUSUBI, MAEGASHIRA, 1, 8)
    assert txt == "M1 claims with 8 wins needing a created slot were honoured 10 of 17 since 1990"


def test_explain_reconstructs_rerank_and_isolates_structural_shift(small_train, cands_for, tidy):
    """experiments.explain: the step-by-step reranker reconstruction reproduces
    GBMRerank.score (with and without twin units), and the miss decomposition
    charges an over-created komusubi slot to the structure, not to the rows below."""
    from experiments.explain import decompose, rerank_detail

    kw = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}
    for twin_unit, target in (("sk", 200401), ("all", 202609)):
        m = GBMRerank(**kw, twin_unit=twin_unit)
        m.fit(small_train)
        cands = cands_for(target)
        base = m.base_score(cands)
        d = rerank_detail(m, cands, base)
        assert np.array_equal(d["score"], m.score(cands))
        assert len(d["cluster_of"]) == len(cands) and (d["cluster_size"] >= 1).all()
        if twin_unit == "all":
            assert any(len(u) == 2 for u in d["units"])

    target = 202309
    cands = cands_for(target)
    actual = tidy[tidy["basho"] == target]
    truth = cands["position_next"].to_numpy(dtype=float)
    pred = resolve(cands, truth, 42, overrides={"count": {KOMUSUBI: int((actual["rank_class"] == KOMUSUBI).sum()) + 1}})
    pb = pred.rename(columns={c: f"{c}_base" for c in CELL + ["pred_pos"]})
    p = cands.assign(score=truth.argsort().argsort(), base=truth).merge(pred, on="rikishi_id").merge(pb, on="rikishi_id")
    p = decompose(p, actual)
    mak = p[p["class_next"] == MAEGASHIRA]
    extra = p[(p["pred_class"] == KOMUSUBI) & (p["class_next"] == MAEGASHIRA)]
    assert len(extra) == 1 and extra["stage"].iloc[0] == "structural"
    rest = mak.drop(extra.index)
    assert (rest["block_offset"] == -1).all()  # one extra label above every maegashira
    # the shifted labels push some make-koshi men past their ceiling, which the
    # resolver repairs by swapping neighbours: those rows are its misses, not cascades
    shifted, swapped = rest[rest["err"] == 0], rest[rest["err"] != 0]
    assert len(shifted) > len(swapped) > 0
    assert (shifted["cell_err"] == -1).all() and (shifted["local_err"] == 0).all()
    assert shifted["cascade"].all() and (shifted["stage"] == "cascade").all()
    assert not swapped["cascade"].any()
    assert ((swapped["stage"] == "resolver") | (swapped["hit"] & (swapped["stage"] == ""))).all()
    assert (p.loc[p["class_next"] <= SEKIWAKE, "block_offset"] == 0).all()
    assert p.loc[p["class_next"] <= SEKIWAKE, "hit"].all()


def test_rerank_offline_rebuilds_borda_from_pairs(small_train, cands_for):
    """experiments.rerank_offline: the reranker rebuilt from a frame's base
    scores and its pair table reproduces GBMRerank.score; Kemeny and Copeland
    are valid permutations; excluded men keep their base-score slot."""
    from experiments.explain import PAIR_GAP, pair_table, rerank_detail
    from experiments.rerank_offline import excluded, rerank

    kw = {**SMALL, "seed": 0, "near_ties": False, "pair": {"n_estimators": 10}}
    m = GBMRerank(**kw)
    m.fit(small_train)
    for target in (200401, 202309):
        cands = cands_for(target)
        base = m.base_score(cands)
        g = cands.assign(base=base, score=m.score(cands), cluster=0)
        pairs = pair_table(m, cands, base, rerank_detail(m, cands, base))
        assert pairs["gap"].max() <= PAIR_GAP
        p_of = dict(zip(zip(pairs["rid_i"], pairs["rid_j"]), pairs["p"]))
        assert np.array_equal(rerank(g, p_of, "borda"), g["score"].to_numpy())
        for how in ("kemeny", "copeland"):
            assert sorted(rerank(g, p_of, how).tolist()) == list(range(len(g)))
        ex = excluded(g)
        assert ex.sum() >= 4 and (g.loc[ex, "rank_class"] <= OZEKI).any()
        out = rerank(g, p_of, "borda", ex)
        assert sorted(out.tolist()) == list(range(len(g)))
        # excluded men hold the base-score slot they would have taken anyway
        by_base = np.lexsort((g["position"].to_numpy(), base))
        rank_base = np.empty(len(g)); rank_base[by_base] = np.arange(len(g))
        assert (out[ex] == rank_base[ex]).all()
