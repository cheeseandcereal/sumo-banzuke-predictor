"""banzuke/resolver.py: E/W layout rules and reproduction of real banzuke
from the true next order."""
import numpy as np
import pandas as pd
import pytest

from banzuke.build import JURYO, KOMUSUBI, MAEGASHIRA, OZEKI, SEKIWAKE, YOKOZUNA
from banzuke.resolver import block_slots, resolve
from conftest import slot_agreement

# targets where the resolver's conventions coincide with the committee's
# choices; each spans a different era / structural situation:
# 200401 makuuchi grows 40->42, 201105 shrinks 42->41 (post-scandal),
# 201307 Sokokurai's court reinstatement (in actual, not in cands),
# 202309 created S2E slot (E2), 202401 and 202609 recent.
TRUE_ORDER_TARGETS = [200401, 201105, 201307, 202309, 202401, 202609]


# --- block_slots: E1 rules --------------------------------------------------

def test_maegashira_strict_alternation_even():
    assert block_slots(MAEGASHIRA, 6, 0, 0) == [
        (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]


@pytest.mark.parametrize("ne,nw", [(0, 0), (0, 10), (10, 0), (3, 3)])
def test_maegashira_odd_block_ignores_column_balance(ne, nw):
    # strict E,W alternation for maegashira: the last of an odd block is
    # always East, whatever the columns above look like
    assert block_slots(MAEGASHIRA, 5, ne, nw) == [
        (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]


def test_juryo_and_yokozuna_strictly_alternate():
    # only O/S/K use the lighter-column rule
    assert block_slots(JURYO, 3, 0, 5) == [(1, 0), (1, 1), (2, 0)]
    assert block_slots(YOKOZUNA, 3, 0, 5) == [(1, 0), (1, 1), (2, 0)]


def test_odd_ozeki_block_goes_to_lighter_column_doc_example():
    # docs/EXPERIMENTS.md E1: "Y1 O3 lays out Y1E / O1E O1W O2W"
    y = block_slots(YOKOZUNA, 1, 0, 0)
    assert y == [(1, 0)]
    ne, nw = sum(s == 0 for _, s in y), sum(s == 1 for _, s in y)
    assert block_slots(OZEKI, 3, ne, nw) == [(1, 0), (1, 1), (2, 1)]


@pytest.mark.parametrize("cls", [OZEKI, SEKIWAKE, KOMUSUBI])
def test_odd_sanyaku_block_last_slot_to_lighter_column(cls):
    # heavier East above -> extra slot West; heavier West above -> East
    assert block_slots(cls, 3, 2, 1) == [(1, 0), (1, 1), (2, 1)]
    assert block_slots(cls, 3, 1, 2) == [(1, 0), (1, 1), (2, 0)]
    assert block_slots(cls, 1, 1, 0) == [(1, 1)]
    assert block_slots(cls, 1, 0, 1) == [(1, 0)]
    # balanced columns above: East (ne <= nw)
    assert block_slots(cls, 1, 0, 0) == [(1, 0)]
    assert block_slots(cls, 3, 4, 4) == [(1, 0), (1, 1), (2, 0)]


@pytest.mark.parametrize("cls", [OZEKI, SEKIWAKE, KOMUSUBI])
def test_even_sanyaku_block_alternates_regardless_of_balance(cls):
    assert block_slots(cls, 2, 5, 0) == [(1, 0), (1, 1)]
    assert block_slots(cls, 4, 0, 5) == [(1, 0), (1, 1), (2, 0), (2, 1)]


def test_numbering_counts_per_side():
    # E,W,W -> 1E, 1W, 2W (not 1E, 1W, 3W): numbers count per side
    slots = block_slots(OZEKI, 3, 1, 0)
    assert slots == [(1, 0), (1, 1), (2, 1)]
    # E,W,E,W,E -> per-side numbers 1,1,2,2,3
    assert [n for n, _ in block_slots(MAEGASHIRA, 5, 0, 0)] == [1, 1, 2, 2, 3]
    # a lone West slot is still number 1 on its side
    assert block_slots(KOMUSUBI, 1, 3, 0) == [(1, 1)]


def test_empty_block():
    assert block_slots(SEKIWAKE, 0, 0, 0) == []


# --- resolve on the true next order ------------------------------------------

@pytest.mark.parametrize("target", TRUE_ORDER_TARGETS)
def test_true_order_reproduces_actual_banzuke(resolve_true_order, target):
    cands, actual, pred = resolve_true_order(target)
    hit, n = slot_agreement(pred, actual)
    print(f"\n{target}: resolver on true order reproduces {hit}/{n} makuuchi slots "
          f"({hit / n:.3f})")
    assert hit / n >= 0.85


@pytest.mark.parametrize("target", TRUE_ORDER_TARGETS)
def test_true_order_output_structure(resolve_true_order, target):
    cands, actual, pred = resolve_true_order(target)
    mak_size = int((actual["division"] == 0).sum())
    # one row per candidate, pred_pos a permutation in emission order
    assert len(pred) == len(cands)
    assert set(pred["rikishi_id"]) == set(cands["rikishi_id"])
    assert pred["pred_pos"].tolist() == list(range(len(cands)))
    # exactly mak_size makuuchi rows, classes non-decreasing down the sheet
    assert int((pred["pred_class"] < JURYO).sum()) == mak_size
    assert (np.diff(pred["pred_class"].to_numpy()) >= 0).all()
    # sanyaku minimums
    assert (pred["pred_class"] == SEKIWAKE).sum() >= 2
    assert (pred["pred_class"] == KOMUSUBI).sum() >= 2
    # yokozuna are never demoted
    yoko = cands.loc[cands["rank_class"] == YOKOZUNA, "rikishi_id"]
    assert (pred.set_index("rikishi_id").loc[yoko, "pred_class"] == YOKOZUNA).all()
    # every (class, number, side) cell is unique
    assert not pred.duplicated(["pred_class", "pred_number", "pred_side"]).any()


def test_true_order_perfect_ordering_when_membership_matches(resolve_true_order):
    # in the 200401 case the resolver reproduces every makuuchi slot; the
    # joint positions agree through makuuchi, and the relative order agrees
    # for the whole sheet (juryo positions themselves differ only because
    # makushita promotees are not candidates)
    cands, actual, pred = resolve_true_order(200401)
    m = cands.merge(pred, on="rikishi_id").sort_values("pred_pos")
    mak = m[m["pred_class"] < JURYO]
    assert (mak["pred_pos"] == mak["position_next"]).all()
    assert (np.diff(m["position_next"].to_numpy()) > 0).all()


def test_true_order_201911_known_sanyaku_count_deviation(resolve_true_order):
    """201911 is a documented limit of the conventions, not an ordering
    error: the committee created a 4th komusubi slot for Asanoyama (M2W,
    10-5), below the resolver's M2 claim threshold of 11 wins
    (resolver.py m_claim). The resolver therefore lays out 3 komusubi and the
    labels of everything from K2 down differ, while the joint order is exact
    and the juryo boundary is right."""
    cands, actual, pred = resolve_true_order(201911)
    hit, n = slot_agreement(pred, actual)
    print(f"\n201911: {hit}/{n} exact slots (3 vs 4 komusubi)")
    assert (pred["pred_class"] == KOMUSUBI).sum() == 3
    assert (actual["rank_class"] == KOMUSUBI).sum() == 4
    mak = actual[actual["division"] == 0].merge(pred, on="rikishi_id")
    assert len(mak) == n == 42
    assert (mak["pred_pos"] == mak["position"]).all()  # order is exact
    asanoyama = mak[mak["shikona"] == "Asanoyama"].iloc[0]
    assert (asanoyama["rank_class"], asanoyama["rank_number"], asanoyama["side"]) == (KOMUSUBI, 2, 1)
    assert (asanoyama["pred_class"], asanoyama["pred_number"], asanoyama["pred_side"]) == (MAEGASHIRA, 1, 0)
    exact = ((mak["pred_class"] == mak["rank_class"])
             & (mak["pred_number"] == mak["rank_number"])
             & (mak["pred_side"] == mak["side"]))
    # Y2 + O3 + S2 + K1E/K1W are right, everything from position 9 down is off
    assert hit == 9
    assert (exact == (mak["position"] < 9)).all()


def test_resolve_respects_make_koshi_ceiling(resolve_true_order):
    # no-promotion ceiling: a make-koshi S/K/M never lands above his prior
    # cell (class, number, side), for any of the reproduced targets
    for target in TRUE_ORDER_TARGETS:
        cands, actual, pred = resolve_true_order(target)
        m = cands.merge(pred, on="rikishi_id")
        mk = m[m["rank_class"].isin([SEKIWAKE, KOMUSUBI, MAEGASHIRA]) & (m["kk"] == 0)]
        prior = list(zip(mk["rank_class"], mk["rank_number"], mk["side"]))
        now = list(zip(mk["pred_class"], mk["pred_number"], mk["pred_side"]))
        assert all(p <= q for p, q in zip(prior, now)), target


def test_true_order_agreement_distribution_2004_plus(make_cands, tidy, bashos):
    """Resolver on the true order for every 42-man-era target: most sheets
    are reproduced exactly. The sub-0.85 cases are sanyaku-count
    differences (the committee created or withheld an S/K slot the
    resolver's claim rules do not), with two exceptions: 200607 (the Y
    promotion rule fires for Hakuho's 14-1 yusho after a 13-2 jun-yusho,
    but the committee waited one more basho) and 200809 (M8E left vacant
    after Wakanoho's dismissal, a mid-sheet gap the layout cannot express)."""
    from banzuke.metrics import evaluate

    rows = []
    for target in bashos:
        if target < 200401:
            continue
        cands = make_cands(target)
        actual = tidy[tidy["basho"] == target]
        mak_size = int((actual["division"] == 0).sum())
        pred = resolve(cands, cands["position_next"].to_numpy(dtype=float), mak_size)
        ev = evaluate(pred, cands, actual)
        rows.append((target, ev["exact"], ev["sanyaku_exact"], ev["promo_f1"], ev["demo_f1"]))
    r = pd.DataFrame(rows, columns=["target", "exact", "sanyaku_exact", "promo_f1", "demo_f1"])
    low = r[r["exact"] < 0.85]
    print(f"\n{len(r)} targets: median exact {r['exact'].median():.3f}, mean "
          f"{r['exact'].mean():.3f}, >=0.85 on {(r['exact'] >= 0.85).sum()}; "
          f"below 0.85: {low['target'].tolist()}")
    assert len(r) == 135
    assert r["exact"].median() == 1.0
    assert (r["exact"] >= 0.85).mean() >= 0.75
    assert (r["exact"] == 1.0).mean() >= 0.5
    # sub-threshold sheets are sanyaku-count disagreements, bar two
    assert set(low.loc[low["sanyaku_exact"] == 1.0, "target"]) <= {200607, 200809}
    # the juryo boundary is almost always right given the order
    assert ((r["promo_f1"] == 1.0) & (r["demo_f1"] == 1.0)).mean() >= 0.95


def test_forced_claimants_precede_fills_within_class_block(make_cands):
    """Characterization (possibly unintended, see resolver.py fill_class:
    `members = list(forced)` then fills appended, and the slot loop that
    assigns cells in `members` order): inside an S/K block a *forced*
    claimant is laid out ahead of a non-forced fill even when the scores
    rank the fill higher. 200605 -> 200607: Kisenosato (M1E 8-7, forced K
    claim, true position 9) gets K1E over Asasekiryu (M2E 10-5, a fill,
    true position 8), whereas the committee did K1E Asasekiryu / K1W
    Kisenosato. Making Asasekiryu a claimant too (11 wins) restores score
    order, showing the mechanism."""
    cands = make_cands(200607)
    scores = cands["position_next"].to_numpy(dtype=float)
    cells = resolve(cands, scores, 42).merge(cands[["rikishi_id", "shikona"]], on="rikishi_id")
    cells = cells.set_index("shikona")[["pred_class", "pred_number", "pred_side"]]
    assert cells.loc["Asasekiryu"].tolist() == [KOMUSUBI, 1, 1]  # actual: K1E
    assert cells.loc["Kisenosato"].tolist() == [KOMUSUBI, 1, 0]  # actual: K1W
    bumped = cands.copy()
    bumped.loc[bumped["shikona"] == "Asasekiryu", "wins"] = 11
    cells2 = resolve(bumped, scores, 42).merge(bumped[["rikishi_id", "shikona"]], on="rikishi_id")
    cells2 = cells2.set_index("shikona")[["pred_class", "pred_number", "pred_side"]]
    assert cells2.loc["Asasekiryu"].tolist() == [KOMUSUBI, 1, 0]
    assert cells2.loc["Kisenosato"].tolist() == [KOMUSUBI, 1, 1]


def test_resolve_scores_are_only_an_ordering(cands_200401):
    # any strictly monotone transform of the scores gives the same sheet
    scores = cands_200401["position_next"].to_numpy(dtype=float)
    a = resolve(cands_200401, scores, 42)
    b = resolve(cands_200401, 10 * scores + 3, 42)
    pd.testing.assert_frame_equal(a, b)
