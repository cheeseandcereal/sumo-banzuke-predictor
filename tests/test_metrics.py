"""banzuke/metrics.py: evaluate() against the resolver's reproduction of
real banzuke, a perfect prediction, and set-F1 edge cases."""
import numpy as np
import pandas as pd
import pytest

from banzuke.build import JURYO
from banzuke.metrics import _set_f1, evaluate
from conftest import slot_agreement
from test_resolver import TRUE_ORDER_TARGETS


def test_set_f1_edges():
    assert _set_f1(set(), set()) == 1.0
    assert _set_f1({1}, set()) == 0.0
    assert _set_f1({1, 2}, {1, 2}) == 1.0
    assert _set_f1({1, 2}, {2, 3}) == pytest.approx(0.5)
    assert _set_f1({1}, {1, 2, 3}) == pytest.approx(0.5)


@pytest.mark.parametrize("target", TRUE_ORDER_TARGETS)
def test_evaluate_true_order_resolution(resolve_true_order, target):
    cands, actual, pred = resolve_true_order(target)
    ev = evaluate(pred, cands, actual)
    hit, n = slot_agreement(pred, actual)
    print(f"\n{target}: exact={ev['exact']:.3f} ({ev['exact_n']}/{ev['n_slots']}) "
          f"promo_f1={ev['promo_f1']:.3f} demo_f1={ev['demo_f1']:.3f} "
          f"tau={ev['tau']:.3f} mae={ev['mae']:.3f}")
    assert ev["n_slots"] == n == int((actual["division"] == 0).sum())
    assert ev["exact_n"] == hit
    assert ev["exact"] == pytest.approx(hit / n)
    newcomers = set(actual.loc[actual["division"] == 0, "rikishi_id"]) - set(cands["rikishi_id"])
    if not newcomers:
        assert ev["promo_f1"] >= 0.9
        assert ev["demo_f1"] >= 0.9
    else:
        # a non-candidate on the next banzuke takes a slot the resolver must
        # hand to the next candidate in line: one unavoidable false promotion
        # (2a/(2a+1) with a actual promotions), see the 201307 test below
        a = len(set(actual.loc[actual["division"] == 0, "rikishi_id"])
                & set(cands.loc[cands["division"] == 1, "rikishi_id"]))
        assert ev["promo_f1"] == pytest.approx(2 * a / (2 * a + 1))
        assert ev["demo_f1"] == 1.0
    # the true order was fed in, so the ordering metrics are (near) perfect
    assert ev["tau"] > 0.99
    assert ev["within1"] >= 0.9
    assert ev["gtb_points"] >= 2 * ev["exact_n"]
    assert 0 <= ev["sanyaku_acc"] <= 1 and ev["sanyaku_exact"] in (0.0, 1.0)


def test_evaluate_true_order_membership_exact_when_no_newcomer(resolve_true_order):
    # when everybody on the next banzuke was a candidate and the resolver
    # agrees on the sanyaku block, boundary membership is reproduced exactly
    for target in (200401, 202309, 202401, 202609):
        cands, actual, pred = resolve_true_order(target)
        ev = evaluate(pred, cands, actual)
        assert ev["promo_f1"] == 1.0 and ev["demo_f1"] == 1.0, target
        assert ev["mae"] == 0 and ev["tau"] == 1.0 and ev["exact"] == 1.0, target
        assert ev["gtb_points"] == 2 * ev["n_slots"], target


def test_evaluate_201307_newcomer_is_an_automatic_miss(resolve_true_order):
    # Sokokurai was reinstated by court order onto the 201307 banzuke without
    # being on 201305's: not a candidate, so an unpredictable slot that is
    # counted in n_slots (a miss) but excluded from mae/tau
    cands, actual, pred = resolve_true_order(201307)
    mak = actual[actual["division"] == 0]
    newcomers = set(mak["rikishi_id"]) - set(cands["rikishi_id"])
    assert newcomers and actual.set_index("rikishi_id").loc[list(newcomers), "shikona"].tolist() == ["Sokokurai"]
    ev = evaluate(pred, cands, actual)
    assert ev["n_slots"] == 42 and ev["exact_n"] == 39
    assert ev["tau"] == 1.0  # order of everyone else is right
    # the two men below him sit one position higher in the prediction
    assert ev["mae"] == pytest.approx(2 / 41)
    # 4 juryo candidates were promoted; the resolver also promotes a 5th
    # (Takanoiwa) into the slot Sokokurai received
    assert ev["promo_f1"] == pytest.approx(8 / 9) and ev["demo_f1"] == 1.0


def test_evaluate_perfect_prediction(make_cands, tidy):
    target = 202401
    cands = make_cands(target)
    actual = tidy[tidy["basho"] == target]
    perfect = actual.merge(cands[["rikishi_id"]], on="rikishi_id").sort_values("position")
    pred = pd.DataFrame({
        "rikishi_id": perfect["rikishi_id"].to_numpy(),
        "pred_class": perfect["rank_class"].to_numpy(),
        "pred_number": perfect["rank_number"].to_numpy(),
        "pred_side": perfect["side"].to_numpy(),
        "pred_pos": perfect["position"].to_numpy(),
    })
    ev = evaluate(pred, cands, actual)
    assert ev["exact"] == 1.0 and ev["exact_n"] == ev["n_slots"] == 42
    assert ev["gtb_points"] == 84 and ev["within1"] == 1.0
    assert ev["mae"] == 0 and ev["tau"] == 1.0
    assert ev["promo_f1"] == ev["demo_f1"] == 1.0
    assert ev["sanyaku_acc"] == 1.0 and ev["sanyaku_exact"] == 1.0


def test_evaluate_shifted_prediction(make_cands, tidy):
    # push the whole order down by one slot: every makuuchi cell wrong,
    # everyone off by exactly one position (mae 1, within1 1, tau 1)
    target = 202401
    cands = make_cands(target)
    actual = tidy[tidy["basho"] == target]
    perfect = actual.merge(cands[["rikishi_id"]], on="rikishi_id").sort_values("position")
    rids = np.roll(perfect["rikishi_id"].to_numpy(), 1)
    pred = pd.DataFrame({
        "rikishi_id": rids,
        "pred_class": perfect["rank_class"].to_numpy(),
        "pred_number": perfect["rank_number"].to_numpy(),
        "pred_side": perfect["side"].to_numpy(),
        "pred_pos": perfect["position"].to_numpy(),
    })
    ev = evaluate(pred, cands, actual)
    assert ev["exact_n"] == 0
    # right rank wrong side does not occur either: each cell holds the man
    # from the cell above; GTB hits only where E/W pairs share a number
    assert ev["gtb_points"] == int(((perfect["rank_class"].to_numpy()[1:] == perfect["rank_class"].to_numpy()[:-1])
                                   & (perfect["rank_number"].to_numpy()[1:] == perfect["rank_number"].to_numpy()[:-1])
                                   & (perfect["position"].to_numpy()[1:] < 42)).sum())
    assert ev["within1"] == 1.0  # everybody is off by exactly one position
    assert ev["mae"] == pytest.approx(1.0)
    assert ev["tau"] == 1.0  # a uniform shift preserves the relative order
    # the man rolled from the bottom of the sheet into Y1E is either a wrong
    # promotion (was juryo) or a missed demotion (was makuuchi)
    assert min(ev["promo_f1"], ev["demo_f1"]) < 1.0
    assert (pred["pred_class"] < JURYO).sum() == 42
