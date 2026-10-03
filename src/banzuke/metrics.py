"""Score a predicted banzuke against the actual one."""
import pandas as pd
from scipy.stats import kendalltau

from banzuke.ranks import KOMUSUBI, MAEGASHIRA, SEKIWAKE


def _set_f1(pred: set, actual: set) -> float:
    if not pred and not actual:
        return 1.0
    return 2 * len(pred & actual) / (len(pred) + len(actual))


def evaluate(pred: pd.DataFrame, cands: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """pred: resolver output. cands: candidate rows at N. actual: tidy rows at N+1."""
    prev_div = cands.set_index("rikishi_id")["division"]
    all_mak = actual[actual["division"] == 0].merge(pred, on="rikishi_id", how="left")
    # rikishi absent from the previous banzuke (e.g. Sokokurai's 2013 court
    # reinstatement) are unpredictable: automatic misses, excluded from mae/tau
    mak = all_mak[all_mak["pred_pos"].notna()]

    right_rank = (mak["pred_class"] == mak["rank_class"]) & (
        mak["pred_number"] == mak["rank_number"])
    exact = right_rank & (mak["pred_side"] == mak["side"])
    hits = right_rank & ~exact  # GTB "hit": right rank, wrong side
    abs_err = (mak["pred_pos"] - mak["position"]).abs()
    mae = abs_err.mean()
    tau = kendalltau(mak["pred_pos"], mak["position"]).statistic

    pred_mak = set(pred.loc[pred["pred_class"] <= MAEGASHIRA, "rikishi_id"])
    actual_mak = set(mak["rikishi_id"])
    was_juryo = set(prev_div[prev_div == 1].index)
    was_mak = set(prev_div[prev_div == 0].index)
    promo_f1 = _set_f1(pred_mak & was_juryo, actual_mak & was_juryo)
    demo_f1 = _set_f1(was_mak - pred_mak, was_mak - actual_mak)

    sk = mak[mak["rank_class"].isin([SEKIWAKE, KOMUSUBI])]
    sanyaku_acc = (sk["pred_class"] == sk["rank_class"]).mean() if len(sk) else 1.0
    sets_match = all(
        set(mak.loc[mak["rank_class"] == c, "rikishi_id"])
        == set(mak.loc[mak["pred_class"] == c, "rikishi_id"])
        for c in (SEKIWAKE, KOMUSUBI)
    )

    return {
        "exact_n": int(exact.sum()),
        "exact": exact.sum() / len(all_mak),
        "gtb_points": int(2 * exact.sum() + hits.sum()),  # dichne.com scoring
        "within1": (abs_err <= 1).sum() / len(all_mak),
        "mae": mae,
        "tau": tau,
        "promo_f1": promo_f1,
        "demo_f1": demo_f1,
        "sanyaku_acc": sanyaku_acc,
        "sanyaku_exact": float(sets_match),
    }
