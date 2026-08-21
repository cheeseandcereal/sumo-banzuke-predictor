"""Turn a model's candidate ordering into a labeled banzuke.

Philosophy: rule knowledge lives in model features; this stage applies only
the near-inviolable structure. Hard-coded here:
- yokozuna are never demoted; Y/O membership changes follow the classic
  promotion/kadoban conventions (rare events, unlearnable from data)
- a sanyaku incumbent with kachi-koshi does not drop out of his class
- minimum 2 sekiwake and 2 komusubi; extra slots emerge when forced
Everything else (who fills vacancies, all of maegashira, the juryo
boundary) comes purely from the model's ordering.
"""
import numpy as np
import pandas as pd

from banzuke.build import YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO


def resolve(cands: pd.DataFrame, scores: np.ndarray, mak_size: int) -> pd.DataFrame:
    """cands: transition rows at basho N. scores: lower = ranked higher.
    Returns one row per candidate: rikishi_id, pred_class/number/side, pred_pos."""
    df = cands.copy()
    df["_score"] = scores
    df = df.sort_values(["_score", "position"], kind="stable").reset_index(drop=True)

    idx = np.arange(len(df))
    cls = df["rank_class"].to_numpy()
    kk = df["kk"].to_numpy() == 1
    wins = df["wins"].to_numpy()
    yusho = df["yusho"].to_numpy() == 1
    yusho1 = df["yusho1"].to_numpy() == 1
    junyusho1 = df["junyusho1"].to_numpy() == 1
    w1 = df["w1"].to_numpy()
    kadoban = df["kadoban"].to_numpy() == 1
    demoted_ozeki = df["demoted_ozeki"].to_numpy() == 1
    run3 = df["ozeki_run3"].to_numpy()

    # yokozuna: incumbents (never demoted) + rule-promoted ozeki
    y_promo = (cls == OZEKI) & yusho & (yusho1 | (junyusho1 & (w1 >= 12)))
    y_members = list(idx[cls == YOKOZUNA]) + list(idx[y_promo])

    # ozeki: incumbents minus kadoban make-koshi, plus promotions/returns
    kadoban_out = (cls == OZEKI) & kadoban & ~kk
    o_stay = (cls == OZEKI) & ~kadoban_out & ~y_promo
    o_promo = (
        np.isin(cls, (SEKIWAKE, KOMUSUBI))
        & (wins >= 10)
        & (np.nan_to_num(run3) >= np.where(yusho, 32, 33))
    )
    o_return = demoted_ozeki & (wins >= 10)
    o_members = list(idx[o_stay]) + list(idx[(o_promo | o_return) & ~o_stay])

    taken = set(y_members) | set(o_members)

    def fill_class(forced_mask, minimum):
        forced = [i for i in idx if i not in taken and forced_mask[i]]
        count = max(minimum, len(forced))
        fills = [i for i in idx if i not in taken and not forced_mask[i]]
        members = forced + fills[: count - len(forced)]
        taken.update(members)
        return members

    # komusubi with 11+ wins historically always get a sekiwake slot created
    s_members = fill_class(kadoban_out | ((cls == SEKIWAKE) & kk)
                           | ((cls == KOMUSUBI) & (wins >= 11)), 2)
    k_members = fill_class((cls == KOMUSUBI) & kk, 2)

    remaining = [i for i in idx if i not in taken]
    n_m = max(0, mak_size - len(y_members) - len(o_members) - len(s_members) - len(k_members))
    blocks = [
        (YOKOZUNA, y_members), (OZEKI, o_members), (SEKIWAKE, s_members),
        (KOMUSUBI, k_members), (MAEGASHIRA, remaining[:n_m]), (JURYO, remaining[n_m:]),
    ]

    # E/W layout, derived empirically (see docs/EXPERIMENTS.md): strict E,W
    # alternation, except an odd O/S/K block sends its last member to the
    # lighter column so the banzuke sheet balances. Numbers count per side.
    out_idx, out_cls, out_num, out_side = [], [], [], []
    ne = nw = 0
    for c, members in blocks:
        n_e = n_w = 0
        for j, i in enumerate(members):
            if (c in (OZEKI, SEKIWAKE, KOMUSUBI) and j == len(members) - 1
                    and len(members) % 2 == 1):
                side = 0 if ne <= nw else 1
            else:
                side = j % 2
            num = (n_e if side == 0 else n_w) + 1
            n_e, n_w = n_e + (side == 0), n_w + (side == 1)
            out_idx.append(i)
            out_cls.append(c)
            out_num.append(num)
            out_side.append(side)
        if c <= KOMUSUBI:
            ne, nw = ne + n_e, nw + n_w
    pred = pd.DataFrame({
        "rikishi_id": df["rikishi_id"].to_numpy()[out_idx],
        "pred_class": out_cls,
        "pred_number": out_num,
        "pred_side": out_side,
    })
    pred["pred_pos"] = np.arange(len(pred))
    return pred
