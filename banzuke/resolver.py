"""Turn a model's candidate ordering into a labeled banzuke.

Philosophy: rule knowledge lives in model features; this stage applies only
the near-inviolable structure. Hard-coded here:
- yokozuna are never demoted; Y/O membership changes follow the classic
  promotion/kadoban conventions (rare events, unlearnable from data)
- a sanyaku incumbent with kachi-koshi does not drop out of his class
- minimum 2 sekiwake and 2 komusubi; extra slots emerge when forced
- no-promotion ceiling for make-koshi S/K/M, including E/W
Vacancies, maegashira placement, and the juryo boundary follow the model's
ordering subject to these constraints.

Human overrides (see banzuke.overrides) may force class membership,
sanyaku counts, or exact cells. They take precedence over the
conventions above; each convention broken is reported via `warnings`.
"""
import numpy as np
import pandas as pd

from banzuke.build import YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO
from banzuke.overrides import OverrideError, fmt_slot

CLS_NAMES = ("yokozuna", "ozeki", "sekiwake", "komusubi", "maegashira", "juryo")


def block_slots(c, n, ne, nw):
    """Slot labels, in precedence order, for a block of n members.
    E/W layout derived empirically (see docs/EXPERIMENTS.md): strict E,W
    alternation, except an odd O/S/K block sends its last slot to the
    lighter column (ne/nw = E/W counts of the blocks above) so the
    banzuke sheet balances. Numbers count per side."""
    slots, n_e, n_w = [], 0, 0
    for j in range(n):
        if c in (OZEKI, SEKIWAKE, KOMUSUBI) and j == n - 1 and n % 2 == 1:
            side = 0 if ne <= nw else 1
        else:
            side = j % 2
        num = (n_e if side == 0 else n_w) + 1
        n_e, n_w = n_e + (side == 0), n_w + (side == 1)
        slots.append((num, side))
    return slots


def resolve(cands: pd.DataFrame, scores: np.ndarray, mak_size: int,
            overrides: dict | None = None, warnings: list | None = None) -> pd.DataFrame:
    """cands: transition rows at basho N. scores: lower = ranked higher.
    overrides: structural overrides (class/count/pins keyed by rikishi_id).
    warnings: optional list collecting broken-convention messages.
    Returns one row per candidate: rikishi_id, pred_class/number/side, pred_pos."""
    df = cands.copy()
    df["_score"] = scores
    df = df.sort_values(["_score", "position"], kind="stable").reset_index(drop=True)
    ov = overrides or {}
    warn = warnings if warnings is not None else []

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
    shik = df["shikona"].to_numpy()

    rid2i = {r: i for i, r in enumerate(df["rikishi_id"])}
    cassert = {rid2i[r]: c for r, c in (ov.get("class") or {}).items() if r in rid2i}
    pins = {rid2i[r]: s for r, s in (ov.get("pins") or {}).items() if r in rid2i}
    counts = ov.get("count") or {}
    prior = list(zip(cls, df["rank_number"], df["side"]))
    limited = np.isin(cls, (SEKIWAKE, KOMUSUBI, MAEGASHIRA)) & ~kk
    bounded = limited.copy()
    for i in idx:
        if i in pins or (i in cassert and cassert[i] < cls[i]):
            bounded[i] = False

    def eligible(i, c, slot):
        return not bounded[i] or (c, *slot) >= prior[i]

    def feasible(c, slots, members):
        """Reserve pins, then fit the most restrictive ceilings from the bottom."""
        reserved = {pins[i][1:] for i in members if i in pins}
        if not reserved.issubset(slots):
            return False
        free = sorted(set(slots) - reserved, reverse=True)
        unpinned = [i for i in members if i not in pins]
        unpinned.sort(key=lambda i: prior[i] if bounded[i] else (-1, 0, 0), reverse=True)
        for i, slot in zip(unpinned, free):
            if not eligible(i, c, slot):
                return False
        return len(unpinned) <= len(free)

    def away(i, c):
        """Asserted to some class other than c."""
        return cassert.get(i, c) != c

    # yokozuna: incumbents (never demoted) + rule-promoted ozeki
    y_promo = (cls == OZEKI) & yusho & (yusho1 | (junyusho1 & (w1 >= 12)))
    # ozeki: incumbents minus kadoban make-koshi, plus promotions/returns
    kadoban_out = (cls == OZEKI) & kadoban & ~kk
    o_stay = (cls == OZEKI) & ~kadoban_out & ~y_promo
    o_promo = (
        np.isin(cls, (SEKIWAKE, KOMUSUBI))
        & (wins >= 10)
        & (np.nan_to_num(run3) >= np.where(yusho, 32, 33))
    )
    o_return = demoted_ozeki & (wins >= 10)

    def rule_class(c, incumbent_mask, promo_mask):
        for i in idx[incumbent_mask]:
            if away(i, c) and cassert[i] > c:
                warn.append(f"convention broken: {CLS_NAMES[c]} {shik[i]} demoted by override")
        for i in idx[promo_mask & ~incumbent_mask]:
            if away(i, c):
                warn.append(f"override cancels rule promotion of {shik[i]} to {CLS_NAMES[c]}")
        members = [i for i in idx if incumbent_mask[i] and not away(i, c)]
        members += [i for i in idx if promo_mask[i] and not incumbent_mask[i]
                    and not away(i, c) and i not in members]
        members += [i for i in idx if cassert.get(i) == c and i not in members]
        return members

    y_members = rule_class(YOKOZUNA, cls == YOKOZUNA, y_promo)
    for i in idx[kadoban_out]:
        if cassert.get(i) == OZEKI:
            warn.append(f"convention broken: kadoban make-koshi ozeki {shik[i]} "
                        "retained by override")
    o_members = rule_class(OZEKI, o_stay, (o_promo | o_return) & ~o_stay)

    taken = set(y_members) | set(o_members)
    upper_blocks = [(YOKOZUNA, y_members), (OZEKI, o_members)]

    def next_slots(c, n):
        ne = nw = 0
        for pc, members in upper_blocks:
            slots = block_slots(pc, len(members), ne, nw)
            ne += sum(s[1] == 0 for s in slots)
            nw += sum(s[1] == 1 for s in slots)
        return block_slots(c, n, ne, nw)

    def fill_class(c, forced_mask):
        for i in idx:
            if (forced_mask[i] and i not in taken and away(i, c) and cassert[i] > c):
                warn.append(f"convention broken: {shik[i]} had a forced "
                            f"{CLS_NAMES[c]} claim, overridden")
        forced = [i for i in idx if i not in taken and forced_mask[i] and not away(i, c)]
        forced += [i for i in idx if i not in taken and cassert.get(i) == c
                   and i not in forced]
        target = counts.get(c)
        if target is None:
            target = max(2, len(forced))
        elif target < len(forced):
            raise OverrideError(
                f"--count {CLS_NAMES[c][0].upper()}={target} conflicts with "
                f"{len(forced)} forced/asserted members "
                f"({', '.join(shik[i] for i in forced)}); move someone with --class")
        if target < 2:
            warn.append(f"convention broken: fewer than 2 {CLS_NAMES[c]}")
        fills = [i for i in idx
                 if i not in taken and not forced_mask[i] and i not in cassert]
        slots = next_slots(c, target)
        if not feasible(c, slots, forced):
            raise OverrideError(f"forced {CLS_NAMES[c]} members/pins cannot fit "
                                "without a make-koshi promotion; adjust --class/--pin/--count")
        members = list(forced)
        for i in fills:
            if len(members) >= target:
                break
            if feasible(c, slots, members + [i]):
                members.append(i)
        if len(members) < target:
            raise OverrideError(f"cannot fill {target} {CLS_NAMES[c]} slots "
                                "without a make-koshi promotion")
        taken.update(members)
        upper_blocks.append((c, members))
        return members

    # komusubi with 11+ wins historically always get a sekiwake slot created
    s_members = fill_class(SEKIWAKE, kadoban_out | ((cls == SEKIWAKE) & kk)
                           | ((cls == KOMUSUBI) & (wins >= 11)))
    # upper maegashira whose scores land sanyaku >=85% of the time (1990+)
    # get komusubi slots created for them when the zone is full
    num = df["rank_number"].to_numpy()
    m_claim = (cls == MAEGASHIRA) & (
        ((num == 1) & (wins >= 8)) | ((num == 2) & (wins >= 11))
        | ((num == 3) & (wins >= 10)) | ((num == 4) & (wins >= 12))
        | ((num == 5) & (wins >= 13))
    )
    k_members = fill_class(KOMUSUBI, ((cls == KOMUSUBI) & kk) | m_claim)

    remaining = [i for i in idx if i not in taken]
    n_m = max(0, mak_size - len(y_members) - len(o_members)
              - len(s_members) - len(k_members))
    forced_m = [i for i in remaining if cassert.get(i) == MAEGASHIRA]
    if len(forced_m) > n_m:
        raise OverrideError(f"{len(forced_m)} rikishi asserted into maegashira "
                            f"but only {n_m} slots exist")
    m_set = set(forced_m)
    m_slots = next_slots(MAEGASHIRA, n_m)
    if not feasible(MAEGASHIRA, m_slots, forced_m):
        raise OverrideError("asserted maegashira members/pins cannot fit without "
                            "a make-koshi promotion; adjust --class/--pin/--mak-size")
    for i in remaining:
        if len(m_set) >= n_m:
            break
        if i not in cassert and feasible(MAEGASHIRA, m_slots, [*m_set, i]):
            m_set.add(i)
    m_members = [i for i in remaining if i in m_set]
    j_members = [i for i in remaining if i not in m_set]
    if len(m_members) < n_m:
        raise OverrideError(f"cannot fill {n_m} maegashira slots "
                            "without a make-koshi promotion")

    blocks = [
        (YOKOZUNA, y_members), (OZEKI, o_members), (SEKIWAKE, s_members),
        (KOMUSUBI, k_members), (MAEGASHIRA, m_members), (JURYO, j_members),
    ]
    out_idx, out_cls, out_num, out_side = [], [], [], []
    ne = nw = 0
    for c, members in blocks:
        slots = block_slots(c, len(members), ne, nw)
        assign = {}  # slot -> member
        for i in members:
            if i in pins:
                pc, pn, ps = pins[i]
                if (pn, ps) not in slots:
                    have = ", ".join(fmt_slot(c, *s) for s in slots) or "none"
                    raise OverrideError(
                        f"pin {shik[i]}={fmt_slot(pc, pn, ps)}: the {CLS_NAMES[c]} "
                        f"block has slots [{have}]; grow it with --count/--class "
                        f"or adjust --mak-size")
                assign[(pn, ps)] = i
        free_members = [i for i in members if i not in pins]
        free_slots = [s for s in slots if s not in assign]
        for slot in free_slots:
            i = next((i for i in free_members if eligible(i, c, slot)), None)
            if i is None:
                raise OverrideError(f"cannot fill {fmt_slot(c, *slot)} "
                                    "without a make-koshi promotion")
            assign[slot] = i
            free_members.remove(i)
        for s in slots:  # emit in precedence order
            i = assign[s]
            if limited[i] and (c, *s) < prior[i]:
                warn.append(f"convention broken: make-koshi {shik[i]} promoted "
                            f"from {fmt_slot(*prior[i])} to {fmt_slot(c, *s)} by override")
            out_idx.append(assign[s])
            out_cls.append(c)
            out_num.append(s[0])
            out_side.append(s[1])
        if c <= KOMUSUBI:
            ne += sum(s[1] == 0 for s in slots)
            nw += sum(s[1] == 1 for s in slots)

    pred = pd.DataFrame({
        "rikishi_id": df["rikishi_id"].to_numpy()[out_idx],
        "pred_class": out_cls,
        "pred_number": out_num,
        "pred_side": out_side,
    })
    pred["pred_pos"] = np.arange(len(pred))
    return pred
