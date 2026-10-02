"""Re-measure the committee conventions the resolver hard-codes.

Every resolver rule is an empirical regularity read off history once
(docs/EXPERIMENTS.md E1, E2, E5, E10, E16, E22). No rule can adapt before its
first exception, so this audit recounts each one over the whole record and
the recent past and names its last violation: drift becomes loud instead of
silent. A few regularities the resolver does not enforce are tracked too
("watch"). Cells read violations/cases.

    banzuke conventions      # also printed by `banzuke analyze`
"""
import numpy as np
import pandas as pd

from banzuke.ranks import KOMUSUBI, MAEGASHIRA, OZEKI, SEKIWAKE, YOKOZUNA
from banzuke.resolver import block_slots, forced_claims, rule_masks

LAYOUT = {YOKOZUNA: "layout: yokozuna block", OZEKI: "layout: ozeki block (odd -> lighter column)",
          SEKIWAKE: "layout: sekiwake block (odd -> lighter column)",
          KOMUSUBI: "layout: komusubi block (odd -> lighter column)",
          MAEGASHIRA: "layout: maegashira strict E/W alternation"}


def cases(trans: pd.DataFrame) -> pd.DataFrame:
    """One row per case a convention applies to: convention, where, basho, held."""
    out = []

    def add(name, where, basho, held):
        out.append(pd.DataFrame({"convention": name, "where": where,
                                 "basho": np.asarray(basho), "held": np.asarray(held, dtype=bool)}))

    t = trans[trans["class_next"].notna()].reset_index(drop=True)
    b = t["basho"].to_numpy()
    cls, nxt = t["rank_class"].to_numpy(), t["class_next"].to_numpy()
    num, wins = t["rank_number"].to_numpy(), t["wins"].to_numpy()
    kk = t["kk"].to_numpy() == 1
    yusho = t["yusho"].to_numpy() == 1
    claims = forced_claims(t)
    rm = rule_masks(t)
    y_promo, o_promo, o_return = rm["y_promo"], rm["o_promo"], rm["o_return"]
    # the old yokozuna rule without the "fought as ozeki" condition (E22 R7)
    y_fires = (cls == OZEKI) & yusho & ((t["yusho1"].to_numpy() == 1)
                                        | ((t["junyusho1"].to_numpy() == 1) & (t["w1"].to_numpy() >= 12)))

    def rule(name, applies, held):
        add(name, "resolver", b[applies], held[applies])

    rule("yokozuna never demoted", cls == YOKOZUNA, nxt == YOKOZUNA)
    rule("ozeki yusho after yusho / 12+ jun-yusho fought as ozeki -> yokozuna", y_promo, nxt == YOKOZUNA)
    rule("yusho / jun-yusho fought at sekiwake does not count for that rule", y_fires & ~y_promo, nxt > YOKOZUNA)
    rule("yokozuna promotions met that rule", (cls == OZEKI) & (nxt == YOKOZUNA), y_promo)
    rule("kadoban ozeki with make-koshi drops out", claims["kadoban_out"], nxt > OZEKI)
    rule("other ozeki keep the rank", (cls == OZEKI) & ~claims["kadoban_out"], nxt <= OZEKI)
    rule("S/K 10+ wins, 33-win run (32 with yusho) over three sanyaku basho -> ozeki", rm["o_run"], nxt == OZEKI)
    rule("S/K 12+ wins, 33-win run with one M1-M3 basho -> ozeki", rm["o_run_m"] & ~rm["o_run"], nxt == OZEKI)
    rule("ozeki promotions met those rules", np.isin(cls, (SEKIWAKE, KOMUSUBI)) & (nxt == OZEKI), o_promo)
    rule("demoted ozeki with 10+ wins returns", o_return, nxt == OZEKI)
    rule("kachi-koshi sekiwake stays sanyaku", (cls == SEKIWAKE) & kk, nxt <= SEKIWAKE)
    rule("kachi-koshi komusubi stays sanyaku", (cls == KOMUSUBI) & kk, nxt <= KOMUSUBI)
    rule("komusubi 11+ wins -> sekiwake", (cls == KOMUSUBI) & (wins >= 11), nxt <= SEKIWAKE)
    m1 = (cls == MAEGASHIRA) & (num == 1) & (wins >= 8)
    rule("M1 8+ wins -> sanyaku", m1, nxt <= KOMUSUBI)
    rule("M2 11+ / M3 10+ / M4 12+ / M5 13+ -> sanyaku", claims["m_claim"] & ~m1, nxt <= KOMUSUBI)
    # make-koshi exits (E20/E22 R2, R3)
    rule("sekiwake with 7 wins and two other sekiwake candidates -> komusubi", rm["k_from_s"], nxt == KOMUSUBI)
    rule("sekiwake with <= 6 wins leaves sanyaku", (cls == SEKIWAKE) & (wins <= 6), nxt >= MAEGASHIRA)
    rule("komusubi with <= 6 wins leaves sanyaku", (cls == KOMUSUBI) & (wins <= 6), nxt >= MAEGASHIRA)
    k1e = (num == 1) & (t["side"].to_numpy() == 0)
    rule("komusubi with 7 wins below K1E leaves sanyaku", (cls == KOMUSUBI) & (wins == 7) & ~k1e, nxt >= MAEGASHIRA)

    def cell(c, n, s):  # lexicographic (class, number, side) as one integer
        return c.astype(int) * 1000 + n.astype(int) * 10 + s.astype(int)

    prior = cell(cls, num, t["side"].to_numpy())
    after = cell(nxt, t["number_next"].to_numpy(), t["side_next"].to_numpy())
    rule("make-koshi S/K/M never above the prior cell",
         np.isin(cls, (SEKIWAKE, KOMUSUBI, MAEGASHIRA)) & ~kk, after >= prior)

    # regularities found in the 2019+ case analysis (docs/EXPERIMENTS.md E19/E20)
    # that the resolver does not enforce: "watch" rows so the first exception is loud
    def watch(name, applies, held):
        add(name, "watch", b[applies], held[applies])

    watch("S/K/M with 0 wins and 8+ absences drops 17+ cells (COVID exemptions violate)",
          np.isin(cls, (SEKIWAKE, KOMUSUBI, MAEGASHIRA)) & (wins == 0) & (t["absences"].to_numpy() >= 8),
          t["delta"].to_numpy() >= 17)

    # order within a block (E22 R14, R4, R12): pairs of one basho
    def same_basho_pairs(mask_a, mask_b=None, keep=None):
        cols = ["basho", "rikishi_id", "wins", "yusho", "fought", "position", "position_next"]
        a = t.loc[mask_a, cols]
        bb = t.loc[mask_a if mask_b is None else mask_b, cols]
        p = a.merge(bb, on="basho", suffixes=("_a", "_b"))
        p = p[p["rikishi_id_a"] != p["rikishi_id_b"]]
        return p[keep(p)] if keep is not None else p

    t["fought"] = ((t["wins"] + t["losses"]) > 0).astype(int)
    for c, name in ((YOKOZUNA, "yokozuna who stay are ordered by wins, yusho, fought, prior position"),
                    (OZEKI, "ozeki who stay are ordered by wins, yusho, fought, prior position")):
        key = lambda p, s: list(zip(-p[f"wins_{s}"], -p[f"yusho_{s}"], -p[f"fought_{s}"], p[f"position_{s}"]))
        p = same_basho_pairs((cls == c) & (nxt == c))
        p = p[[ka < kb for ka, kb in zip(key(p, "a"), key(p, "b"))]]
        add(name, "resolver", p["basho"], p["position_next_a"] < p["position_next_b"])
    p = same_basho_pairs(claims["kadoban_out"] & (nxt == SEKIWAKE), ~claims["kadoban_out"] & (nxt == SEKIWAKE))
    add("demoted ozeki is the bottom sekiwake", "resolver", p["basho"], p["position_next_a"] > p["position_next_b"])
    p = same_basho_pairs((cls > KOMUSUBI) & (nxt == KOMUSUBI), (cls == KOMUSUBI) & kk & (nxt == KOMUSUBI))
    add("komusubi newcomers rank below kachi-koshi komusubi incumbents", "resolver",
        p["basho"], p["position_next_a"] > p["position_next_b"])

    tw = t[t["side"] == 0].merge(t[t["side"] == 1], suffixes=("_e", "_w"),
                                 on=["basho", "rank_class", "rank_number", "wins", "losses", "absences"])
    mj, sk = tw["rank_class"] >= MAEGASHIRA, tw["rank_class"].isin((SEKIWAKE, KOMUSUBI))
    add("identical-record M/J E/W twins keep their order", "resolver",
        tw.loc[mj, "basho"], tw.loc[mj, "position_next_e"] < tw.loc[mj, "position_next_w"])
    add("identical-record S/K E/W twins stay adjacent", "Ar twin_unit",
        tw.loc[sk, "basho"], (tw.loc[sk, "position_next_w"] - tw.loc[sk, "position_next_e"]).abs() == 1)

    for basho, g in trans.groupby("basho"):  # per banzuke, latest included
        g = g.sort_values("position")
        n_cls = g["rank_class"].value_counts()
        add("at least 2 sekiwake and 2 komusubi", "resolver", [basho],
            [n_cls.get(SEKIWAKE, 0) >= 2 and n_cls.get(KOMUSUBI, 0) >= 2])
        ne = nw = 0  # actual E/W counts of the blocks above
        for c, name in LAYOUT.items():
            blk = g[g["rank_class"] == c]
            actual = list(zip(blk["rank_number"], blk["side"]))
            add(name, "resolver", [basho], [actual == block_slots(c, len(blk), ne, nw)])
            ne, nw = ne + int((blk["side"] == 0).sum()), nw + int((blk["side"] == 1).sum())
    return pd.concat(out, ignore_index=True)


def table(c: pd.DataFrame, recent=30) -> pd.DataFrame:
    """Violations/cases per window and the last violating basho. Flags:
    '!' a rule clean since 2004 was violated inside the last `recent` basho;
    '~' a rule with known exceptions is violated more often there than over
    2004+ as a whole."""
    bashos = sorted(c["basho"].unique())
    cut = bashos[-recent]
    windows = [("all", bashos[0]), ("2004+", 200401), ("last 60", bashos[-60]), (f"last {recent}", cut)]
    rows = []
    for name in c["convention"].drop_duplicates():
        s = c[c["convention"] == name]
        row = {"convention": name, "where": s["where"].iloc[0]}
        for w, since in windows:
            q = s[s["basho"] >= since]
            row[w] = f"{int((~q['held']).sum())}/{len(q)}"
        viol = s.loc[~s["held"], "basho"]
        row["last violation"] = str(int(viol.max())) if len(viol) else ""
        modern, late = s[s["basho"] >= 200401], s[s["basho"] >= cut]
        v_modern, v_late = int((~modern["held"]).sum()), int((~late["held"]).sum())
        if v_late and v_late == v_modern:
            row["flag"] = "!"
        elif v_late and v_late / len(late) > v_modern / len(modern):
            row["flag"] = "~"
        else:
            row["flag"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def report(trans: pd.DataFrame) -> None:
    tab = table(cases(trans))
    print("=== convention audit: violations/cases ('!' first exception since 2004 is recent,"
          " '~' recent rate above the 2004+ rate) ===")
    print(tab.to_string(index=False))
