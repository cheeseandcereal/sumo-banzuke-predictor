#!/usr/bin/env python3
"""Residual analysis for one model over a target window (default 2004 through
the latest basho): where do exact-slot misses come from, and are they
systematically biased?

Usage: uv run python analyze.py [--model Ar] [--start 200401] [--end 202609]
                                [--seeds 0-2] [--set base.n_estimators=200 ...]

Sections (the first seed drives everything but the last):
  1. miss decomposition, signed bias by group, committee anchoring
  2. sanyaku slot-count taxonomy: predicted vs actual S/K block sizes per
     basho, exact slots by error bucket
  3. under-creation: who got the S/K slots the resolver's rules did not
     create; what lowering the maegashira claim thresholds by one win would
     cover and what false claims it would add; threshold precision from the
     full transitions table
  4. over-creation: who the resolver forced into S/K that did not get there,
     by rule
  5. within-block ordering: inverted pairs inside the predicted S/K blocks
     and the exact slots a perfect within-block order would recover
  6. seed-spread calibration (>1 seed): does seed disagreement predict error?
"""
import argparse
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from banzuke.build import OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA
from banzuke.harness import fingerprint, parse_seeds, parse_sets, run_backtest

PROCESSED = Path(__file__).parent / "data" / "processed"
SCRATCH = Path(__file__).parent / "results" / "scratch"
RECENT = 202001  # the confirm window (docs/EXPERIMENTS.md)
CLS = "YOSKMJ"
BUCKETS = ["ok", "under_S", "under_K", "under_both", "over_S", "over_K", "over_both", "mixed"]
# resolver.py: maegashira win totals that force a komusubi slot, and the
# proposal of lowering each by one win (M1's 8 is already the kachi-koshi floor)
M_CLAIM = {1: 8, 2: 11, 3: 10, 4: 12, 5: 13}
M_CLAIM_LOWER = {2: 10, 3: 9, 4: 11, 5: 12}


def rank_str(c, n, s) -> str:
    return f"{CLS[int(c)]}{int(n)}{'ew'[int(s)]}"


def record_str(r) -> str:
    rec = f"{int(r.wins)}-{int(r.losses)}"
    return rec + (f"-{int(r.absences)}" if r.absences else "")


def exact_hit(df: pd.DataFrame) -> pd.Series:
    return ((df["pred_class"] == df["class_next"]) & (df["pred_number"] == df["number_next"])
            & (df["pred_side"] == df["side_next"]))


def forced_rule(r):
    """(class, rule) the resolver forces candidate row r into, or None when the
    placement is left to the model's ordering. Mirrors resolver.resolve; the
    sekiwake rules are checked first because fill_class(SEKIWAKE) runs first."""
    kk = r.kk == 1
    num = int(r.rank_number)
    if r.rank_class == OZEKI and r.kadoban == 1 and not kk:
        return SEKIWAKE, "kadoban make-koshi ozeki -> S"
    if r.rank_class == SEKIWAKE and kk:
        return SEKIWAKE, "kachi-koshi S incumbent keeps S"
    if r.rank_class == KOMUSUBI and r.wins >= 11:
        return SEKIWAKE, "K with 11+ wins -> forced S"
    if r.rank_class == KOMUSUBI and kk:
        return KOMUSUBI, "kachi-koshi K incumbent keeps K"
    if r.rank_class == MAEGASHIRA and r.wins >= M_CLAIM.get(num, 99):
        return KOMUSUBI, f"M{num} {M_CLAIM[num]}+ claim -> K"
    return None


def is_forced_into(r, c) -> bool:
    fr = forced_rule(r)
    return fr is not None and fr[0] == c


def block_rule(r, c) -> str:
    """Why r sits in the predicted block of class c: the forcing rule, or a fill."""
    return forced_rule(r)[1] if is_forced_into(r, c) else "fill (model order)"


# ---------------------------------------------------------------- 1. residuals
def residuals(m: pd.DataFrame):
    miss = m[~m["hit"]]
    side_only = (miss["pred_class"] == miss["class_next"]) & (
        miss["pred_number"] == miss["number_next"])
    near = ~side_only & (miss["err"].abs() <= 1)
    print("miss decomposition:")
    print(f"  side-only (E/W flip):      {side_only.mean():.3f}")
    print(f"  near (|err| <= 1 pos):     {near.mean():.3f}")
    print(f"  far  (|err| >  1 pos):     {(~side_only & ~near).mean():.3f}")

    print("\nsigned bias (mean err, + = under-promoted) by group:")
    groups = {
        "juryo promotee": (m["division"] == 1),
        "yusho winner": m["yusho"] == 1,
        "sansho winner": m["sansho"] > 0,
        "big win (11+)": m["wins"] >= 11,
        "big loss (<=4)": (m["wins"] <= 4) & (m["absences"] < 8),
        "mostly absent": m["absences"] >= 8,
        "kk streak >=3": m["kk_streak"] >= 3,
        "sanyaku now": m["rank_class"] <= KOMUSUBI,
        "M1-M5": (m["rank_class"] == MAEGASHIRA) & (m["rank_number"] <= 5),
        "M11+": (m["rank_class"] == MAEGASHIRA) & (m["rank_number"] >= 11),
    }
    for name, mask in groups.items():
        sub = m[mask]
        print(f"  {name:16s} n={len(sub):5d}  err={sub['err'].mean():+.2f}"
              f"  |err|={sub['err'].abs().mean():.2f}  exact={sub['hit'].mean():.3f}")

    # committee anchoring: inverted adjacent pairs vs prior order
    inv_prior, adj_prior = [], []
    for t, g in m.groupby("target"):
        g = g.sort_values("position_next")
        a = g["position_next"].to_numpy()
        p = g["pred_pos"].to_numpy()
        prior = g["position"].to_numpy()
        for i in range(len(g) - 1):
            if a[i + 1] - a[i] <= 1:
                (inv_prior if p[i] > p[i + 1] else adj_prior).append(prior[i] < prior[i + 1])
    print(f"\nadjacent actual pairs where model inverted the order: {len(inv_prior)}")
    print(f"  actual-higher had better prior rank: {np.mean(inv_prior):.3f}"
          f"  (non-inverted baseline: {np.mean(adj_prior):.3f})")


# ------------------------------------------------------- 2. slot-count taxonomy
def bucket(dS: int, dK: int) -> str:
    """dS, dK = actual - predicted block size."""
    if dS == 0 and dK == 0:
        return "ok"
    if dS >= 0 and dK >= 0:
        return "under_both" if dS and dK else ("under_S" if dS else "under_K")
    if dS <= 0 and dK <= 0:
        return "over_both" if dS and dK else ("over_S" if dS else "over_K")
    return "mixed"


def count_taxonomy(p0: pd.DataFrame, m: pd.DataFrame, tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per target: predicted S/K block sizes (all candidates), actual
    sizes read from the target banzuke, error bucket, exact slots."""
    hits = m.groupby("target")["hit"].sum()
    rows = []
    for t, g in p0.groupby("target"):
        act = tidy[(tidy["basho"] == t) & (tidy["division"] == 0)]
        nS_p, nK_p = int((g["pred_class"] == SEKIWAKE).sum()), int((g["pred_class"] == KOMUSUBI).sum())
        nS_a, nK_a = int((act["rank_class"] == SEKIWAKE).sum()), int((act["rank_class"] == KOMUSUBI).sum())
        rows.append({"target": t, "nS_pred": nS_p, "nS_act": nS_a, "nK_pred": nK_p,
                     "nK_act": nK_a, "dS": nS_a - nS_p, "dK": nK_a - nK_p,
                     "exact_n": int(hits.get(t, 0))})
    pb = pd.DataFrame(rows)
    pb["bucket"] = [bucket(a, b) for a, b in zip(pb["dS"], pb["dK"])]
    return pb


def print_taxonomy(pb: pd.DataFrame, label: str):
    if not len(pb):
        return
    ok = pb.loc[pb["bucket"] == "ok", "exact_n"].mean()
    off = pb["bucket"] != "ok"
    sizes = Counter((pb["dS"].abs() + pb["dK"].abs())[off])
    print(f"\n  {label}: {len(pb)} basho, count wrong in {off.sum()} ({off.mean():.3f}); "
          "total block-size error " + ", ".join(
              f"{k} in {v} basho" for k, v in sorted(sizes.items())))
    print(f"  {'bucket':11s} {'n':>4s} {'share':>6s} {'exact_n':>8s} {'vs ok':>6s}")

    def line(name, sub):
        if not len(sub):
            print(f"  {name:11s} {0:4d} {0:6.3f} {'-':>8s} {'-':>6s}")
            return
        mean = sub["exact_n"].mean()
        print(f"  {name:11s} {len(sub):4d} {len(sub) / len(pb):6.3f} {mean:8.1f} {mean - ok:+6.1f}")

    for b in BUCKETS:
        line(b, pb[pb["bucket"] == b])
    line("any under", pb[pb["bucket"].str.startswith("under")])
    line("any over", pb[pb["bucket"].str.startswith("over")])
    line("any error", pb[off])
    lost = (ok - pb.loc[off, "exact_n"]).sum()
    print(f"  exact slots below the ok mean, summed over count-error basho: {lost:.0f} "
          f"({lost / len(pb):.2f} per basho over the window)")


# ---------------------------------------------------- 3. under-creation detail
def missed_claimants(p0: pd.DataFrame, pb: pd.DataFrame, trans: pd.DataFrame):
    tgt = pb[pb["bucket"].str.startswith("under") | (pb["bucket"] == "mixed")]
    bmap = tgt.set_index("target")["bucket"]
    mc = p0[p0["target"].isin(bmap.index) & p0["class_next"].isin([SEKIWAKE, KOMUSUBI])
            & (p0["pred_class"] == MAEGASHIRA)].sort_values(["target", "position_next"])
    print(f"\n=== under-creation: actual S/K the model left in maegashira "
          f"({len(mc)} rikishi in {mc['target'].nunique()} of {len(tgt)} under_*/mixed basho) ===")
    print(f"  {'target':6s} {'bucket':10s} {'shikona':14s} {'prior':5s} {'record':8s} "
          f"{'actual':6s} {'pred':6s}")
    for r in mc.itertuples():
        print(f"  {r.target:6d} {bmap[r.target]:10s} {r.shikona:14s} "
              f"{rank_str(r.rank_class, r.rank_number, r.side):5s} {record_str(r):8s} "
              f"{rank_str(r.class_next, r.number_next, r.side_next):6s} "
              f"{rank_str(r.pred_class, r.pred_number, r.pred_side):6s}")

    if not len(mc):
        print("  (none)")
    else:
        print("\n  missed claimants by prior rank (rows) and wins (columns):")
        order = sorted(set(zip(mc["rank_class"], mc["rank_number"])))
        prior = [f"{CLS[int(c)]}{int(n)}" for c, n in zip(mc["rank_class"], mc["rank_number"])]
        tab = pd.crosstab(pd.Series(prior, index=mc.index, name="prior"), mc["wins"])
        tab = tab.reindex([f"{CLS[int(c)]}{int(n)}" for c, n in order])
        tab["total"] = tab.sum(axis=1)
        print("  " + tab.to_string().replace("\n", "\n  "))
        by_class = Counter(CLS[int(c)] for c in mc["rank_class"])
        print("  by prior class: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_class.items())))

    lower = mc["rank_number"].map(M_CLAIM_LOWER)
    covered = (mc["rank_class"] == MAEGASHIRA) & (mc["wins"] >= lower.fillna(99))
    print(f"\n  covered by lowering the claims one win (M2 10+, M3 9+, M4 11+, M5 12+): "
          f"{covered.sum()} of {len(mc)} missed claimants")
    print(f"  the new marginal claimants across all {p0['target'].nunique()} targets "
          f"(seed-0 forecast), by what the rule would change:")
    print(f"  {'claim':7s} {'n':>4s} {'->S/K':>6s} {'false':>6s} | {'fix':>4s} {'new wrong slot':>15s} "
          f"{'no change':>10s}")
    tot = Counter()
    for n, w in M_CLAIM_LOWER.items():
        sub = p0[(p0["rank_class"] == MAEGASHIRA) & (p0["rank_number"] == n) & (p0["wins"] == w)]
        reached = sub["class_next"] <= KOMUSUBI
        pred_m = sub["pred_class"] == MAEGASHIRA
        row = {"n": len(sub), "reached": int(reached.sum()), "false": int((~reached).sum()),
               "fix": int((reached & pred_m).sum()), "wrong": int((~reached & pred_m).sum()),
               "same": int((~pred_m).sum())}
        tot.update(row)
        print(f"  M{n} ={w:<3d} {row['n']:4d} {row['reached']:6d} {row['false']:6d} | "
              f"{row['fix']:4d} {row['wrong']:15d} {row['same']:10d}")
    print(f"  {'total':7s} {tot['n']:4d} {tot['reached']:6d} {tot['false']:6d} | "
          f"{tot['fix']:4d} {tot['wrong']:15d} {tot['same']:10d}")
    print("  (fix: reached S/K but forecast M; new wrong slot: forecast M and stayed M;"
          " no change: forecast already S/K)")

    t = trans[(trans["rank_class"] == MAEGASHIRA) & trans["position_next"].notna()]
    print("\n  precision of claim thresholds from all transitions (share reaching S/K or "
          "better next basho); * = proposed:")
    print(f"  {'rule':11s} {'2004+ n':>8s} {'prec':>6s} {'2020+ n':>8s} {'prec':>6s}")
    for n in range(1, 6):
        specs = [(f"M{n} {M_CLAIM[n]}+", t["wins"] >= M_CLAIM[n]),
                 (f"M{n} ={M_CLAIM[n]}", t["wins"] == M_CLAIM[n])]
        if n in M_CLAIM_LOWER:
            specs += [(f"M{n} {M_CLAIM_LOWER[n]}+ *", t["wins"] >= M_CLAIM_LOWER[n]),
                      (f"M{n} ={M_CLAIM_LOWER[n]} *", t["wins"] == M_CLAIM_LOWER[n])]
        for name, mask in specs:
            line = f"  {name:11s}"
            for since in (200401, RECENT):
                sub = t[mask & (t["rank_number"] == n) & (t["next_basho"] >= since)]
                prec = (sub["class_next"] <= KOMUSUBI).mean() if len(sub) else np.nan
                line += f" {len(sub):8d} {prec:6.3f}"
            print(line)


# ----------------------------------------------------- 4. over-creation detail
def over_placements(p0: pd.DataFrame, pb: pd.DataFrame):
    tgt = pb[pb["bucket"].str.startswith("over") | (pb["bucket"] == "mixed")]
    bmap = tgt.set_index("target")["bucket"]
    op = p0[p0["target"].isin(bmap.index) & p0["pred_class"].isin([SEKIWAKE, KOMUSUBI])
            & (p0["class_next"] != p0["pred_class"])].sort_values(["target", "pred_pos"])
    print(f"\n=== over-creation: predicted S/K whose actual class differs "
          f"({len(op)} rikishi in {op['target'].nunique()} of {len(tgt)} over_*/mixed basho) ===")
    print(f"  {'target':6s} {'bucket':10s} {'shikona':14s} {'prior':5s} {'record':8s} "
          f"{'pred':6s} {'actual':6s} rule")
    rules, actual = [], []
    for r in op.itertuples():
        rule = block_rule(r, r.pred_class)
        rules.append(rule)
        actual.append("Y/O" if r.class_next <= OZEKI else CLS[int(r.class_next)])
        print(f"  {r.target:6d} {bmap[r.target]:10s} {r.shikona:14s} "
              f"{rank_str(r.rank_class, r.rank_number, r.side):5s} {record_str(r):8s} "
              f"{rank_str(r.pred_class, r.pred_number, r.pred_side):6s} "
              f"{rank_str(r.class_next, r.number_next, r.side_next):6s} {rule}")
    if len(op):
        tab = pd.crosstab(pd.Series(rules, name="rule"), pd.Series(actual, name="actual"))
        tab = tab.reindex(columns=[c for c in ("Y/O", "S", "K", "M", "J") if c in tab.columns])
        tab["total"] = tab.sum(axis=1)
        print("\n  by rule that placed him, vs actual class:")
        print("  " + tab.sort_values("total", ascending=False).to_string().replace("\n", "\n  "))
        mj = Counter(r for r, a in zip(rules, actual) if a in ("M", "J"))
        print("  actual maegashira/juryo only: " + ", ".join(
            f"{k}: {v}" for k, v in mj.most_common()))


# --------------------------------------------------- 5. within-block ordering
def block_ordering(p0: pd.DataFrame):
    per, comp, comp_gain = [], Counter(), Counter()
    fill_above = 0
    for t, g in p0.groupby("target"):
        row = {"target": t, "inv": 0, "gain": 0, "set_ok": 0, "order_wrong": 0, "layout": 0}
        actual_sets = {c: set(g.loc[g["class_next"] == c, "rikishi_id"]) for c in (SEKIWAKE, KOMUSUBI)}
        for c in (SEKIWAKE, KOMUSUBI):
            blk = g[g["pred_class"] == c].sort_values("pred_pos")
            pos = blk.loc[blk["class_next"] == c, "position_next"].to_numpy()
            inv = sum(1 for i, j in combinations(range(len(pos)), 2) if pos[i] > pos[j])
            row["inv"] += inv
            if set(blk["rikishi_id"]) != actual_sets[c]:
                continue
            row["set_ok"] += 1
            wrong = int((~exact_hit(blk)).sum())
            if not wrong:
                continue
            if inv == 0:  # same order, different E/W layout of the block
                row["layout"] += wrong
                continue
            row["order_wrong"] += 1
            row["gain"] += wrong
            forced = np.array([is_forced_into(r, c) for r in blk.itertuples()])
            kind = "all forced" if forced.all() else "all fill" if not forced.any() else "forced+fill"
            comp[kind] += 1
            comp_gain[kind] += wrong
            if kind == "forced+fill":
                pn = blk["position_next"].to_numpy()
                if pn[~forced].min() < pn[forced].max():
                    fill_above += 1
        per.append(row)
    pb = pd.DataFrame(per)
    print("\n=== within-block ordering of the predicted S and K blocks ===")
    for label, sub in (("all", pb), (f"{RECENT}+", pb[pb["target"] >= RECENT])):
        if not len(sub):
            continue
        print(f"  {label}: {len(sub)} basho; inverted pairs (both actually in the class): "
              f"{sub['inv'].sum()} total, {sub['inv'].mean():.2f}/basho, "
              f"{(sub['inv'] > 0).sum()} basho with any")
        print(f"    blocks with exactly the right members: {sub['set_ok'].sum()} of {2 * len(sub)}; "
              f"of these wrong order: {sub['order_wrong'].sum()} -> exact slots recoverable: "
              f"{sub['gain'].sum()} ({sub['gain'].sum() / len(sub):.2f}/basho); "
              f"right order but different E/W layout: {sub['layout'].sum()} slots")
    print("  set-right/order-wrong blocks by composition (blocks / slots): " + ", ".join(
        f"{k}: {v} / {comp_gain[k]}" for k, v in comp.most_common()) + (
            f"; forced+fill blocks where a fill actually ranked above a forced member: {fill_above}"
            if comp["forced+fill"] else ""))


# ------------------------------------------------- 6. seed-spread calibration
def seed_spread(preds: pd.DataFrame, m: pd.DataFrame, seeds: tuple):
    piv = preds.pivot_table(index=["target", "rikishi_id"], columns="seed", values="pred_pos")
    spread = (piv.max(axis=1) - piv.min(axis=1)).rename("spread")
    mm = m.join(spread, on=["target", "rikishi_id"])
    mm["aerr"] = (mm["pred_pos"] - mm["position_next"]).abs()
    rho, p = spearmanr(mm["spread"], mm["aerr"])
    per_seed = preds[preds["class_next"] <= MAEGASHIRA].assign(hit=lambda d: exact_hit(d))
    per_seed = per_seed.groupby("seed")["hit"].mean()
    print(f"\n=== seed-spread calibration: seeds {list(seeds)}; spread = max-min pred_pos "
          f"across seeds, err = |pred_pos(seed {seeds[0]}) - actual| ===")
    print("  exact rate per seed: " + ", ".join(f"{s}: {v:.3f}" for s, v in per_seed.items()))
    print(f"  Spearman corr(spread, err) = {rho:.3f} (p = {p:.1e}, n = {len(mm)} makuuchi slots)")
    print(f"  {'spread':7s} {'n':>6s} {'share':>6s} {'mean err':>9s} {'exact':>6s}")
    for lo, hi, name in ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10**6, "5+")):
        sub = mm[(mm["spread"] >= lo) & (mm["spread"] <= hi)]
        if len(sub):
            print(f"  {name:7s} {len(sub):6d} {len(sub) / len(mm):6.3f} {sub['aerr'].mean():9.2f} "
                  f"{sub['hit'].mean():6.3f}")
        else:
            print(f"  {name:7s} {0:6d} {0:6.3f} {'-':>9s} {'-':>6s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Ar")
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--end", type=int, default=None, help="default: latest basho")
    ap.add_argument("--seeds", default="0", metavar="SPEC",
                    help="bag replicates to fit, e.g. 0-2 (default 0); the first drives the "
                         "residual sections, all of them the seed-spread section")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in backtest.py")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="target-level worker processes")
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    kwargs = parse_sets(args.set)
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    args.end = args.end or int(tidy["basho"].max())
    fp = fingerprint(json.dumps({"kwargs": kwargs, "seeds": list(seeds)}, sort_keys=True))
    cache = SCRATCH / f"preds_{args.model}_{args.start}_{args.end}_{fp}.parquet"
    if cache.exists() and not args.fresh:
        preds = pd.read_parquet(cache)
    else:
        targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= args.end]
        _, preds = run_backtest([args.model], targets, trans, tidy, return_preds=True,
                                seeds=tuple(seeds), model_kwargs=kwargs, workers=args.workers)
        preds.to_parquet(cache, index=False)

    p0 = preds[preds["seed"] == seeds[0]].copy()
    # rows whose actual outcome is a makuuchi slot
    m = p0[p0["class_next"] <= MAEGASHIRA].copy()
    m["err"] = m["pred_pos"] - m["position_next"]  # + = predicted too low
    m["hit"] = exact_hit(m)
    print(f"=== {args.model} {args.start}-{args.end} seed {seeds[0]}: "
          f"{len(m)} makuuchi slots, exact {m['hit'].mean():.3f} ===\n")
    residuals(m)

    # sanyaku-count mismatches shift whole blocks below them
    pb = count_taxonomy(p0, m, tidy)
    print("\n=== sanyaku slot-count taxonomy (actual - predicted block sizes) ===")
    print_taxonomy(pb, "all")
    print_taxonomy(pb[pb["target"] >= RECENT], f"{RECENT}+")
    off = pb[pb["bucket"] != "ok"]
    items = [f"{r.target}" + (f" S{r.dS:+d}" if r.dS else "") + (f" K{r.dK:+d}" if r.dK else "")
             for r in off.itertuples()]
    print("  count-error basho:")
    for i in range(0, len(items), 6):
        print("    " + "  ".join(items[i:i + 6]))

    missed_claimants(p0, pb, trans)
    over_placements(p0, pb)
    block_ordering(p0)
    if len(seeds) > 1:
        seed_spread(preds, m, seeds)


if __name__ == "__main__":
    main()
