#!/usr/bin/env python3
"""Residual analysis for one model: where do exact-slot misses come from, and
are they systematically biased?

Usage: uv run python analyze.py [--model Ar] [--start 200401] [--end 202609]
                                [--seeds 0-2] [--set base.n_estimators=200]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from banzuke.build import SEKIWAKE, KOMUSUBI, MAEGASHIRA
from banzuke.harness import fingerprint, parse_seeds, parse_sets, run_backtest

PROCESSED = Path(__file__).parent / "data" / "processed"
SCRATCH = Path(__file__).parent / "results" / "scratch"
CLS = "YOSKMJ"


def cell(c, n, s):
    return f"{CLS[int(c)]}{int(n)}{'ew'[int(s)]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Ar")
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--end", type=int, default=None, help="default: latest basho")
    ap.add_argument("--seeds", default="0", metavar="SPEC",
                    help="bag replicates, e.g. 0-2 (default 0); the first drives the residual "
                         "sections, all of them the seed-spread section")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in backtest.py")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    kwargs = parse_sets(args.set)
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    end = args.end or int(tidy["basho"].max())
    fp = fingerprint(kwargs=kwargs, seeds=seeds)
    cache = SCRATCH / f"preds_{args.model}_{args.start}_{end}_{fp}.parquet"
    if cache.exists() and not args.fresh:
        preds = pd.read_parquet(cache)
    else:
        targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= end]
        _, preds = run_backtest([args.model], targets, trans, tidy, return_preds=True,
                                seeds=seeds, model_kwargs=kwargs, workers=args.workers)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(cache, index=False)

    p0 = preds[preds["seed"] == seeds[0]]
    # rows whose actual outcome is a makuuchi slot
    m = p0[p0["class_next"] <= MAEGASHIRA].copy()
    m["err"] = m["pred_pos"] - m["position_next"]  # + = predicted too low
    m["hit"] = ((m["pred_class"] == m["class_next"]) & (m["pred_number"] == m["number_next"])
                & (m["pred_side"] == m["side_next"]))
    miss = m[~m["hit"]]
    print(f"=== {args.model} {args.start}-{end}: {len(m)} makuuchi slots, "
          f"exact {m['hit'].mean():.3f} ===\n")

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

    # sanyaku slot counts: a wrong S/K block size shifts every slot below it
    sk = [SEKIWAKE, KOMUSUBI]
    pred_n = p0[p0["pred_class"].isin(sk)].groupby(["target", "pred_class"]).size()
    act_n = p0[p0["class_next"].isin(sk)].groupby(["target", "class_next"]).size()
    diff = (act_n.rename_axis(["target", "cls"]).sub(pred_n.rename_axis(["target", "cls"]),
                                                      fill_value=0).unstack(fill_value=0))
    hits = m.groupby("target")["hit"].sum()
    ok = (diff == 0).all(axis=1)
    print(f"\nsanyaku slot counts right in {ok.mean():.3f} of basho: exact_n "
          f"{hits[ok].mean():.1f} when right, {hits[~ok].mean():.1f} when wrong "
          f"({(diff > 0).any(axis=1).sum()} basho with slots the rules did not create, "
          f"{(diff < 0).any(axis=1).sum()} with slots the committee did not)")
    created = p0[p0["class_next"].isin(sk) & (p0["pred_class"] == MAEGASHIRA)]
    if len(created):
        print("  received a created slot while forecast in maegashira:")
        for r in created.sort_values("target").itertuples():
            rec = f"{int(r.wins)}-{int(r.losses)}" + (f"-{int(r.absences)}" if r.absences else "")
            print(f"    {r.target}  {r.shikona:14s} {cell(r.rank_class, r.rank_number, r.side):>4s}"
                  f" {rec:7s} -> {cell(r.class_next, r.number_next, r.side_next)}"
                  f" (forecast {cell(r.pred_class, r.pred_number, r.pred_side)})")

    # committee anchoring: inverted adjacent pairs vs prior order
    inv_prior, adj_prior = [], []
    for _, g in m.groupby("target"):
        g = g.sort_values("position_next")
        a, p, prior = (g[c].to_numpy() for c in ("position_next", "pred_pos", "position"))
        for i in range(len(g) - 1):
            if a[i + 1] - a[i] <= 1:
                (inv_prior if p[i] > p[i + 1] else adj_prior).append(prior[i] < prior[i + 1])
    print(f"\nadjacent actual pairs where model inverted the order: {len(inv_prior)}")
    print(f"  actual-higher had better prior rank: {np.mean(inv_prior):.3f}"
          f"  (non-inverted baseline: {np.mean(adj_prior):.3f})")

    if len(seeds) > 1:  # does seed disagreement predict error?
        piv = preds.pivot_table(index=["target", "rikishi_id"], columns="seed", values="pred_pos")
        mm = m.join((piv.max(axis=1) - piv.min(axis=1)).rename("spread"),
                    on=["target", "rikishi_id"])
        rho = spearmanr(mm["spread"], mm["err"].abs()).statistic
        print(f"\nseed spread (max-min position over seeds {list(seeds)}) vs |err|: "
              f"Spearman {rho:.3f}")
        for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 99)):
            sub = mm[mm["spread"].between(lo, hi)]
            if len(sub):
                print(f"  spread {lo}{'+' if hi > lo else ' '}: n={len(sub):5d} "
                      f"({len(sub) / len(mm):.3f})  |err|={sub['err'].abs().mean():.2f}  "
                      f"exact={sub['hit'].mean():.3f}")


if __name__ == "__main__":
    main()
