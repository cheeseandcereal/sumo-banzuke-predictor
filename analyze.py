#!/usr/bin/env python3
"""Residual analysis for one model over the dev window: where do exact-slot
misses come from, and are they systematically biased?

Usage: uv run python analyze.py [--model Ar] [--start 200401] [--end 202311]
                                [--set base.n_estimators=200 ...]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from banzuke.build import KOMUSUBI, MAEGASHIRA
from banzuke.harness import fingerprint, parse_sets, run_backtest

PROCESSED = Path(__file__).parent / "data" / "processed"
SCRATCH = Path(__file__).parent / "results" / "scratch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Ar")
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--end", type=int, default=202311)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in backtest.py")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    kwargs = parse_sets(args.set)
    fp = fingerprint(json.dumps({"kwargs": kwargs, "seed": args.seed}, sort_keys=True))
    cache = SCRATCH / f"preds_{args.model}_{args.start}_{args.end}_{fp}.parquet"
    if cache.exists() and not args.fresh:
        preds = pd.read_parquet(cache)
    else:
        tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
        trans = pd.read_parquet(PROCESSED / "transitions.parquet")
        targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= args.end]
        _, preds = run_backtest([args.model], targets, trans, tidy, return_preds=True,
                                seeds=(args.seed,), model_kwargs=kwargs, workers=args.workers)
        preds.to_parquet(cache, index=False)

    # rows whose actual outcome is a makuuchi slot
    m = preds[preds["class_next"] <= MAEGASHIRA].copy()
    m["err"] = m["pred_pos"] - m["position_next"]  # + = predicted too low
    m["hit"] = (
        (m["pred_class"] == m["class_next"])
        & (m["pred_number"] == m["number_next"])
        & (m["pred_side"] == m["side_next"])
    )
    miss = m[~m["hit"]]
    print(f"=== {args.model} {args.start}-{args.end}: "
          f"{len(m)} makuuchi slots, exact {m['hit'].mean():.3f} ===\n")

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

    # sanyaku-count mismatches shift whole blocks below them
    sk = m[m["class_next"].isin([2, 3]) | m["pred_class"].isin([2, 3])]
    per_basho = []
    for t, g in m.groupby("target"):
        na = ((g["class_next"] == 2) | (g["class_next"] == 3)).sum()
        np_ = ((g["pred_class"] == 2) | (g["pred_class"] == 3)).sum()
        per_basho.append({"target": t, "count_ok": na == np_, "exact_n": g["hit"].sum()})
    pb = pd.DataFrame(per_basho)
    print(f"\nsanyaku slot-count correct: {pb['count_ok'].mean():.3f} of basho")
    print(f"  exact_n when count ok:   {pb.loc[pb['count_ok'], 'exact_n'].mean():.1f}")
    print(f"  exact_n when count off:  {pb.loc[~pb['count_ok'], 'exact_n'].mean():.1f}")

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


if __name__ == "__main__":
    main()
