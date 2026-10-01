#!/usr/bin/env python3
"""Run the model bake-off over a window of historical basho.

Per-(configuration, model, seed, basho) results are cached in results/scratch/
and reused on re-runs. The cache key covers the dataset, the source of every
module that affects results, the lockfile and the configuration, so editing
code, rebuilding data or changing a parameter invalidates it automatically;
--fresh forces recomputation.

Examples:
    uv run python backtest.py                        # all models, 2004+
    uv run python backtest.py --models Ar,Aq --seeds 0-1 --end 201911
    uv run python backtest.py --models Ar --set base.n_estimators=600 --baseline Ar

--set KEY=VALUE (repeatable) overrides a model option: dotted keys address
the LightGBM stages (base.*, pair.*), bare keys are model options (n_seeds,
gap, cluster_max, context, near_ties, half_life, ...). --baseline MODEL runs
that model's defaults alongside and pairs every row against them
(MODEL:label for another configuration present in the results).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from banzuke.harness import (DEFAULT_WORKERS, fingerprint, parse_seeds, parse_sets, run_backtest,
                             summarize)
from banzuke.models import MODELS

PROCESSED = Path(__file__).parent / "data" / "processed"
RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "scratch" / "backtest_cache.parquet"

# committee behavior drifts; screen/confirm are the tuning windows of
# docs/EXPERIMENTS.md protocol v2
WINDOWS = [("full", None, None), ("screen 2004-2019", 200401, 201911),
           ("confirm 2020+", 202001, None)]
SHOW = ["n", "exact_n", "gtb_points", "mae", "within1", "promo_f1", "demo_f1",
        "sanyaku_exact", "d_exact", "ci_exact", "p_exact", "wl", "d_mae", "ci_mae", "p_mae"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model names")
    ap.add_argument("--start", type=int, default=200401,
                    help="first target basho (default 200401: start of the 42-man, "
                         "post-kosho era; earlier committee behavior differs)")
    ap.add_argument("--end", type=int, default=None, help="last target basho (default: latest)")
    ap.add_argument("--seeds", default="0", metavar="SPEC",
                    help="bag replicates to fit, e.g. 0-1 (default 0); averaged per basho")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, see above")
    ap.add_argument("--baseline", default=None, metavar="LABEL",
                    help="configuration paired comparisons refer to (default: the leader)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--out", default=None, help="results file stem, e.g. results/dev")
    ap.add_argument("--summarize", default=None, metavar="CSVS",
                    help="skip running; summarize existing per-basho csv(s), comma-separated")
    ap.add_argument("--train-start", type=int, default=None, metavar="BASHO",
                    help="ignore training transitions before this basho "
                         "(e.g. 201001); default: all history since 1959")
    ap.add_argument("--fresh", action="store_true", help="recompute instead of using the cache")
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    if args.summarize:
        print_summaries(pd.concat([pd.read_csv(f) for f in args.summarize.split(",")]),
                        args.baseline)
        return

    names = args.models.split(",")
    unknown = set(names) - set(MODELS)
    if unknown:
        ap.error(f"unknown models: {unknown}; available: {list(MODELS)}")
    seeds = parse_seeds(args.seeds)
    configs = {",".join(args.set) or "base": parse_sets(args.set)}
    if args.baseline in MODELS or (args.baseline or "").endswith(":base"):
        configs.setdefault("base", {})

    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    end = args.end or int(tidy["basho"].max())
    targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= end]
    if not targets:
        ap.error(f"no target basho in {args.start}..{end}")
    if args.train_start:
        n_train = int(tidy["basho"].between(args.train_start, targets[0]).sum()) - 1
        if n_train < 30:
            print(f"warning: first target {targets[0]} has only {n_train} training "
                  f"basho with --train-start {args.train_start}", file=sys.stderr)

    store = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    if "seed" not in store:
        store = pd.DataFrame()  # older cache layout
    results = []
    for label, kwargs in configs.items():
        fp = fingerprint(kwargs=kwargs, train_start=args.train_start)
        cached = pd.DataFrame()
        if len(store) and not args.fresh:
            cached = store[(store["fp"] == fp) & store["model"].isin(names)
                           & store["seed"].isin(seeds) & store["basho"].isin(targets)]
            cached = cached.drop(columns="fp").assign(config=label)
            if len(cached):
                print(f"[{label}] reusing {len(cached)} cached rows", file=sys.stderr)
        skip = set(zip(cached["model"], cached["seed"], cached["basho"])) if len(cached) else None
        computed = run_backtest(names, targets, trans, tidy, skip=skip, seeds=seeds,
                                train_start=args.train_start, model_kwargs=kwargs,
                                config=label, workers=args.workers)
        if len(computed):
            store = pd.concat([store, computed.assign(fp=fp)], ignore_index=True)
        results += [f for f in (cached, computed) if len(f)]
    results = pd.concat(results, ignore_index=True)

    store = store.drop_duplicates(["fp", "model", "seed", "basho"], keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(CACHE, index=False)

    print_summaries(results, args.baseline)
    if args.out:
        RESULTS.mkdir(exist_ok=True)
        results.to_csv(f"{args.out}_per_basho.csv", index=False)
        pd.concat([summarize(results, args.baseline, s, u).assign(window=w)
                   for w, s, u in WINDOWS
                   if results["basho"].between(s or 0, u or 10**8).any()]
                  ).round(4).to_csv(f"{args.out}_summary.csv")
        print(f"\nwrote {args.out}_per_basho.csv, {args.out}_summary.csv")


def print_summaries(results, baseline=None):
    for label, since, until in WINDOWS:
        sub = results[(results["basho"] >= (since or 0)) & (results["basho"] <= (until or 10**8))]
        if not len(sub):
            continue
        s = summarize(results, baseline, since, until)
        print(f"\n=== {label} ({sub['basho'].nunique()} basho, {sub['seed'].nunique()} "
              f"seed(s); paired vs {s.attrs['baseline']}) ===")
        print(s[SHOW].round(3).to_string())


if __name__ == "__main__":
    main()
