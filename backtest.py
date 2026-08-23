#!/usr/bin/env python3
"""Run the model bake-off over a window of historical basho.

Per-(model, basho) results are cached in results/scratch/ and reused on
re-runs. The cache key covers the dataset content and the source of every
module that affects results, so editing code or rebuilding data
invalidates it automatically; --fresh forces recomputation.

Examples:
    uv run python backtest.py                        # all models, dev window
    uv run python backtest.py --models R,A --start 202001 --end 202311
"""
import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import pandas as pd

from banzuke import features, metrics, models, resolver
from banzuke.harness import run_backtest, summarize
from banzuke.models import MODELS

PROCESSED = Path(__file__).parent / "data" / "processed"
RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "scratch" / "backtest_cache.parquet"


def fingerprint() -> str:
    h = hashlib.sha256()
    for f in ("tidy.parquet", "transitions.parquet", "bouts.parquet"):
        h.update((PROCESSED / f).read_bytes())
    for mod in (features, metrics, models, resolver):
        h.update(inspect.getsource(mod).encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model names")
    ap.add_argument("--start", type=int, default=200401,
                    help="first target basho (default 200401: start of the 42-man, "
                         "post-kosho era; earlier committee behavior differs)")
    ap.add_argument("--end", type=int, default=None,
                    help="last target basho (default: latest fetched; during model "
                         "development 202311 was used to protect the 2024+ holdout)")
    ap.add_argument("--out", default=None, help="results file stem, e.g. results/dev")
    ap.add_argument("--summarize", default=None, metavar="CSVS",
                    help="skip running; summarize existing per-basho csv(s), comma-separated")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and rebuild the per-(model, basho) result cache")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    if args.summarize:
        results = pd.concat([pd.read_csv(f) for f in args.summarize.split(",")])
        print_summaries(results)
        return

    names = args.models.split(",")
    unknown = set(names) - set(MODELS)
    if unknown:
        ap.error(f"unknown models: {unknown}; available: {list(MODELS)}")

    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    end = args.end or int(tidy["basho"].max())
    targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= end]

    fp = fingerprint()
    cached = pd.DataFrame()
    if CACHE.exists() and not args.fresh:
        store = pd.read_parquet(CACHE)
        cached = store[(store["fp"] == fp) & store["model"].isin(names)
                       & store["basho"].isin(targets)].drop(columns="fp")
        if len(cached):
            print(f"reusing {len(cached)} cached (model, basho) results; "
                  f"--fresh to recompute", file=sys.stderr)

    skip = set(zip(cached["model"], cached["basho"])) if len(cached) else None
    computed = run_backtest(names, targets, trans, tidy, skip=skip)
    results = pd.concat([f for f in (cached, computed) if len(f)], ignore_index=True)

    if len(computed):
        keep = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
        if len(keep):
            keep = keep[keep["fp"] == fp]  # drop rows from stale code/data
        store = pd.concat([keep, computed.assign(fp=fp)], ignore_index=True)
        store = store.drop_duplicates(["fp", "model", "basho"], keep="last")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        store.to_parquet(CACHE, index=False)

    print_summaries(results)

    if args.out:
        RESULTS.mkdir(exist_ok=True)
        results.to_csv(f"{args.out}_per_basho.csv", index=False)
        summaries = [summarize(results, since=s).assign(window=w)
                     for w, s in WINDOWS if len(results[results["basho"] >= (s or 0)])]
        pd.concat(summaries).round(4).to_csv(f"{args.out}_summary.csv")
        print(f"\nwrote {args.out}_per_basho.csv, {args.out}_summary.csv")


# committee behavior drifts; recent windows matter most for model selection
WINDOWS = [("full", None), ("2014+", 201401), ("2020+", 202001)]


def print_summaries(results: pd.DataFrame):
    for label, since in WINDOWS:
        sub = results[results["basho"] >= (since or 0)]
        if not len(sub):
            continue
        print(f"\n=== {label} ({sub['basho'].nunique()} basho) ===")
        print(summarize(results, since=since).round(3).to_string())


if __name__ == "__main__":
    main()
