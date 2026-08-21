#!/usr/bin/env python3
"""Run the model bake-off over a window of historical basho.

Examples:
    uv run python backtest.py                        # all models, dev window
    uv run python backtest.py --models R,A --start 202001 --end 202311
"""
import argparse
from pathlib import Path

import pandas as pd

from banzuke.harness import run_backtest, summarize
from banzuke.models import MODELS

PROCESSED = Path(__file__).parent / "data" / "processed"
RESULTS = Path(__file__).parent / "results"


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

    results = run_backtest(names, targets, trans, tidy)
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
