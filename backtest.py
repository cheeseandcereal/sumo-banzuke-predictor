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
    ap.add_argument("--start", type=int, default=200401, help="first target basho")
    ap.add_argument("--end", type=int, default=202311, help="last target basho")
    ap.add_argument("--out", default=None, help="results file stem, e.g. results/dev")
    args = ap.parse_args()

    names = args.models.split(",")
    unknown = set(names) - set(MODELS)
    if unknown:
        ap.error(f"unknown models: {unknown}; available: {list(MODELS)}")

    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    targets = [b for b in sorted(tidy["basho"].unique()) if args.start <= b <= args.end]

    results = run_backtest(names, targets, trans, tidy)
    summary = summarize(results)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())

    if args.out:
        RESULTS.mkdir(exist_ok=True)
        results.to_csv(f"{args.out}_per_basho.csv", index=False)
        summary.round(4).to_csv(f"{args.out}_summary.csv")
        print(f"\nwrote {args.out}_per_basho.csv, {args.out}_summary.csv")


if __name__ == "__main__":
    main()
