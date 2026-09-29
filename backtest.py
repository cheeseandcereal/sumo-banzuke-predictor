#!/usr/bin/env python3
"""Run the model bake-off over a window of historical basho.

Per-(config, model, seed, basho) results are cached in results/scratch/ and
reused on re-runs. The cache key covers the dataset content, the source of
every module that affects results, the lockfile and the resolved
configuration, so editing code, rebuilding data or changing a parameter
invalidates it automatically; --fresh forces recomputation.

Examples:
    uv run python backtest.py                         # all models, 2004+, bag replicate 0
    uv run python backtest.py --models Ar --seeds 0-4 --start 200401 --end 201911
    uv run python backtest.py --models Ar --set base.n_estimators=200 \\
        --set pair.n_estimators=150 --name n200 --baseline Ar
    uv run python backtest.py --models Ar --config sweep.json --baseline Ar

--set KEY=VALUE overrides a model option: dotted keys address the LightGBM
stages (base.*, pair.*); bare keys are model options (gap, cluster_max,
pair_window, n_seeds, context, pairs, oof_gap, half_life, blend_l2, ...).
Values are parsed as JSON where possible (true, 0.25, 5), else strings.
A --config file maps labels to {key: value} dicts and runs each; the label
"base" (no overrides) is added when --baseline names it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from banzuke.harness import fingerprint, parse_seeds, parse_sets, run_backtest, summarize
from banzuke.models import MODELS

PROCESSED = Path(__file__).parent / "data" / "processed"
RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "scratch" / "backtest_cache.parquet"

# committee behavior drifts; recent windows matter most for model selection.
# screen/confirm are the protocol-v2 tuning windows (docs/EXPERIMENTS.md)
WINDOWS = [("full", None, None), ("screen 2004-2019", 200401, 201911),
           ("confirm 2020+", 202001, None), ("2014+", 201401, None)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model names")
    ap.add_argument("--start", type=int, default=200401,
                    help="first target basho (default 200401: start of the 42-man, "
                         "post-kosho era; earlier committee behavior differs)")
    ap.add_argument("--end", type=int, default=None,
                    help="last target basho (default: latest fetched)")
    ap.add_argument("--seeds", default="0", metavar="SPEC",
                    help="bag replicates to fit, e.g. 0-1 or 0,2 (default 0); replicate k "
                         "uses LightGBM seeds k*n_seeds..(k+1)*n_seeds-1; averaged per basho")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override (repeatable), see above")
    ap.add_argument("--name", default=None,
                    help="label for the --set configuration (default: derived)")
    ap.add_argument("--config", default=None, metavar="JSON",
                    help="file of {label: {key: value}} configurations to run")
    ap.add_argument("--baseline", default=None, metavar="LABEL",
                    help="label paired comparisons refer to (default: leader); "
                         "MODEL or MODEL:config")
    ap.add_argument("--mak-size-policy", choices=("prior", "actual"), default="prior",
                    help="size the forecast like the previous banzuke (prior, default) "
                         "or read the target's actual size (actual)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="target-level worker processes (default: up to 16, capped at CPUs)")
    ap.add_argument("--out", default=None, help="results file stem, e.g. results/dev")
    ap.add_argument("--summarize", default=None, metavar="CSVS",
                    help="skip running; summarize existing per-basho csv(s), comma-separated")
    ap.add_argument("--train-start", type=int, default=None, metavar="BASHO",
                    help="ignore training transitions before this basho "
                         "(e.g. 201001); default: all history since 1959")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore cached per-(config, model, seed, basho) rows and recompute")
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    if args.summarize:
        results = pd.concat([pd.read_csv(f) for f in args.summarize.split(",")])
        print_summaries(results, args.baseline)
        return

    names = args.models.split(",")
    unknown = set(names) - set(MODELS)
    if unknown:
        ap.error(f"unknown models: {unknown}; available: {list(MODELS)}")
    seeds = parse_seeds(args.seeds)

    configs: dict[str, dict] = {}
    if args.config:
        configs.update(json.loads(Path(args.config).read_text()))
    if args.set or not configs:
        kwargs = parse_sets(args.set)
        label = args.name or (",".join(args.set) if args.set else "base")
        configs[label] = kwargs
    base_label = args.baseline.split(":", 1)[1] if args.baseline and ":" in args.baseline \
        else ("base" if args.baseline in MODELS else None)
    if base_label == "base" and "base" not in configs:
        configs = {"base": {}, **configs}

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

    store = pd.read_parquet(CACHE) if CACHE.exists() and not args.fresh else pd.DataFrame()
    if len(store) and "seed" not in store:
        store = pd.DataFrame()  # pre-2026-09 cache layout
    all_results, new_rows = [], []
    for label, kwargs in configs.items():
        extra = json.dumps({"kwargs": kwargs, "train_start": args.train_start,
                            "mak_size_policy": args.mak_size_policy}, sort_keys=True)
        fp = fingerprint(extra)
        cached = pd.DataFrame()
        if len(store):
            cached = store[(store["fp"] == fp) & store["model"].isin(names)
                           & store["seed"].isin(seeds) & store["basho"].isin(targets)]
            cached = cached.drop(columns="fp").assign(config=label)
            if len(cached):
                print(f"[{label}] reusing {len(cached)} cached rows; --fresh to recompute",
                      file=sys.stderr)
        skip = set(zip(cached["model"], cached["seed"], cached["basho"])) if len(cached) else None
        if len(configs) > 1:
            print(f"[{label}] {kwargs}", file=sys.stderr)
        computed = run_backtest(names, targets, trans, tidy, skip=skip, seeds=seeds,
                                train_start=args.train_start, model_kwargs=kwargs,
                                mak_size_policy=args.mak_size_policy, config=label,
                                workers=args.workers)
        if len(computed):
            new_rows.append(computed.assign(fp=fp))
        all_results.append(pd.concat([f for f in (cached, computed) if len(f)],
                                     ignore_index=True))
    results = pd.concat(all_results, ignore_index=True)

    if new_rows:  # replace this run's rows, keep every other configuration's
        keep = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
        if len(keep) and "seed" not in keep:
            keep = pd.DataFrame()
        if len(keep):
            keep = keep[~keep["fp"].isin({r["fp"].iloc[0] for r in new_rows})]
        store = pd.concat([keep, *new_rows], ignore_index=True)
        store = store.drop_duplicates(["fp", "model", "seed", "basho"], keep="last")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        store.to_parquet(CACHE, index=False)

    print_summaries(results, args.baseline)

    if args.out:
        RESULTS.mkdir(exist_ok=True)
        results.to_csv(f"{args.out}_per_basho.csv", index=False)
        summaries = []
        for w, s, u in WINDOWS:
            sub = results[(results["basho"] >= (s or 0)) & (results["basho"] <= (u or 10**8))]
            if len(sub):
                summaries.append(summarize(results, args.baseline, s, u).assign(window=w))
        pd.concat(summaries).round(4).to_csv(f"{args.out}_summary.csv")
        print(f"\nwrote {args.out}_per_basho.csv, {args.out}_summary.csv")


SHOW = ["n", "exact_n", "gtb_points", "mae", "within1", "promo_f1", "demo_f1",
        "sanyaku_exact", "d_exact", "ci_exact", "p_exact", "wl", "d_mae", "ci_mae", "p_mae"]


def print_summaries(results: pd.DataFrame, baseline=None):
    for label, since, until in WINDOWS:
        sub = results[(results["basho"] >= (since or 0)) & (results["basho"] <= (until or 10**8))]
        if not len(sub):
            continue
        s = summarize(results, baseline, since, until)
        seeds = sub["seed"].nunique() if "seed" in sub else 1
        print(f"\n=== {label} ({sub['basho'].nunique()} basho, {seeds} seed"
              f"{'s' if seeds != 1 else ''}; paired vs {s.attrs['baseline']}) ===")
        cols = [c for c in SHOW if c in s] + [c for c in ("pair_acc", "pair_prior_acc",
                                                          "pair_outside") if c in s]
        print(s[cols].round(3).to_string())


if __name__ == "__main__":
    main()
