"""Run the model bake-off over a window of historical basho.

Per-(configuration, model, seed, basho) results are cached in results/scratch/
and reused on re-runs. The cache key covers the dataset, the source of every
module that affects results, the lockfile and the configuration, so editing
code, rebuilding data or changing a parameter invalidates it automatically;
--fresh forces recomputation.

    banzuke backtest                                   # all models, 2004+
    banzuke backtest --models Ar,Aq --seeds 0-1 --end 201911
    banzuke backtest --models Ar --set base.n_estimators=600 --baseline Ar

--set KEY=VALUE (repeatable) overrides a model option: dotted keys address
the LightGBM stages (base.*, pair.*), bare keys are model options (n_seeds,
gap, cluster_max, context, near_ties, half_life, ...). --baseline MODEL runs
that model's defaults alongside and pairs every row against them
(MODEL:label for another configuration present in the results).
"""
import sys

from banzuke.cli._common import (CommandError, add_seeds, add_set, add_train_start, add_window,
                                 add_workers, parse_seeds, parse_sets, subcommand, targets_in)

# committee behavior drifts; screen/confirm are the tuning windows of
# docs/EXPERIMENTS.md protocol v2
WINDOWS = [("full", None, None), ("screen 2004-2019", 200401, 201911),
           ("confirm 2020+", 202001, None)]
SHOW = ["n", "exact_n", "gtb_points", "mae", "within1", "promo_f1", "demo_f1",
        "sanyaku_exact", "d_exact", "ci_exact", "p_exact", "wl", "d_mae", "ci_mae", "p_mae"]


def add_parser(sub):
    ap = subcommand(sub, "backtest", __doc__, "rolling-origin backtest of the models")
    ap.add_argument("--models", default=None, metavar="NAMES",
                    help="comma-separated model names (default: all)")
    add_window(ap, 200401, "first target basho (200401: start of the 42-man, post-kosho era; "
                           "earlier committee behavior differs)")
    add_seeds(ap, "0", "bag replicates to fit, averaged per basho")
    add_set(ap)
    ap.add_argument("--baseline", default=None, metavar="LABEL",
                    help="configuration paired comparisons refer to (default: the leader)")
    add_workers(ap)
    ap.add_argument("--out", default=None, metavar="STEM",
                    help="write STEM_per_basho.csv and STEM_summary.csv, e.g. results/dev")
    ap.add_argument("--summarize", default=None, metavar="CSVS",
                    help="skip running; summarize existing per-basho csv(s), comma-separated")
    add_train_start(ap)
    ap.add_argument("--fresh", action="store_true", help="recompute instead of using the cache")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    import pandas as pd

    from banzuke.harness import run_cached, summarize
    from banzuke.models import MODELS
    from banzuke.paths import RESULTS, load_tidy, load_transitions

    pd.set_option("display.width", 250)
    if args.summarize:
        print_summaries(pd.concat([pd.read_csv(f) for f in args.summarize.split(",")]),
                        args.baseline)
        return

    names = args.models.split(",") if args.models else list(MODELS)
    unknown = set(names) - set(MODELS)
    if unknown:
        raise CommandError(f"unknown models: {unknown}; available: {list(MODELS)}")
    seeds = parse_seeds(args.seeds)
    configs = {",".join(args.set) or "base": parse_sets(args.set)}
    if args.baseline in MODELS or (args.baseline or "").endswith(":base"):
        configs.setdefault("base", {})

    tidy, trans = load_tidy(), load_transitions()
    targets, end = targets_in(tidy, args.start, args.end)
    if args.train_start:
        n_train = int(tidy["basho"].between(args.train_start, targets[0]).sum()) - 1
        if n_train < 30:
            print(f"warning: first target {targets[0]} has only {n_train} training "
                  f"basho with --train-start {args.train_start}", file=sys.stderr)

    results = run_cached(names, targets, trans, tidy, configs, seeds=seeds,
                         train_start=args.train_start, workers=args.workers, fresh=args.fresh)
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
    from banzuke.harness import summarize

    for label, since, until in WINDOWS:
        sub = results[(results["basho"] >= (since or 0)) & (results["basho"] <= (until or 10**8))]
        if not len(sub):
            continue
        s = summarize(results, baseline, since, until)
        print(f"\n=== {label} ({sub['basho'].nunique()} basho, {sub['seed'].nunique()} "
              f"seed(s); paired vs {s.attrs['baseline']}) ===")
        print(s[SHOW].round(3).to_string())
