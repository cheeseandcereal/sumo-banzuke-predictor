"""The human benchmark: how the "Guess the Banzuke" (GTB) field scored on the
target basho the backtest predicts, and where a model's forecasts would have
placed in it.

GTB players (dichne.com) predict the next banzuke after each basho and are
scored like `gtb_points`: 2 per bullseye (exact cell), 1 per hit (right
rank, wrong side). The archive at sumodb.sumogames.de/gtb is fetched once
per basho into cache/gtb/ (git-ignored, never committed). Per target basho
and averaged over the window: the number of entries, the median (P50),
75th-percentile (P75) and winning (P100) points and bullseyes.

    banzuke gtb                                   # the confirm window, 2020+
    banzuke gtb --start 201401 --end 201911       # another window
    banzuke gtb --percentiles 25,50,75,90,100
    banzuke gtb --model Ar --seeds 0-9            # the model's place in each field
    banzuke gtb --csv cache/all_per_basho.csv --model Ar
    banzuke gtb --out cache/gtb                   # cache/gtb_field.csv, cache/gtb_entries.csv

--model takes the model's rows from the backtest cache (rows of the same
--seeds, --set and --train-start are reused, the rest computed) or, with
--csv, from per-basho files written by `banzuke backtest --out`. Seeds are
averaged within a basho, then each basho's score is placed in that basho's
field: the place it would have taken (ties share the better place), the
share of entries it beats (ties counted half) and the points behind the
winner.
"""
from pathlib import Path

from banzuke.cli._common import (CommandError, add_fresh, add_model, add_seeds, add_set,
                                 add_train_start, add_window, add_threads, model_class, parse_seeds,
                                 set_configs, subcommand, targets_in)
from banzuke.ranks import CONFIRM_ERA


def add_parser(sub):
    ap = subcommand(sub, "gtb", __doc__, "the human Guess-the-Banzuke field, basho by basho")
    add_window(ap, CONFIRM_ERA, "first target basho (default: the confirm window of docs/EXPERIMENTS.md)")
    ap.add_argument("--percentiles", default="50,75,100", metavar="LIST",
                    help="percentiles of the field to report, comma-separated (100 = the winner)")
    add_model(ap, None, "place this model's backtest forecasts in each basho's field")
    add_seeds(ap, "0", "bag replicates of --model, averaged per basho")
    add_set(ap)
    add_train_start(ap)
    add_threads(ap, "worker processes for backtest rows not in the cache")
    ap.add_argument("--csv", default=None, metavar="FILES",
                    help="take the model's rows from these per-basho csv(s) instead "
                         "(comma-separated; `banzuke backtest --out`)")
    ap.add_argument("--out", default=None, metavar="STEM",
                    help="write STEM_field.csv (the table) and STEM_entries.csv (every entry)")
    add_fresh(ap, "refetch the archive pages")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def parse_percentiles(spec) -> tuple:
    """'50,75,100' -> (50.0, 75.0, 100.0)."""
    try:
        out = tuple(float(p) for p in str(spec).split(","))
    except ValueError:
        out = ()
    if not out or any(not 0 <= p <= 100 for p in out):
        raise CommandError(f"--percentiles expects numbers in 0..100, comma-separated, got {spec!r}")
    return out


def run(args):
    import pandas as pd

    from banzuke import gtb
    from banzuke.paths import load_tidy

    pd.set_option("display.width", 250)
    percentiles = parse_percentiles(args.percentiles)
    if args.model and not args.csv:
        model_class(args.model)
    tidy = load_tidy()
    targets, end = targets_in(tidy, args.start, args.end)
    results = label = None
    if args.csv:
        results, label = csv_rows(args.csv, args.model, targets)
    entries = gtb.load_results(targets, fresh=args.fresh)
    if not len(entries):
        raise CommandError(f"no GTB results posted for any target basho in {args.start}..{end}")
    if args.model and not args.csv:
        results, label = backtest_rows(args, targets, tidy)
    table = gtb.field(entries, percentiles)
    if results is not None:
        table = table.join(gtb.place_in_field(entries, results, label))
    print_report(table, label)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(f"{args.out}_field.csv")
        entries.to_csv(f"{args.out}_entries.csv", index=False)
        print(f"\nwrote {args.out}_field.csv, {args.out}_entries.csv")


def csv_rows(files, model, targets):
    """Per-basho rows of one label from `banzuke backtest --out` files."""
    import pandas as pd

    from banzuke.harness import label_of

    results = pd.concat([pd.read_csv(f) for f in files.split(",")], ignore_index=True)
    labels = sorted(set(label_of(results)))
    label = model or (labels[0] if len(labels) == 1 else None)
    if label not in labels:
        raise CommandError(f"--csv holds {', '.join(labels)}; pick one with --model")
    results = results[(label_of(results) == label) & results["basho"].isin(targets)]
    if not len(results):
        raise CommandError(f"--csv has no rows of {label} inside the window")
    return results, label


def backtest_rows(args, targets, tidy):
    """The model's rows from the backtest cache, computing the missing ones."""
    from banzuke.harness import label_of, run_cached
    from banzuke.paths import load_transitions

    configs = set_configs(args)
    try:
        results = run_cached([args.model], targets, load_transitions(), tidy, configs,
                             seeds=parse_seeds(args.seeds), train_start=args.train_start,
                             threads=args.threads)
    except (TypeError, ValueError) as e:
        raise CommandError(f"--set: {e}")
    return results, str(label_of(results)[0])


def print_report(table, label=None):
    import pandas as pd

    n, start, end = len(table), table.index.min(), table.index.max()
    print(f"=== Guess the Banzuke field: {n} banzuke {start}-{end} "
          f"(points = 2 x bullseyes + hits; p100 = the winner) ===")
    mean = table.mean(numeric_only=True).to_frame("mean").T
    out = pd.concat([table.rename(index=str), mean])
    fmt = {c: "{:.0f}".format if c in ("entries", "place") else "{:.0%}".format if c == "pct"
           else "{:.1f}".format for c in out.columns if c != "winner"}
    print(out.to_string(formatters=fmt, na_rep=""))
    if label:
        p = table.dropna(subset=["place"])
        print(f"\n{label}: {p[f'{label}_pts'].mean():.1f} points/basho ({p[f'{label}_be'].mean():.1f} "
              f"exact), mean place {p['place'].mean():.0f} of {p['entries'].mean():.0f}, beats "
              f"{p['pct'].mean():.0%} of the field; won {int((p['place'] == 1).sum())} of {len(p)} "
              f"basho, above the median entry in {int((p['pct'] > 0.5).sum())}, "
              f"{p['behind'].mean():.1f} points behind the winner on average")
