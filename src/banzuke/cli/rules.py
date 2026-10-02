"""Candidate committee rules measured on cached Ar frames (built by
`banzuke explain build`), without retraining. A rule edits the model's order,
asserts class membership through the resolver's override mechanism, or edits
the frame columns a resolver convention reads; the frame is re-resolved and
re-evaluated, exactly paired with V0 (the cached order re-resolved by the
current resolver).

    banzuke rules [--start 200401] [--rules R11,R13] [--workers 14]
    banzuke rules --cache results/scratch/explain/<old fingerprint>

A resolver edit changes the fingerprint, so `--cache` names the frames to
reuse. The `cached` variant scores the sheet as the cache's resolver made it,
so V0 - cached is the value of the resolver change itself, and a rule the
resolver now applies natively must be a no-op (`changed` 0 in every frame).
New rules are written in banzuke.experiments.rules (see RULE_FUNCS).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from banzuke.cli._common import CommandError, subcommand
from banzuke.experiments import rules
from banzuke.experiments.explain import cache_dir
from banzuke.harness import DEFAULT_WORKERS
from banzuke.paths import require_processed


def add_parser(sub):
    ap = subcommand(sub, "rules", __doc__, "measure candidate committee rules on cached frames")
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--rules", default=",".join(rules.RULE_FUNCS),
                    help="comma-separated RULE_FUNCS keys")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cache", default=None, metavar="DIR",
                    help="explain cache to reuse (default: the current fingerprint's)")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    d = Path(args.cache) if args.cache else cache_dir({}, list(range(args.seeds)))
    files = sorted(d.glob("rows_*.parquet"))
    if not files:
        raise CommandError(f"no cached frames under {d}; run `banzuke explain build` or pass --cache")
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = rows[rows["target"] >= args.start]
    tidy = pd.read_parquet(require_processed() / "tidy.parquet")
    names = args.rules.split(",")
    unknown = sorted(set(names) - set(rules.RULE_FUNCS))
    if unknown:
        raise CommandError(f"unknown rules {unknown}; available: {list(rules.RULE_FUNCS)}")
    variants = {r: [rules.RULE_FUNCS[r]] for r in names}
    if len(names) > 1:  # the rules applied together, in the order given
        variants["ALL"] = [rules.RULE_FUNCS[r] for r in names]
    tasks = [(t, g, tidy[tidy["basho"] == t], variants) for t, g in rows.groupby("target")]
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(rules.run_target, tasks, chunksize=2), 1):
            out += r
            print(f"\r{k}/{len(tasks)} targets", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    res = pd.DataFrame(out)
    res.to_parquet(rules.SCRATCH / "rules_results.parquet", index=False)
    v0 = res[res["variant"] == "V0"]
    print(f"V0 reproduces the cached sheet in {v0['repro'].mean():.3f} of frames; "
          f"variant resolves failed: {int(res['fail'].sum())}")
    noop = (res[res["variant"].isin(variants)].groupby("variant")["changed"].agg(lambda c: int((c > 0).sum())))
    print("frames changed per variant: " + ", ".join(f"{v} {n}" for v, n in noop.items()))
    per = res.groupby(["variant", "target"]).mean(numeric_only=True).reset_index()
    pd.set_option("display.width", 220)
    order = [v for v in list(variants) if v in set(per["variant"])]
    if not v0["repro"].all():
        # the cache was built by an older resolver: V0 (current) is the baseline,
        # `cached` the old resolver's sheet, V0 - cached the value of the change
        order = ["cached"] + order
    for w, lo, hi in rules.WINDOWS:
        tab, n, ex0, mae0 = rules.paired(per, order, lo, hi)
        print(f"\n=== {w}: {n} basho, V0 exact {ex0:.3f} MAE {mae0:.4f}; paired deltas vs V0 (seeds averaged) ===")
        print(tab.round(4).to_string())
