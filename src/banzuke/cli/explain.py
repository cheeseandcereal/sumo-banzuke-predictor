"""Per-basho explain dumps, miss docket and pair calibration for the default
model Ar (docs/EXPERIMENTS.md E19).

    banzuke explain build --start 201901 --end 202609 --seeds 3 --workers 14
    banzuke explain sheet 202101 [--seed 0] [--all]      # side-by-side banzuke
    banzuke explain explain 202101 [--seed 0] [--all]    # per-rikishi stage table
    banzuke explain docket --start 201901               # every missed cell, classified
    banzuke explain calibration                         # pair probability reliability

`build` mirrors the backtest harness frame by frame and records, besides the
standard prediction frame, the reranker's units and clusters, every pair
probability within PAIR_GAP base points, the sheet the resolver would have
produced from the base order alone, and a per-row miss decomposition. The
readers concatenate every build under
results/scratch/explain/<fingerprint>/.

Stage labels (misses only): cascade (right once the structure above is),
structural (wrong Y/O/S/K membership), boundary (wrong side of the juryo
line), rerank (the base order had the cell right, the reranked order lost
it), base (the base order was wrong and the reranker did not fix it),
resolver (the final order was right, the resolver's rules or layout moved
the cell).
"""
import pandas as pd

from banzuke.cli._common import CommandError, subcommand
from banzuke.experiments import explain
from banzuke.harness import DEFAULT_WORKERS
from banzuke.paths import require_processed


def add_parser(sub):
    ap = subcommand(sub, "explain", __doc__, "cached backtest frames: sheets, misses, calibration")
    ap.add_argument("cmd", choices=("build", "sheet", "explain", "docket", "calibration"))
    ap.add_argument("target", nargs="?", type=int)
    ap.add_argument("--start", type=int, default=201901)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="sheet/explain: which seed's frame")
    ap.add_argument("--all", action="store_true", help="sheet/explain: every cached target >= --start")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    processed = require_processed()
    if args.cmd == "build":
        return explain.build(args)
    if args.cmd == "docket":
        return explain.docket(args)
    if args.cmd == "calibration":
        return explain.calibration(args)
    rows, frames = explain.load("rows", args), explain.load("frames", args)
    tidy = pd.read_parquet(processed / "tidy.parquet")
    targets = sorted(t for t in rows["target"].unique() if t >= args.start) if args.all else [args.target]
    if targets == [None]:
        raise CommandError("give a target or --all")
    if args.cmd == "sheet":
        out = explain.SCRATCH / "sheets"
        out.mkdir(exist_ok=True)
        for t in targets:
            text = explain.sheet_text(rows, frames, tidy, t, args.seed)
            (out / f"{t}.txt").write_text(text + "\n")
            if not args.all:
                print(text)
    else:
        pairs = explain.load("pairs", args)
        trans = pd.read_parquet(processed / "transitions.parquet")
        out = explain.SCRATCH / "basho"
        out.mkdir(exist_ok=True)
        for t in targets:
            text = explain.explain_text(rows, pairs, frames, tidy, trans, t, args.seed)
            (out / f"{t}.txt").write_text(text + "\n")
            if not args.all:
                print(text)
    if args.all:
        print(f"wrote {len(targets)} files under {out}")
