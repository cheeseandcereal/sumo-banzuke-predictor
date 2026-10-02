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
from banzuke.cli._common import DEFAULT_WORKERS, CommandError, parse_sets, subcommand


def add_parser(sub):
    ap = subcommand(sub, "explain", __doc__, "cached backtest frames: sheets, misses, calibration")
    ap.add_argument("cmd", choices=("build", "sheet", "explain", "docket", "calibration"))
    ap.add_argument("target", nargs="?", type=int, help="sheet/explain: the basho to show")
    ap.add_argument("--start", type=int, default=201901, metavar="BASHO",
                    help="build: first target; docket/--all: earliest target to include")
    ap.add_argument("--end", type=int, default=None, metavar="BASHO",
                    help="build: last target (default: latest)")
    ap.add_argument("--seeds", type=int, default=3, metavar="N",
                    help="frames per target (seeds 0..N-1); part of the cache key")
    ap.add_argument("--seed", type=int, default=0, help="sheet/explain: which seed's frame")
    ap.add_argument("--all", action="store_true", help="sheet/explain: every cached target >= --start")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in `banzuke backtest`; part of the cache key")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="build: worker processes")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke.experiments import explain
    from banzuke.paths import load_tidy, load_transitions

    kwargs, seeds = parse_sets(args.set), list(range(args.seeds))
    try:
        if args.cmd == "build":
            tidy = load_tidy()
            end = args.end or int(tidy["basho"].max())
            explain.build(load_transitions(), tidy, args.start, end, seeds, kwargs, args.workers)
            return
        if args.cmd == "docket":
            explain.docket(args.start, kwargs, seeds)
            return
        if args.cmd == "calibration":
            explain.calibration(kwargs, seeds)
            return
        rows, frames = explain.load("rows", kwargs, seeds), explain.load("frames", kwargs, seeds)
        tidy = load_tidy()
        targets = (sorted(t for t in rows["target"].unique() if t >= args.start) if args.all
                   else [args.target])
        if targets == [None]:
            raise CommandError("give a target or --all")
        if args.cmd == "sheet":
            out = explain.SCRATCH / "sheets"
            out.mkdir(parents=True, exist_ok=True)
            for t in targets:
                text = explain.sheet_text(rows, frames, tidy, t, args.seed)
                (out / f"{t}.txt").write_text(text + "\n")
                if not args.all:
                    print(text)
        else:
            pairs, trans = explain.load("pairs", kwargs, seeds), load_transitions()
            out = explain.SCRATCH / "basho"
            out.mkdir(parents=True, exist_ok=True)
            for t in targets:
                text = explain.explain_text(rows, pairs, frames, tidy, trans, t, args.seed)
                (out / f"{t}.txt").write_text(text + "\n")
                if not args.all:
                    print(text)
        if args.all:
            print(f"wrote {len(targets)} files under {out}")
    except explain.NoCache as e:
        raise CommandError(str(e))
