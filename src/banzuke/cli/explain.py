"""Per-basho explain dumps, miss docket and pair calibration for the default
model Ar (docs/EXPERIMENTS.md E19).

    banzuke explain build --start 201901 --end 202609 --seeds 0-2 --threads 14
    banzuke explain sheet 202101 [--seed 0]      # side-by-side predicted / actual banzuke
    banzuke explain sheet --all --start 201901   # one file per cached target
    banzuke explain detail 202101 [--seed 0]     # per-rikishi stage table, clusters, precedent
    banzuke explain docket --start 201901        # every missed cell, classified
    banzuke explain calibration                  # pair probability reliability

`build` mirrors the backtest harness frame by frame and records, besides the
standard prediction frame, the reranker's units and clusters, every pair
probability within PAIR_GAP base points, the sheet the resolver would have
produced from the base order alone, and a per-row miss decomposition. The
readers concatenate every build under cache/explain/<fingerprint>/;
--seeds and --set are part of that fingerprint, so a reader must repeat the
values the build used.

Stage labels (misses only): cascade (right once the structure above is),
structural (wrong Y/O/S/K membership), boundary (wrong side of the juryo
line), rerank (the base order had the cell right, the reranked order lost
it), base (the base order was wrong and the reranker did not fix it),
resolver (the final order was right, the resolver's rules or layout moved
the cell).
"""
from banzuke.cli._common import (CommandError, action_parser, add_seeds, add_set, add_window,
                                 add_threads, parse_seeds, parse_sets, subcommand, targets_in)


def _cache_key(ap):
    add_seeds(ap, "0-2", "frames per target, one per seed; part of the cache key")
    add_set(ap, "model option override, repeatable; part of the cache key")


def _which(ap):
    ap.add_argument("target", nargs="?", type=int, help="the basho whose banzuke to show")
    ap.add_argument("--seed", type=int, default=0, help="which seed's frame")
    ap.add_argument("--all", action="store_true",
                    help="every cached target from --start on, written to files instead of printed")
    ap.add_argument("--start", type=int, default=201901, metavar="BASHO",
                    help="with --all: earliest target to include")


def add_parser(sub):
    ap = subcommand(sub, "explain", __doc__, "cached backtest frames: sheets, misses, calibration")
    ap.set_defaults(_parser=ap)
    what = ap.add_subparsers(dest="action", metavar="ACTION", required=True)

    p = action_parser(what, "build", "run the frames and cache them (minutes)")
    add_window(ap=p, start=201901)
    _cache_key(p)
    add_threads(p)
    p.set_defaults(run=run_build, _parser=p)

    p = action_parser(what, "sheet", "predicted and actual banzuke side by side")
    _which(p)
    _cache_key(p)
    p.set_defaults(run=run_sheet, _parser=p, text=_sheet)

    p = action_parser(what, "detail", "per-rikishi stage table, reranker clusters and precedent")
    _which(p)
    _cache_key(p)
    p.set_defaults(run=run_sheet, _parser=p, text=_detail)

    p = action_parser(what, "docket", "every missed makuuchi cell, classified by stage and zone")
    p.add_argument("--start", type=int, default=201901, metavar="BASHO", help="earliest target")
    _cache_key(p)
    p.set_defaults(run=run_docket, _parser=p)

    p = action_parser(what, "calibration", "reliability of the reranker's pair probabilities")
    _cache_key(p)
    p.set_defaults(run=run_calibration, _parser=p)
    return ap


def _key(args):
    return parse_sets(args.set), list(parse_seeds(args.seeds))


def run_build(args):
    from banzuke.experiments import explain
    from banzuke.paths import load_tidy, load_transitions

    kwargs, seeds = _key(args)
    tidy = load_tidy()
    _, end = targets_in(tidy, args.start, args.end)
    explain.build(load_transitions(), tidy, args.start, end, seeds, kwargs, args.threads)


def run_docket(args):
    from banzuke.experiments import explain

    kwargs, seeds = _key(args)
    try:
        explain.docket(args.start, kwargs, seeds)
    except explain.NoCache as e:
        raise CommandError(str(e))


def run_calibration(args):
    from banzuke.experiments import explain

    kwargs, seeds = _key(args)
    try:
        explain.pair_calibration(kwargs, seeds)
    except explain.NoCache as e:
        raise CommandError(str(e))


def _sheet(explain, data, target, seed):
    return explain.sheet_text(data["rows"], data["frames"], data["tidy"], target, seed), "sheets"


def _detail(explain, data, target, seed):
    if "pairs" not in data:
        from banzuke.paths import load_transitions

        data["pairs"], data["trans"] = explain.load("pairs", *data["key"]), load_transitions()
    return (explain.explain_text(data["rows"], data["pairs"], data["frames"], data["tidy"],
                                 data["trans"], target, seed), "basho")


def run_sheet(args):
    from banzuke.experiments import explain
    from banzuke.paths import load_tidy

    if args.target is None and not args.all:
        raise CommandError("give a target basho or --all")
    key = _key(args)
    try:
        data = {"key": key, "rows": explain.load("rows", *key), "frames": explain.load("frames", *key),
                "tidy": load_tidy()}
        targets = (sorted(t for t in data["rows"]["target"].unique() if t >= args.start) if args.all
                   else [args.target])
        for t in targets:
            text, sub = args.text(explain, data, t, args.seed)
            if not args.all:
                print(text)
                continue
            out = explain.SCRATCH / sub
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{t}.txt").write_text(text + "\n")
    except explain.NoCache as e:
        raise CommandError(str(e))
    if args.all:
        print(f"wrote {len(targets)} files under {out}")
