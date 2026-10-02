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
from banzuke.cli._common import CommandError, add_seeds, add_workers, parse_seeds, subcommand


def add_parser(sub):
    ap = subcommand(sub, "rules", __doc__, "measure candidate committee rules on cached frames")
    ap.add_argument("--start", type=int, default=200401, metavar="BASHO",
                    help="earliest target to score")
    ap.add_argument("--rules", default=None, metavar="NAMES",
                    help="comma-separated RULE_FUNCS keys (default: all)")
    add_workers(ap)
    add_seeds(ap, "0-2", "the frames per target the explain cache was built with")
    ap.add_argument("--cache", default=None, metavar="DIR",
                    help="explain cache to reuse (default: the current fingerprint's)")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke.experiments import rules
    from banzuke.experiments.explain import NoCache
    from banzuke.paths import load_tidy

    names = args.rules.split(",") if args.rules else list(rules.RULE_FUNCS)
    try:
        rows = rules.load_frames(args.start, parse_seeds(args.seeds), args.cache)
        res, variants = rules.measure(rows, load_tidy(), names, args.workers)
    except (NoCache, KeyError) as e:
        raise CommandError(str(e).strip("'"))
    rules.report(res, variants)
