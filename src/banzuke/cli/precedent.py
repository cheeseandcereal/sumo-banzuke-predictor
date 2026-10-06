"""Committee precedent queries over the transitions, always with counts.

    banzuke precedent landing "K1 5-10"                  # where did that rank and record land
    banzuke precedent landing "M1 8-7" --where "year>=2010"
    banzuke precedent landing "K1 5-10" --list           # and who they were, newest first
    banzuke precedent pair "K 5-10" "M9-10 10-5"         # how often was the first above the second
    banzuke precedent pair "K 5-10" "M9-10 10-5" --limit 0   # listing every pair, not just 40
    banzuke precedent cells "S 7-8"                      # same as landing

Spec grammar: <class>[<number>|<lo>-<hi>][E|W] <record>
    class   Y O S K M J
    record  9-6 | 5-7-3 (exact W-L[-A]) | 8+ (at least 8 wins) | kk | mk | *
Windows: all (1959+), 2004+ (modern, 42-man makuuchi), last60 (the 60 most
recent basho). `--where` is a pandas query on the transition columns.
`--list` adds the rows (pairs) behind the counts, newest basho first: the
basho whose record it is, the man, his cell and record there and `-> cell`
on the banzuke that followed. A row ends with the move in ranks (+ = up
the banzuke, 0.5 per cell; the `delta` column `--where` sees is the same
number), a pair with whether the first man landed above or below the
second.
"""
from banzuke.cli._common import CommandError, action_parser, subcommand

SPEC_HELP = 'rank and record, e.g. "K1 5-10", "M9-10 10-5", "S kk"'
LIST_DEFAULT = 40


def add_parser(sub):
    ap = subcommand(sub, "precedent", __doc__, "where the committee put a rank and record")
    ap.set_defaults(_parser=ap)
    what = ap.add_subparsers(dest="action", metavar="ACTION", required=True)
    for name in ("landing", "cells"):
        p = action_parser(what, name, "where rows matching each SPEC landed, per window")
        p.add_argument("spec", nargs="+", metavar="SPEC", help=SPEC_HELP)
        _options(p, "rows")
        p.set_defaults(run=run_landing, _parser=p)
    p = action_parser(what, "pair", "share of same-basho pairs where the first SPEC landed above the second")
    p.add_argument("spec", nargs=2, metavar="SPEC", help=SPEC_HELP)
    _options(p, "pairs")
    p.set_defaults(run=run_pair, _parser=p)
    return ap


def _options(ap, noun):
    ap.add_argument("--where", default=None, metavar="QUERY",
                    help="pandas query on the transition columns, e.g. \"year>=2010\"")
    ap.add_argument("--list", action="store_true",
                    help=f"also list the {noun} behind the counts, newest first "
                         f"({LIST_DEFAULT} unless --limit)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help=f"{noun} to list, 0 = all; implies --list")


def _limit(args):
    """Rows to list: 0 = all, None = no listing; a negative --limit is a usage error."""
    if args.limit is None:
        return LIST_DEFAULT if args.list else None
    if args.limit < 0:
        raise CommandError(f"--limit must be 0 (all) or a positive count, got {args.limit}")
    return args.limit


def _labeled():
    from banzuke.experiments import precedent
    from banzuke.paths import load_transitions

    return precedent, precedent.labeled(load_transitions())


# a malformed spec (ValueError) or --where query (what pandas.eval raises)
BAD_INPUT = (ValueError, SyntaxError, NameError, KeyError, TypeError)


def run_landing(args):
    limit = _limit(args)
    precedent, t = _labeled()
    try:
        for spec in args.spec:
            print(precedent.fmt_landing(spec, precedent.landing(t, spec, args.where)))
            if limit is not None:
                rows = t[precedent.select(t, spec, args.where)]
                if len(rows):
                    print(precedent.fmt_rows(rows, limit))
    except BAD_INPUT as e:
        raise CommandError(f"{e} (spec {spec!r}, --where {args.where!r})") from e


def run_pair(args):
    limit = _limit(args)
    precedent, t = _labeled()
    try:
        print(precedent.fmt_pair(*args.spec, precedent.pair_rate(t, *args.spec, args.where)))
        if limit is not None:
            pairs = precedent.pairs(t, *args.spec, args.where)
            if len(pairs):
                print(precedent.fmt_pairs(pairs, limit))
    except BAD_INPUT as e:
        raise CommandError(f"{e} (specs {args.spec}, --where {args.where!r})") from e
