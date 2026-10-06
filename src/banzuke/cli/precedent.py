"""Committee precedent queries over the transitions, always with counts.

    banzuke precedent landing "K1 5-10"                  # where did that rank and record land
    banzuke precedent landing "M1 8-7" --where "year>=2010"
    banzuke precedent pair "K 5-10" "M9-10 10-5"         # how often was the first above the second
    banzuke precedent cells "S 7-8"                      # same as landing

Spec grammar: <class>[<number>|<lo>-<hi>][E|W] <record>
    class   Y O S K M J
    record  9-6 | 5-7-3 (exact W-L[-A]) | 8+ (at least 8 wins) | kk | mk | *
Windows: all (1959+), 2004+ (modern, 42-man makuuchi), last60 (the 60 most
recent basho). `--where` is a pandas query on the transition columns.
"""
from banzuke.cli._common import CommandError, action_parser, subcommand

SPEC_HELP = 'rank and record, e.g. "K1 5-10", "M9-10 10-5", "S kk"'


def add_parser(sub):
    ap = subcommand(sub, "precedent", __doc__, "where the committee put a rank and record")
    ap.set_defaults(_parser=ap)
    what = ap.add_subparsers(dest="action", metavar="ACTION", required=True)
    for name in ("landing", "cells"):
        p = action_parser(what, name, "where rows matching each SPEC landed, per window")
        p.add_argument("spec", nargs="+", metavar="SPEC", help=SPEC_HELP)
        _where(p)
        p.set_defaults(run=run_landing, _parser=p)
    p = action_parser(what, "pair", "share of same-basho pairs where the first SPEC landed above the second")
    p.add_argument("spec", nargs=2, metavar="SPEC", help=SPEC_HELP)
    _where(p)
    p.set_defaults(run=run_pair, _parser=p)
    return ap


def _where(ap):
    ap.add_argument("--where", default=None, metavar="QUERY",
                    help="pandas query on the transition columns, e.g. \"year>=2010\"")


def _labeled():
    from banzuke.experiments import precedent
    from banzuke.paths import load_transitions

    return precedent, precedent.labeled(load_transitions())


# a malformed spec (ValueError) or --where query (what pandas.eval raises)
BAD_INPUT = (ValueError, SyntaxError, NameError, KeyError, TypeError)


def run_landing(args):
    precedent, t = _labeled()
    try:
        for spec in args.spec:
            print(precedent.fmt_landing(spec, precedent.landing(t, spec, args.where)))
    except BAD_INPUT as e:
        raise CommandError(f"{e} (spec {spec!r}, --where {args.where!r})") from e


def run_pair(args):
    precedent, t = _labeled()
    try:
        print(precedent.fmt_pair(*args.spec, precedent.pair_rate(t, *args.spec, args.where)))
    except BAD_INPUT as e:
        raise CommandError(f"{e} (specs {args.spec}, --where {args.where!r})") from e
