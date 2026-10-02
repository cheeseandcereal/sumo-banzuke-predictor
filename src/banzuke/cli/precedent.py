"""Committee precedent queries over the transitions, always with counts.

    banzuke precedent landing "K1 5-10"
    banzuke precedent landing "M1 8-7" --where "year>=2010"
    banzuke precedent pair "K 5-10" "M9-10 10-5"
    banzuke precedent cells "S 7-8"      # landing cells, counted

Spec grammar: <class>[<number>|<lo>-<hi>][E|W] <record>
    class   Y O S K M J
    record  9-6 | 5-7-3 (exact W-L[-A]) | 8+ (at least 8 wins) | kk | mk | *
Windows: all (1959+), 2004+ (modern, 42-man makuuchi), last60 (the 60 most
recent transitions). `--where` is a pandas query on the transition columns.
"""
from banzuke.cli._common import CommandError, subcommand


def add_parser(sub):
    ap = subcommand(sub, "precedent", __doc__, "where the committee put a rank and record")
    ap.add_argument("cmd", choices=("landing", "pair", "cells"),
                    help="landing/cells: where each SPEC landed; pair: how often the first "
                         "SPEC ranked above the second")
    ap.add_argument("spec", nargs="+", metavar="SPEC", help='e.g. "K1 5-10", "M9-10 10-5"')
    ap.add_argument("--where", default=None, metavar="QUERY",
                    help="pandas query on the transition columns, e.g. \"year>=2010\"")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke.experiments import precedent
    from banzuke.paths import load_transitions

    t = precedent.labeled(load_transitions())
    try:
        if args.cmd in ("landing", "cells"):
            for spec in args.spec:
                print(precedent.fmt_landing(spec, precedent.landing(t, spec, args.where)))
        else:
            if len(args.spec) != 2:
                raise CommandError("pair needs two specs")
            print(precedent.fmt_pair(*args.spec, precedent.pair_rate(t, *args.spec, args.where)))
    except ValueError as e:  # a malformed spec
        raise CommandError(str(e))
