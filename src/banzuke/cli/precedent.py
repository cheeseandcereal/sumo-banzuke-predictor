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
import pandas as pd

from banzuke.cli._common import CommandError, subcommand
from banzuke.experiments import precedent
from banzuke.paths import require_processed


def add_parser(sub):
    ap = subcommand(sub, "precedent", __doc__, "where the committee put a rank and record")
    ap.add_argument("cmd", choices=("landing", "pair", "cells"))
    ap.add_argument("spec", nargs="+")
    ap.add_argument("--where", default=None, help="pandas query on transition columns")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    t = precedent.labeled(pd.read_parquet(require_processed() / "transitions.parquet"))
    if args.cmd in ("landing", "cells"):
        for spec in args.spec:
            print(precedent.fmt_landing(spec, precedent.landing(t, spec, args.where)))
    else:
        if len(args.spec) != 2:
            raise CommandError("pair needs two specs")
        print(precedent.fmt_pair(*args.spec, precedent.pair_rate(t, *args.spec, args.where)))
