"""Re-measure every committee convention the resolver hard-codes against the
committee's actual decisions: violations/cases over the whole record, 2004+,
the last 60 and the last 30 basho, with the last violating basho. '!' marks a
rule clean since 2004 that was violated recently, '~' one violated more often
recently than over 2004+ as a whole. `banzuke analyze` ends with the same
table.
"""
from banzuke.cli._common import subcommand


def add_parser(sub):
    ap = subcommand(sub, "conventions", __doc__, "audit the resolver's conventions against history")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke import conventions
    from banzuke.paths import load_transitions

    conventions.report(load_transitions())
