"""The `banzuke` command: forecasting, data maintenance, evaluation and the
research tools behind one entry point.

    banzuke predict                 # the next banzuke from the latest results
    banzuke data update             # fetch new basho, rebuild the datasets
    banzuke backtest / analyze / conventions / gtb
    banzuke explain / precedent / rules

`banzuke COMMAND --help` documents each one.
"""
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from banzuke.cli._common import CommandError
from banzuke.paths import MissingDataError

DESCRIPTION = """\
Predict the next makuuchi banzuke from basho results.

A clone is ready to use: `banzuke predict` trains on the committed datasets
under data/ (about 30 s, nothing to precompute) and prints the forecast with
confidence markers; `banzuke data update` fetches newly completed basho
first. The evaluation commands (backtest, analyze, conventions, gtb) and the
research tools (explain, precedent, rules) are documented in docs/.
"""


def _version():
    try:
        return version("sumo-banzuke")
    except PackageNotFoundError:  # running from a checkout that is not installed
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="banzuke", description=DESCRIPTION,
                                 epilog="`banzuke COMMAND --help` documents each command.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", required=True, title="commands")
    from banzuke.cli import (analyze, backtest, conventions, data, explain, gtb, precedent, predict,
                             rules)
    for mod in (predict, data, backtest, analyze, conventions, gtb, explain, precedent, rules):
        mod.add_parser(sub)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.run(args) or 0
    except (CommandError, MissingDataError) as e:
        args._parser.error(str(e))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0
