"""Fetch raw sumo data from sumo-api.com and rebuild the processed datasets.

    banzuke data update    # fetch what is new, then rebuild (completed basho are skipped)
    banzuke data fetch     # update the raw JSON under data/ only
    banzuke data build     # rebuild data/processed/ from the existing JSON only

Run after a banzuke release or a completed basho. Raw responses land in
data/banzuke/{basho}_{Makuuchi,Juryo}.json and data/basho/{basho}.json; the
build writes tidy, bouts and transitions .parquet to data/processed/, then
the default model's out-of-fold table oof.parquet (one process per thread;
skipped when the committed one still matches, or with --skip-oof). Commit
all four with the raw JSON.
"""
from banzuke.cli._common import action_parser, add_threads, subcommand


def add_parser(sub):
    ap = subcommand(sub, "data", __doc__, "fetch raw data and rebuild the processed datasets")
    ap.set_defaults(_parser=ap)
    what = ap.add_subparsers(dest="action", metavar="ACTION", required=True)
    steps = {"update": ("fetch new raw data, then rebuild", (True, True)),
             "fetch": ("fetch raw API data without rebuilding", (True, False)),
             "build": ("rebuild the Parquet datasets from the raw JSON", (False, True))}
    for name, (help_, (fetch_, build_)) in steps.items():
        p = action_parser(what, name, help_)
        p.set_defaults(run=run, _parser=p, fetch=fetch_, build=build_)
        if build_:
            p.add_argument("--skip-oof", action="store_true",
                           help="leave the out-of-fold table (oof.parquet) as it is")
            add_threads(p, "processes for the out-of-fold table")
    return ap


def run(args):
    if args.fetch:
        from banzuke.fetch import fetch_all

        fetch_all()
    if args.build:
        from banzuke.build import main as build

        build(oof=not args.skip_oof, threads=args.threads)
