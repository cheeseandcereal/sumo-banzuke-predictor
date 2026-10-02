"""Residual analysis for one model: where do exact-slot misses come from, are
they systematically biased, and do the confidence signals
(banzuke.confidence) separate reliable rows from shaky ones? Ends with the
convention audit (banzuke.conventions): every hard-coded resolver rule
re-measured against the committee's decisions.

    banzuke analyze [--model Ar] [--start 200401] [--end 202609]
                    [--seeds 0-2] [--set base.n_estimators=200]

Predictions are cached per (model, window, configuration) under
results/scratch/; --fresh recomputes them.
"""
from banzuke.cli._common import DEFAULT_WORKERS, CommandError, parse_seeds, parse_sets, subcommand


def add_parser(sub):
    ap = subcommand(sub, "analyze", __doc__, "residual analysis and confidence calibration")
    ap.add_argument("--model", default="Ar", metavar="NAME", help="ordering model")
    ap.add_argument("--start", type=int, default=200401, metavar="BASHO", help="first target basho")
    ap.add_argument("--end", type=int, default=None, metavar="BASHO",
                    help="last target basho (default: latest)")
    ap.add_argument("--seeds", default="0", metavar="SPEC",
                    help="bag replicates, e.g. 0-2; the first drives the residual "
                         "sections, all of them the seed-spread section")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in `banzuke backtest`")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="worker processes")
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke import analysis, conventions
    from banzuke.models import MODELS
    from banzuke.paths import load_tidy, load_transitions

    if args.model not in MODELS:
        raise CommandError(f"unknown model {args.model!r}; available: {list(MODELS)}")
    seeds = parse_seeds(args.seeds)
    kwargs = parse_sets(args.set)
    tidy, trans = load_tidy(), load_transitions()
    end = args.end or int(tidy["basho"].max())
    preds = analysis.cached_predictions(args.model, args.start, end, seeds, kwargs, trans, tidy,
                                        workers=args.workers, fresh=args.fresh)
    analysis.residuals(preds, args.model, args.start, end, seeds)
    analysis.calibration(preds, trans)
    print()
    conventions.report(trans)
