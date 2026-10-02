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
from banzuke.cli._common import (add_seeds, add_set, add_window, add_workers, model_class, parse_seeds,
                                 parse_sets, subcommand)


def add_parser(sub):
    ap = subcommand(sub, "analyze", __doc__, "residual analysis and confidence calibration")
    ap.add_argument("--model", default="Ar", metavar="NAME", help="ordering model")
    add_window(ap, 200401)
    add_seeds(ap, "0", "bag replicates; the first drives the residual sections, all of them "
                       "the seed-spread section")
    add_set(ap)
    add_workers(ap)
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def run(args):
    from banzuke import analysis, conventions
    from banzuke.paths import load_tidy, load_transitions

    model_class(args.model)
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
