"""Shared pieces of the `banzuke` subcommands: the error type the dispatcher
reports without a traceback, the help formatter, the grammar of the options
several commands share (--set, --seeds, --threads) and the helpers that
declare them with one wording everywhere."""
import argparse
import json
import os

DEFAULT_THREADS = os.process_cpu_count() or 1


class CommandError(Exception):
    """A usage error detected after parsing (unknown model, no targets in the
    window, an override that cannot be applied...). The dispatcher prints the
    subcommand's usage and the message and exits with status 2, like an
    argparse error."""


class Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Module docstrings as written, plus `(default: x)` on options that have one."""

    def _get_help_string(self, action):
        # ArgumentDefaultsHelpFormatter appends the default even when the help
        # text already explains it; skip actions whose default is None/False/empty
        d = action.default
        if d is None or d is False or d == "" or d == [] or d is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def subcommand(sub, name, module_doc, help, **kwargs):
    """A subparser whose description is the command module's docstring."""
    return sub.add_parser(name, help=help, description=module_doc, formatter_class=Formatter,
                          **kwargs)


def action_parser(sub, name, help, description=None, **kwargs):
    """A nested action of a command (`banzuke explain sheet`): the help line
    doubles as the description unless a longer one is given."""
    return sub.add_parser(name, help=help, description=description or help,
                          formatter_class=Formatter, **kwargs)


# --- options several commands share -------------------------------------------

def add_model(ap, default=None, help="ordering model (docs/MODEL.md lists them)"):
    ap.add_argument("--model", default=default, metavar="NAME", help=help)


def add_fresh(ap, help="recompute instead of using the cache"):
    ap.add_argument("--fresh", action="store_true", help=help)


def add_set(ap, help=None):
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help=help or "model option override, repeatable: dotted keys address the "
                                 "LightGBM stages (base.*, pair.*), bare keys are model options "
                                 "(n_seeds, gap, cluster_max, context, near_ties, extra, ...)")


def add_seeds(ap, default, help):
    ap.add_argument("--seeds", default=default, metavar="SPEC", help=help + " (e.g. 0, 0-2, 0,3)")


def add_threads(ap, help="CPU threads: worker processes, one single-threaded fit each"):
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=help)


def add_window(ap, start, start_help="first target basho", end_help="last target basho"):
    ap.add_argument("--start", type=int, default=start, metavar="BASHO", help=start_help)
    ap.add_argument("--end", type=int, default=None, metavar="BASHO",
                    help=end_help + " (default: latest)")


def add_train_start(ap):
    ap.add_argument("--train-start", type=int, default=None, metavar="BASHO",
                    help="ignore training transitions before this basho "
                         "(default: all history since 1959; docs/EXPERIMENTS.md E9)")


def targets_in(tidy, start, end):
    """The basho of `tidy` inside start..end (end None = latest); a usage error
    when there are none."""
    end = end or int(tidy["basho"].max())
    targets = [b for b in sorted(tidy["basho"].unique()) if start <= b <= end]
    if not targets:
        raise CommandError(f"no target basho in {start}..{end}")
    return targets, end


def model_class(name):
    """The model class registered under `name`, or a usage error."""
    from banzuke.models import MODELS

    if name not in MODELS:
        raise CommandError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]


def model_classes(names):
    """Comma-separated model names -> classes, in order; None = every model."""
    from banzuke.models import MODELS

    return [model_class(n) for n in names.split(",")] if names else list(MODELS.values())


def build_model(model_cls, kwargs, trans, threads=1, train_start=None):
    """prepare() and construct a model from --set kwargs; an option the model
    does not declare, or a bad value, is a usage error rather than a traceback."""
    try:
        kwargs = model_cls.prepare(kwargs, trans, threads, train_start)
        return model_cls(**kwargs, threads=threads)
    except (TypeError, ValueError) as e:
        raise CommandError(f"--set: {e}") from e


# --- option value grammar -----------------------------------------------------

def parse_seeds(spec) -> tuple:
    """'0-4' or '0,2' -> (0, 1, 2, 3, 4) / (0, 2)."""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        try:
            out.extend(range(int(a), int(b or a) + 1))
        except ValueError:
            raise CommandError(f"--seeds expects N, A-B or a comma-separated list, got {spec!r}") from None
    return tuple(out)


def set_configs(args) -> dict:
    """The --set options as one labelled configuration, {label: model kwargs};
    the label is the --set text itself, or "base" without any."""
    return {",".join(args.set) or "base": parse_sets(args.set)}


def parse_sets(items) -> dict:
    """['base.n_estimators=200', 'gap=0.25', 'context=false'] -> model kwargs;
    dotted keys address the LightGBM stages, values are JSON where possible."""
    kwargs: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise CommandError(f"--set expects KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if "." in key:
            stage, k = key.split(".", 1)
            kwargs.setdefault(stage, {})[k] = value
        else:
            kwargs[key] = value
    return kwargs
