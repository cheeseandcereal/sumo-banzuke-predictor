"""Shared pieces of the `banzuke` subcommands: the error type the dispatcher
reports without a traceback, the help formatter, and the grammar of the
options several commands share (--set, --seeds, --workers)."""
import argparse
import json
import os

# leave two cores to the rest of the machine (one worker saturates one core)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 2)


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
        if action.default in (None, False, "", []) or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def subcommand(sub, name, module_doc, help, **kwargs):
    """A subparser whose description is the command module's docstring."""
    return sub.add_parser(name, help=help, description=module_doc, formatter_class=Formatter,
                          **kwargs)


def parse_seeds(spec) -> tuple:
    """'0-4' or '0,2' -> (0, 1, 2, 3, 4) / (0, 2)."""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        try:
            out.extend(range(int(a), int(b or a) + 1))
        except ValueError:
            raise CommandError(f"--seeds expects N, A-B or a comma-separated list, got {spec!r}")
    return tuple(out)


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
