"""Shared pieces of the `banzuke` subcommands."""
import argparse


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
