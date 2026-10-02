"""`python -m banzuke` is the `banzuke` command."""
import sys

from banzuke.cli import main

if __name__ == "__main__":  # the guard matters: worker processes re-import __main__
    sys.exit(main())
