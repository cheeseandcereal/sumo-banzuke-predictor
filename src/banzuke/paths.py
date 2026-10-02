"""Where the repository keeps its data and generated files.

The package is installed from a checkout and reads the committed raw API
responses and processed Parquet datasets under `data/`. Everything it
regenerates on demand (keyed caches, per-run dumps) goes under `cache/`, the
one git-ignored directory. ROOT is that checkout: the directory above `src/`,
or `$BANZUKE_ROOT` when set (a copy of the data elsewhere).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BANZUKE_ROOT") or Path(__file__).resolve().parents[2])

DATA = ROOT / "data"
RAW_BANZUKE = DATA / "banzuke"      # {basho}_{Makuuchi,Juryo}.json
RAW_BASHO = DATA / "basho"          # {basho}.json: yusho, special prizes
PROCESSED = DATA / "processed"      # tidy, bouts, transitions, oof .parquet
OOF_TABLE = PROCESSED / "oof.parquet"  # the default model's out-of-fold base scores
LOCKFILE = ROOT / "uv.lock"

CACHE = ROOT / "cache"              # git-ignored: keyed caches and per-run dumps
BACKTEST_CACHE = CACHE / "backtest_cache.parquet"
OOF_CACHE = CACHE / "oof"
EXPLAIN_CACHE = CACHE / "explain"


class MissingDataError(FileNotFoundError):
    """The processed datasets are not where the package expects them."""


def require_processed() -> Path:
    """PROCESSED, or a MissingDataError naming the command that creates it."""
    if not (PROCESSED / "transitions.parquet").is_file():
        raise MissingDataError(
            f"processed data not found under {PROCESSED}; run `banzuke data update` "
            "(or `banzuke data build` if the raw JSON is present)")
    return PROCESSED


def load_tidy():
    import pandas as pd

    return pd.read_parquet(require_processed() / "tidy.parquet")


def load_transitions():
    import pandas as pd

    return pd.read_parquet(require_processed() / "transitions.parquet")


def load_bouts():
    import pandas as pd

    return pd.read_parquet(require_processed() / "bouts.parquet")
