"""Where the repository keeps its data and generated files.

The package is installed from a checkout and reads the committed raw API
responses and processed Parquet datasets under `data/`, writing caches and
reports under `results/`. ROOT is that checkout: the directory above `src/`,
or `$BANZUKE_ROOT` when set (a copy of the data elsewhere).
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("BANZUKE_ROOT") or Path(__file__).resolve().parents[2])

DATA = ROOT / "data"
RAW_BANZUKE = DATA / "banzuke"      # {basho}_{Makuuchi,Juryo}.json
RAW_BASHO = DATA / "basho"          # {basho}.json: yusho, special prizes
PROCESSED = DATA / "processed"      # tidy, bouts, transitions .parquet
LOCKFILE = ROOT / "uv.lock"

RESULTS = ROOT / "results"
SCRATCH = RESULTS / "scratch"       # git-ignored caches and per-run dumps
BACKTEST_CACHE = SCRATCH / "backtest_cache.parquet"
OOF_CACHE = SCRATCH / "oof"
EXPLAIN_CACHE = SCRATCH / "explain"


class MissingDataError(FileNotFoundError):
    """The processed datasets are not where the package expects them."""


def require_processed() -> Path:
    """PROCESSED, or a MissingDataError naming the command that creates it."""
    if not (PROCESSED / "transitions.parquet").is_file():
        raise MissingDataError(
            f"processed data not found under {PROCESSED}; run `banzuke data update` "
            "(or `banzuke data build` if the raw JSON is present)")
    return PROCESSED


def load_tidy() -> pd.DataFrame:
    return pd.read_parquet(require_processed() / "tidy.parquet")


def load_transitions() -> pd.DataFrame:
    return pd.read_parquet(require_processed() / "transitions.parquet")


def load_bouts() -> pd.DataFrame:
    return pd.read_parquet(require_processed() / "bouts.parquet")
