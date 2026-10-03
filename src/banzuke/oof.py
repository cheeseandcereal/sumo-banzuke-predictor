"""Rolling out-of-fold base scores: the reranker's training input.

Rows of source basho b are scored by the base stage fitted on transitions
labelled by b (next_basho <= b), the backtest's own base prediction for
target next(b), so no row is scored by a model that saw its label. The
default model's table is committed as data/processed/oof.parquet (built by
`banzuke data build`, its oof_key in the frame's attrs) and rebuilt in place
when that key no longer matches; any other base configuration is cached
under cache/oof/<key>.parquet. docs/MODEL.md 4.4.
"""
import hashlib
import inspect
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from banzuke.features import FEATURES
from banzuke.models import BASE_PARAMS, GBMRegression, GBMRerank, _cols

OOF_MIN_HISTORY = 60  # labelled basho required before a basho gets an out-of-fold score


def _base_kwargs(kwargs):
    """The options that reach the base stage (the pair stage shares the table)."""
    keys = ("n_seeds", "base", *GBMRegression.OPTIONS)
    return {k: kwargs[k] for k in keys if k in kwargs}


def oof_key(trans, model_cls, kwargs, min_history=OOF_MIN_HISTORY, train_start=None):
    """Identity of an out-of-fold table: the base configuration, BASE_PARAMS,
    the base-stage and _oof_one source, and the labelled data's content."""
    lab = trans[trans["position_next"].notna()]
    base_spec = [model_cls.objective, _base_kwargs(kwargs), BASE_PARAMS, min_history, train_start]
    key = hashlib.sha256(json.dumps(base_spec, sort_keys=True).encode())
    key.update((inspect.getsource(GBMRegression) + inspect.getsource(_oof_one)).encode())
    cols = FEATURES + _cols(kwargs.get("extra"))
    key.update(pd.util.hash_pandas_object(lab[["basho", "rikishi_id", "position_next"] + cols],
                                          index=False).to_numpy().tobytes())
    return key.hexdigest()[:16]


def oof_base_scores(trans, model_cls, kwargs, min_history=OOF_MIN_HISTORY, threads=1,
                    cache_dir=None, train_start=None):
    """The table for model_cls(**kwargs)'s base stage, read from the committed
    file or the cache when current, else computed (one process per thread)
    and written there."""
    # bound late: the tests point paths.OOF_TABLE/OOF_CACHE at a temp dir
    from banzuke.paths import OOF_CACHE, OOF_TABLE

    lab = trans[trans["position_next"].notna()]
    kwargs = {k: v for k, v in kwargs.items() if k not in ("oof", "seed")}
    key = oof_key(trans, model_cls, kwargs, min_history, train_start)
    committed = cache_dir is None and key == oof_key(trans, GBMRerank, {})
    path = OOF_TABLE if committed else Path(cache_dir or OOF_CACHE) / f"{key}.parquet"
    if path.exists():
        out = pd.read_parquet(path)
        if not committed or out.attrs.get("oof_key") == key:
            return out
        print(f"{path} is stale (data or base stage changed): rebuilding it", file=sys.stderr)

    todo = sorted(lab["basho"].unique())[min_history:]
    print(f"computing out-of-fold base scores for {len(todo)} basho ({threads} processes; "
          "cached afterwards)...", file=sys.stderr, flush=True)
    args = [(b, model_cls, kwargs, train_start) for b in todo]
    if threads > 1:
        with ProcessPoolExecutor(threads, mp_context=get_context("spawn"),
                                 initializer=_oof_init, initargs=(lab,)) as ex:
            parts = list(ex.map(_oof_one, args, chunksize=4))
    else:
        _oof_init(lab)
        parts = [_oof_one(a) for a in args]
    out = pd.concat(parts, ignore_index=True)
    out.attrs["oof_key"] = key
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


_OOF_LAB = None  # the labelled frame, set once per worker process


def _oof_init(lab):
    global _OOF_LAB
    _OOF_LAB = lab


def _oof_one(args):
    b, model_cls, kwargs, train_start = args
    train = _OOF_LAB[_OOF_LAB["next_basho"] <= b]
    if train_start:
        train = train[train["basho"] >= train_start]
    rows = _OOF_LAB[_OOF_LAB["basho"] == b]
    m = model_cls(**kwargs)
    GBMRegression.fit(m, train)  # base stage only
    return pd.DataFrame({"basho": rows["basho"].to_numpy(),
                         "rikishi_id": rows["rikishi_id"].to_numpy(),
                         "oof": GBMRegression.score(m, rows)})
