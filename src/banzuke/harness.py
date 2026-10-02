"""Rolling-origin backtest: for each target basho, train every model on
strictly-prior transitions, predict the target banzuke, score it. Targets
are independent and run in worker processes (one LightGBM thread each)."""
import hashlib
import inspect
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context

import numpy as np
import pandas as pd

from banzuke.metrics import evaluate
from banzuke.models import MODELS
from banzuke.resolver import resolve

METRICS = ["exact", "exact_n", "gtb_points", "within1", "mae", "tau",
           "promo_f1", "demo_f1", "sanyaku_acc", "sanyaku_exact"]
# leave two cores to the rest of the machine (one worker saturates one core)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 2)

_TRANS = _TIDY = None


def _init_worker(trans, tidy):
    global _TRANS, _TIDY
    _TRANS, _TIDY = trans, tidy


def _run_target(args):
    target, jobs, train_start, kw_by_model, config, return_preds = args
    trans, tidy = _TRANS, _TIDY
    bashos = sorted(tidy["basho"].unique())
    prev = bashos[bashos.index(target) - 1]
    train = trans[(trans["next_basho"] < target) & trans["position_next"].notna()]
    if train_start:
        train = train[train["basho"] >= train_start]
    cands = trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
    actual = tidy[tidy["basho"] == target]
    mak_size = int(cands["mak_size"].iloc[0])  # sized like the previous banzuke
    rows, preds = [], []
    for name, seed in jobs:
        model = MODELS[name](seed=seed, **kw_by_model[name])
        model.fit(train)
        score = model.score(cands)
        pred = resolve(cands, score, mak_size)
        rows.append({"config": config, "model": name, "seed": seed, "basho": target,
                     **evaluate(pred, cands, actual)})
        if return_preds:
            base = model.base_score(cands) if hasattr(model, "base_score") else score
            p = cands.assign(score=score, base=base).merge(pred, on="rikishi_id")
            p["config"], p["model"], p["seed"], p["target"] = config, name, seed, target
            preds.append(p)
    return rows, preds


def run_backtest(model_names, targets, trans, tidy, progress=True, return_preds=False,
                 skip=None, train_start=None, seeds=(0,), model_kwargs=None,
                 config="base", workers=1):
    """skip: set of (model_name, seed, target) combos to leave out (cached).
    train_start: ignore training transitions from basho before this.
    seeds: one fit per seed (bag replicate); summarize() averages them per basho.
    model_kwargs: per-stage params and options; each model takes the keys it
    declares, and its prepare() hook may add inputs computed once (OOF scores)."""
    kw_by_model = {}
    for name in model_names:
        cls = MODELS[name]
        kw = {k: v for k, v in (model_kwargs or {}).items()
              if k in ("n_seeds", "base", "pair") or k in cls.OPTIONS}
        kw_by_model[name] = cls.prepare(kw, trans, workers, train_start)
    per_target = []
    for target in targets:
        jobs = [(n, s) for n in model_names for s in seeds
                if not (skip and (n, s, target) in skip)]
        if jobs:
            per_target.append((target, jobs, train_start, kw_by_model, config, return_preds))

    rows, preds, t0 = [], [], time.time()
    if workers > 1 and len(per_target) > 1:
        pool = ProcessPoolExecutor(min(workers, len(per_target)), mp_context=get_context("spawn"),
                                   initializer=_init_worker, initargs=(trans, tidy))
        outputs = pool.map(_run_target, per_target[::-1])  # latest (largest) targets first
    else:
        _init_worker(trans, tidy)
        pool, outputs = nullcontext(), map(_run_target, per_target)
    with pool:
        for n, (r, p) in enumerate(outputs, 1):
            rows.extend(r)
            preds.extend(p)
            if progress:
                print(f"\r{n}/{len(per_target)} basho, {time.time() - t0:.0f}s",
                      end="", file=sys.stderr, flush=True)
    if progress and per_target:
        print(file=sys.stderr)
    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(["config", "model", "seed", "basho"]).reset_index(drop=True)
    if return_preds:
        return results, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return results


def label_of(results):
    """One label per (model, config): the model letter, or model:config."""
    cfg = results["config"].astype(str)
    return np.where(cfg == "base", results["model"], results["model"] + ":" + cfg)


def block_bootstrap_ci(diff, reps=10000, block=6, seed=0):
    """95% CI of the mean of a per-basho paired difference; circular block
    bootstrap over `block` consecutive basho keeps neighbouring tournaments'
    dependence."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, (n + block - 1) // block))
    idx = (starts[:, :, None] + np.arange(block)) % n
    means = diff[idx.reshape(reps, -1)[:, :n]].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(results, baseline=None, since=None, until=None):
    """Per-label means plus paired comparisons against `baseline` (a label;
    default: the exact-slot leader). Seeds are averaged within (label, basho)
    first, so the paired unit is the basho; for exact slots and MAE reports
    the mean difference, its block-bootstrap 95% CI, a Wilcoxon signed-rank p
    and the win-loss count."""
    from scipy.stats import wilcoxon

    r = results[(results["basho"] >= (since or 0)) & (results["basho"] <= (until or 10**8))]
    per = r.assign(label=label_of(r)).groupby(["label", "basho"])[METRICS].mean()
    summary = per.groupby("label").mean().sort_values("exact", ascending=False)
    summary.insert(0, "n", per.groupby("label").size())
    if baseline not in summary.index:
        if baseline is not None:
            print(f"warning: baseline {baseline!r} not in results; using leader", file=sys.stderr)
        baseline = summary.index[0]
    summary.attrs["baseline"] = baseline
    for metric, tag, nd in (("exact_n", "exact", 2), ("mae", "mae", 3)):
        pivot = per[metric].unstack("label")
        d, ci, p, wl = [], [], [], []
        for m in summary.index:
            diff = (pivot[m] - pivot[baseline]).dropna()
            d.append(diff.mean())
            if m == baseline or len(diff) < 2:
                ci.append("")
                p.append(np.nan)
                wl.append("")
                continue
            lo, hi = block_bootstrap_ci(diff)
            ci.append(f"{lo:+.{nd}f},{hi:+.{nd}f}")
            p.append(wilcoxon(diff, zero_method="pratt").pvalue if (diff != 0).any() else 1.0)
            wl.append(f"{int((diff > 0).sum())}-{int((diff < 0).sum())}")
        summary[f"d_{tag}"], summary[f"ci_{tag}"], summary[f"p_{tag}"] = d, ci, p
        if tag == "exact":
            summary["wl"] = wl
    return summary


def fingerprint(**extra) -> str:
    """Identity of everything that determines a backtest row: processed data,
    the source of every module that affects results, the lockfile, and the
    caller's resolved configuration."""
    from banzuke import features, harness, metrics, models, resolver
    from banzuke.paths import LOCKFILE, PROCESSED

    h = hashlib.sha256()
    for f in ("tidy.parquet", "transitions.parquet", "bouts.parquet"):
        h.update((PROCESSED / f).read_bytes())
    for mod in (features, harness, metrics, models, resolver):
        h.update(inspect.getsource(mod).encode())
    h.update(LOCKFILE.read_bytes())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def parse_seeds(spec) -> tuple:
    """'0-4' or '0,2' -> (0, 1, 2, 3, 4) / (0, 2)."""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return tuple(out)


def parse_sets(items) -> dict:
    """['base.n_estimators=200', 'gap=0.25', 'context=false'] -> model kwargs;
    dotted keys address the LightGBM stages, values are JSON where possible."""
    kwargs: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
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
