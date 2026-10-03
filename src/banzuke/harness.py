"""Rolling-origin backtest: for each target basho, train every model on
strictly-prior transitions, predict the target banzuke, score it. Targets
are independent and run in worker processes (one LightGBM thread each)."""
import hashlib
import inspect
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from banzuke.metrics import evaluate
from banzuke.models import MODELS
from banzuke.ranks import in_window
from banzuke.resolver import resolve

METRICS = ["exact", "exact_n", "gtb_points", "within1", "mae", "tau",
           "promo_f1", "demo_f1", "sanyaku_acc", "sanyaku_exact"]

_TRANS = _TIDY = None  # the frames every target reads, set once per worker process


def _init_worker(trans, tidy):
    global _TRANS, _TIDY
    _TRANS, _TIDY = trans, tidy


def pmap(fn, tasks, threads, init, initargs, label=None, chunksize=1):
    """fn over tasks in up to `threads` spawned processes (in this process when
    one is enough), each initialised once with init(*initargs). Yields the
    results in task order; label ("basho") turns on a progress line."""
    t0 = time.time()
    if threads > 1 and len(tasks) > 1:
        pool = ProcessPoolExecutor(min(threads, len(tasks)), mp_context=get_context("spawn"),
                                   initializer=init, initargs=initargs)
        with pool:
            outputs = pool.map(fn, tasks, chunksize=chunksize)
            yield from _progress(outputs, len(tasks), label, t0)
    else:
        init(*initargs)
        yield from _progress(map(fn, tasks), len(tasks), label, t0)


def _progress(outputs, n, label, t0):
    for i, out in enumerate(outputs, 1):
        yield out
        if label:
            print(f"\r{i}/{n} {label}, {time.time() - t0:.0f}s", end="", file=sys.stderr, flush=True)
    if label and n:
        print(file=sys.stderr)


def split(trans, tidy, target, train_start=None):
    """The frames of one backtest target: (prev, train, cands, actual, mak_size).
    Training rows are the transitions decided before `target`, candidates the
    previous banzuke's rows still on a sheet, the makuuchi sized like it."""
    bashos = sorted(tidy["basho"].unique())
    prev = bashos[bashos.index(target) - 1]
    train = trans[(trans["next_basho"] < target) & trans["position_next"].notna()]
    if train_start:
        train = train[train["basho"] >= train_start]
    cands = trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
    return prev, train, cands, tidy[tidy["basho"] == target], int(cands["mak_size"].iloc[0])


def _run_target(args):
    target, jobs, train_start, kw_by_model, config, return_preds = args
    _, train, cands, actual, mak_size = split(_TRANS, _TIDY, target, train_start)
    rows, preds = [], []
    for name, seed in jobs:
        model = MODELS[name](seed=seed, **kw_by_model[name])
        model.fit(train)
        score = model.score(cands)
        pred = resolve(cands, score, mak_size)
        rows.append({"config": config, "model": name, "seed": seed, "basho": target,
                     **evaluate(pred, cands, actual)})
        if return_preds:
            base = model.base_score(cands) if model.base_score else score
            p = cands.assign(score=score, base=base).merge(pred, on="rikishi_id")
            p["config"], p["model"], p["seed"], p["target"] = config, name, seed, target
            preds.append(p)
    return rows, preds


def run_backtest(model_names, targets, trans, tidy, progress=True, return_preds=False,
                 skip=None, train_start=None, seeds=(0,), model_kwargs=None,
                 config="base", threads=1):
    """skip: set of (model_name, seed, target) combos to leave out (cached).
    train_start: ignore training transitions from basho before this.
    seeds: one fit per seed (bag replicate); summarize() averages them per basho.
    model_kwargs: per-stage params and options; each model takes the keys it
    declares, and its prepare() hook may add inputs computed once (OOF scores)."""
    kw_by_model, used = {}, {"n_seeds", "base", "pair"}
    for name in model_names:
        model_cls = MODELS[name]
        kw = {k: v for k, v in (model_kwargs or {}).items()
              if k in ("n_seeds", "base", "pair") or k in model_cls.OPTIONS}
        used |= set(model_cls.OPTIONS)
        kw_by_model[name] = model_cls.prepare(kw, trans, threads, train_start)
    unknown = set(model_kwargs or {}) - used
    if unknown:
        raise ValueError(f"no model in {list(model_names)} has options {sorted(unknown)}")
    per_target = []
    for target in targets:
        jobs = [(n, s) for n in model_names for s in seeds
                if not (skip and (n, s, target) in skip)]
        if jobs:
            per_target.append((target, jobs, train_start, kw_by_model, config, return_preds))

    rows, preds = [], []
    # latest (largest) targets first: the longest fits start earliest
    for r, p in pmap(_run_target, per_target[::-1], threads, _init_worker, (trans, tidy),
                     label="basho" if progress else None):
        rows.extend(r)
        preds.extend(p)
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

    r = results[in_window(results["basho"], since, until)]
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


def run_cached(model_names, targets, trans, tidy, configs, seeds=(0,), train_start=None,
               threads=1, fresh=False, cache=None):
    """run_backtest() for each labelled configuration in `configs` ({label:
    model_kwargs}), reusing rows of the on-disk store keyed by fingerprint()
    and writing the new ones back. fresh recomputes everything."""
    from banzuke.paths import BACKTEST_CACHE

    cache = cache or BACKTEST_CACHE
    store = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    if "seed" not in store:
        store = pd.DataFrame()  # older cache layout
    results = []
    for label, kwargs in configs.items():
        fp = fingerprint(kwargs=kwargs, train_start=train_start)
        cached = pd.DataFrame()
        if len(store) and not fresh:
            cached = store[(store["fp"] == fp) & store["model"].isin(model_names)
                           & store["seed"].isin(seeds) & store["basho"].isin(targets)]
            cached = cached.drop(columns="fp").assign(config=label)
            if len(cached):
                print(f"[{label}] reusing {len(cached)} cached rows", file=sys.stderr)
        skip = set(zip(cached["model"], cached["seed"], cached["basho"])) if len(cached) else None
        computed = run_backtest(model_names, targets, trans, tidy, skip=skip, seeds=seeds,
                                train_start=train_start, model_kwargs=kwargs,
                                config=label, threads=threads)
        if len(computed):
            store = pd.concat([store, computed.assign(fp=fp)], ignore_index=True)
        results += [f for f in (cached, computed) if len(f)]
    store = store.drop_duplicates(["fp", "model", "seed", "basho"], keep="last")
    cache.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(cache, index=False)
    return pd.concat(results, ignore_index=True)
