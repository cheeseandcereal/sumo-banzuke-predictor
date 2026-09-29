"""Rolling-origin backtest: for each target basho, train every model on
strictly-prior transitions, predict the target banzuke, score it.

Targets are independent, so they run in worker processes; each worker
receives the data once and fits with one thread (BASE_PARAMS n_jobs=1).
"""
import hashlib
import inspect
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from banzuke.metrics import evaluate
from banzuke.models import MODELS
from banzuke.resolver import resolve

METRICS = ["exact", "exact_n", "gtb_points", "within1", "mae", "tau",
           "promo_f1", "demo_f1", "sanyaku_acc", "sanyaku_exact"]
PAIR_METRICS = ["pair_n", "pair_acc", "pair_prior_acc", "pair_logloss", "pair_outside"]

_TRANS = _TIDY = None


def _init_worker(trans, tidy, limit_threads=True):
    global _TRANS, _TIDY
    _TRANS, _TIDY = trans, tidy
    if limit_threads:  # worker processes only; never pin the caller's BLAS
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(1)
        except ImportError:
            pass


def _pair_diagnostics(model, cands):
    """How well the reranker adjudicated the near-tie pairs it saw, against
    the next banzuke: accuracy, the prior-order baseline (i stays above j),
    log loss, and the share of pairs farther apart than the training window."""
    lp = getattr(model, "last_pairs", None)
    if not lp:
        return {}
    i, j, p, in_window = lp
    nxt = cands["position_next"].to_numpy()
    ok = ~np.isnan(nxt[i]) & ~np.isnan(nxt[j])
    if not ok.any():
        return {}
    y = (nxt[i][ok] < nxt[j][ok]).astype(float)
    pc = np.clip(p[ok], 1e-6, 1 - 1e-6)
    return {
        "pair_n": int(ok.sum()),
        "pair_acc": float(((pc > 0.5) == (y == 1)).mean()),
        "pair_prior_acc": float(y.mean()),
        "pair_logloss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "pair_outside": float((~in_window).mean()),
    }


def _run_target(args):
    (target, jobs, train_start, kw_by_model, mak_size_policy, config,
     return_preds) = args
    trans, tidy = _TRANS, _TIDY
    bashos = sorted(tidy["basho"].unique())
    prev = bashos[bashos.index(target) - 1]
    train = trans[(trans["next_basho"] < target) & trans["position_next"].notna()]
    if train_start:
        train = train[train["basho"] >= train_start]
    cands = trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
    actual = tidy[tidy["basho"] == target]
    if mak_size_policy == "actual":
        mak_size = int((actual["division"] == 0).sum())
    else:
        mak_size = int(cands["mak_size"].iloc[0])
    rows, preds = [], []
    for name, seed in jobs:
        model = MODELS[name](seed=seed, **kw_by_model[name])
        model.fit(train)
        pred = resolve(cands, model.score(cands), mak_size)
        rows.append({"config": config, "model": name, "seed": seed, "basho": target,
                     **evaluate(pred, cands, actual), **_pair_diagnostics(model, cands)})
        if return_preds:
            p = cands.merge(pred, on="rikishi_id")
            p["config"], p["model"], p["seed"], p["target"] = config, name, seed, target
            preds.append(p)
    return rows, preds


def run_backtest(model_names, targets, trans, tidy, progress=True, return_preds=False,
                 skip=None, train_start=None, seeds=(0,), model_kwargs=None,
                 mak_size_policy="prior", config="base", workers=1):
    """skip: set of (model_name, seed, target) combos to leave out (cached).
    train_start: ignore training transitions from basho before this.
    seeds: one full fit per seed; summarize() averages seeds within a basho.
    model_kwargs: per-stage params and options; each model receives the keys
    it declares in OPTIONS (plus n_seeds/base/pair), and models with a
    `prepare` hook get target-independent inputs computed once here (e.g.
    rolling OOF scores).
    mak_size_policy: 'prior' sizes the predicted makuuchi like the previous
    banzuke (all a forecaster can know); 'actual' reads the target's size.
    workers: target-level processes."""
    model_kwargs = dict(model_kwargs or {})
    kw_by_model = {}
    for name in model_names:
        cls = MODELS[name]
        # each model takes the options it knows (so one --set list can drive a
        # multi-model run); prepared inputs (e.g. OOF scores) stay per model
        kw = {k: v for k, v in model_kwargs.items()
              if k in ("n_seeds", "base", "pair") or k in cls.OPTIONS}
        prep = getattr(cls, "prepare", None)
        kw_by_model[name] = prep(kw, trans, workers, train_start) if prep else kw
    per_target = []
    for target in targets:
        jobs = [(n, s) for n in model_names for s in seeds
                if not (skip and (n, s, target) in skip)]
        if jobs:
            per_target.append((target, jobs, train_start, kw_by_model, mak_size_policy,
                               config, return_preds))
    rows, preds = [], []
    t0 = time.time()

    def report(n):
        if progress:
            print(f"\r{n}/{len(per_target)} basho, {time.time() - t0:.0f}s",
                  end="", file=sys.stderr, flush=True)

    if workers > 1 and len(per_target) > 1:
        with ProcessPoolExecutor(min(workers, len(per_target)),
                                 mp_context=get_context("spawn"),
                                 initializer=_init_worker, initargs=(trans, tidy)) as ex:
            # longest jobs (latest targets, most training data) first
            for n, (r, p) in enumerate(ex.map(_run_target, per_target[::-1]), 1):
                rows.extend(r)
                preds.extend(p)
                report(n)
    else:
        _init_worker(trans, tidy, limit_threads=False)
        for n, a in enumerate(per_target, 1):
            r, p = _run_target(a)
            rows.extend(r)
            preds.extend(p)
            report(n)
    if progress and per_target:
        print(file=sys.stderr)
    results = pd.DataFrame(rows)
    if len(results):
        results = results.sort_values(["config", "model", "seed", "basho"]).reset_index(drop=True)
    return (results, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()) \
        if return_preds else results


def parse_seeds(spec: str) -> tuple:
    """'0-4' or '0,2' -> (0, 1, 2, 3, 4) / (0, 2): bag replicates to fit."""
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def parse_sets(items) -> dict:
    """CLI option overrides ['base.n_estimators=200', 'gap=0.25', 'pairs=oof']
    -> model kwargs; dotted keys address the LightGBM stages."""
    import json

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


def label_of(results):
    """One label per (model, config): the model letter, or model:config."""
    cfg = results["config"] if "config" in results else pd.Series("base", index=results.index)
    return np.where(cfg == "base", results["model"], results["model"] + ":" + cfg.astype(str))


def block_bootstrap_ci(diff, reps=10000, block=6, seed=0):
    """95% CI of the mean of a per-basho paired difference series, circular
    block bootstrap (blocks of `block` consecutive basho keep the temporal
    dependence between neighbouring tournaments)."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, (n + block - 1) // block))
    idx = (starts[:, :, None] + np.arange(block)) % n
    means = diff[idx.reshape(reps, -1)[:, :n]].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return (float(lo), float(hi))


def summarize(results: pd.DataFrame, baseline: str | None = None, since: int | None = None,
              until: int | None = None) -> pd.DataFrame:
    """Per-label means with paired comparisons against `baseline` (a label;
    default: the exact-slot leader, as in older logs). Seeds are averaged
    within (label, basho) first, so the paired unit is the basho. Reports the
    mean difference, its block-bootstrap 95% CI, a Wilcoxon signed-rank p
    (zeros kept, Pratt), and the win-loss count, for exact slots and MAE."""
    from scipy.stats import wilcoxon

    r = results
    if since:
        r = r[r["basho"] >= since]
    if until:
        r = r[r["basho"] <= until]
    r = r.assign(label=label_of(r))
    cols = METRICS + [c for c in PAIR_METRICS if c in r]
    per = r.groupby(["label", "basho"])[cols].mean()  # seed-average
    summary = per.groupby("label").mean().sort_values("exact", ascending=False)
    summary.insert(0, "n", per.groupby("label").size())
    if baseline is None or baseline not in summary.index:
        if baseline is not None:
            print(f"warning: baseline {baseline!r} not in results; using leader",
                  file=sys.stderr)
        baseline = summary.index[0]
    ex = per["exact_n"].unstack("label")
    ma = per["mae"].unstack("label")
    gt = per["gtb_points"].unstack("label")

    def paired(pivot, m, ci=True):
        d = (pivot[m] - pivot[baseline]).dropna()
        if len(d) < 2 or m == baseline:
            return d.mean() if len(d) else np.nan, (np.nan, np.nan), np.nan, "0-0"
        p = wilcoxon(d, zero_method="pratt").pvalue if (d != 0).any() else 1.0
        return (d.mean(), block_bootstrap_ci(d) if ci else (np.nan, np.nan), p,
                f"{int((d > 0).sum())}-{int((d < 0).sum())}")

    out = {k: [] for k in ("d_exact", "ci_exact", "p_exact", "wl", "d_gtb", "d_mae",
                           "ci_mae", "p_mae")}
    for m in summary.index:
        de, cie, pe, wl = paired(ex, m)
        dg, _, _, _ = paired(gt, m, ci=False)
        dm, cim, pm, _ = paired(ma, m)
        out["d_exact"].append(de)
        out["ci_exact"].append(_fmt_ci(cie))
        out["p_exact"].append(pe)
        out["wl"].append(wl)
        out["d_gtb"].append(dg)
        out["d_mae"].append(dm)
        out["ci_mae"].append(_fmt_ci(cim, 3))
        out["p_mae"].append(pm)
    for k, v in out.items():
        summary[k] = v
    summary.attrs["baseline"] = baseline
    return summary


def _fmt_ci(ci, nd=2):
    lo, hi = ci
    return "" if np.isnan(lo) else f"{lo:+.{nd}f},{hi:+.{nd}f}"


def fingerprint(extra: str = "") -> str:
    """Identity of everything that determines a backtest row: processed data,
    the source of every module that affects results, the lockfile, plus a
    caller-supplied string (resolved configuration, policy flags)."""
    from banzuke import features, harness, metrics, models, resolver
    from banzuke.build import PROCESSED

    h = hashlib.sha256()
    for f in ("tidy.parquet", "transitions.parquet", "bouts.parquet"):
        h.update((PROCESSED / f).read_bytes())
    for mod in (features, harness, metrics, models, resolver):
        h.update(inspect.getsource(mod).encode())
    lock = PROCESSED.parent.parent / "uv.lock"
    if lock.exists():
        h.update(lock.read_bytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]
