"""Rolling-origin backtest: for each target basho, train every model on
strictly-prior transitions, predict the target banzuke, score it."""
import sys
import time

import pandas as pd

from banzuke.metrics import evaluate
from banzuke.models import MODELS
from banzuke.resolver import resolve


def run_backtest(model_names, targets, trans, tidy, progress=True, return_preds=False,
                 skip=None, train_start=None, seed=0, model_kwargs=None,
                 mak_size_policy="prior"):
    """skip: set of (model_name, target) combos to leave out (already cached).
    train_start: ignore training transitions from basho before this.
    seed / model_kwargs: passed to every model's constructor.
    mak_size_policy: 'prior' sizes the predicted makuuchi like the previous
    banzuke (all a forecaster can know); 'actual' reads the target's size."""
    bashos = sorted(tidy["basho"].unique())
    prev_map = {b: p for p, b in zip(bashos, bashos[1:])}
    rows, preds = [], []
    t0 = time.time()
    for n, target in enumerate(targets, 1):
        prev = prev_map[target]
        train = trans[(trans["next_basho"] < target) & trans["position_next"].notna()]
        if train_start:
            train = train[train["basho"] >= train_start]
        cands = trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
        actual = tidy[tidy["basho"] == target]
        if mak_size_policy == "actual":
            mak_size = int((actual["division"] == 0).sum())
        else:
            mak_size = int(cands["mak_size"].iloc[0])
        for name in model_names:
            if skip and (name, target) in skip:
                continue
            model = MODELS[name](seed=seed, **(model_kwargs or {}))
            model.fit(train)
            score = model.score(cands)
            pred = resolve(cands, score, mak_size)
            rows.append(
                {"model": name, "basho": target, **evaluate(pred, cands, actual)}
            )
            if return_preds:
                base = model.base_score(cands) if hasattr(model, "base_score") else score
                p = cands.assign(score=score, base=base).merge(pred, on="rikishi_id")
                p["model"], p["target"] = name, target
                preds.append(p)
        if progress:
            print(f"\r{n}/{len(targets)} basho, {time.time() - t0:.0f}s",
                  end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    results = pd.DataFrame(rows)
    return (results, pd.concat(preds, ignore_index=True)) if return_preds else results


def summarize(results: pd.DataFrame, since: int | None = None) -> pd.DataFrame:
    from scipy.stats import binomtest

    if since:
        results = results[results["basho"] >= since]
    if "seed" in results.columns:  # average over seeds within (model, basho)
        results = (results.drop(columns="seed").groupby(["model", "basho"])
                   .mean(numeric_only=True).reset_index())
    summary = (
        results.groupby("model")
        .agg(
            exact=("exact", "mean"),
            exact_n=("exact_n", "mean"),
            gtb_points=("gtb_points", "mean"),
            within1=("within1", "mean"),
            mae=("mae", "mean"),
            tau=("tau", "mean"),
            promo_f1=("promo_f1", "mean"),
            demo_f1=("demo_f1", "mean"),
            sanyaku_acc=("sanyaku_acc", "mean"),
            sanyaku_exact=("sanyaku_exact", "mean"),
        )
        .sort_values("exact", ascending=False)
    )
    # paired sign test on exact slots vs the leader
    best = summary.index[0]
    pivot = results.pivot(index="basho", columns="model", values="exact_n")
    pvals, wl = {}, {}
    for m in summary.index:
        diff = pivot[m] - pivot[best]
        w, l = int((diff > 0).sum()), int((diff < 0).sum())
        wl[m] = f"{w}-{l}"
        pvals[m] = binomtest(w, w + l).pvalue if w + l else 1.0
    summary["wl_vs_best"] = pd.Series(wl)
    summary["p_vs_best"] = pd.Series(pvals)
    return summary
