"""Rolling-origin backtest: for each target basho, train every model on
strictly-prior transitions, predict the target banzuke, score it."""
import sys
import time

import pandas as pd

from banzuke.metrics import evaluate
from banzuke.models import MODELS
from banzuke.resolver import resolve


def run_backtest(model_names, targets, trans, tidy, progress=True) -> pd.DataFrame:
    bashos = sorted(tidy["basho"].unique())
    prev_map = {b: p for p, b in zip(bashos, bashos[1:])}
    rows = []
    t0 = time.time()
    for n, target in enumerate(targets, 1):
        prev = prev_map[target]
        train = trans[(trans["next_basho"] < target) & trans["position_next"].notna()]
        cands = trans[(trans["basho"] == prev) & ~trans["dropped"]].reset_index(drop=True)
        actual = tidy[tidy["basho"] == target]
        mak_size = int((actual["division"] == 0).sum())
        for name in model_names:
            model = MODELS[name]()
            model.fit(train)
            pred = resolve(cands, model.score(cands), mak_size)
            rows.append(
                {"model": name, "basho": target, **evaluate(pred, cands, actual)}
            )
        if progress:
            print(f"\r{n}/{len(targets)} basho, {time.time() - t0:.0f}s",
                  end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, since: int | None = None) -> pd.DataFrame:
    from scipy.stats import binomtest

    if since:
        results = results[results["basho"] >= since]
    summary = (
        results.groupby("model")
        .agg(
            exact=("exact", "mean"),
            exact_n=("exact_n", "mean"),
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
