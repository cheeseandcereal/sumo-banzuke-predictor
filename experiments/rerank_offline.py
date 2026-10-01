#!/usr/bin/env python3
"""E24: reranker aggregation variants measured offline on cached Ar frames
(experiments.explain build, PAIR_GAP >= the widest cluster, so every
in-cluster pair probability is on disk). The reranker is rebuilt from `rows`
(base scores, positions, twin units) and `pairs` (p), checked against the
cached order, then re-aggregated per variant, re-resolved by the current
resolver and re-evaluated: exactly paired with V0, no model needed.

    uv run python -m experiments.rerank_offline [--cache DIR] [--start 200401] [--workers 14]

Variants:
  borda      the production aggregation, rebuilt (must reproduce the cache)
  exclude    rule-decided men (Y/O incumbents, rule promotions and returns,
             kadoban make-koshi ozeki) leave the reranker: the others are
             clustered and aggregated without them, and they are merged back
             at their base-score position
  kemeny     within a cluster, the unit order maximizing the summed
             probability of the pairs it places consistently (<= 6 units,
             720 permutations), Borda order as the tiebreak
  copeland   pairwise wins (p > .5), Borda as the tiebreak
  kemeny+exclude, copeland+exclude
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from banzuke.build import OZEKI, PROCESSED
from banzuke.harness import DEFAULT_WORKERS
from banzuke.metrics import evaluate
from banzuke.models import GBMRerank, _twin_units
from banzuke.overrides import OverrideError
from banzuke.resolver import resolve, rule_masks
from experiments.explain import cache_dir
from experiments.rules import WINDOWS, paired

SCRATCH = Path(__file__).resolve().parent.parent / "results" / "scratch" / "explain"
VARIANTS = ["borda", "exclude", "kemeny", "copeland", "kemeny+exclude", "copeland+exclude"]
OPTS = {k: GBMRerank.OPTIONS[k] for k in ("gap", "cluster_max", "twin_unit")}


def clusters_of(g):
    """Units (row-index lists), clusters (unit-index lists), unit base scores
    and positions, exactly as GBMRerank.score forms them."""
    base, pos = g["base"].to_numpy(), g["position"].to_numpy()
    units = [[i] for i in range(len(g))]
    if OPTS["twin_unit"]:
        for e, w in _twin_units(g, OPTS["twin_unit"]):
            units[e], units[w] = [e, w], []
    units = [u for u in units if u]
    ubase = np.array([base[u].mean() for u in units])
    upos = np.array([pos[u].min() for u in units])
    order = np.lexsort((upos, ubase))
    clusters, cur = [], [order[0]]
    for prev, i in zip(order, order[1:]):
        if ubase[i] - ubase[prev] <= OPTS["gap"] and len(cur) < OPTS["cluster_max"]:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return units, clusters, ubase, upos


def unit_probs(g, units, clusters, upos, p_of):
    """{(a, b): P(unit a stays above unit b)} for every in-cluster unit pair,
    a the currently higher unit: the mean of the member pairs' p."""
    rid = g["rikishi_id"].to_numpy()
    out = {}
    for cl in clusters:
        for a, b in combinations(cl, 2):
            if upos[a] > upos[b]:
                a, b = b, a
            out[(a, b)] = float(np.mean([p_of[(rid[i], rid[j])] for i in units[a] for j in units[b]]))
    return out


def aggregate(cl, pu, ubase, how):
    """Order of one cluster's units under an aggregation rule."""
    def p(a, b):  # P(a above b)
        return pu[(a, b)] if (a, b) in pu else 1 - pu[(b, a)]

    borda = {u: sum(p(u, v) for v in cl if v != u) for u in cl}
    borda_order = sorted(cl, key=lambda u: (-borda[u], ubase[u]))
    if how == "borda" or len(cl) == 1:
        return borda_order
    if how == "copeland":
        wins = {u: sum((p(u, v) > .5) + .5 * (p(u, v) == .5) for v in cl if v != u) for u in cl}
        return sorted(cl, key=lambda u: (-wins[u], -borda[u], ubase[u]))
    if how == "kemeny":
        rank = {u: k for k, u in enumerate(borda_order)}
        best, best_key = None, None
        for perm in permutations(cl):
            agreed = sum(p(perm[i], perm[j]) for i in range(len(perm)) for j in range(i + 1, len(perm)))
            key = (-round(agreed, 9), sum(abs(rank[u] - k) for k, u in enumerate(perm)))  # closest to Borda
            if best_key is None or key < best_key:
                best, best_key = perm, key
        return list(best)
    raise ValueError(how)


def rerank_rows(g, p_of, how):
    """Final order (row indices of g, top first) under `how`."""
    units, clusters, ubase, upos = clusters_of(g)
    pu = unit_probs(g, units, clusters, upos, p_of)
    pos = g["position"].to_numpy()
    return [i for cl in clusters for u in aggregate(cl, pu, ubase, how)
            for i in sorted(units[u], key=lambda i: pos[i])]


def rerank(g, p_of, how, exclude=None):
    """Integer ranks (lower = higher) for the rows of g. With `exclude`, the
    masked rows take no part: the rest are reranked alone, each final slot
    takes the k-th smallest kept base score, and the excluded rows are merged
    back by their own base score (prior position as the tiebreak)."""
    if exclude is None or not exclude.any():
        final = rerank_rows(g, p_of, how)
    else:
        keep = np.flatnonzero(~exclude)
        sub = g.iloc[keep].reset_index(drop=True)
        order = [keep[i] for i in rerank_rows(sub, p_of, how)]
        base, pos = g["base"].to_numpy(), g["position"].to_numpy()
        key = base.astype(float).copy()
        key[order] = np.sort(base[keep])
        final = list(np.lexsort((pos, key)))
    out = np.empty(len(g))
    out[final] = np.arange(len(final))
    return out


def excluded(g):
    rm = rule_masks(g)
    return ((g["rank_class"].to_numpy() <= OZEKI) | rm["y_promo"] | rm["o_promo"]
            | rm["o_return"] | rm["kadoban_out"])


def run_target(task):
    target, g_all, pairs_all, actual = task
    mak_size = int(g_all["mak_size"].iloc[0])
    rows = []
    for seed, g in g_all.groupby("seed"):
        g = g.reset_index(drop=True)
        pairs = pairs_all[pairs_all["seed"] == seed]
        p_of = dict(zip(zip(pairs["rid_i"], pairs["rid_j"]), pairs["p"]))
        cached = np.empty(len(g))
        cached[np.lexsort((g["position"].to_numpy(), g["score"].to_numpy()))] = np.arange(len(g))
        repro = bool(np.array_equal(rerank(g, p_of, "borda"), cached))

        def score(order):
            try:
                pred = resolve(g, order, mak_size)
            except OverrideError:
                return None, None
            return evaluate(pred, g, actual), pred.set_index("rikishi_id")["pred_pos"].reindex(g["rikishi_id"]).to_numpy()

        m0, pp0 = score(cached)
        rows.append({"target": target, "seed": seed, "variant": "V0", "events": 0, "repro": repro, "fail": False, **m0})
        ex = excluded(g)
        units, clusters, _, _ = clusters_of(g)
        n_ex = sum(1 for cl in clusters if len(cl) > 1 for u in cl for i in units[u] if ex[i])
        for name in VARIANTS:
            how, _, flag = name.partition("+")
            use_ex = bool(flag) or how == "exclude"
            order = rerank(g, p_of, "borda" if how == "exclude" else how, ex if use_ex else None)
            m, pp = score(order)
            fail = m is None
            changed = 0 if fail else int((pp != pp0).sum())
            rows.append({"target": target, "seed": seed, "variant": name, "repro": repro, "fail": fail,
                         "events": n_ex if use_ex else int((order != cached).sum()),
                         "changed": changed, **(m0 if fail else m)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--cache", default=None, metavar="DIR",
                    help="explain cache to reuse (default: the current fingerprint's)")
    args = ap.parse_args()
    d = Path(args.cache) if args.cache else cache_dir({}, list(range(args.seeds)))
    files = sorted(d.glob("rows_*.parquet"))
    if not files:
        sys.exit(f"no cached frames under {d}")
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pairs = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("pairs_*.parquet"))], ignore_index=True)
    rows = rows[rows["target"] >= args.start]
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    tasks = [(t, g, pairs[pairs["target"] == t], tidy[tidy["basho"] == t]) for t, g in rows.groupby("target")]
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(run_target, tasks, chunksize=2), 1):
            out += r
            print(f"\r{k}/{len(tasks)} targets", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    res = pd.DataFrame(out)
    res.to_parquet(SCRATCH / "rerank_offline_results.parquet", index=False)
    v0 = res[res["variant"] == "V0"]
    print(f"rebuilt Borda reproduces the cached order in {v0['repro'].mean():.3f} of frames; "
          f"failed resolves: {int(res['fail'].sum())}")
    chg = res[res["variant"] != "V0"].groupby("variant")["changed"].agg(lambda c: int((c > 0).sum()))
    print("frames changed per variant: " + ", ".join(f"{v} {n}" for v, n in chg.items()))
    per = res.groupby(["variant", "target"]).mean(numeric_only=True).reset_index()
    pd.set_option("display.width", 220)
    for w, lo, hi in WINDOWS:
        tab, n, ex0, mae0 = paired(per, VARIANTS, lo, hi)
        print(f"\n=== {w}: {n} basho, V0 exact {ex0:.3f} MAE {mae0:.4f}; paired deltas vs V0 (seeds averaged) ===")
        print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
