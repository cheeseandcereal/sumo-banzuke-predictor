"""Candidate committee rules measured on cached Ar frames (built by
`banzuke explain build`), without retraining. A rule edits the model's order
(pseudo scores), asserts class membership through the resolver's override
mechanism, or edits the frame columns a resolver convention reads; the frame
is re-resolved and re-evaluated, exactly paired with V0 (the cached order
re-resolved by the current resolver). Used for E20 (whose rules now live in
banzuke.resolver) and for the E22 port check.

    banzuke rules [--start 200401] [--rules R11,R13] [--workers 14]
    banzuke rules --cache cache/explain/<old fingerprint>

A resolver edit changes the fingerprint, so `--cache` names the frames to
reuse. The `cached` variant scores the sheet as the cache's resolver made it,
so V0 - cached is the value of the resolver change itself, and a rule the
resolver now applies natively must be a no-op (`changed` 0 in every frame).

Writing a rule: `fn(f, pseudo, ov) -> (pseudo, ov, n_events)` with `f` a
Frame (numpy views of the candidate columns, `f.claims` the forced S/K
claims, `f.g` the frame the resolver will read), `pseudo` the order as
fractional ranks (lower = higher; `f.place` moves a man to a cell), `ov`
{row: class} assertions. Register it in RULE_FUNCS. The two kept below are
the leads the log left open (E20, E22); the adopted rules are in the
resolver.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from banzuke.ranks import KOMUSUBI, MAEGASHIRA, SEKIWAKE
from banzuke.experiments.explain import NoCache, cache_dir
from banzuke.harness import block_bootstrap_ci
from banzuke.metrics import evaluate
from banzuke.overrides import OverrideError
from banzuke.paths import EXPLAIN_CACHE as SCRATCH
from banzuke.resolver import forced_claims, resolve

WINDOWS = [("full 2004-2026", 0, 999999), ("2004-2018", 0, 201811), ("2019+", 201901, 999999),
           ("confirm 2020+", 202001, 999999)]


class Frame:
    """One cached (target, seed) frame with V0 pseudo scores = order index."""

    def __init__(self, g):
        self.g = g.copy()
        self.n = len(g)
        self.order0 = np.lexsort((g["position"].to_numpy(), g["score"].to_numpy()))
        self.rank0 = np.empty(self.n)
        self.rank0[self.order0] = np.arange(self.n)
        for c in ("rank_class", "rank_number", "side", "wins", "losses", "absences", "position",
                  "pred_class", "pred_pos", "kadoban", "yusho", "yusho1", "junyusho1", "w1",
                  "roll3", "ozeki_run3", "sanyaku3", "class1", "num1", "class2", "num2"):
            setattr(self, c, g[c].to_numpy())
        self.kk = self.wins >= 8
        self.claims = forced_claims(g)
        self.rid = g["rikishi_id"].to_numpy()
        self.row_of = {r: i for i, r in enumerate(self.rid)}
        self.mak_size = int(g["mak_size"].iloc[0])

    def place(self, pseudo, i, target_pos):
        """Move row i so it lands at joint position target_pos: just ahead of
        whoever the model predicted there."""
        pseudo[i] = min(target_pos, self.n - 1) - 0.5
        return pseudo


# --- rules: each returns (pseudo scores, {row: class}, events) ------------------------

def _k_block(f, pseudo, ov):
    overrides = {"class": {f.rid[i]: int(c) for i, c in ov.items()}} if ov else None
    try:
        pred = resolve(f.g, pseudo, f.mak_size, overrides=overrides)
    except OverrideError:
        return None
    k = pred.loc[pred["pred_class"] == KOMUSUBI, "rikishi_id"]
    return [f.row_of[r] for r in k]


def r13_weak_m1_claims(f, pseudo, ov):
    """M1 claimants with 8-9 wins never create a third komusubi slot: the
    weakest (by model order) go back to maegashira until the block is 2
    (class assertions; E15b "zone full", honoured 58% historically)."""
    n = 0
    while True:
        block = _k_block(f, pseudo, ov)
        if block is None or len(block) <= 2:
            return pseudo, ov, n
        weak = [i for i in block if f.rank_class[i] == MAEGASHIRA and f.rank_number[i] == 1
                and f.wins[i] <= 9 and i not in ov]
        if not weak:
            return pseudo, ov, n
        ov[max(weak, key=lambda i: pseudo[i])] = MAEGASHIRA
        n += 1


def r11_east_keeps(f, pseudo, ov):
    """S/K East incumbent keeps the cell over the West twin-rank kachi-koshi
    incumbent unless West has 4+ more wins (order edit; a 2016+ lean, 1/7
    passed, not a rule yet)."""
    n = 0
    for c in (SEKIWAKE, KOMUSUBI):
        rows = np.flatnonzero((f.rank_class == c) & f.kk)
        for num in np.unique(f.rank_number[rows]):
            pair = rows[f.rank_number[rows] == num]
            if len(pair) != 2:
                continue
            e, w = pair[np.argsort(f.side[pair])]
            if pseudo[w] < pseudo[e] and f.wins[w] - f.wins[e] < 4:
                pseudo[e], pseudo[w] = pseudo[w], pseudo[e]
                n += 1
    return pseudo, ov, n


RULE_FUNCS = {"R11": r11_east_keeps, "R13": r13_weak_m1_claims}


# --- evaluation ----------------------------------------------------------------------

def run_target(task):
    target, g_all, actual, variants = task
    mak_size = int(g_all["mak_size"].iloc[0])
    rows = []
    for seed, g in g_all.groupby("seed"):
        g = g.reset_index(drop=True)
        rid = g["rikishi_id"].to_numpy()

        def score(f, pseudo, ov):
            overrides = {"class": {rid[i]: int(c) for i, c in ov.items()}} if ov else None
            try:
                pred = resolve(f.g, pseudo, mak_size, overrides=overrides)
            except OverrideError as e:
                return None, str(e)
            return evaluate(pred, g, actual), pred

        f = Frame(g)
        m0, pred0 = score(f, f.rank0.copy(), {})
        pp0 = pred0.set_index("rikishi_id")["pred_pos"].reindex(rid).to_numpy()
        repro = bool(np.array_equal(pp0, g["pred_pos"].to_numpy()))
        rows.append({"target": target, "seed": seed, "variant": "V0", "events": 0, "repro": repro, "fail": False, **m0})
        cached = g[["rikishi_id", "pred_class", "pred_number", "pred_side", "pred_pos"]]
        rows.append({"target": target, "seed": seed, "variant": "cached", "events": 0, "repro": repro, "fail": False,
                     "changed": int((g["pred_pos"].to_numpy() != pp0).sum()), **evaluate(cached, g, actual)})
        for name, funcs in variants.items():
            f = Frame(g)  # fresh copy: rules may edit frame columns
            pseudo, ov, n = f.rank0.copy(), {}, 0
            for fn in funcs:
                pseudo, ov, k = fn(f, pseudo, ov)
                n += k
            m, pred = score(f, pseudo, ov)
            fail = m is None
            if fail:
                m = m0
            changed = 0 if fail or n == 0 else int((pred.set_index("rikishi_id")["pred_pos"].reindex(rid).to_numpy()
                                                    != pp0).sum())
            rows.append({"target": target, "seed": seed, "variant": name, "events": n, "repro": True,
                         "fail": fail, "changed": changed, **m})
    return rows


def paired(per, names, lo, hi):
    sub = per[(per["target"] >= lo) & (per["target"] <= hi)]
    b = sub[sub["variant"] == "V0"].set_index("target").sort_index()
    out = []
    for v in names:
        x = sub[sub["variant"] == v].set_index("target").reindex(b.index)
        dx, dm = x["exact_n"] - b["exact_n"], x["mae"] - b["mae"]
        lo_x, hi_x = block_bootstrap_ci(dx)
        lo_m, hi_m = block_bootstrap_ci(dm)
        out.append({"variant": v, "ev/basho": x["events"].mean(), "frames touched": float((x["events"] > 0).mean()),
                    "d_exact": dx.mean(), "ci": f"{lo_x:+.2f},{hi_x:+.2f}",
                    "W-L": f"{int((dx > 0).sum())}-{int((dx < 0).sum())}",
                    "p": wilcoxon(dx, zero_method="pratt").pvalue if (dx != 0).any() else 1.0,
                    "d_mae": dm.mean(), "ci_mae": f"{lo_m:+.3f},{hi_m:+.3f}",
                    "d_gtb": (x["gtb_points"] - b["gtb_points"]).mean(),
                    "d_sanyaku": (x["sanyaku_exact"] - b["sanyaku_exact"]).mean()})
    return pd.DataFrame(out).set_index("variant"), len(b), b["exact_n"].mean(), b["mae"].mean()


# --- driver ------------------------------------------------------------------------

def load_frames(start=200401, seeds=(0, 1, 2), cache=None):
    """Cached Ar frames for targets >= start: the current fingerprint's explain
    build, or the directory `cache` (an older build to re-resolve)."""
    d = Path(cache) if cache else cache_dir({}, list(seeds))
    files = sorted(d.glob("rows_*.parquet"))
    if not files:
        raise NoCache(f"no cached frames under {d}; run `banzuke explain build` or pass --cache")
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return rows[rows["target"] >= start]


def measure(rows, tidy, names, workers=1):
    """Every rule in `names` alone and, with several, all of them together
    ("ALL", in the order given), re-resolved frame by frame against V0.
    Returns (per-frame results, variants); the results are also written to
    rules_results.parquet under the explain cache."""
    unknown = sorted(set(names) - set(RULE_FUNCS))
    if unknown:
        raise KeyError(f"unknown rules {unknown}; available: {list(RULE_FUNCS)}")
    variants = {r: [RULE_FUNCS[r]] for r in names}
    if len(names) > 1:
        variants["ALL"] = [RULE_FUNCS[r] for r in names]
    tasks = [(t, g, tidy[tidy["basho"] == t], variants) for t, g in rows.groupby("target")]
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(run_target, tasks, chunksize=2), 1):
            out += r
            print(f"\r{k}/{len(tasks)} targets", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    res = pd.DataFrame(out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    res.to_parquet(SCRATCH / "rules_results.parquet", index=False)
    return res, variants


def report(res, variants):
    v0 = res[res["variant"] == "V0"]
    print(f"V0 reproduces the cached sheet in {v0['repro'].mean():.3f} of frames; "
          f"variant resolves failed: {int(res['fail'].sum())}")
    noop = (res[res["variant"].isin(variants)].groupby("variant")["changed"].agg(lambda c: int((c > 0).sum())))
    print("frames changed per variant: " + ", ".join(f"{v} {n}" for v, n in noop.items()))
    per = res.groupby(["variant", "target"]).mean(numeric_only=True).reset_index()
    pd.set_option("display.width", 220)
    order = [v for v in list(variants) if v in set(per["variant"])]
    if not v0["repro"].all():
        # the cache was built by an older resolver: V0 (current) is the baseline,
        # `cached` the old resolver's sheet, V0 - cached the value of the change
        order = ["cached"] + order
    for w, lo, hi in WINDOWS:
        tab, n, ex0, mae0 = paired(per, order, lo, hi)
        print(f"\n=== {w}: {n} basho, V0 exact {ex0:.3f} MAE {mae0:.4f}; paired deltas vs V0 (seeds averaged) ===")
        print(tab.round(4).to_string())
