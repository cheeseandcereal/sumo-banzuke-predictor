#!/usr/bin/env python3
"""E20: candidate committee rules measured on cached Ar frames (built by
experiments.explain). Each rule edits the model's order (pseudo scores) or
asserts class membership through the resolver's override mechanism; the frame
is re-resolved and re-evaluated, so every comparison is exactly paired with V0
(the cached order re-resolved by the current resolver) and needs no retraining.

    uv run python -m experiments.rules [--start 200401] [--rules R1,R4] [--workers 14]
    uv run python -m experiments.rules --cache results/scratch/explain/<old fingerprint>

A resolver edit changes the fingerprint, so `--cache` names the frames to
reuse. The `cached` variant scores the sheet as the cache's resolver made it,
so V0 - cached is the value of the resolver change itself, and a rule the
resolver now applies natively must be a no-op (`changed` 0 in every frame).

Rules (precedent counts in results/scratch/explain/review/adhoc.md):
  R1  zero wins with 8+ absences (S/K/M): full kyujo lands position+24,
      partial (some bouts lost) position+23
  R2  make-koshi sekiwake: 7 wins -> komusubi with priority over weak M1 claims
      (the K block grows only for 10+ claimants), <=6 wins -> maegashira
  R13 M1 claimants with 8-9 wins never create a third komusubi slot
      (weakest dropped to M until the block is 2; E15b "zone full")
  R3  make-koshi komusubi: <=6 wins, or 7 wins at K1W -> maegashira
  R4  demoted ozeki ranks below every other sekiwake claimant (bottom S slot)
  R5  a yokozuna with no bouts ranks below every yokozuna who fought
  R6  make-koshi ozeki who stay ozeki are ordered by wins
  R7  yokozuna rule needs the previous yusho / jun-yusho fought as ozeki
  R14 yokozuna who stay and ozeki who stay are ordered by wins, yusho winner
      first, a man who fought above a full-kyujo man, then prior position
      (contains R5 and R6)
  R8  ozeki rule also fires on 33+ over three basho with 12+ now when exactly
      one of the three was fought at M1-M3
  R9  an M1W claimant with <= 9 wins never gets a created (third) K slot
  R10 make-koshi with <= 6 wins drops at least 2 cells
  R11 S/K East incumbent keeps the cell over the West twin-rank KK incumbent
      unless West has 4+ more wins (2016+ lean, not a 100% rule)
  R12 komusubi newcomers rank below kachi-koshi komusubi incumbents
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from banzuke.build import YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO, PROCESSED
from banzuke.harness import DEFAULT_WORKERS, block_bootstrap_ci
from banzuke.metrics import evaluate
from banzuke.overrides import OverrideError
from banzuke.resolver import forced_claims, resolve
from experiments.explain import cache_dir

SCRATCH = Path(__file__).resolve().parent.parent / "results" / "scratch" / "explain"
WINDOWS = [("full 2004-2026", 0, 999999), ("2004-2018", 0, 201811), ("2019+", 201901, 999999),
           ("confirm 2020+", 202001, 999999)]
RULES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14"]
BUNDLE = ["R1", "R2", "R3", "R4", "R7", "R8", "R12", "R14"]  # 100%-precedent (or near) rules; R14 contains R5 and R6


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


# --- rules: each returns (pseudo scores or None, {row: class}) ---------------------

def r1_kyujo(f, pseudo, ov):
    base = np.isin(f.rank_class, (SEKIWAKE, KOMUSUBI, MAEGASHIRA)) & (f.wins == 0) & (f.absences >= 8)
    n = 0
    for i in np.flatnonzero(base):
        pseudo = f.place(pseudo, i, f.position[i] + (24 if f.absences[i] >= 14 else 23))
        n += 1
    return pseudo, ov, n


def _k_block(f, pseudo, ov):
    overrides = {"class": {f.rid[i]: int(c) for i, c in ov.items()}} if ov else None
    try:
        pred = resolve(f.g, pseudo, f.mak_size, overrides=overrides)
    except OverrideError:
        return None
    k = pred.loc[pred["pred_class"] == KOMUSUBI, "rikishi_id"]
    return [f.row_of[r] for r in k]


def _trim_weak_m1(f, pseudo, ov):
    """Drop M1 claimants with <= 9 wins (weakest by model order first) to M
    while the komusubi block exceeds 2. Returns the number dropped."""
    n = 0
    while True:
        block = _k_block(f, pseudo, ov)
        if block is None or len(block) <= 2:
            return n
        weak = [i for i in block if f.rank_class[i] == MAEGASHIRA and f.rank_number[i] == 1
                and f.wins[i] <= 9 and i not in ov]
        if not weak:
            return n
        ov[max(weak, key=lambda i: pseudo[i])] = MAEGASHIRA
        n += 1


def r2_mk_sekiwake(f, pseudo, ov):
    """S 7 wins -> K when at least two kachi-koshi sekiwake candidates exist
    (KK S incumbents, KK K incumbents, M claimants, demoted ozeki); the one
    modern exception (Goeido 201209, kept S1W) had nobody to take the slot."""
    cands = int((f.claims["s"] | ((f.rank_class == KOMUSUBI) & f.kk) | f.claims["m_claim"]).sum())
    n = 0
    seven = False
    for i in np.flatnonzero((f.rank_class == SEKIWAKE) & ~f.kk):
        if f.wins[i] == 7 and cands < 2:
            continue
        ov[i] = KOMUSUBI if f.wins[i] == 7 else MAEGASHIRA
        seven |= f.wins[i] == 7
        n += 1
    if seven:
        n += _trim_weak_m1(f, pseudo, ov)
    return pseudo, ov, n


def r13_weak_m1_claims(f, pseudo, ov):
    return pseudo, ov, _trim_weak_m1(f, pseudo, ov)


def r3_mk_komusubi(f, pseudo, ov):
    n = 0
    for i in np.flatnonzero((f.rank_class == KOMUSUBI) & ~f.kk):
        if f.wins[i] <= 6 or (f.wins[i] == 7 and f.side[i] == 1):
            ov[i] = MAEGASHIRA
            n += 1
    return pseudo, ov, n


def r4_demoted_ozeki(f, pseudo, ov):
    out = np.flatnonzero(f.claims["kadoban_out"])
    others = np.flatnonzero(f.claims["s"] & ~f.claims["kadoban_out"])
    if not len(out) or not len(others):
        return pseudo, ov, 0
    floor = pseudo[others].max()
    for k, i in enumerate(out):
        pseudo[i] = floor + 0.1 * (k + 1)
    return pseudo, ov, len(out)


def r5_yokozuna_kyujo(f, pseudo, ov):
    y = np.flatnonzero(f.rank_class == YOKOZUNA)
    fought = y[(f.wins[y] + f.losses[y]) > 0]
    idle = y[(f.wins[y] + f.losses[y]) == 0]
    if not len(fought) or not len(idle):
        return pseudo, ov, 0
    n = 0
    for i in idle:
        if pseudo[i] < pseudo[fought].max():
            pseudo[i] = pseudo[fought].max() + 0.1 * (n + 1)
            n += 1
    return pseudo, ov, n


def r6_mk_ozeki_by_wins(f, pseudo, ov):
    o = np.flatnonzero((f.rank_class == OZEKI) & ~f.kk & ~f.claims["kadoban_out"])
    if len(o) < 2:
        return pseudo, ov, 0
    want = o[np.lexsort((f.position[o], -f.wins[o]))]
    have = o[np.argsort(pseudo[o], kind="stable")]
    if np.array_equal(want, have):
        return pseudo, ov, 0
    slots = np.sort(pseudo[o])
    for i, s in zip(want, slots):
        pseudo[i] = s
    return pseudo, ov, 1


def r7_yokozuna_rule(f, pseudo, ov):
    """The previous yusho / jun-yusho must have been fought as ozeki (0/3 when
    it was at sekiwake; 6/8 when at ozeki, 2004+). Implemented by blanking the
    rule's inputs on the frame so the resolver's own Y logic does not fire."""
    fires = (f.rank_class == OZEKI) & (f.yusho == 1) & ((f.yusho1 == 1) | ((f.junyusho1 == 1) & (f.w1 >= 12)))
    rows = np.flatnonzero(fires & (f.class1 != OZEKI))
    if len(rows):
        f.g.loc[f.g.index[rows], ["yusho1", "junyusho1"]] = 0
    return pseudo, ov, len(rows)


def r14_yo_order_by_wins(f, pseudo, ov):
    """Yokozuna who stay, and ozeki who stay, are ordered by wins, then the
    yusho winner first, then a man who fought above a full-kyujo man, then
    prior position (all 139 Y pairs and 554 O pairs since 2004)."""
    n = 0
    for c in (YOKOZUNA, OZEKI):
        rows = np.flatnonzero((f.rank_class == c) & ~f.claims["kadoban_out"])
        if len(rows) < 2:
            continue
        fought = (f.wins[rows] + f.losses[rows]) > 0
        want = rows[np.lexsort((f.position[rows], ~fought, -f.yusho[rows], -f.wins[rows]))]
        have = rows[np.argsort(pseudo[rows], kind="stable")]
        if not np.array_equal(want, have):
            slots = np.sort(pseudo[rows])
            for i, s_ in zip(want, slots):
                pseudo[i] = s_
            n += 1
    return pseudo, ov, n


def r8_ozeki_rule(f, pseudo, ov):
    sk = np.isin(f.rank_class, (SEKIWAKE, KOMUSUBI))
    m_prev = ((f.class1 == MAEGASHIRA) & (f.num1 <= 3) & (f.class2 <= KOMUSUBI)) | (
        (f.class2 == MAEGASHIRA) & (f.num2 <= 3) & (f.class1 <= KOMUSUBI))
    extra = sk & (f.wins >= 12) & (f.roll3 >= 33) & np.isnan(f.ozeki_run3) & m_prev
    rows = np.flatnonzero(extra)
    if len(rows):  # let the resolver's own ozeki rule see the run
        f.g.loc[f.g.index[rows], "ozeki_run3"] = f.roll3[rows]
    return pseudo, ov, len(rows)


def r9_m1w_claim(f, pseudo, ov):
    k_block = np.flatnonzero(f.pred_class == KOMUSUBI)
    if len(k_block) <= 2:
        return pseudo, ov, 0
    n = 0
    for i in k_block:
        if f.rank_class[i] == MAEGASHIRA and f.rank_number[i] == 1 and f.side[i] == 1 and f.wins[i] <= 9:
            ov[i] = MAEGASHIRA
            n += 1
    return pseudo, ov, n


def r10_min_drop(f, pseudo, ov):
    rows = np.isin(f.rank_class, (SEKIWAKE, KOMUSUBI, MAEGASHIRA)) & (f.wins <= 6) & (f.pred_pos <= f.position + 1)
    n = 0
    for i in np.flatnonzero(rows):
        pseudo = f.place(pseudo, i, f.position[i] + 2)
        n += 1
    return pseudo, ov, n


def r11_east_keeps(f, pseudo, ov):
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


def r12_k_newcomers_below(f, pseudo, ov):
    k_block = np.flatnonzero(f.pred_class == KOMUSUBI)
    inc = [i for i in k_block if f.rank_class[i] == KOMUSUBI and f.kk[i]]
    new = [i for i in k_block if f.rank_class[i] > KOMUSUBI]
    if not inc or not new:
        return pseudo, ov, 0
    if pseudo[new].min() > pseudo[inc].max():
        return pseudo, ov, 0
    slots = np.sort(pseudo[k_block])
    want = sorted(inc, key=lambda i: pseudo[i]) + sorted(new, key=lambda i: pseudo[i])
    want += [i for i in sorted(k_block, key=lambda i: pseudo[i]) if i not in want]
    for i, s in zip(want, slots):
        pseudo[i] = s
    return pseudo, ov, 1


RULE_FUNCS = {"R1": r1_kyujo, "R2": r2_mk_sekiwake, "R3": r3_mk_komusubi, "R4": r4_demoted_ozeki,
              "R5": r5_yokozuna_kyujo, "R6": r6_mk_ozeki_by_wins, "R7": r7_yokozuna_rule,
              "R8": r8_ozeki_rule, "R9": r9_m1w_claim, "R10": r10_min_drop, "R11": r11_east_keeps,
              "R12": r12_k_newcomers_below, "R13": r13_weak_m1_claims, "R14": r14_yo_order_by_wins}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cache", default=None, metavar="DIR",
                    help="explain cache to reuse (default: the current fingerprint's)")
    args = ap.parse_args()
    d = Path(args.cache) if args.cache else cache_dir({}, list(range(args.seeds)))
    files = sorted(d.glob("rows_*.parquet"))
    if not files:
        sys.exit(f"no cached frames under {d}; run `python -m experiments.explain build` or pass --cache")
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = rows[rows["target"] >= args.start]
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    names = args.rules.split(",")
    variants = {r: [RULE_FUNCS[r]] for r in names}
    if len(names) > 1:
        variants["BUNDLE"] = [RULE_FUNCS[r] for r in BUNDLE if r in names]
        variants["BUNDLE-R1"] = [RULE_FUNCS[r] for r in BUNDLE if r in names and r != "R1"]
        variants["ALL"] = [RULE_FUNCS[r] for r in names]
    tasks = [(t, g, tidy[tidy["basho"] == t], variants) for t, g in rows.groupby("target")]
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(run_target, tasks, chunksize=2), 1):
            out += r
            print(f"\r{k}/{len(tasks)} targets", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    res = pd.DataFrame(out)
    res.to_parquet(SCRATCH / "rules_results.parquet", index=False)
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


if __name__ == "__main__":
    main()
