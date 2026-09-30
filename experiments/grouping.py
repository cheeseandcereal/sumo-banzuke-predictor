#!/usr/bin/env python3
"""E16: same-rank E/W "twins" with identical records. Does the committee keep
them adjacent more often than the model does, and does gluing them together
in the model's order help?

Everything is post-processing on analyze.py's cached Ar predictions: a
variant permutes the model's order, re-resolves and re-evaluates, so every
comparison is exactly paired (same fitted scores, same seed) with zero
retraining.

    uv run python -m experiments.grouping describe [--screen-end 201911]
    uv run python -m experiments.grouping backtest [--primary twins:1:mid]

Vocabulary: a group is a set of candidates with identical W-L-A records that
were adjacent on the previous banzuke. `twins` = E and W of one rank number,
`xrank` = adjacent across a rank number (M8w + M9e), `both` = union-find of
the two (chains). An interloper is a candidate the model placed between
group members; k = how many. An anchor rule says where interlopers go when
the group is glued: `mid` (above iff base score beats the group's mean
base), `prior` (above iff ranked above the group before), `climber` (the
riser goes above, the faller below), `below`/`above` (all one way), `oracle`.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from banzuke.build import MAEGASHIRA
from banzuke.metrics import evaluate
from banzuke.overrides import OverrideError
from banzuke.resolver import resolve

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SCRATCH = ROOT / "results" / "scratch"
KINDS = ("twins", "xrank", "both")
ANCHORS = ("mid", "prior", "climber", "below", "above")
INF = 10 ** 6
WINDOWS = [("full", 0, 999999), ("2004-2019", 0, 201911), ("2014+", 201401, 999999),
           ("2020+", 202001, 999999)]


def load_preds(start, end, seeds, workers=1):
    cache = SCRATCH / f"preds_Ar_{start}_{end}_s{seeds}.parquet"
    tidy = pd.read_parquet(PROCESSED / "tidy.parquet")
    if not cache.exists():
        import inspect

        from banzuke.harness import run_backtest

        trans = pd.read_parquet(PROCESSED / "transitions.parquet")
        targets = [b for b in sorted(tidy["basho"].unique()) if start <= b <= end]
        kw = {"seeds": range(seeds)}
        if "workers" in inspect.signature(run_backtest).parameters:  # tuning branch
            kw["workers"] = workers
        _, preds = run_backtest(["Ar"], targets, trans, tidy, return_preds=True, **kw)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(cache, index=False)
    return pd.read_parquet(cache), tidy


# --- groups and glue ---------------------------------------------------------

def find_groups(g, kind, wins_only=False, mak_only=False):
    """Groups (lists of row indices in prior order) of identical-record
    candidates adjacent on the previous banzuke, per the kind above."""
    pos, cls, num = (g[c].to_numpy() for c in ("position", "rank_class", "rank_number"))
    cols = ["wins"] if wins_only else ["wins", "losses", "absences"]
    rec = list(zip(*(g[c].to_numpy() for c in cols)))
    ok = (cls == MAEGASHIRA) if mak_only else (cls >= MAEGASHIRA)
    by_pos = np.argsort(pos, kind="stable")
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    pairs = []
    for a, b in zip(by_pos, by_pos[1:]):
        if pos[b] - pos[a] != 1 or not (ok[a] and ok[b]) or rec[a] != rec[b]:
            continue
        twin = cls[a] == cls[b] and num[a] == num[b]
        if kind == "both" or (kind == "twins") == twin:
            pairs.append((a, b))
            parent[find(b)] = find(a)
    groups = {}
    for a, b in pairs:
        groups.setdefault(find(a), set()).update((a, b))
    return sorted((sorted(m, key=lambda i: pos[i]) for m in groups.values()),
                  key=lambda m: pos[m[0]])


def adjacent(vals):
    return bool(np.all(np.diff(np.sort(vals)) == 1))


def above_set(xs, members, anchor, base, pos, nxt, glued=None):
    """Interlopers sent above the glued group under an anchor rule. Interlopers
    that already form a glued block (glued: row -> block key) move as one unit
    so an earlier glue is never undone."""
    units = {}
    for x in xs:
        units.setdefault(glued.get(x, x) if glued else x, []).append(x)
    up = set()
    for u in units.values():
        if anchor == "mid":
            go = base[u].mean() < base[members].mean()
        elif anchor == "prior":
            go = pos[u[0]] < pos[members[0]]
        elif anchor == "climber":
            go = pos[u[0]] > pos[members[-1]]
        elif anchor == "oracle":
            go = nxt[u].min() < nxt[members].min()
        else:
            go = anchor == "above"
        if go:
            up.update(u)
    return up


def glue_one(order, members, v, base, pos, nxt, glued=None):
    """Make `members` contiguous (prior order) in `order`, sending each
    interloper above or below per v["anchor"]. None if nothing changes.
    A k=0 group whose members the model inverted is simply put back in
    prior order (identical records never invert historically)."""
    rank = {i: r for r, i in enumerate(order)}
    lo, hi = min(rank[i] for i in members), max(rank[i] for i in members)
    seg = order[lo:hi + 1]
    mset = set(members)
    xs = [i for i in seg if i not in mset]
    if v.get("decide") == "oracle":
        if not adjacent(nxt[members]):
            return None
    else:
        if len(xs) > v.get("K", INF):
            return None
        band = v.get("band", 0)
        if band and any(abs(base[x] - base[members].mean()) < band for x in xs):
            return None
    up = above_set(xs, members, v["anchor"], base, pos, nxt, glued)
    new = [x for x in xs if x in up] + list(members) + [x for x in xs if x not in up]
    return None if new == seg else order[:lo] + new + order[hi + 1:]


def apply_variant(order, groups, v, base, pos, nxt):
    order, n, glued = list(order), 0, {}
    for members in groups:  # top-down; groups are disjoint but their segments can overlap
        new = glue_one(order, members, v, base, pos, nxt, glued)
        if new is not None:
            order, n = new, n + 1
            glued.update({i: tuple(members) for i in members})
    return order, n


def vname(v):
    parts = [v["kind"], "oracle" if v.get("decide") == "oracle"
             else f"K{'inf' if v.get('K', INF) >= INF else v['K']}", v["anchor"]]
    if v.get("band"):
        parts.append(f"band{v['band']}")
    if v.get("wins_only"):
        parts.append("wins")
    if v.get("mak_only"):
        parts.append("mak")
    return "/".join(parts)


def variant_grid(default_anchor):
    vs = [dict(kind=k, K=K, anchor=a) for k in KINDS for K in (1, 2, INF) for a in ANCHORS]
    vs += [dict(kind="twins", K=INF, anchor="mid", band=b) for b in (0.25, 0.5, 1.0)]
    for K in (1, INF):
        vs.append(dict(kind="twins", K=K, anchor=default_anchor, wins_only=True))
        vs.append(dict(kind="twins", K=K, anchor=default_anchor, mak_only=True))
    vs += [dict(kind="twins", decide="oracle", anchor=a) for a in ("mid", "prior", "oracle")]
    vs.append(dict(kind="both", decide="oracle", anchor="oracle"))
    return vs


# --- per-group description (no resolver) --------------------------------------

def frame_arrays(g):
    order = list(np.lexsort((g["position"].to_numpy(), g["score"].to_numpy())))
    rank = np.empty(len(g), dtype=int)
    rank[order] = np.arange(len(g))
    nxt = g["position_next"].to_numpy()
    return dict(order=order, rank=rank, base=g["base"].to_numpy(), pos=g["position"].to_numpy(),
                nxt=nxt, by_nxt={int(p): i for i, p in enumerate(nxt)},
                cls=g["rank_class"].to_numpy(), num=g["rank_number"].to_numpy(),
                w=g["wins"].to_numpy(), l=g["losses"].to_numpy(), a=g["absences"].to_numpy(),
                cnext=g["class_next"].to_numpy(), rid=g["rikishi_id"].to_numpy())


def group_row(kind, members, A):
    rank, order, base, pos, nxt = A["rank"], A["order"], A["base"], A["pos"], A["nxt"]
    first = members[0]
    lo, hi = rank[members].min(), rank[members].max()
    mset = set(members)
    xs = [order[r] for r in range(lo, hi + 1) if order[r] not in mset]
    mn, mx = nxt[members].min(), nxt[members].max()
    row = {
        "kind": kind, "size": len(members), "k": len(xs), "rid": A["rid"][first],
        "rank_class": A["cls"][first], "rank_number": A["num"][first],
        "wins": A["w"][first], "losses": A["l"][first], "absences": A["a"][first],
        "base_gap": base[members].max() - base[members].min(),
        "inverted": any(rank[i] > rank[j] for i, j in zip(members, members[1:])),
        "actual_adj": adjacent(nxt[members]), "actual_gap": mx - mn,
        "order_kept": all(nxt[i] < nxt[j] for i, j in zip(members, members[1:])),
        "touches_mak": bool((A["cnext"][members] <= MAEGASHIRA).any()),
    }
    truth = {x for x in xs if nxt[x] < mn}
    for anc in ANCHORS:
        row[f"q_{anc}"] = ((above_set(xs, members, anc, base, pos, nxt) == truth)
                           if xs and row["actual_adj"] else np.nan)
    if len(xs) == 1:
        x = xs[0]
        row.update(x_origin="faller" if pos[x] < pos[first] else "climber",
                   x_actual="above" if nxt[x] < mn else "below" if nxt[x] > mx else "between",
                   x_mid=base[x] - base[members].mean(), x_dwins=A["w"][x] - A["w"][first])
    # committee side: who actually landed inside a split group
    inside = [A["by_nxt"][p] for p in range(int(mn) + 1, int(mx))
              if p in A["by_nxt"] and A["by_nxt"][p] not in mset]
    row["n_intruders"] = int(mx - mn + 1 - len(members))
    if len(inside) == 1:
        c = inside[0]
        row.update(c_origin="faller" if pos[c] < pos[first] else "climber",
                   c_dwins=A["w"][c] - A["w"][first], c_in_model=bool(lo < rank[c] < hi))
    return row


def pair_table(preds, kinds=KINDS):
    rows = []
    for (target, seed), g in preds.groupby(["target", "seed"]):
        g = g.reset_index(drop=True)
        A = frame_arrays(g)
        for kind in kinds:
            for members in find_groups(g, kind):
                rows.append({"target": target, "seed": seed, **group_row(kind, members, A)})
    pt = pd.DataFrame(rows)
    pt["div"] = np.where(pt["rank_class"] == MAEGASHIRA, "M", "J")  # by the group's first member
    pt["rec"] = (pt["wins"].astype(str) + "-" + pt["losses"].astype(str)
                 + np.where(pt["absences"] > 0, "-" + pt["absences"].astype(str), ""))
    pt["zone"] = np.select([pt["div"] == "J", pt["rank_number"] <= 5, pt["rank_number"] <= 11],
                           ["J", "M1-5", "M6-11"], "M12+")
    pt["model_adj"] = (pt["k"] == 0) & ~pt["inverted"]
    return pt


# --- describe ----------------------------------------------------------------

def window_mask(df, lo, hi):
    return (df["target"] >= lo) & (df["target"] <= hi)


def committee_tables(pt):
    print("=== committee: identical-record groups adjacent on the previous banzuke ===")
    print("(seed 0 frames; adj = still adjacent next basho; kept = prior order kept;"
          " 1-intruder = share of splits with exactly one slot inside;"
          " faller = share of known intruders who fell from above)")
    rows = []
    for w, lo, hi in WINDOWS:
        s = pt[window_mask(pt, lo, hi) & (pt["size"] == 2)]
        if not len(s):
            continue
        for kind in ("twins", "xrank"):
            for div in ("M", "J"):
                q = s[(s["kind"] == kind) & (s["div"] == div)]
                sp = q[~q["actual_adj"]]
                origin = sp.loc[sp["n_intruders"] == 1, "c_origin"].dropna()
                rows.append({"window": w, "kind": kind, "div": div, "n": len(q),
                             "adj": q["actual_adj"].mean(), "kept": q["order_kept"].mean(),
                             "1-intruder": (sp["n_intruders"] == 1).mean() if len(sp) else np.nan,
                             "faller": origin.eq("faller").mean() if len(origin) else np.nan})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    tw = pt[pt["kind"] == "twins"]
    print("\ntwins (M+J) adjacency by record (n >= 8):")
    t = tw.groupby("rec").agg(n=("actual_adj", "size"), adj=("actual_adj", "mean"))
    print(t[t["n"] >= 8].sort_values("n", ascending=False).round(3).to_string())
    print("\ntwins adjacency by zone:")
    print(tw.groupby("zone").agg(n=("actual_adj", "size"), adj=("actual_adj", "mean"))
          .round(3).to_string())


def model_tables(pt):
    print("\n=== model (Ar, seed 0): interlopers in the final order ===")
    for kind in KINDS:
        s = pt[pt["kind"] == kind]
        print(f"\n{kind}: p(kept adjacent | k interlopers)")
        rows = []
        for w, lo, hi in WINDOWS:
            q = s[window_mask(s, lo, hi)]
            if not len(q):
                continue
            r = {"window": w, "n": len(q), "inverted": int(q["inverted"].sum()),
                 "model_adj": q["model_adj"].mean(), "actual_adj": q["actual_adj"].mean()}
            for kk in (0, 1, 2):
                m = q[q["k"] == kk] if kk < 2 else q[q["k"] >= 2]
                lab = f"k={kk}" if kk < 2 else "k>=2"
                r[f"n {lab}"] = len(m)
                r[f"p {lab}"] = m["actual_adj"].mean() if len(m) else np.nan
            rows.append(r)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    tw = pt[pt["kind"] == "twins"]
    print("\ntwins: model vs committee adjacency (all windows)")
    print(pd.crosstab(tw["model_adj"].rename("model_adj"), tw["actual_adj"].rename("actual_adj")))
    print("\nanchor accuracy q on kept groups with k >= 1 (all interlopers on the right side)")
    rows = []
    for kind in KINDS:
        for w, lo, hi in WINDOWS:
            q = pt[(pt["kind"] == kind) & window_mask(pt, lo, hi) & pt["actual_adj"] & (pt["k"] >= 1)]
            if len(q):
                rows.append({"kind": kind, "window": w, "n": len(q),
                             **{a: q[f"q_{a}"].mean() for a in ANCHORS}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    one = tw[tw["k"] == 1]
    print("\ntwins with one interloper: where the committee put him")
    print(pd.crosstab([one["x_origin"], np.sign(one["x_dwins"]).map({-1: "fewer wins", 1: "more wins", 0: "same"})],
                      one["x_actual"]))
    print("\n|base_x - group mid| by outcome (band sizing), twins k=1:")
    print(one.groupby("x_actual")["x_mid"].apply(lambda s: s.abs().describe()[["count", "25%", "50%", "75%"]])
          .unstack().round(2).to_string())


def choose_primary(pt, screen_end):
    """Fix the pre-registered variant from the screening window only: twins
    touching makuuchi (the metric is makuuchi-only), anchor = best q, K = 1
    (the trio accounting below is exact for one interloper only; larger k
    is left to the exploratory grid)."""
    s = pt[(pt["kind"] == "twins") & (pt["target"] <= screen_end) & (pt["k"] >= 1) & pt["touches_mak"]]
    kept_mask = s["actual_adj"] & s["order_kept"]
    kept = s[kept_mask]
    q = {a: float(kept[f"q_{a}"].mean()) for a in ANCHORS}
    anchor = max(q, key=q.get)
    print(f"\n=== primary variant, fixed on targets <= {screen_end} ===")
    print("q by anchor on kept twin splits:", {a: round(v, 3) for a, v in q.items()}, f"(n={len(kept)})")
    print("expected change per glued split under the chosen anchor, from p(k) and q(k), k=1 trio"
          " accounting: exact (given block placed right) = 3pq + p - 2; |err| = 2 - 4pq (lower is better)")
    for kk in (1, 2, 3):
        m = s[s["k"] == kk] if kk < 3 else s[s["k"] >= 3]
        km = kept_mask[m.index]
        p = km.mean() if len(m) else np.nan
        qq = m.loc[km, f"q_{anchor}"].mean() if km.any() else np.nan
        lab = f"k={kk}" if kk < 3 else "k>=3"
        note = "" if kk == 1 else "  (k=1 formula, indicative only)"
        print(f"  {lab:5s} n={len(m):3d} p={p:.3f} q={qq:.3f} exact={3 * p * qq + p - 2:+.2f}"
              f" |err|={2 - 4 * p * qq:+.2f}{note}")
    name = f"twins:1:{anchor}"
    print(f"primary = {name}   (pass as --primary to backtest)")
    return name


def describe(args):
    preds, _ = load_preds(args.start, args.end, args.seeds, args.workers)
    pt = pair_table(preds[preds["seed"] == 0])
    committee_tables(pt)
    model_tables(pt)
    choose_primary(pt, args.screen_end)


# --- backtest ----------------------------------------------------------------

def run_target(task):
    target, gt, actual, variants = task
    mak_size = int((actual["division"] == 0).sum())
    rows, events, fails = [], [], 0
    for seed, g in gt.groupby("seed"):
        g = g.reset_index(drop=True)
        A = frame_arrays(g)
        base, pos, nxt, order0 = A["base"], A["pos"], A["nxt"], A["order"]
        mak = A["cnext"] <= MAEGASHIRA
        memo = {}

        def score_order(order):
            key = tuple(order)
            if key not in memo:
                sc = np.empty(len(order))
                sc[list(order)] = np.arange(len(order))
                try:
                    pred = resolve(g, sc, mak_size)
                except OverrideError:
                    return None
                pp = pred.set_index("rikishi_id")["pred_pos"].reindex(g["rikishi_id"]).to_numpy()
                memo[key] = (evaluate(pred, g, actual), pp)
            return memo[key]

        m0, pp0 = score_order(order0)
        cached = g[["rikishi_id", "pred_class", "pred_number", "pred_side", "pred_pos"]]
        repro = bool(np.array_equal(pp0, g["pred_pos"].to_numpy()))
        err0 = np.abs(pp0 - nxt)[mak].sum()
        rows.append({"target": target, "seed": seed, "variant": "V0", "n_events": 0, "repro": repro, **m0})
        rows.append({"target": target, "seed": seed, "variant": "cached", "n_events": 0,
                     **evaluate(cached, g, actual)})
        gcache = {}
        for v in variants:
            gk = (v["kind"], v.get("wins_only", False), v.get("mak_only", False))
            if gk not in gcache:
                gcache[gk] = find_groups(g, *gk)
            order, n = apply_variant(order0, gcache[gk], v, base, pos, nxt)
            r = score_order(order)
            if r is None:
                fails, r = fails + 1, (m0, pp0)
            rows.append({"target": target, "seed": seed, "variant": vname(v), "n_events": n, **r[0]})
        # each group glued in isolation, per anchor: clean per-event deltas
        for kind in KINDS:
            for members in gcache.setdefault((kind, False, False), find_groups(g, kind)):
                ev = {"target": target, "seed": seed, **group_row(kind, members, A)}
                for anc in ANCHORS + ("oracle",):
                    new = glue_one(order0, members, {"anchor": anc}, base, pos, nxt)
                    r = score_order(new) if new is not None else (m0, pp0)
                    if r is None:
                        ev[f"dx_{anc}"] = ev[f"de_{anc}"] = np.nan
                    else:
                        ev[f"dx_{anc}"] = r[0]["exact_n"] - m0["exact_n"]
                        ev[f"de_{anc}"] = np.abs(r[1] - nxt)[mak].sum() - err0
                events.append(ev)
    return rows, events, fails


def boot_ci(d, reps=2000, seed=0):
    d = np.asarray(d, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (reps, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_table(per, variants, lo, hi):
    """per: seed-averaged per-(variant, target) metrics. Paired deltas vs V0."""
    sub = per[window_mask(per, lo, hi)]
    b = sub[sub["variant"] == "V0"].set_index("target").sort_index()
    rows = []
    for v in variants:
        x = sub[sub["variant"] == v].set_index("target").reindex(b.index)
        dx, dm = x["exact_n"] - b["exact_n"], x["mae"] - b["mae"]
        w, l = int((dx > 0).sum()), int((dx < 0).sum())
        ci_x, ci_m = boot_ci(dx), boot_ci(dm)
        rows.append({
            "variant": v, "ev/basho": x["n_events"].mean(),
            "d_exact": dx.mean(), "ci_lo": ci_x[0], "ci_hi": ci_x[1], "W-L": f"{w}-{l}",
            "p_sign": binomtest(w, w + l).pvalue if w + l else 1.0,
            "p_wilc": wilcoxon(dx).pvalue if (dx != 0).any() else 1.0,
            "d_mae": dm.mean(), "mae_lo": ci_m[0], "mae_hi": ci_m[1],
            "d_within1": (x["within1"] - b["within1"]).mean(),
            "d_gtb": (x["gtb_points"] - b["gtb_points"]).mean(),
        })
    out = pd.DataFrame(rows).set_index("variant")
    return out, len(b), b["exact_n"].mean(), b["mae"].mean()


def events_report(ev):
    print("\n=== per-event deltas: each group glued in isolation ===")
    print("(k >= 1 groups touching makuuchi, deltas averaged over seeds so n counts groups;"
          " dx = exact slots, de = summed |err|, lower is better; W-L-T by sign of dx)")
    e = ev[(ev["k"] >= 1) & ev["touches_mak"]]
    dcols = [c for c in e.columns if c.startswith(("dx_", "de_"))]
    e = e.groupby(["target", "kind", "rid"]).agg({**{c: "mean" for c in dcols}, "actual_adj": "first"}).reset_index()
    for kind in KINDS:
        s = e[e["kind"] == kind]
        rows = []
        for anc in ANCHORS + ("oracle",):
            for lab, m in (("all", s), ("kept", s[s["actual_adj"]]), ("split", s[~s["actual_adj"]])):
                dx, de = m[f"dx_{anc}"], m[f"de_{anc}"]
                rows.append({"anchor": anc, "events": lab, "n": len(m), "dx_mean": dx.mean(),
                             "W-L-T": f"{int((dx > 0).sum())}-{int((dx < 0).sum())}-{int((dx == 0).sum())}",
                             "de_mean": de.mean()})
        print(f"\n{kind}:")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    tw = ev[(ev["kind"] == "twins") & (ev["k"] == 1) & ev["touches_mak"]]
    if len(tw):
        print("\ntwins k=1 by (interloper origin, actual outcome), seed-events: n, mean dx/de")
        agg = tw.groupby(["x_origin", "x_actual"]).agg(
            n=("dx_mid", "size"), dx_mid=("dx_mid", "mean"), dx_prior=("dx_prior", "mean"),
            dx_climber=("dx_climber", "mean"), de_mid=("de_mid", "mean"))
        print(agg.round(3).to_string())


def backtest(args):
    preds, tidy = load_preds(args.start, args.end, args.seeds, args.workers)
    primary = None
    if args.primary:
        kind, K, anchor = args.primary.split(":")
        primary = dict(kind=kind, K=INF if K == "inf" else int(K), anchor=anchor)
    variants = variant_grid(primary["anchor"] if primary else "mid")
    names = [vname(v) for v in variants]
    if primary:
        pn = vname(primary)
        if pn in names:
            names.remove(pn)
        else:
            variants.append(primary)
        names.insert(0, pn)
    tasks = [(t, g, tidy[tidy["basho"] == t], variants) for t, g in preds.groupby("target")]
    rows, events, fails = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, (r, e, f) in enumerate(ex.map(run_target, tasks, chunksize=2), 1):
            rows += r
            events += e
            fails += f
            print(f"\r{n}/{len(tasks)} targets", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    res, ev = pd.DataFrame(rows), pd.DataFrame(events)
    res.to_parquet(SCRATCH / "grouping_results.parquet", index=False)
    ev.to_parquet(SCRATCH / "grouping_events.parquet", index=False)
    if fails:
        print(f"warning: {fails} variant resolves failed (fell back to baseline)")
    v0 = res[res["variant"] == "V0"]
    repro = v0["repro"].astype(bool)
    if not repro.all():
        # the cache was built by an older resolver: V0 is the current code's
        # baseline, `cached` the old one, and their paired difference is the
        # value of whatever changed in between
        print(f"note: {int((~repro).sum())} of {len(v0)} cached frames do not reproduce"
              " under the current resolver; V0 (current) is the baseline for every variant"
              " and the paired delta to `cached` is reported per window below")
    per = res.groupby(["variant", "target"]).mean(numeric_only=True).reset_index()
    pd.set_option("display.width", 250)
    for w, lo, hi in WINDOWS:
        if not window_mask(per, lo, hi).any():
            continue
        tab, n, ex0, mae0 = paired_table(per, names, lo, hi)
        star = ""
        if primary:
            insample = lo <= args.screen_end
            star = (f"  (* = primary {pn}, chosen on targets <= {args.screen_end}"
                    f"{', in-sample here' if insample else ', out-of-sample here'})")
            tab.index = ["* " + i if i == pn else i for i in tab.index]
        print(f"\n=== {w}: {n} basho, V0 exact_n {ex0:.3f}, mae {mae0:.4f}; paired deltas vs V0{star} ===")
        if not repro.all():
            c = paired_table(per, ["cached"], lo, hi)[0].iloc[0]
            print(f"cached (old resolver) vs V0: d_exact {-c['d_exact']:+.4f} for V0"
                  f" [CI {-c['ci_hi']:+.4f}..{-c['ci_lo']:+.4f}], d_mae {-c['d_mae']:+.5f}, W-L for V0 "
                  f"{c['W-L'].split('-')[1]}-{c['W-L'].split('-')[0]}")
        print(tab.round(4).to_string())
    events_report(ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("describe", "backtest"))
    ap.add_argument("--start", type=int, default=200401)
    ap.add_argument("--end", type=int, default=202609)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--screen-end", type=int, default=201911,
                    help="describe: last target of the screening window that fixes the primary")
    ap.add_argument("--primary", default=None, help="backtest: kind:K:anchor, e.g. twins:1:mid")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    describe(args) if args.cmd == "describe" else backtest(args)


if __name__ == "__main__":
    main()
