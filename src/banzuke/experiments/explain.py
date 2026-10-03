"""E19: per-basho explain dumps, miss docket and pair calibration for Ar.

    banzuke explain build --start 201901 --end 202609 --seeds 3 --threads 14
    banzuke explain sheet 202101 [--seed 0] [--all]      # side-by-side banzuke
    banzuke explain detail 202101 [--seed 0] [--all]     # per-rikishi stage table
    banzuke explain docket --start 201901               # every missed cell, classified
    banzuke explain calibration                         # pair probability reliability

`build` mirrors the backtest harness frame by frame (train on next_basho < T,
candidates from the T-1 banzuke, rolling OOF scores for the pair stage) and
records, besides the standard prediction frame, the reranker's units and
clusters, every pair probability within PAIR_GAP base points, the sheet the
resolver would have produced from the base order alone, and a per-row miss
decomposition: block offset from structural Y/O/S/K count errors, local error,
and the stage the miss was born in. Outputs live in
cache/explain/<fingerprint>/{rows,pairs,frames}_<start>_<end>.parquet
and are concatenated by every reader.

Stage labels (misses only): cascade (right once the structure above is), structural
(wrong Y/O/S/K membership), boundary (wrong side of the juryo line), rerank (the
base order had the cell right, the reranked order lost it, or moved it farther
from the actual order), base (the base order was wrong and the reranker did not
fix it), resolver (the final order was right, the resolver's rules or layout moved
the cell).
"""
import numpy as np
import pandas as pd

from banzuke import confidence
from banzuke.experiments import precedent
from banzuke.harness import fingerprint, pmap, split
from banzuke.metrics import evaluate
from banzuke.models import GBMRerank, gap_pairs
from banzuke.overrides import OverrideError
from banzuke.paths import EXPLAIN_CACHE as SCRATCH
from banzuke.ranks import (CLS_CHARS as CLS, JURYO, KOMUSUBI, MAEGASHIRA, OZEKI, RECENT_ERA, SEKIWAKE,
                           YOKOZUNA, fmt_cell as cell, fmt_rank, fmt_record as rec)
from banzuke.resolver import forced_claims, resolve

PAIR_GAP = 5.0          # record pair probabilities for every pair this close in base score
                        # (clusters span up to 4.4 points: every in-cluster pair is present)
VERSION = 3             # bump to invalidate caches when the dump format changes
UPPER = (YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI)
STAGES = ("cascade", "structural", "boundary", "rerank", "base", "resolver")

_TRANS = _TIDY = None


def _int(x):
    return "" if x is None or np.isnan(x) else f"{int(x):+d}"


def cache_dir(kwargs, seeds):
    return SCRATCH / fingerprint(kwargs=kwargs, seeds=list(seeds), explain=VERSION)


# --- pair probabilities ---------------------------------------------------------

def pair_table(model, cands, base, detail):
    """Every pair within PAIR_GAP base points, oriented i = currently higher
    ranked: P(i above j), cluster membership, the actual outcome."""
    pos = cands["position"].to_numpy()
    i_arr, j_arr = gap_pairs(base, pos, PAIR_GAP)
    if not len(i_arr):
        return pd.DataFrame()
    p = model.pair.proba(cands, i_arr, j_arr, base)
    nxt = cands["position_next"].to_numpy()
    rid = cands["rikishi_id"].to_numpy()
    same = detail["cluster_of"][i_arr] == detail["cluster_of"][j_arr]
    return pd.DataFrame({
        "rid_i": rid[i_arr], "rid_j": rid[j_arr], "p": p, "gap": base[j_arr] - base[i_arr],
        "same_cluster": same, "cluster": np.where(same, detail["cluster_of"][i_arr], -1),
        "twin_i": [len(detail["units"][u]) > 1 for u in detail["unit_of"][i_arr]],
        "twin_j": [len(detail["units"][u]) > 1 for u in detail["unit_of"][j_arr]],
        "i_above": nxt[i_arr] < nxt[j_arr],
    })


# --- per-row decomposition -----------------------------------------------------

def _ranks(values, tiebreak):
    order = np.lexsort((tiebreak, values))
    r = np.empty(len(values), dtype=float)
    r[order] = np.arange(len(values))
    return r


def decompose(p, actual):
    """p: one frame (cands + score/base/pred_*/pred_pos/pred_*_base). actual: the
    real banzuke rows at the target. Adds ranks, errors, hit flags, block
    offset, local error, cascade and stage.

    Two error scales: `err` = pred_pos - position_next (joint cell index, what
    MAE uses) and `cell_err` = where the predicted *label* sits in the actual
    layout minus the actual position. A wrong S/K count moves no position but
    shifts every label below it, so structure shows up only in cell_err:
    `block_offset` is that label shift (negative = the model created extra
    slots above this row's class), `local_err` = cell_err - block_offset."""
    p = p.copy()
    pos, nxt = p["position"].to_numpy(), p["position_next"].to_numpy(dtype=float)
    p["base_rank"] = _ranks(p["base"].to_numpy(), pos)
    p["final_rank"] = p["score"].astype(int)
    known = ~np.isnan(nxt)
    act = np.full(len(p), np.nan)
    act[known] = _ranks(nxt[known], pos[known])
    p["act_rank"] = act
    p["err"] = p["pred_pos"] - p["position_next"]
    p["hit"] = ((p["pred_class"] == p["class_next"]) & (p["pred_number"] == p["number_next"])
                & (p["pred_side"] == p["side_next"]))
    p["hit_base"] = ((p["pred_class_base"] == p["class_next"]) & (p["pred_number_base"] == p["number_next"])
                     & (p["pred_side_base"] == p["side_next"]))
    lab = {(int(c), int(n), int(s)): int(q) for c, n, s, q in zip(
        actual["rank_class"], actual["rank_number"], actual["side"], actual["position"])}
    p["cell_pos"] = [lab.get((int(c), int(n), int(s)), q) if c < JURYO else q
                     for c, n, s, q in zip(p["pred_class"], p["pred_number"], p["pred_side"], p["pred_pos"])]
    p["cell_err"] = p["cell_pos"] - p["position_next"]
    n_pred = {c: int((p["pred_class"] == c).sum()) for c in UPPER}
    n_act = {c: int((actual["rank_class"] == c).sum()) for c in UPPER}
    cum, run = {}, 0
    for c in UPPER:
        cum[c] = run
        run += n_act[c] - n_pred[c]
    cum[MAEGASHIRA] = run
    cn = p["class_next"].to_numpy()
    p["block_offset"] = [cum[int(c)] if not np.isnan(c) and c <= MAEGASHIRA else np.nan for c in cn]
    p["local_err"] = p["cell_err"] - p["block_offset"]
    mak = p["class_next"] <= MAEGASHIRA
    miss = mak & ~p["hit"]
    cascade = miss & (p["local_err"] == 0) & (p["block_offset"] != 0)
    structural = miss & ~cascade & (p["pred_class"] != p["class_next"]) & (
        (p["pred_class"] <= KOMUSUBI) | (p["class_next"] <= KOMUSUBI))
    boundary = ((p["pred_class"] == JURYO) & mak) | ((p["pred_class"] < JURYO) & (p["class_next"] == JURYO))
    order_err = p["final_rank"] - p["act_rank"]
    base_err = p["base_rank"] - p["act_rank"]
    rerank = miss & (p["hit_base"] | (order_err.abs() > base_err.abs()))
    resolver = miss & (order_err == 0)
    stage = np.select(
        [cascade, structural, boundary & miss, boundary & ~mak, rerank, resolver, miss],
        ["cascade", "structural", "boundary", "boundary", "rerank", "resolver", "base"], "")
    p["stage"] = stage
    p["rerank_fixed"] = mak & p["hit"] & ~p["hit_base"]
    p["cascade"] = cascade
    p["flip"] = miss & (p["pred_class"] == p["class_next"]) & (p["pred_number"] == p["number_next"])
    return p


# --- build ---------------------------------------------------------------------

def _init(trans, tidy):
    global _TRANS, _TIDY
    _TRANS, _TIDY = trans, tidy


def _names(df, mask):
    return ", ".join(df.loc[mask, "shikona"])


def _build_target(args):
    target, seeds, kw = args
    prev, train, cands, actual, mak_size = split(_TRANS, _TIDY, target)
    claims = forced_claims(cands)
    rows, pairs, frames = [], [], []
    for seed in seeds:
        model = GBMRerank(seed=seed, **kw)
        model.fit(train)
        detail = model.rerank(cands)
        base, score = detail["base"], detail["score"]
        pred = resolve(cands, score, mak_size)
        try:
            pred_base = resolve(cands, base, mak_size)
        except OverrideError:
            pred_base = pred.assign(pred_class=np.nan, pred_number=np.nan, pred_side=np.nan, pred_pos=np.nan)
        pb = pred_base.rename(columns={c: f"{c}_base" for c in ("pred_class", "pred_number", "pred_side", "pred_pos")})
        p = cands.assign(score=score, base=base).merge(pred, on="rikishi_id").merge(pb, on="rikishi_id")
        p["cluster"], p["cluster_size"] = detail["cluster_of"], detail["cluster_size"]
        p["unit"] = detail["unit_of"]
        p["twin_unit"] = [len(detail["units"][u]) > 1 for u in detail["unit_of"]]
        p["borda"] = detail["borda"][detail["unit_of"]]
        p = decompose(p, actual)
        p["config"], p["model"], p["seed"], p["target"] = "base", "Ar", seed, target
        rows.append(p)
        pt = pair_table(model, cands, base, detail)
        pt["seed"], pt["target"] = seed, target
        pairs.append(pt)

        m = evaluate(pred, cands, actual)
        mak = p["class_next"] <= MAEGASHIRA
        created = {}
        for c, key in ((SEKIWAKE, "s"), (KOMUSUBI, "k")):
            got = (p["pred_class"] == c) & claims[key] & (p["rank_class"] > c)
            created[f"created_{CLS[c]}"] = _names(p, got & (p["pred_class"] == c))
            created[f"declined_{CLS[c]}"] = _names(p, got & (p["class_next"] > c))
            created[f"committee_{CLS[c]}"] = _names(p, (p["class_next"] == c) & (p["pred_class"] > c))
        frames.append({
            "target": target, "seed": seed, "prev": prev, "train_max": int(train["next_basho"].max()),
            **m,
            **{f"n_pred_{CLS[c]}": int((p["pred_class"] == c).sum()) for c in UPPER},
            **{f"n_act_{CLS[c]}": int((actual["rank_class"] == c).sum()) for c in UPPER},
            "n_cascade": int(p["cascade"].sum()),
            **{f"n_{s}": int((p.loc[mak, "stage"] == s).sum()) for s in STAGES},
            "n_rerank_fixed": int(p["rerank_fixed"].sum()),
            "n_flip": int(p["flip"].sum()),
            **created,
            "promo_pred": _names(p, (p["division"] == 1) & (p["pred_class"] < JURYO)),
            "promo_act": _names(p, (p["division"] == 1) & (p["class_next"] < JURYO)),
            "demo_pred": _names(p, (p["division"] == 0) & (p["pred_class"] == JURYO)),
            "demo_act": _names(p, (p["division"] == 0) & (p["class_next"] == JURYO)),
            "reinstated": ", ".join(actual.loc[~actual["rikishi_id"].isin(cands["rikishi_id"])
                                               & (actual["division"] == 0), "shikona"]),
        })
    return pd.concat(rows, ignore_index=True), pd.concat(pairs, ignore_index=True), pd.DataFrame(frames)


def build(trans, tidy, start, end, seeds, kwargs=None, threads=1):
    """Dump rows/pairs/frames for every target in start..end (one frame per
    seed) under cache_dir(kwargs, seeds). Returns the frames table."""
    kwargs = kwargs or {}
    seeds = list(seeds)
    targets = [b for b in sorted(tidy["basho"].unique()) if start <= b <= end]
    out = cache_dir(kwargs, seeds)
    out.mkdir(parents=True, exist_ok=True)
    kw = GBMRerank.prepare(kwargs, trans, threads)
    tasks = [(t, seeds, kw) for t in targets[::-1]]
    rows, pairs, frames = [], [], []
    for r, p, f in pmap(_build_target, tasks, threads, _init, (trans, tidy), label="targets"):
        rows.append(r)
        pairs.append(p)
        frames.append(f)
    tag = f"{start}_{end}"
    for name, parts in (("rows", rows), ("pairs", pairs), ("frames", frames)):
        pd.concat(parts, ignore_index=True).to_parquet(out / f"{name}_{tag}.parquet", index=False)
    fr = pd.concat(frames, ignore_index=True)
    print(f"wrote {out} ({tag}); exact {fr['exact_n'].mean():.2f} MAE {fr['mae'].mean():.3f} "
          f"over {fr['target'].nunique()} targets")
    return fr


# --- readers -------------------------------------------------------------------

class NoCache(FileNotFoundError):
    """No explain build exists for this configuration."""


def load(kind, kwargs=None, seeds=(0, 1, 2)):
    """Concatenate every build of `kind` (rows, pairs, frames) under the cache
    dir of this configuration; NoCache names the directory otherwise."""
    d = cache_dir(kwargs or {}, list(seeds))
    files = sorted(d.glob(f"{kind}_*.parquet"))
    if not files:
        raise NoCache(f"no {kind} cache under {d}; run `banzuke explain build` first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def with_signals(rows, target, seed):
    """Frame plus confidence signals (over all cached seeds of the target)."""
    g = rows[rows["target"] == target]
    if not (g["seed"] == seed).any():
        raise NoCache(f"target {target} seed {seed} not in cache")
    return confidence.with_signals(g, seed)


# --- sheet -----------------------------------------------------------------------

def header(p, fr, actual):
    lines = [f"=== {fr['target']} banzuke (from {fr['prev']} results), Ar seed {fr['seed']}: "
             f"exact {fr['exact_n']}/42, GTB {fr['gtb_points']}, MAE {fr['mae']:.3f}, "
             f"within1 {fr['within1']:.2f}, promo/demo F1 {fr['promo_f1']:.2f}/{fr['demo_f1']:.2f} ==="]
    sp = " ".join(f"{CLS[c]}{fr[f'n_pred_{CLS[c]}']}" for c in UPPER)
    sa = " ".join(f"{CLS[c]}{fr[f'n_act_{CLS[c]}']}" for c in UPPER)
    notes = []
    for c in (SEKIWAKE, KOMUSUBI):
        d = fr[f"n_pred_{CLS[c]}"] - fr[f"n_act_{CLS[c]}"]
        if d > 0:
            notes.append(f"{CLS[c]} over-created by {d} (declined claim: {fr[f'declined_{CLS[c]}'] or 'none'})")
        elif d < 0:
            notes.append(f"{CLS[c]} under-created by {-d} (committee gave it to: {fr[f'committee_{CLS[c]}'] or '?'})")
    for c in (YOKOZUNA, OZEKI):
        if fr[f"n_pred_{CLS[c]}"] != fr[f"n_act_{CLS[c]}"]:
            notes.append(f"{CLS[c]} count wrong")
    lines.append(f"structure  pred {sp} | actual {sa}" + (": " + "; ".join(notes) if notes else "  (right)"))
    if fr["created_S"] or fr["created_K"]:
        lines.append(f"created    S: {fr['created_S'] or '-'} | K: {fr['created_K'] or '-'}")
    lines.append(f"boundary   promoted pred [{fr['promo_pred']}] actual [{fr['promo_act']}]")
    lines.append(f"           demoted  pred [{fr['demo_pred']}] actual [{fr['demo_act']}]")
    if fr["reinstated"]:
        lines.append(f"           reinstated (not a candidate): {fr['reinstated']}")
    lines.append(f"misses     cascade {fr['n_cascade']}, structural {fr['n_structural']}, boundary {fr['n_boundary']}, "
                 f"rerank {fr['n_rerank']}, base {fr['n_base']}, resolver {fr['n_resolver']}; "
                 f"E/W flips {fr['n_flip']}; reranker fixed {fr['n_rerank_fixed']}")
    return lines


def sheet_text(rows, frames, tidy, target, seed):
    p = with_signals(rows, target, seed)
    fr = frames[(frames["target"] == target) & (frames["seed"] == seed)].iloc[0]
    actual = tidy[(tidy["basho"] == target) & (tidy["division"] == 0)]
    lines = header(p, fr, actual)
    info = p.set_index("rikishi_id")
    pred_cells = {(int(r.pred_class), int(r.pred_number), int(r.pred_side)): r.rikishi_id
                  for r in p.itertuples() if r.pred_class < JURYO}
    act_cells = {(int(r.rank_class), int(r.rank_number), int(r.side)): r.rikishi_id for r in actual.itertuples()}

    def who(rid, other_cell, arrow):
        if rid not in info.index:
            a = actual[actual["rikishi_id"] == rid].iloc[0]
            return f"   {a['shikona']:<13} (new: not a candidate)"
        r = info.loc[rid]
        tag = f"({cell(r['rank_class'], r['rank_number'], r['side'])} {rec(r['wins'], r['losses'], r['absences'])})"
        s = f"{r['marker']:<3}{r['shikona']:<13} {tag:<13}"
        if other_cell:
            s += f" {arrow}{other_cell}"
        return s

    ranks = sorted({k[:2] for k in pred_cells} | {k[:2] for k in act_cells})
    p["marker"] = p["tier"] + np.where(p["big_move"], "~", "")
    info = p.set_index("rikishi_id")
    lines.append("")
    lines.append(f"{'rank':<5} {'PREDICTED EAST':<40} {'PREDICTED WEST':<40} | {'ACTUAL EAST':<40} {'ACTUAL WEST':<40}")
    for c, n in ranks:
        cols = []
        for side in (0, 1):
            rid = pred_cells.get((c, n, side))
            if rid is None:
                cols.append("")
                continue
            r = info.loc[rid]
            a_cell = cell(r["class_next"], r["number_next"], r["side_next"])
            cols.append(who(rid, "" if r["hit"] else a_cell, "->"))
        for side in (0, 1):
            rid = act_cells.get((c, n, side))
            if rid is None:
                cols.append("")
                continue
            if rid in info.index:
                r = info.loc[rid]
                p_cell = cell(r["pred_class"], r["pred_number"], r["pred_side"])
                cols.append(who(rid, "" if r["hit"] else p_cell, "<-"))
            else:
                cols.append(who(rid, "", ""))
        lines.append(f"{fmt_rank(c, n):<5} {cols[0]:<40} {cols[1]:<40} | {cols[2]:<40} {cols[3]:<40}".rstrip())
    n_mak = int(p["mak_size"].iloc[0])
    near = (p["pred_pos"] <= n_mak + 6) | (p["position_next"] <= n_mak + 6)
    jur = p[((p["pred_class"] == JURYO) | (p["class_next"] == JURYO)) & near].sort_values("pred_pos")
    lines.append("")
    lines.append("juryo line (candidates predicted or landing in juryo, within 6 cells of it, by predicted position):")
    for r in jur.itertuples():
        lines.append(f"  {r.shikona:<14} ({cell(r.rank_class, r.rank_number, r.side):>4} {rec(r.wins, r.losses, r.absences):<7})"
                     f" pred {cell(r.pred_class, r.pred_number, r.pred_side):<5} actual {cell(r.class_next, r.number_next, r.side_next):<5}"
                     f" base {r.base:6.1f} spread {int(r.spread)}")
    lines.append("")
    lines.append("markers: ? low / ?? very low confidence (tight base gap, seed disagreement), ~ big move;"
                 " ->X where the man actually landed, <-X where the model put him")
    return "\n".join(lines)


# --- explain -------------------------------------------------------------------

def explain_text(rows, pairs, frames, tidy, trans, target, seed):
    p = with_signals(rows, target, seed)
    fr = frames[(frames["target"] == target) & (frames["seed"] == seed)].iloc[0]
    actual = tidy[(tidy["basho"] == target) & (tidy["division"] == 0)]
    lines = header(p, fr, actual)
    p = p.sort_values("pred_pos")
    lines.append("")
    lines.append(f"{'pp':>3} {'name':<14} {'prior':<5} {'rec':<7} {'base':>6} {'brk':>3} {'fin':>3} {'cl':>3} {'sz':>2} "
                 f"{'pred':<5} {'act':<5} {'err':>4} {'loc':>4} {'stage':<10} {'spr':>3} {'tier':<4}")
    for r in p.itertuples():
        loc, err = _int(r.local_err), _int(r.err)
        lines.append(f"{int(r.pred_pos):3d} {r.shikona:<14} {cell(r.rank_class, r.rank_number, r.side):<5} "
                     f"{rec(r.wins, r.losses, r.absences):<7} {r.base:6.1f} {int(r.base_rank):3d} {int(r.final_rank):3d} "
                     f"{int(r.cluster):3d} {int(r.cluster_size):2d} {cell(r.pred_class, r.pred_number, r.pred_side):<5} "
                     f"{cell(r.class_next, r.number_next, r.side_next):<5} {err:>4} {loc:>4} "
                     f"{(r.stage or ('hit' if r.hit else '')):<10} {int(r.spread):3d} {r.tier:<4}".rstrip())
    mak = p[p["class_next"] <= MAEGASHIRA]
    miss = mak[~mak["hit"]]
    lines.append("")
    lines.append(f"misses ({len(miss)} of {len(mak)} makuuchi cells):")
    groups = [("cascade (right once the structure above is)", miss["stage"] == "cascade"),
              ("structural / boundary", miss["stage"].isin(["structural", "boundary"])),
              ("far local (|local err| >= 2)", (miss["local_err"].abs() >= 2) & ~miss["stage"].isin(["cascade", "structural", "boundary"])),
              ("near local (|local err| == 1)", (miss["local_err"].abs() == 1) & ~miss["stage"].isin(["cascade", "structural", "boundary"])),
              ("E/W flip (same position error 0)", (miss["local_err"] == 0) & ~miss["stage"].isin(["cascade", "structural", "boundary"]))]
    for title, m in groups:
        sub = miss[m]
        if not len(sub):
            continue
        lines.append(f"  {title}: {len(sub)}")
        if title.startswith("cascade"):
            lines.append("    " + ", ".join(sub["shikona"]))
            continue
        for r in sub.itertuples():
            lines.append(f"    {r.shikona:<14} {cell(r.rank_class, r.rank_number, r.side):<5} {rec(r.wins, r.losses, r.absences):<7} "
                         f"pred {cell(r.pred_class, r.pred_number, r.pred_side):<5} act {cell(r.class_next, r.number_next, r.side_next):<5} "
                         f"err {_int(r.err)} local {_int(r.local_err)} stage {r.stage} spread {int(r.spread)}")
    pr = pairs[(pairs["target"] == target) & (pairs["seed"] == seed) & pairs["same_cluster"]]
    info = p.set_index("rikishi_id")
    lines.append("")
    lines.append("reranker clusters (>= 2 units) with pair probabilities P(upper-ranked-now above):")
    for c, g in p[p["cluster_size"] >= 2].groupby("cluster"):
        g = g.sort_values("score")
        lines.append(f"  cluster {c}: " + " > ".join(
            f"{r.shikona} ({cell(r.rank_class, r.rank_number, r.side)} {rec(r.wins, r.losses, r.absences)}, base {r.base:.1f}, borda {r.borda:.2f})"
            for r in g.itertuples()))
        for q in pr[pr["cluster"] == c].itertuples():
            a, b = info.loc[q.rid_i], info.loc[q.rid_j]
            verdict = "right" if (q.p >= 0.5) == q.i_above else "wrong"
            lines.append(f"      {a['shikona']:<14} above {b['shikona']:<14} p={q.p:.2f}  actual: "
                         f"{'yes' if q.i_above else 'no ':<3} ({verdict}); gap {q.gap:+.2f}")
    far = miss[(miss["local_err"].abs() >= 2) | miss["stage"].isin(["structural", "boundary"])]
    if len(far):
        t = precedent.labeled(trans)
        hist = t[t["next_basho"] < target]  # decided before this banzuke: no leak of the case itself
        lines.append("")
        lines.append("precedent for far/structural misses (prior cell + record, decided before this basho):")
        for r in far.itertuples():
            spec = {"rank_class": int(r.rank_class), "num_lo": int(r.rank_number), "num_hi": int(r.rank_number),
                    "wins": int(r.wins), "losses": int(r.losses), "absences": int(r.absences)}
            res = precedent.landing(hist, spec)
            lines.append("  " + precedent.fmt_landing(
                f"{r.shikona} {cell(r.rank_class, r.rank_number, r.side)} {rec(r.wins, r.losses, r.absences)} "
                f"(pred {cell(r.pred_class, r.pred_number, r.pred_side)}, act {cell(r.class_next, r.number_next, r.side_next)})",
                res).replace("\n", "\n  "))
    return "\n".join(lines)


# --- docket ----------------------------------------------------------------------

def docket_rows(rows, pairs, start, seed=0):
    """One line per missed makuuchi cell (plus boundary misses) for targets >= start."""
    r = rows[(rows["target"] >= start)]
    spread = r.pivot_table(index=["target", "rikishi_id"], columns="seed", values="pred_pos")
    spread = (spread.max(axis=1) - spread.min(axis=1)).rename("spread")
    p = r[r["seed"] == seed].join(spread, on=["target", "rikishi_id"])
    miss = p[((p["class_next"] <= MAEGASHIRA) & ~p["hit"]) | (p["stage"] == "boundary")].copy()
    # the model's probability for the row against the man who actually took its predicted cell
    occ = p.set_index(["target", "class_next", "number_next", "side_next"])["rikishi_id"]
    key = pd.MultiIndex.from_arrays([miss["target"], miss["pred_class"], miss["pred_number"], miss["pred_side"]])
    miss["occupant"] = occ.reindex(key).to_numpy()
    pr = pairs[(pairs["seed"] == seed) & (pairs["target"] >= start)]
    fwd = pr.set_index(["target", "rid_i", "rid_j"])
    rev = pr.set_index(["target", "rid_j", "rid_i"])
    k1 = pd.MultiIndex.from_arrays([miss["target"], miss["rikishi_id"], miss["occupant"]])
    p_f = fwd["p"].reindex(k1).to_numpy()
    p_r = 1 - rev["p"].reindex(k1).to_numpy()
    miss["p_vs_occupant"] = np.where(np.isnan(p_f), p_r, p_f)
    miss["gap_vs_occupant"] = np.where(np.isnan(p_f), -rev["gap"].reindex(k1).to_numpy(), fwd["gap"].reindex(k1).to_numpy())
    names = p.set_index(["target", "rikishi_id"])["shikona"]
    miss["occupant_name"] = names.reindex(pd.MultiIndex.from_arrays([miss["target"], miss["occupant"]])).to_numpy()
    zone = np.select([miss["class_next"] <= OZEKI, miss["class_next"] <= KOMUSUBI,
                      (miss["class_next"] == MAEGASHIRA) & (miss["number_next"] <= 5),
                      (miss["class_next"] == MAEGASHIRA) & (miss["number_next"] <= 11),
                      miss["class_next"] == MAEGASHIRA], ["Y/O", "S/K", "M1-5", "M6-11", "M12+"], "J")
    out = pd.DataFrame({
        "target": miss["target"], "rikishi_id": miss["rikishi_id"], "shikona": miss["shikona"],
        "prior": [cell(c, n, s) for c, n, s in zip(miss["rank_class"], miss["rank_number"], miss["side"])],
        "record": [rec(w, l, a) for w, l, a in zip(miss["wins"], miss["losses"], miss["absences"])],
        "pred": [cell(c, n, s) for c, n, s in zip(miss["pred_class"], miss["pred_number"], miss["pred_side"])],
        "actual": [cell(c, n, s) for c, n, s in zip(miss["class_next"], miss["number_next"], miss["side_next"])],
        "err": miss["err"], "cell_err": miss["cell_err"], "block_offset": miss["block_offset"],
        "local_err": miss["local_err"],
        "zone": zone, "stage": miss["stage"], "cluster_size": miss["cluster_size"],
        "base": miss["base"].round(2), "spread": miss["spread"].astype(int),
        "occupant": miss["occupant_name"], "p_vs_occupant": miss["p_vs_occupant"].round(3),
        "gap_vs_occupant": miss["gap_vs_occupant"].round(2),
        "kind": np.select([miss["stage"].isin(["cascade", "structural", "boundary"]),
                           miss["local_err"].abs() >= 2, miss["local_err"].abs() == 1],
                          [miss["stage"], "far", "near"], "flip"),
    })
    return out.sort_values(["target", "err"], key=lambda s: s.abs() if s.name == "err" else s, ascending=[True, False])


def docket(start, kwargs=None, seeds=(0, 1, 2)):
    """Print the miss docket for targets >= start and write docket_auto.csv /
    notable_auto.csv under the explain cache."""
    rows, pairs, frames = (load(k, kwargs, seeds) for k in ("rows", "pairs", "frames"))
    d = docket_rows(rows, pairs, start)
    out = SCRATCH / "case_docket"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "docket_auto.csv", index=False)
    fr = frames[frames["target"] >= start]
    n_t = fr["target"].nunique()
    print(f"=== docket {start}+: {n_t} targets, seed 0 rows, {len(d)} missed cells ===")
    per = fr.groupby("seed")[["exact_n", "mae", "gtb_points"]].mean()
    print("per seed:\n" + per.round(3).to_string())
    print(f"\nstructure wrong in {int((fr[fr.seed == 0].eval('n_pred_S != n_act_S or n_pred_K != n_act_K or n_pred_Y != n_act_Y or n_pred_O != n_act_O')).sum())} of {n_t} frames (seed 0)")
    print("\nmissed makuuchi cells by stage x zone (seed 0):")
    mk = d[d["zone"] != "J"]
    print(pd.crosstab(mk["stage"], mk["zone"], margins=True).to_string())
    print("\nby kind:")
    print(mk["kind"].value_counts().to_string())
    print("\nslots per frame by stage:")
    print((mk.groupby("stage").size() / n_t).round(2).to_string())
    print("\nreranker: fixed vs introduced per frame (seed 0): "
          f"{fr.loc[fr.seed == 0, 'n_rerank_fixed'].mean():.2f} fixed, {fr.loc[fr.seed == 0, 'n_rerank'].mean():.2f} introduced/worsened")
    notable = d[(d["kind"].isin(["structural", "boundary", "far"]))
                | ((d["kind"] == "near") & ((d["p_vs_occupant"] - 0.5).abs() >= 0.3))].copy()
    notable["reason"] = np.where(notable["kind"] == "near", "confident near miss", notable["kind"])
    notable.to_csv(out / "notable_auto.csv", index=False)
    print(f"\nnotable skeleton: {len(notable)} rows -> {out / 'notable_auto.csv'} "
          f"({(notable['reason'] == 'structural').sum()} structural, {(notable['reason'] == 'boundary').sum()} boundary, "
          f"{(notable['reason'] == 'far').sum()} far, {(notable['reason'] == 'confident near miss').sum()} confident near)")
    print(f"full docket -> {out / 'docket_auto.csv'}")


# --- calibration -----------------------------------------------------------------

def pair_calibration(kwargs=None, seeds=(0, 1, 2)):
    """Reliability of the pair probabilities, by window and pair type."""
    rows, pairs = load("rows", kwargs, seeds), load("pairs", kwargs, seeds)
    info = rows.set_index(["target", "seed", "rikishi_id"])
    k_i = pd.MultiIndex.from_arrays([pairs["target"], pairs["seed"], pairs["rid_i"]])
    k_j = pd.MultiIndex.from_arrays([pairs["target"], pairs["seed"], pairs["rid_j"]])
    pr = pairs.copy()
    for col in ("rank_class", "rank_number", "wins", "division", "class_next"):
        pr[f"{col}_i"], pr[f"{col}_j"] = info[col].reindex(k_i).to_numpy(), info[col].reindex(k_j).to_numpy()
    pr = pr[(pr["class_next_i"] <= MAEGASHIRA) | (pr["class_next_j"] <= MAEGASHIRA)]
    pr["window"] = np.where(pr["target"] >= RECENT_ERA, "2019+", "2004-2018")
    pr["bin"] = pd.cut(pr["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    pr["sanyaku"] = (pr["rank_class_i"] <= KOMUSUBI) | (pr["rank_class_j"] <= KOMUSUBI)
    pr["promotee"] = (pr["division_i"] == 1) | (pr["division_j"] == 1)
    pr["dwins"] = pd.cut(pr["wins_i"] - pr["wins_j"], [-99, -3, -1, 0, 2, 99], labels=["<=-3", "-2..-1", "0", "1..2", ">=3"])
    pr["twin"] = pr["twin_i"] | pr["twin_j"]
    pr["gapbin"] = pd.cut(pr["gap"].abs(), [0, .25, .5, 1, 2.01], include_lowest=True)
    print(f"=== pair calibration: {len(pr)} pairs within {PAIR_GAP} base points, all seeds ===")
    print("(p = P(currently higher ranked stays above); outcome = he did)\n")

    def table(g, by):
        t = g.groupby(by, observed=True).agg(n=("p", "size"), p=("p", "mean"), outcome=("i_above", "mean"))
        t["gap"] = (t["outcome"] - t["p"]).round(3)
        return t.round(3)

    for w, g in pr.groupby("window"):
        print(f"--- {w}: reliability by p bin ---")
        print(table(g, "bin").to_string())
        ll = -np.mean(np.where(g["i_above"], np.log(g["p"].clip(1e-6)), np.log((1 - g["p"]).clip(1e-6))))
        acc = ((g["p"] >= .5) == g["i_above"]).mean()
        print(f"log loss {ll:.3f}, accuracy {acc:.3f}, in-cluster accuracy "
              f"{((g.loc[g.same_cluster, 'p'] >= .5) == g.loc[g.same_cluster, 'i_above']).mean():.3f}\n")
    for by in ("sanyaku", "promotee", "dwins", "twin", "gapbin", "same_cluster"):
        print(f"--- by {by} (2019+ / 2004-2018): n, mean p, outcome, gap ---")
        print(table(pr, ["window", by]).to_string())
        print()
    conf = pr[(pr["p"] - .5).abs() >= .3]
    print(f"confident pairs (|p-.5| >= .3): {len(conf)}, accuracy {((conf['p'] >= .5) == conf['i_above']).mean():.3f}")
