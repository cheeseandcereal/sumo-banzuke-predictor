"""Confidence markers and review items for a predicted banzuke.

Signals per rikishi: a tight ordering boundary (small base-score gap to a
neighbour the model itself ranked, not a rule), seed spread >= 1, and
seed spread >= 2. Big moves get their own flag. Structural items cover
created S/K slots, with historical precedent for the claim behind them.
Pure functions, no I/O.
"""
import numpy as np
import pandas as pd

from banzuke.build import OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO
from banzuke.overrides import CLS_CHARS, OverrideError, fmt_slot
from banzuke.resolver import forced_claims, resolve

# Ar model, 2024-2026 backtest, 3 seeds, exact / far (>1 position off):
# 0 signals 60% / 11%, 1 signal 35% / 22%, 2+ signals 25% / 37%.
# Incumbents moving 8+ half-ranks: climbs 36% exact, drops 25% / 43% far;
# juryo promotees landing 8+ cells above the last makuuchi cell 25% / 67%.
TIGHT_GAP = 0.25  # base-score gap below which a boundary is a tight call
BIG_MOVE = 8      # half-ranks


def boundaries(pred, base, final, skip=()):
    """One row per consecutive pair in the resolved order. `decided` marks
    boundaries the model ordered itself: final order agrees, outside Y/O
    (base scores there are compressed and meaningless), no override."""
    p = pred.sort_values("pred_pos")
    rid, cls = p["rikishi_id"].to_numpy(), p["pred_class"].to_numpy()
    b = pd.DataFrame({"upper": rid[:-1], "lower": rid[1:]})
    b["gap"] = base.reindex(b["lower"]).to_numpy() - base.reindex(b["upper"]).to_numpy()
    b["decided"] = (
        (final.reindex(b["upper"]).to_numpy() < final.reindex(b["lower"]).to_numpy())
        & (cls[:-1] > OZEKI) & (cls[1:] > OZEKI)
        & ~b["upper"].isin(skip) & ~b["lower"].isin(skip)
    )
    b["tight"] = b["decided"] & (b["gap"] < TIGHT_GAP)
    b["inverted"] = b["decided"] & (b["gap"] < 0)
    return b


def signals(pred, base, final, seed_preds=(), skip=()):
    """Per-row signal frame indexed like `pred`: gap, inverted, tight,
    spread, move, big_move, n_signals, tier ("", "?", "??"), marker."""
    d = boundaries(pred, base, final, skip).query("decided")
    per = pd.concat([d.assign(rid=d["upper"]), d.assign(rid=d["lower"])])
    agg = per.groupby("rid").agg(gap=("gap", "min"), inverted=("inverted", "any"),
                                 tight=("tight", "any"))
    rid = pred["rikishi_id"]
    out = pd.DataFrame(index=pred.index)
    out["gap"] = rid.map(agg["gap"]).fillna(np.inf)
    out["inverted"] = rid.isin(agg.index[agg["inverted"]]).to_numpy()
    out["tight"] = rid.isin(agg.index[agg["tight"]]).to_numpy()
    out["spread"] = 0
    if len(seed_preds) >= 2:
        pos = pd.concat(seed_preds).groupby("rikishi_id")["pred_pos"]
        out["spread"] = rid.map(pos.max() - pos.min()).fillna(0).astype(int)
    out["move"] = (pred["pred_pos"] - pred["position"]).astype(float)
    n_mak = int((pred["pred_class"] < JURYO).sum())
    big = np.where(pred["division"] == 0, out["move"].abs() >= BIG_MOVE,
                   pred["pred_pos"] <= n_mak - 1 - BIG_MOVE)
    out["big_move"] = big & (pred["pred_class"] < JURYO).to_numpy()
    out["n_signals"] = (out["tight"].astype(int) + (out["spread"] >= 1)
                        + (out["spread"] >= 2))
    out["tier"] = np.select([out["n_signals"] >= 2, out["n_signals"] == 1], ["??", "?"], "")
    out["marker"] = out["tier"] + np.where(out["big_move"], "~", "")
    return out


def _gap(g):
    s = f"{g:.1f}"
    return s.replace("0.", ".", 1) if s.lstrip("-").startswith("0.") else s


def review(pred, sig, base, final, skip=()):
    """Review items, one per run of consecutive tight boundaries starting in
    makuuchi, sorted least-confident first (most "??" members, then gap)."""
    p = pred.sort_values("pred_pos").reset_index(drop=True)
    b = boundaries(p, base, final, skip)
    sg = sig.set_axis(pred["rikishi_id"].to_numpy())
    name = dict(zip(p["rikishi_id"], p["shikona"]))
    slot = [fmt_slot(c, n, s) for c, n, s in
            zip(p["pred_class"], p["pred_number"], p["pred_side"])]
    items = []
    t = b.index[b["tight"]]
    for _, run in pd.Series(t).groupby((t - np.arange(len(t))).to_numpy()):
        i, j = run.iloc[0], run.iloc[-1]
        if p["pred_class"].iat[i] == JURYO:
            continue
        rids = p["rikishi_id"].to_numpy()[i:j + 2]
        bnd = b.iloc[i:j + 1]
        rows = sg.loc[rids]
        gaps = [_gap(abs(g)) for g in bnd["gap"]]  # reversals are spelled out below
        parts = [f"base scores {gaps[0]} apart" if len(gaps) == 1
                 else "base score gaps " + ", ".join(gaps)]
        hints = []
        inv = bnd[bnd["inverted"]]
        if len(inv):
            parts.append("reranker reversed the base order (" + ", ".join(
                f"{name[lo]} below {name[up]}" for up, lo in zip(inv["upper"], inv["lower"]))
                + ")")
            hints += [("above", f"{name[lo]} > {name[up]}")
                      for up, lo in zip(inv["upper"], inv["lower"])]
        if rows["spread"].max() >= 1:
            parts.append(f"position varies by up to {rows['spread'].max()} across seeds")
        for r in rids[rows["big_move"].to_numpy()]:
            if any(name[r] in s for _, s in hints):
                continue  # the inversion hint already tests this mover
            if r == rids[-1]:
                hints.append(("above", f"{name[r]} > {name[rids[0]]}"))
            else:
                hints.append(("below", f"{name[r]} < {name[rids[-1]]}"))
        n2 = int((rows["tier"] == "??").sum())
        items.append({
            "marker": "??" if n2 else "?",
            "range": f"{slot[i]}-{slot[j + 1]}",
            "members": [name[r] for r in rids],
            "text": "; ".join(parts),
            "hints": list(dict.fromkeys(hints)),
            "key": (-n2, bnd["gap"].min()),
        })
    return sorted(items, key=lambda it: it["key"])


def claim_rates(trans, since=199001):
    """How often forced S/K claims were honoured when honouring them needed
    a created slot (block > 2). One row per (claimed, rank_class,
    rank_number, wins) cell; komusubi claimants pool under rank_number 0."""
    t = trans[(trans["basho"] >= since) & trans["class_next"].notna()]
    rows = []
    for _, g in t.groupby("basho"):
        cl = forced_claims(g)
        if cl["s"].sum() > 2:
            rows.append(g[(g["rank_class"] == KOMUSUBI) & (g["wins"] >= 11)]
                        .assign(claimed=SEKIWAKE))
        if (cl["k"] & ~cl["s"]).sum() > 2:
            rows.append(g[cl["m_claim"]].assign(claimed=KOMUSUBI))
    c = pd.concat(rows)
    c["honoured"] = c["class_next"] <= c["claimed"]
    c["rank_number"] = c["rank_number"].where(c["rank_class"] != KOMUSUBI, 0)
    out = (c.groupby(["claimed", "rank_class", "rank_number", "wins"])
           .agg(n=("honoured", "size"), honoured=("honoured", "sum")).reset_index())
    out.attrs["since"] = since
    return out


def precedent(rates, claimed, rank_class, rank_number, wins):
    """Text for one claim cell; falls back to all win totals when n < 5."""
    since = rates.attrs.get("since", 199001) // 100
    if rank_class == KOMUSUBI:
        rank_number = 0
    lab = "K" if rank_class == KOMUSUBI else f"M{rank_number}"
    pool = rates[(rates["claimed"] == claimed) & (rates["rank_class"] == rank_class)
                 & (rates["rank_number"] == rank_number)]
    cell = pool[pool["wins"] == wins]
    n, h = int(cell["n"].sum()), int(cell["honoured"].sum())
    if n >= 5:
        return (f"{lab} claims with {wins} wins needing a created slot "
                f"were honoured {h} of {n} since {since}")
    pn, ph = int(pool["n"].sum()), int(pool["honoured"].sum())
    if pn == 0:
        return f"no precedent since {since}"
    return (f"{lab} claims needing a created slot were honoured {ph} of {pn} "
            f"since {since} (all win totals; {wins}-win cell has {n})")


def _cells(p):
    return dict(zip(p["rikishi_id"], zip(p["pred_class"], p["pred_number"], p["pred_side"])))


def _was(r):
    return fmt_slot(r["rank_class"], r["rank_number"], r["side"])


def structural(pred, cands, scores, mak_size, ov, rates):
    """Items for S/K slots created by forced claims of promotees, each with
    precedent and the footprint of denying the claim (counterfactual)."""
    claims = forced_claims(cands)
    info = cands.set_index("rikishi_id")
    cells = _cells(pred)
    items = []
    for c, mask in ((SEKIWAKE, claims["s"]), (KOMUSUBI, claims["k"])):
        if c in ov["count"] or c in ov["class"].values():
            continue
        members = pred.loc[pred["pred_class"] == c, "rikishi_id"]
        if len(members) <= 2:
            continue
        holders = set(cands.loc[mask, "rikishi_id"])
        for rid in members:
            r = info.loc[rid]
            if rid not in holders or r["rank_class"] <= c:
                continue
            deny = {**ov, "class": {**ov["class"], rid: MAEGASHIRA}}
            try:
                cf = _cells(resolve(cands, scores, mak_size, overrides=deny))
            except OverrideError:
                foot = "counterfactual infeasible"
            else:
                foot = f"without it {sum(cells[k] != cf[k] for k in cells)} cells shift"
                stays = [f"{info.at[k, 'shikona']} ({_was(info.loc[k])})"
                         for k in cells if cells[k][0] == JURYO and cf[k][0] < JURYO]
                if stays:
                    foot += f" and {', '.join(stays)} stays in makuuchi"
            rec = f"{int(r['wins'])}-{int(r['losses'])}"
            if r["absences"]:
                rec += f"-{int(r['absences'])}"
            prec = precedent(rates, c, r["rank_class"], r["rank_number"], r["wins"])
            items.append({
                "marker": "!", "rid": rid, "range": fmt_slot(*cells[rid]),
                "text": (f"{CLS_CHARS[c]}{len(members)} created for {r['shikona']} "
                         f"({_was(r)} {rec}): {prec}; {foot}"),
                "hints": [("class", f"{r['shikona']}=M")],
            })
    return items
