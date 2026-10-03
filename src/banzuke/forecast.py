"""Assemble a forecast of the next banzuke: candidates with the context the
notes need, the overrides spliced into the model's order, the resolver run
once per bag member for the confidence signals, and the notes (promotions,
kadoban, ozeki runs, juryo demotions). Pure functions over frames; `banzuke
predict` trains the model, calls these and renders the sheet.
"""
import numpy as np
import pandas as pd

from banzuke import confidence
from banzuke.overrides import OverrideError, splice
from banzuke.ranks import (
    CLS_NAMES,
    JURYO,
    KOMUSUBI,
    MAEGASHIRA,
    OZEKI,
    SEKIWAKE,
    YOKOZUNA,
    fmt_cell,
    fmt_record,
)
from banzuke.resolver import resolve


def latest_basho(trans: pd.DataFrame) -> int:
    """The most recent basho with results (a yusho); a fetched-but-unplayed
    banzuke after it is the one being predicted."""
    return int(trans.loc[trans["yusho"].eq(1), "basho"].max())


def training_rows(trans: pd.DataFrame, latest: int, train_start=None) -> pd.DataFrame:
    """Labeled transitions decided by `latest`; the unplayed next banzuke must
    not supply the transition being predicted."""
    train = trans[trans["position_next"].notna() & (trans["next_basho"] <= latest)]
    if train_start:
        train = train[train["basho"] >= train_start]
    return train


def candidates(trans: pd.DataFrame, latest: int, retired=(), protected=()) -> pd.DataFrame:
    """Everyone on the `latest` banzuke who is still around, with context().
    retired: shikona to leave out (announced retirements). protected: shikona
    whose full absence the JSA exempted; sets their rank_protected feature.
    Names are case-insensitive; an unknown one is an OverrideError."""
    cands = trans[(trans["basho"] == latest) & ~trans["dropped"]].reset_index(drop=True)
    gone = _named(cands, retired, "--retired")
    cands = cands[~gone].reset_index(drop=True)
    cands.loc[_named(cands, protected, "--protected"), "rank_protected"] = 1
    return context(cands, trans)


def mak_size_of(cands: pd.DataFrame) -> int:
    """The makuuchi size of the banzuke the candidates stand on; the next
    sheet is sized like it unless told otherwise."""
    return int(cands["mak_size"].iloc[0])


def _named(cands, names, flag):
    """Boolean mask of the candidates named (case-insensitive)."""
    wanted = {n.strip().lower() for n in names if n.strip()}
    mask = cands["shikona"].str.lower().isin(wanted)
    missing = sorted(wanted - set(cands.loc[mask, "shikona"].str.lower()))
    if missing:
        raise OverrideError(f"{flag}: unknown shikona {', '.join(missing)}")
    return mask


def context(cands, trans):
    """Columns the notes need beyond the candidates' own rows: the previous
    basho's record (rec1, NaN for men who were not on that banzuke) and the
    two-basho sanyaku win total (ozeki_run2)."""
    latest = int(cands["basho"].iloc[0])
    previous = trans.loc[trans["basho"] < latest, "basho"].max()
    prev = trans[trans["basho"] == previous].set_index("rikishi_id")
    out = cands.copy()
    out["rec1"] = out["rikishi_id"].map(
        {r: fmt_record(w, lo, a)
         for r, w, lo, a in zip(prev.index, prev["wins"], prev["losses"], prev["absences"])})
    sk = prev[prev["rank_class"].isin((SEKIWAKE, KOMUSUBI)) & (prev["wins"] >= 8)]
    out["ozeki_run2"] = out["wins"] + out["rikishi_id"].map(sk["wins"])
    return out


def predict(cands, point, per_seed, ov, mak_size, base=None, rates=None):
    """Splice relative overrides into the bag's order and each seed's order,
    resolve with the structural overrides. Returns (pred, warnings, review
    items); the `spread` column is a rikishi's position range across single
    seeds. base: mean base score per rikishi (Series); drives the markers."""
    rids = cands["rikishi_id"].to_numpy()
    pos = cands["position"].to_numpy()
    preds, pseudos, warnings = [], [], []
    for k, sc in enumerate([point, *per_seed]):
        order = [rids[i] for i in np.lexsort((pos, sc))]
        rank = {r: j for j, r in enumerate(splice(order, ov["relative"]))}
        pseudos.append(np.array([rank[r] for r in rids], dtype=float))
        preds.append(resolve(cands, pseudos[-1], mak_size, overrides=ov,
                             warnings=warnings if k == 0 else None))
    pred = preds[0].merge(cands, on="rikishi_id")
    if base is None:
        base = pd.Series(point, index=rids)
    final = pd.Series(pseudos[0], index=rids)
    named = {r for _, chain, _ in ov["relative"] for seg in chain for r in seg}
    named |= set(ov["class"]) | set(ov["pins"])
    sig = confidence.signals(pred, base, final, preds[1:], skip=named)
    pred = pred.join(sig)
    items = confidence.review(pred, sig, base, final, preds[1:], skip=named)
    if rates is not None:
        items = confidence.structural(pred, cands, pseudos[0], mak_size, ov, rates) + items
    for it in items:
        if it["marker"] == "!":
            pred.loc[pred["rikishi_id"] == it["rid"], "marker"] = "!"
    return pred, warnings, items


def _yokozuna_why(r):
    if r["yusho"] == 1 and (r["yusho1"] == 1 or r["junyusho1"] == 1):
        last = "yusho" if r["yusho1"] == 1 else "jun-yusho"
        where = "" if r["class1"] == OZEKI else f" as {CLS_NAMES[int(r['class1'])]}"
        return f", yusho {fmt_record(r['wins'], r['losses'], r['absences'])} after {last} {r['rec1']}{where}"
    return " by override"


def _ozeki_why(r):
    c1, c2 = r["class1"], r["class2"]
    if np.isnan(r["roll3"]) or r["rank_class"] not in (SEKIWAKE, KOMUSUBI):
        return " by override"
    n = int(r["roll3"])
    if c1 <= KOMUSUBI and c2 <= KOMUSUBI:
        return f", {n} wins over last 3 basho in sanyaku"
    if (c1 == MAEGASHIRA) != (c2 == MAEGASHIRA) and min(c1, c2) <= KOMUSUBI:
        num = r["num1"] if c1 == MAEGASHIRA else r["num2"]
        return f", {n} wins over last 3 basho, one of them at M{int(num)}"
    return " by override"


def notes(pred, target):
    """Events on the sheet: yokozuna and ozeki promotions with the results
    behind them (or "by override" when no convention explains one), ozeki
    returns, kadoban, ozeki runs, juryo demotions. pred needs context()."""
    out = []
    for _, r in pred.sort_values("pred_pos").iterrows():
        if r["pred_class"] == YOKOZUNA and r["rank_class"] > YOKOZUNA:
            out.append(f"{r['shikona']}: promoted to yokozuna{_yokozuna_why(r)}")
        if r["pred_class"] == OZEKI and r["rank_class"] > OZEKI:
            if r["demoted_ozeki"] == 1:
                out.append(f"{r['shikona']}: returns to ozeki, "
                           f"{fmt_record(r['wins'], r['losses'], r['absences'])} the basho after demotion")
            else:
                out.append(f"{r['shikona']}: promoted to ozeki{_ozeki_why(r)}")
        if r["rank_class"] == OZEKI and r["pred_class"] == OZEKI and r["wins"] < 8:
            out.append(f"{r['shikona']}: kadoban at {target}")
        if (r["rank_class"] in (SEKIWAKE, KOMUSUBI)
                and r["pred_class"] in (SEKIWAKE, KOMUSUBI)
                and r["wins"] >= 8 and r["ozeki_run2"] >= 20):
            out.append(f"{r['shikona']}: ozeki run, {int(r['ozeki_run2'])} wins "
                       f"over last 2 basho in sanyaku")
    demoted = pred[(pred["rank_class"] < JURYO) & (pred["pred_class"] == JURYO)]
    if len(demoted):
        out.append("demoted to juryo: " + ", ".join(
            f"{r['shikona']} ({fmt_cell(r['rank_class'], r['rank_number'], r['side'])})"
            for _, r in demoted.sort_values("pred_pos").iterrows()))
    return out
