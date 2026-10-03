"""Residual analysis of one model's backtest predictions: where exact-slot
misses come from, whether they are systematically biased, and whether the
confidence signals (banzuke.confidence) separate reliable rows from shaky
ones. `banzuke analyze` runs cached_predictions(), residuals() and
calibration(), then the convention audit.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from banzuke import confidence
from banzuke.harness import fingerprint, run_backtest
from banzuke.paths import CACHE
from banzuke.ranks import KOMUSUBI, MAEGASHIRA, OZEKI, SEKIWAKE, fmt_cell, fmt_record


def rate_table(m, groups, title):
    """Aligned n / share / exact / far / mean |err| rows for named masks."""
    print(f"\n{title}:")
    print(f"  {'':24s} {'n':>5s} {'share':>6s} {'exact':>6s} {'far':>6s} {'|err|':>6s}")
    for name, mask in groups.items():
        sub = m[mask]
        if not len(sub):
            print(f"  {name:24s} {0:5d}")
            continue
        print(f"  {name:24s} {len(sub):5d} {len(sub) / len(m):6.3f} {sub['hit'].mean():6.3f}"
              f" {sub['far'].mean():6.3f} {sub['err'].abs().mean():6.2f}")


def calibration(preds, trans):
    """Confidence signals on the first seed's banzuke, using base scores
    averaged over seeds and per-seed orders for the spread."""
    seeds = sorted(preds["seed"].unique())
    m = pd.concat([confidence.with_signals(g, seeds[0]) for _, g in preds.groupby("target")],
                  ignore_index=True)
    m = m[m["class_next"] <= MAEGASHIRA].copy()
    m["err"] = (m["pred_pos"] - m["position_next"]).abs()
    m["hit"] = (
        (m["pred_class"] == m["class_next"])
        & (m["pred_number"] == m["number_next"])
        & (m["pred_side"] == m["side_next"])
    )
    m["far"] = m["err"] > 1
    print(f"\n=== confidence calibration ({len(seeds)} seeds, {len(m)} makuuchi slots) ===")

    tiers = {f"tier {t!r}": m["tier"] == t for t in ("", "?", "??")}
    tiers["big_move"] = m["big_move"]
    tiers["not big_move"] = ~m["big_move"]
    rate_table(m, tiers, "by tier")

    zone = np.select(
        [m["pred_class"] <= OZEKI, m["pred_class"].isin((SEKIWAKE, KOMUSUBI)),
         (m["pred_class"] == MAEGASHIRA) & (m["pred_number"] >= 14)],
        ["Y/O", "S/K", "M14+"], "M1-13")
    bucket = pd.cut(m["gap"].clip(upper=99), [-np.inf, 0, 0.25, 0.5, 1, np.inf], right=False,
                    labels=["inverted", "0-.25", ".25-.5", ".5-1", "1+/none"])
    print("\nby zone x base-score gap bucket (n exact far):")
    tab = m.groupby([pd.Series(zone, index=m.index, name="zone"), bucket], observed=False)
    tab = tab.agg(n=("hit", "size"), exact=("hit", "mean"), far=("far", "mean"))
    print(f"  {'':6s}" + "".join(f"{b:>18s}" for b in bucket.cat.categories))
    for z in ("Y/O", "S/K", "M1-13", "M14+"):
        cells = []
        for b in bucket.cat.categories:
            r = tab.loc[(z, b)]
            cells.append(f"{int(r['n']):4d} {r['exact']:.3f} {r['far']:.3f}" if r["n"]
                         else f"{0:4d}     -     -")
        print(f"  {z:6s}" + "".join(f"{c:>18s}" for c in cells))

    inc, pro = m["division"] == 0, m["division"] == 1
    mv = m["move"].abs()
    moves = {
        "incumbent climb 8+": inc & (m["move"] <= -8),
        "incumbent drop 8+": inc & (m["move"] >= 8),
        "incumbent move 4-8": inc & (mv >= 4) & (mv < 8),
        "incumbent move <4": inc & (mv < 4),
        "promotee big_move": pro & m["big_move"],
        "promotee not big_move": pro & ~m["big_move"],
    }
    rate_table(m, moves, "by move class")

    rho = spearmanr(m["n_signals"], m["err"])
    print(f"\nspearman rho(n_signals, |err|) = {rho.statistic:.3f} (p = {rho.pvalue:.2g})")

    rates = confidence.claim_rates(trans)
    print(f"\nclaim precedent (created S/K slot, since {rates.attrs['since']}):")
    for c, lab in ((SEKIWAKE, "K->S"), (KOMUSUBI, "M->K")):
        r = rates[rates["claimed"] == c]
        print(f"  pooled {lab}: {int(r['honoured'].sum())} of {int(r['n'].sum())}")
    for num in (1, 3):
        r = rates[(rates["claimed"] == KOMUSUBI) & (rates["rank_class"] == MAEGASHIRA)
                  & (rates["rank_number"] == num)]
        for _, row in r.sort_values("wins").iterrows():
            print(f"  M{num} {int(row['wins'])} wins: {int(row['honoured'])} of {int(row['n'])}")


def cached_predictions(model, targets, seeds, kwargs, trans, tidy, threads=1, fresh=False):
    """Per-row backtest predictions of `model` over the target basho for every
    seed, cached under cache/ by configuration and data fingerprint."""
    fp = fingerprint(kwargs=kwargs, seeds=seeds)
    cache = CACHE / f"preds_{model}_{targets[0]}_{targets[-1]}_{fp}.parquet"
    if cache.exists() and not fresh:
        return pd.read_parquet(cache)
    _, preds = run_backtest([model], targets, trans, tidy, return_preds=True,
                            seeds=seeds, model_kwargs=kwargs, threads=threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache, index=False)
    return preds


def residuals(preds, model, start, end, seeds):
    """Miss decomposition, signed bias by group, sanyaku slot counts, committee
    anchoring and (with several seeds) seed spread against error."""
    p0 = preds[preds["seed"] == seeds[0]]
    # rows whose actual outcome is a makuuchi slot
    m = p0[p0["class_next"] <= MAEGASHIRA].copy()
    m["err"] = m["pred_pos"] - m["position_next"]  # + = predicted too low
    m["hit"] = ((m["pred_class"] == m["class_next"]) & (m["pred_number"] == m["number_next"])
                & (m["pred_side"] == m["side_next"]))
    miss = m[~m["hit"]]
    print(f"=== {model} {start}-{end}: {len(m)} makuuchi slots, "
          f"exact {m['hit'].mean():.3f} ===\n")

    side_only = (miss["pred_class"] == miss["class_next"]) & (
        miss["pred_number"] == miss["number_next"])
    near = ~side_only & (miss["err"].abs() <= 1)
    print("miss decomposition:")
    print(f"  side-only (E/W flip):      {side_only.mean():.3f}")
    print(f"  near (|err| <= 1 pos):     {near.mean():.3f}")
    print(f"  far  (|err| >  1 pos):     {(~side_only & ~near).mean():.3f}")

    print("\nsigned bias (mean err, + = under-promoted) by group:")
    groups = {
        "juryo promotee": (m["division"] == 1),
        "yusho winner": m["yusho"] == 1,
        "sansho winner": m["sansho"] > 0,
        "big win (11+)": m["wins"] >= 11,
        "big loss (<=4)": (m["wins"] <= 4) & (m["absences"] < 8),
        "mostly absent": m["absences"] >= 8,
        "kk streak >=3": m["kk_streak"] >= 3,
        "sanyaku now": m["rank_class"] <= KOMUSUBI,
        "M1-M5": (m["rank_class"] == MAEGASHIRA) & (m["rank_number"] <= 5),
        "M11+": (m["rank_class"] == MAEGASHIRA) & (m["rank_number"] >= 11),
    }
    for name, mask in groups.items():
        sub = m[mask]
        print(f"  {name:16s} n={len(sub):5d}  err={sub['err'].mean():+.2f}"
              f"  |err|={sub['err'].abs().mean():.2f}  exact={sub['hit'].mean():.3f}")

    # sanyaku slot counts: a wrong S/K block size shifts every slot below it
    sk = [SEKIWAKE, KOMUSUBI]
    pred_n = p0[p0["pred_class"].isin(sk)].groupby(["target", "pred_class"]).size()
    act_n = p0[p0["class_next"].isin(sk)].groupby(["target", "class_next"]).size()
    diff = (act_n.rename_axis(["target", "cls"]).sub(pred_n.rename_axis(["target", "cls"]),
                                                      fill_value=0).unstack(fill_value=0))
    hits = m.groupby("target")["hit"].sum()
    ok = (diff == 0).all(axis=1)
    print(f"\nsanyaku slot counts right in {ok.mean():.3f} of basho: exact_n "
          f"{hits[ok].mean():.1f} when right, {hits[~ok].mean():.1f} when wrong "
          f"({(diff > 0).any(axis=1).sum()} basho with slots the rules did not create, "
          f"{(diff < 0).any(axis=1).sum()} with slots the committee did not)")
    created = p0[p0["class_next"].isin(sk) & (p0["pred_class"] == MAEGASHIRA)]
    if len(created):
        print("  received a created slot while forecast in maegashira:")
        for r in created.sort_values("target").itertuples():
            rec = fmt_record(r.wins, r.losses, r.absences)
            print(f"    {r.target}  {r.shikona:14s} {fmt_cell(r.rank_class, r.rank_number, r.side):>4s}"
                  f" {rec:7s} -> {fmt_cell(r.class_next, r.number_next, r.side_next)}"
                  f" (forecast {fmt_cell(r.pred_class, r.pred_number, r.pred_side)})")

    # committee anchoring: inverted adjacent pairs vs prior order
    inv_prior, adj_prior = [], []
    for _, g in m.groupby("target"):
        g = g.sort_values("position_next")
        a, p, prior = (g[c].to_numpy() for c in ("position_next", "pred_pos", "position"))
        for i in range(len(g) - 1):
            if a[i + 1] - a[i] <= 1:
                (inv_prior if p[i] > p[i + 1] else adj_prior).append(prior[i] < prior[i + 1])
    print(f"\nadjacent actual pairs where model inverted the order: {len(inv_prior)}")
    print(f"  actual-higher had better prior rank: {np.mean(inv_prior):.3f}"
          f"  (non-inverted baseline: {np.mean(adj_prior):.3f})")

    if len(seeds) > 1:  # does seed disagreement predict error?
        piv = preds.pivot_table(index=["target", "rikishi_id"], columns="seed", values="pred_pos")
        mm = m.join((piv.max(axis=1) - piv.min(axis=1)).rename("spread"),
                    on=["target", "rikishi_id"])
        rho = spearmanr(mm["spread"], mm["err"].abs()).statistic
        print(f"\nseed spread (max-min position over seeds {list(seeds)}) vs |err|: "
              f"Spearman {rho:.3f}")
        for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 99)):
            sub = mm[mm["spread"].between(lo, hi)]
            if len(sub):
                print(f"  spread {lo}{'+' if hi > lo else ' '}: n={len(sub):5d} "
                      f"({len(sub) / len(mm):.3f})  |err|={sub['err'].abs().mean():.2f}  "
                      f"exact={sub['hit'].mean():.3f}")
