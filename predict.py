#!/usr/bin/env python3
"""Predict the next makuuchi banzuke from the latest fetched basho.

Usage:
    uv run python predict.py                      # predict the next banzuke
    uv run python predict.py --retired Tamawashi  # exclude announced retirees
    uv run python predict.py --model Aq --seeds 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from banzuke import models
from banzuke.build import JURYO, KOMUSUBI, OZEKI, SEKIWAKE
from banzuke.resolver import resolve

PROCESSED = Path(__file__).parent / "data" / "processed"
CLS = "YOSKMJ"
BASHO_MONTHS = (1, 3, 5, 7, 9, 11)


def next_basho_id(basho: int) -> int:
    year, month = divmod(basho, 100)
    nxt = BASHO_MONTHS[(BASHO_MONTHS.index(month) + 1) % 6]
    return (year + (nxt == 1)) * 100 + nxt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Ar", choices=list(models.MODELS))
    ap.add_argument("--retired", default="", help="comma-separated shikona to exclude")
    ap.add_argument("--mak-size", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=5,
                    help="ensemble size for the uncertainty column")
    args = ap.parse_args()

    trans = pd.read_parquet(PROCESSED / "transitions.parquet")
    latest = int(trans["basho"].max())
    target = next_basho_id(latest)
    train = trans[trans["position_next"].notna()]
    cands = trans[trans["basho"] == latest].reset_index(drop=True)

    retired = [s.strip().lower() for s in args.retired.split(",") if s.strip()]
    gone = cands["shikona"].str.lower().isin(retired)
    if retired:
        print(f"excluding: {', '.join(cands.loc[gone, 'shikona'])}\n")
    cands = cands[~gone].reset_index(drop=True)

    # seed ensemble: first seed is the prediction, spread is the uncertainty
    preds = []
    for seed in range(args.seeds):
        models.LGB_PARAMS["random_state"] = seed
        model = models.MODELS[args.model]()
        model.fit(train)
        preds.append(resolve(cands, model.score(cands), args.mak_size))
    pred = preds[0].merge(cands, on="rikishi_id")
    spread = (
        pd.concat(preds)
        .groupby("rikishi_id")["pred_pos"]
        .agg(lambda s: s.max() - s.min())
    )
    pred["spread"] = pred["rikishi_id"].map(spread)

    print(f"predicted makuuchi banzuke for {target} "
          f"(from {latest} results, model {args.model}, {args.seeds} seeds)\n")
    mak = pred[pred["pred_class"] < JURYO].sort_values("pred_pos")

    def cell(r):
        if r is None:
            return ""
        prev = f"{CLS[r['rank_class']]}{r['rank_number']}{'EW'[r['side']]}"
        rec = f"{int(r['wins'])}-{int(r['losses'])}"
        if r["absences"]:
            rec += f"-{int(r['absences'])}"
        out = f"{r['shikona']:<14} ({prev:>4} {rec})"
        if r["spread"] > 2:
            out += f" +-{r['spread']:.0f}"
        return out

    slots: dict = {}
    for _, r in mak.iterrows():
        slots.setdefault((r["pred_class"], r["pred_number"]), {})[r["pred_side"]] = r
    width = 34
    print(f"       {'EAST':<{width}}{'WEST'}")
    for (c, n), sides in sorted(slots.items()):
        label = f"{CLS[int(c)]}{int(n)}"
        line = f"  {label:>3}  {cell(sides.get(0)):<{width}}{cell(sides.get(1))}"
        print(line.rstrip())

    notes = []
    for _, r in pred.iterrows():
        if r["rank_class"] == OZEKI and r["pred_class"] == OZEKI and r["wins"] < 8:
            notes.append(f"{r['shikona']}: kadoban at {target}")
        if (r["rank_class"] in (SEKIWAKE, KOMUSUBI) and r["wins"] >= 10
                and (r["roll3"] or 0) >= 26):
            notes.append(f"{r['shikona']}: ozeki run, {int(r['roll3'])} wins "
                         f"over last 3 basho")
    demoted = pred[(pred["rank_class"] < JURYO) & (pred["pred_class"] == JURYO)]
    if len(demoted):
        notes.append("demoted to juryo: " + ", ".join(
            f"{r['shikona']} ({CLS[r['rank_class']]}{r['rank_number']}{'EW'[r['side']]})"
            for _, r in demoted.sort_values("pred_pos").iterrows()))
    if notes:
        print("\nnotes:")
        for n in notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
