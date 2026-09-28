#!/usr/bin/env python3
"""Predict the next makuuchi banzuke from the latest fetched basho.

Usage:
    uv run python predict.py                      # predict the next banzuke
    uv run python predict.py --retired Tamawashi  # exclude announced retirees
    uv run python predict.py --model Aq --seeds 1

Overrides (constrain the assignment; the model fills everything else):
    --above "A > B"     A rises to immediately above B
    --below "A < B"     A drops to immediately below B
    --class X=O         force class membership (YOSKMJ)
    --count S=3         force the sekiwake/komusubi slot count
    --pin X=M2E         exact cell
    --interactive       train once, then iterate on overrides instantly
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from banzuke import models
from banzuke.build import JURYO, KOMUSUBI, OZEKI, SEKIWAKE
from banzuke.overrides import OverrideError, parse, splice, verify
from banzuke.resolver import resolve

PROCESSED = Path(__file__).parent / "data" / "processed"
CLS = "YOSKMJ"
BASHO_MONTHS = (1, 3, 5, 7, 9, 11)
SPEC_KEYS = ("above", "below", "class", "count", "pin")


def next_basho_id(basho: int) -> int:
    year, month = divmod(basho, 100)
    nxt = BASHO_MONTHS[(BASHO_MONTHS.index(month) + 1) % 6]
    return (year + (nxt == 1)) * 100 + nxt


def label(c, n, s):
    return f"{CLS[int(c)]}{int(n)}{'EW'[int(s)]}"


def predict(cands, scores_list, ov, mak_size):
    """Splice relative overrides into each seed's order, resolve with the
    structural overrides. Returns (merged seed-0 pred, warnings)."""
    rids = cands["rikishi_id"].to_numpy()
    pos = cands["position"].to_numpy()
    preds, warnings = [], []
    for k, sc in enumerate(scores_list):
        order = [rids[i] for i in np.lexsort((pos, sc))]
        rank = {r: j for j, r in enumerate(splice(order, ov["relative"]))}
        pseudo = np.array([rank[r] for r in rids], dtype=float)
        preds.append(resolve(cands, pseudo, mak_size, overrides=ov,
                             warnings=warnings if k == 0 else None))
    pred = preds[0].merge(cands, on="rikishi_id")
    spread = (
        pd.concat(preds).groupby("rikishi_id")["pred_pos"].agg(lambda s: s.max() - s.min())
    )
    pred["spread"] = pred["rikishi_id"].map(spread)
    return pred, warnings


def render(pred, ov, warnings, baseline, target, latest, args):
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
        if r["rikishi_id"] in ov["pins"]:
            out += " [pin]"
        now = label(r["pred_class"], r["pred_number"], r["pred_side"])
        was = baseline.get(r["rikishi_id"])
        if was and was != now:
            out += f" <-{was}"
        if r["spread"] > 2:
            out += f" +-{r['spread']:.0f}"
        return out

    slots: dict = {}
    for _, r in mak.iterrows():
        slots.setdefault((r["pred_class"], r["pred_number"]), {})[r["pred_side"]] = r
    east = {k: cell(s.get(0)) for k, s in slots.items()}
    width = max([34] + [len(c) + 2 for c in east.values()])
    print(f"       {'EAST':<{width}}{'WEST'}")
    for key in sorted(slots):
        lab = f"{CLS[int(key[0])]}{int(key[1])}"
        print(f"  {lab:>3}  {east[key]:<{width}}{cell(slots[key].get(1))}".rstrip())

    if ov["relative"]:
        posmap = dict(zip(pred["rikishi_id"], pred["pred_pos"]))
        print("\noverrides:")
        for spec, direction, ok in verify(ov["relative"], posmap):
            state = "satisfied" if ok else "VIOLATED (a structural rule outranked it)"
            print(f"  - {direction} '{spec}': {state}")
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")

    notes = []
    for _, r in pred.iterrows():
        if r["rank_class"] == OZEKI and r["pred_class"] == OZEKI and r["wins"] < 8:
            notes.append(f"{r['shikona']}: kadoban at {target}")
        if (r["rank_class"] in (SEKIWAKE, KOMUSUBI)
                and r["pred_class"] in (SEKIWAKE, KOMUSUBI)
                and r["wins"] >= 8 and r["ozeki_run2"] >= 20):
            notes.append(f"{r['shikona']}: ozeki run, {int(r['ozeki_run2'])} wins "
                         f"over last 2 basho in sanyaku")
    demoted = pred[(pred["rank_class"] < JURYO) & (pred["pred_class"] == JURYO)]
    if len(demoted):
        notes.append("demoted to juryo: " + ", ".join(
            f"{r['shikona']} ({CLS[r['rank_class']]}{r['rank_number']}{'EW'[r['side']]})"
            for _, r in demoted.sort_values("pred_pos").iterrows()))
    if notes:
        print("\nnotes:")
        for n in notes:
            print(f"  - {n}")


def interactive(state, run):
    try:
        import readline  # noqa: F401  (line editing side effect)
    except ImportError:
        pass
    help_line = ("commands: above|below|class|count|pin <spec>, unset <shikona>, "
                 "clear, show, flags, quit")
    print(f"\n{help_line}")
    while True:
        try:
            line = input("override> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()
        if cmd in ("quit", "exit", "q"):
            return
        if cmd == "help":
            print(help_line)
            continue
        if cmd == "show":
            run(state)
            continue
        if cmd == "flags":
            flags = [f'--{k} "{v}"' for k in SPEC_KEYS for v in state[k]]
            print("predict.py " + " ".join(flags) if flags else "(no overrides)")
            continue
        new = {k: list(v) for k, v in state.items()}
        if cmd == "clear":
            new = {k: [] for k in new}
        elif cmd == "unset":
            if not rest:
                print("usage: unset <shikona>")
                continue
            new = {k: [v for v in vs if rest.lower() not in v.lower()]
                   for k, vs in new.items()}
        elif cmd in SPEC_KEYS:
            if not rest:
                print(f"usage: {cmd} <spec>")
                continue
            new[cmd].append(rest)
        else:
            print(f"unknown command '{cmd}'; {help_line}")
            continue
        try:
            run(new)
            state.update(new)
        except OverrideError as e:
            print(f"error: {e} (override not applied)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Ar", choices=list(models.MODELS))
    ap.add_argument("--retired", default="", help="comma-separated shikona to exclude")
    ap.add_argument("--mak-size", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=5,
                    help="ensemble size for the uncertainty column")
    ap.add_argument("--train-start", type=int, default=None, metavar="BASHO",
                    help="ignore training transitions before this basho")
    ap.add_argument("--above", action="append", default=[], metavar='"A > B"')
    ap.add_argument("--below", action="append", default=[], metavar='"A < B"')
    ap.add_argument("--class", dest="cls", action="append", default=[], metavar="X=O")
    ap.add_argument("--count", action="append", default=[], metavar="S=3")
    ap.add_argument("--pin", action="append", default=[], metavar="X=M2E")
    ap.add_argument("--interactive", action="store_true",
                    help="train once, then adjust overrides in a loop")
    args = ap.parse_args()

    transitions_path = PROCESSED / "transitions.parquet"
    if not transitions_path.is_file():
        ap.error("processed data not found; run `uv run python update_data.py` first")
    trans = pd.read_parquet(transitions_path)
    latest = int(trans.loc[trans["yusho"].eq(1), "basho"].max())
    target = next_basho_id(latest)
    train = trans[trans["position_next"].notna()]
    if args.train_start:
        train = train[train["basho"] >= args.train_start]
        print(f"training restricted to {train['basho'].nunique()} basho "
              f"({args.train_start}+)", file=sys.stderr)
    cands = trans[trans["basho"] == latest].reset_index(drop=True)
    previous = trans.loc[trans["basho"] < latest, "basho"].max()
    prev_sk = trans[(trans["basho"] == previous)
                    & trans["rank_class"].isin((SEKIWAKE, KOMUSUBI))
                    & (trans["wins"] >= 8)]
    cands["ozeki_run2"] = cands["wins"] + cands["rikishi_id"].map(
        prev_sk.set_index("rikishi_id")["wins"])

    retired = [s.strip().lower() for s in args.retired.split(",") if s.strip()]
    gone = cands["shikona"].str.lower().isin(retired)
    if retired:
        print(f"excluding: {', '.join(cands.loc[gone, 'shikona'])}\n")
    cands = cands[~gone].reset_index(drop=True)

    # train once per seed; everything downstream is instant
    print("training...", file=sys.stderr)
    scores_list = []
    for seed in range(args.seeds):
        models.LGB_PARAMS["random_state"] = seed
        model = models.MODELS[args.model]()
        model.fit(train)
        scores_list.append(model.score(cands))

    base = resolve(cands, scores_list[0], args.mak_size)
    baseline = {r: label(c, n, s) for r, c, n, s in zip(
        base["rikishi_id"], base["pred_class"], base["pred_number"], base["pred_side"])}

    def run(state):
        ov = parse(cands, above=state["above"], below=state["below"],
                   classes=state["class"], counts=state["count"], pins=state["pin"])
        pred, warnings = predict(cands, scores_list, ov, args.mak_size)
        render(pred, ov, warnings, baseline, target, latest, args)

    state = {"above": args.above, "below": args.below, "class": args.cls,
             "count": args.count, "pin": args.pin}
    try:
        run(state)
    except OverrideError as e:
        if not args.interactive:
            ap.error(str(e))
        print(f"error: {e}")
        state = {k: [] for k in state}
        run(state)
    if args.interactive:
        interactive(state, run)


if __name__ == "__main__":
    main()
