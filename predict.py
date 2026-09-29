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

Markers before each name flag where to look: ! occupies an S/K slot the
rules created, ? low confidence, ?? very low confidence (from tight
ordering calls and seed disagreement), ~ big move. The review list under
the sheet names the decisions behind them with an override to test the
alternative.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from banzuke import confidence, models
from banzuke.build import JURYO, KOMUSUBI, OZEKI, SEKIWAKE
from banzuke.harness import parse_sets
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


def predict(cands, point, per_seed, ov, mak_size, base=None, rates=None):
    """Splice relative overrides into the point forecast's order (the seed
    bag) and into each single seed's order, resolve with the structural
    overrides. Returns (point pred with a `spread` column, warnings, review
    items); spread is the range of a rikishi's resolved position across single
    seeds, a sensitivity diagnostic rather than a calibrated interval.
    base: mean base score per rikishi (Series); drives the confidence markers."""
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
    items = confidence.review(pred, sig, base, final, preds, skip=named)
    if rates is not None:
        items = confidence.structural(pred, cands, pseudos[0], mak_size, ov, rates) + items
    for it in items:
        if it["marker"] == "!":
            pred.loc[pred["rikishi_id"] == it["rid"], "marker"] = "!"
    return pred, warnings, items


def render(pred, ov, warnings, items, baseline, target, latest, args):
    how = f"{args.seeds}-seed bag" if args.seeds > 1 else "seed 0"
    print(f"predicted makuuchi banzuke for {target} "
          f"(from {latest} results, model {args.model}, {how})\n")
    mak = pred[pred["pred_class"] < JURYO].sort_values("pred_pos")

    def cell(r):
        if r is None:
            return ""
        prev = f"{CLS[r['rank_class']]}{r['rank_number']}{'EW'[r['side']]}"
        rec = f"{int(r['wins'])}-{int(r['losses'])}"
        if r["absences"]:
            rec += f"-{int(r['absences'])}"
        out = f"{r['marker']:<3} {r['shikona']:<14} ({prev:>4} {rec})"
        if r["rikishi_id"] in ov["pins"]:
            out += " [pin]"
        now = label(r["pred_class"], r["pred_number"], r["pred_side"])
        was = baseline.get(r["rikishi_id"])
        if was and was != now:
            out += f" <-{was}"
        return out

    slots: dict = {}
    for _, r in mak.iterrows():
        slots.setdefault((r["pred_class"], r["pred_number"]), {})[r["pred_side"]] = r
    east = {k: cell(s.get(0)) for k, s in slots.items()}
    width = max([38] + [len(c) + 2 for c in east.values()])
    print(f"{'':11}{'EAST':<{width}}WEST")
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

    if items:
        print("\nreview (least confident first; ! created slot, ?? very low confidence, "
              "? low confidence, ~ big move):")
        print("  base score: the model's predicted position in cells before reranking; "
              "neighbours under .25 apart are coin flips")
        fmt = (lambda f, s: f"{f} {s}") if args.interactive else (lambda f, s: f'--{f} "{s}"')
        for it in items:
            head = f"  {it['marker']:<2} {it['range']:<10}"
            who = " > ".join(it["members"]) + "   " if "members" in it else ""
            print(f"{head} {who}{it['text']}")
            if it["hints"]:
                print(f"{'':16}try " + "  or  ".join(fmt(f, s) for f, s in it["hints"]))

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
    ap.add_argument("--mak-size", type=int, default=None,
                    help="makuuchi size (default: same as the latest banzuke)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="bag size: the forecast averages this many seeds; single "
                         "seeds also feed the +-N sensitivity column and the markers")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override, as in backtest.py (base.n_estimators=200)")
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
    # labels published by `latest`: a fetched-but-unplayed next banzuke must
    # not supply the transition being predicted
    train = trans[trans["position_next"].notna() & (trans["next_basho"] <= latest)]
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

    # train once; everything downstream is instant
    print("training...", file=sys.stderr)
    kwargs = models.MODELS[args.model].prepare(parse_sets(args.set), trans) \
        if hasattr(models.MODELS[args.model], "prepare") else parse_sets(args.set)
    model = models.MODELS[args.model](seed=0, n_seeds=args.seeds, **kwargs)
    model.fit(train)
    point = model.score(cands)
    per_seed = [model.score(cands, k) for k in range(args.seeds)] if args.seeds > 1 else []
    base = pd.Series(getattr(model, "base_score", model.score)(cands),
                     index=cands["rikishi_id"].to_numpy())
    rates = confidence.claim_rates(trans)

    mak_size = args.mak_size or int(cands["mak_size"].iloc[0])
    base_pred = resolve(cands, point, mak_size)
    baseline = {r: label(c, n, s) for r, c, n, s in zip(
        base_pred["rikishi_id"], base_pred["pred_class"], base_pred["pred_number"],
        base_pred["pred_side"])}

    def run(state):
        ov = parse(cands, above=state["above"], below=state["below"],
                   classes=state["class"], counts=state["count"], pins=state["pin"])
        pred, warnings, items = predict(cands, point, per_seed, ov, mak_size, base, rates)
        render(pred, ov, warnings, items, baseline, target, latest, args)

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
