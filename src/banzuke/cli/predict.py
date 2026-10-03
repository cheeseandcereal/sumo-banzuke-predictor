"""Predict the next makuuchi banzuke from the latest fetched basho.

    banzuke predict                        # predict the next banzuke
    banzuke predict --retired Tamawashi    # exclude announced retirees
    banzuke predict --protected Takayasu   # absence exempted by the JSA (rank frozen)
    banzuke predict --model Aq --seeds 1
    banzuke predict --set twin_unit=sk     # model option, as in `banzuke backtest`

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
import sys

from banzuke.cli._common import (
    CommandError,
    add_model,
    add_set,
    add_threads,
    add_train_start,
    build_model,
    model_class,
    parse_sets,
    subcommand,
)

SPEC_KEYS = ("above", "below", "class", "count", "pin")


def add_parser(sub):
    ap = subcommand(sub, "predict", __doc__, "predict the next banzuke from the latest results")
    add_model(ap, "Ar")
    ap.add_argument("--retired", default="", help="comma-separated shikona to exclude")
    ap.add_argument("--protected", default="",
                    help="comma-separated shikona whose full absence the JSA exempted "
                         "(rank frozen); sets the rank_protected feature")
    ap.add_argument("--mak-size", type=int, default=None,
                    help="makuuchi size (default: same as the latest banzuke)")
    ap.add_argument("--seeds", type=int, default=None, metavar="N",
                    help="bag size (default: the model's, 5 for the GBMs); single seeds "
                         "also feed the confidence markers")
    add_set(ap)
    add_train_start(ap)
    add_threads(ap, "CPU threads: the bag's seeds are fitted concurrently, up to this many; "
                    "also the processes for the out-of-fold table when a non-default base "
                    "stage (--seeds, --set base.*, --train-start) needs its own, or the "
                    "committed one is stale")
    ap.add_argument("--above", action="append", default=[], metavar='"A > B"',
                    help="A rises to immediately above B (repeatable)")
    ap.add_argument("--below", action="append", default=[], metavar='"A < B"',
                    help="A drops to immediately below B (repeatable)")
    ap.add_argument("--class", dest="cls", action="append", default=[], metavar="X=O",
                    help="force class membership, one of YOSKMJ (repeatable)")
    ap.add_argument("--count", action="append", default=[], metavar="S=3",
                    help="force the sekiwake or komusubi slot count (repeatable)")
    ap.add_argument("--pin", action="append", default=[], metavar="X=M2E",
                    help="exact cell (repeatable)")
    ap.add_argument("--interactive", action="store_true",
                    help="train once, then adjust overrides in a loop")
    ap.set_defaults(run=run, _parser=ap)
    return ap


def render(pred, ov, warnings, items, baseline, target, latest, args, n_seeds):
    from banzuke.forecast import notes
    from banzuke.overrides import verify
    from banzuke.ranks import JURYO, fmt_cell, fmt_rank, fmt_record

    how = f"{n_seeds}-seed bag" if n_seeds > 1 else "seed 0"
    print(f"predicted makuuchi banzuke for {target} "
          f"(from {latest} results, model {args.model}, {how})\n")
    mak = pred[pred["pred_class"] < JURYO].sort_values("pred_pos")

    def cell(r):
        if r is None:
            return ""
        prev = fmt_cell(r['rank_class'], r['rank_number'], r['side'])
        rec = fmt_record(r["wins"], r["losses"], r["absences"])
        out = f"{r['marker']:<3} {r['shikona']:<14} ({prev:>4} {rec})"
        if r["rikishi_id"] in ov["pins"]:
            out += " [pin]"
        now = fmt_cell(r["pred_class"], r["pred_number"], r["pred_side"])
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
        print(f"  {fmt_rank(*key):>3}  {east[key]:<{width}}{cell(slots[key].get(1))}".rstrip())

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

    items = notes(pred, target)
    if items:
        print("\nnotes:")
        for n in items:
            print(f"  - {n}")


def interactive(state, run_once):
    from banzuke.overrides import OverrideError

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
            run_once(state)
            continue
        if cmd == "flags":
            flags = [f'--{k} "{v}"' for k in SPEC_KEYS for v in state[k]]
            print("banzuke predict " + " ".join(flags) if flags else "(no overrides)")
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
            run_once(new)
            state.update(new)
        except OverrideError as e:
            print(f"error: {e} (override not applied)")


def run(args):
    import pandas as pd

    from banzuke import confidence, forecast
    from banzuke.overrides import OverrideError, parse
    from banzuke.paths import load_transitions
    from banzuke.ranks import fmt_cell, next_basho
    from banzuke.resolver import resolve

    model_cls = model_class(args.model)
    trans = load_transitions()
    latest = forecast.latest_basho(trans)
    target = next_basho(latest)
    train = forecast.training_rows(trans, latest, args.train_start)
    if args.train_start:
        print(f"training restricted to {train['basho'].nunique()} basho "
              f"({args.train_start}+)", file=sys.stderr)
    retired, protected = args.retired.split(","), args.protected.split(",")
    try:
        cands = forecast.candidates(trans, latest, retired, protected)
    except OverrideError as e:
        raise CommandError(str(e)) from e
    if any(retired):
        wanted = {n.strip().lower() for n in retired}
        on_sheet = trans.loc[trans["basho"] == latest, "shikona"]
        print(f"excluding: {', '.join(n for n in on_sheet if n.lower() in wanted)}\n")
    if any(protected):
        print(f"rank protected: {', '.join(cands.loc[cands['rank_protected'] == 1, 'shikona'])}\n")

    # train once; everything downstream is instant
    print("training...", file=sys.stderr)
    kwargs = {**parse_sets(args.set), **({"n_seeds": args.seeds} if args.seeds else {})}
    model = build_model(model_cls, kwargs, trans, args.threads, args.train_start)
    model.fit(train)
    point = model.score(cands)
    per_seed = [model.score(cands, k) for k in range(len(model.seeds))] \
        if len(model.seeds) > 1 else []
    base = pd.Series((model.base_score or model.score)(cands),
                     index=cands["rikishi_id"].to_numpy())
    rates = confidence.claim_rates(trans)

    mak_size = args.mak_size or forecast.mak_size_of(cands)
    base_pred = resolve(cands, point, mak_size)
    baseline = {r: fmt_cell(c, n, s) for r, c, n, s in zip(
        base_pred["rikishi_id"], base_pred["pred_class"], base_pred["pred_number"],
        base_pred["pred_side"])}

    def run_once(state):
        ov = parse(cands, above=state["above"], below=state["below"],
                   classes=state["class"], counts=state["count"], pins=state["pin"])
        pred, warnings, items = forecast.predict(cands, point, per_seed, ov, mak_size, base, rates)
        render(pred, ov, warnings, items, baseline, target, latest, args, len(model.seeds))

    state = {"above": args.above, "below": args.below, "class": args.cls,
             "count": args.count, "pin": args.pin}
    try:
        run_once(state)
    except OverrideError as e:
        if not args.interactive:
            raise CommandError(str(e)) from e
        print(f"error: {e}")
        state = {k: [] for k in state}
        run_once(state)
    if args.interactive:
        interactive(state, run_once)
