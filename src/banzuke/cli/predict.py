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

from banzuke.cli._common import DEFAULT_WORKERS, CommandError, parse_sets, subcommand

SPEC_KEYS = ("above", "below", "class", "count", "pin")


def add_parser(sub):
    ap = subcommand(sub, "predict", __doc__, "predict the next banzuke from the latest results")
    ap.add_argument("--model", default="Ar", metavar="NAME",
                    help="ordering model (R, L, A, Aq, Aw, B, C, Ar, Ah; see docs/MODEL.md)")
    ap.add_argument("--retired", default="", help="comma-separated shikona to exclude")
    ap.add_argument("--protected", default="",
                    help="comma-separated shikona whose full absence the JSA exempted "
                         "(rank frozen); sets the rank_protected feature")
    ap.add_argument("--mak-size", type=int, default=None,
                    help="makuuchi size (default: same as the latest banzuke)")
    ap.add_argument("--seeds", type=int, default=None, metavar="N",
                    help="bag size (default: the model's, 5 for the GBMs); single seeds "
                         "also feed the confidence markers")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="model option override (e.g. twin_unit=sk, gap=0.5), as in "
                         "`banzuke backtest`")
    ap.add_argument("--train-start", type=int, default=None, metavar="BASHO",
                    help="ignore training transitions before this basho")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="processes for the one-off out-of-fold table (cached afterwards)")
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
    from banzuke.build import JURYO
    from banzuke.forecast import CLS, label, notes, rec
    from banzuke.overrides import verify

    how = f"{n_seeds}-seed bag" if n_seeds > 1 else "seed 0"
    print(f"predicted makuuchi banzuke for {target} "
          f"(from {latest} results, model {args.model}, {how})\n")
    mak = pred[pred["pred_class"] < JURYO].sort_values("pred_pos")

    def cell(r):
        if r is None:
            return ""
        prev = f"{CLS[r['rank_class']]}{r['rank_number']}{'EW'[r['side']]}"
        out = f"{r['marker']:<3} {r['shikona']:<14} ({prev:>4} {rec(r['wins'], r['losses'], r['absences'])})"
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

    from banzuke import confidence, forecast, models
    from banzuke.overrides import OverrideError, parse
    from banzuke.paths import load_transitions
    from banzuke.resolver import resolve

    if args.model not in models.MODELS:
        raise CommandError(f"unknown model {args.model!r}; available: {list(models.MODELS)}")
    trans = load_transitions()
    latest = forecast.latest_basho(trans)
    target = forecast.next_basho_id(latest)
    train = forecast.training_rows(trans, latest, args.train_start)
    if args.train_start:
        print(f"training restricted to {train['basho'].nunique()} basho "
              f"({args.train_start}+)", file=sys.stderr)
    cands = forecast.candidates(trans, latest)

    retired = [s.strip().lower() for s in args.retired.split(",") if s.strip()]
    gone = cands["shikona"].str.lower().isin(retired)
    if retired:
        print(f"excluding: {', '.join(cands.loc[gone, 'shikona'])}\n")
    cands = cands[~gone].reset_index(drop=True)
    protected = [s.strip().lower() for s in args.protected.split(",") if s.strip()]
    if protected:
        keep = cands["shikona"].str.lower().isin(protected)
        missing = sorted(set(protected) - set(cands.loc[keep, "shikona"].str.lower()))
        if missing:
            raise CommandError(f"--protected: unknown shikona {missing}")
        cands.loc[keep, "rank_protected"] = 1
        print(f"rank protected: {', '.join(cands.loc[keep, 'shikona'])}\n")

    # train once; everything downstream is instant
    print("training...", file=sys.stderr)
    cls = models.MODELS[args.model]
    kwargs = {**parse_sets(args.set), **({"n_seeds": args.seeds} if args.seeds else {})}
    kwargs = cls.prepare(kwargs, trans, args.workers, args.train_start)
    model = cls(**kwargs)
    model.fit(train)
    point = model.score(cands)
    per_seed = [model.score(cands, k) for k in range(len(model.seeds))] \
        if len(model.seeds) > 1 else []
    base = pd.Series(getattr(model, "base_score", model.score)(cands),
                     index=cands["rikishi_id"].to_numpy())
    rates = confidence.claim_rates(trans)

    mak_size = args.mak_size or int(cands["mak_size"].iloc[0])
    base_pred = resolve(cands, point, mak_size)
    baseline = {r: forecast.label(c, n, s) for r, c, n, s in zip(
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
            raise CommandError(str(e))
        print(f"error: {e}")
        state = {k: [] for k in state}
        run_once(state)
    if args.interactive:
        interactive(state, run_once)
