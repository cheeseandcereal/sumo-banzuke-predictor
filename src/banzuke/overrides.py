"""Human-in-the-loop overrides for banzuke prediction.

Overrides constrain the assignment, not the model: scores are computed
once, relative overrides splice the model's ordering, and structural
overrides steer the resolver. There is no conditional re-inference; the
sheet shifts structurally around the override.

Spec grammar (shikona case-insensitive):
    above: "A > B"      A rises to immediately above B
           "A > B > C"  A,B become a contiguous block above C
           "A > B,C"    A rises above the best-ranked of the group
    below: "A < B"      A drops to immediately below B (mirror rules)
    class: "X=O"        force class membership (one of YOSKMJ)
    count: "S=3"        force the sekiwake/komusubi slot count
    pin:   "X=M2E"      exact cell (implies the class assertion)
"""
import difflib
import re

from banzuke.ranks import CLS_CHARS


class OverrideError(ValueError):
    pass


def parse_slot(spec):
    m = re.fullmatch(r"([YOSKMJ])(\d+)([EW])", spec.strip().upper())
    if not m:
        raise OverrideError(f"bad slot '{spec.strip()}': expected like M2E")
    return CLS_CHARS.index(m[1]), int(m[2]), int(m[3] == "W")


def _lookup(name, names):
    key = name.strip().lower()
    if key in names:
        return names[key]
    close = difflib.get_close_matches(key, names, n=3)
    hint = f" (did you mean: {', '.join(close)}?)" if close else ""
    raise OverrideError(f"unknown shikona '{name.strip()}'{hint}")


def parse(cands, above=(), below=(), classes=(), counts=(), pins=()):
    """Parse override specs against the candidate frame. Returns a dict:
    relative: [(direction, chain of rikishi_id segments, spec)],
    class: {rikishi_id: class}, count: {class: n},
    pins: {rikishi_id: (class, number, side)}."""
    names = {}
    for rid, s in zip(cands["rikishi_id"], cands["shikona"]):
        names.setdefault(s.lower(), rid)

    relative = []
    for direction, specs, op, other in (("above", above, ">", "<"),
                                        ("below", below, "<", ">")):
        for spec in specs:
            if other in spec:
                raise OverrideError(f"--{direction} takes '{op}' relations, got '{spec}'")
            segs = [s.strip() for s in spec.split(op)]
            if len(segs) < 2 or not all(segs):
                raise OverrideError(f"bad --{direction} spec '{spec}'")
            chain = []
            for i, seg in enumerate(segs):
                parts = [p for p in (x.strip() for x in seg.split(",")) if p]
                if i < len(segs) - 1 and len(parts) != 1:
                    raise OverrideError(
                        f"only the last segment of '{spec}' may be a group")
                chain.append([_lookup(p, names) for p in parts])
            flat = [r for seg in chain for r in seg]
            if len(set(flat)) != len(flat):
                raise OverrideError(f"repeated shikona in '{spec}'")
            relative.append((direction, chain, spec))

    cls_assert, pin_map, count_map = {}, {}, {}
    for spec in pins:
        name, eq, slot = spec.partition("=")
        if not eq:
            raise OverrideError(f"bad --pin spec '{spec}': expected NAME=SLOT")
        rid = _lookup(name, names)
        if rid in pin_map:
            raise OverrideError(f"duplicate pin for '{name.strip()}'")
        pin_map[rid] = parse_slot(slot)
        cls_assert[rid] = pin_map[rid][0]
    if len(set(pin_map.values())) != len(pin_map):
        raise OverrideError("two pins target the same slot")

    for spec in classes:
        name, eq, c = spec.partition("=")
        c = c.strip().upper()
        if not eq or c not in CLS_CHARS or len(c) != 1:
            raise OverrideError(f"bad --class spec '{spec}': expected NAME=C, C in {CLS_CHARS}")
        rid = _lookup(name, names)
        if cls_assert.get(rid, CLS_CHARS.index(c)) != CLS_CHARS.index(c):
            raise OverrideError(f"conflicting class/pin overrides for '{name.strip()}'")
        cls_assert[rid] = CLS_CHARS.index(c)

    for spec in counts:
        c, eq, n = spec.partition("=")
        c = c.strip().upper()
        if not eq or c not in "SK":
            raise OverrideError(f"bad --count spec '{spec}': expected S=n or K=n")
        try:
            count_map[CLS_CHARS.index(c)] = int(n)
        except ValueError:
            raise OverrideError(f"bad --count spec '{spec}': '{n}' is not a number") from None

    return {"relative": relative, "class": cls_assert,
            "count": count_map, "pins": pin_map}


def splice(order, relative):
    """Apply relative overrides to an ordering of rikishi_ids (best first).
    Each constraint moves only its first-named wrestler: the minimal
    weight adjustment sufficient to clear the target. Sequential, so a
    later override sees the effect of an earlier one."""
    order = list(order)
    for direction, chain, _ in relative:
        for i in range(len(chain) - 2, -1, -1):  # right-to-left through the chain
            mover, targets = chain[i][0], chain[i + 1]
            mpos = order.index(mover)
            tpos = [order.index(t) for t in targets]
            satisfied = (mpos < min(tpos)) if direction == "above" else (mpos > max(tpos))
            if satisfied:  # already clears the target: no adjustment
                continue
            order.remove(mover)
            tpos = [order.index(t) for t in targets]
            at = min(tpos) if direction == "above" else max(tpos) + 1
            order.insert(at, mover)
    return order


def verify(relative, pos):
    """Check each relative override against final joint positions
    (rikishi_id -> pred_pos). Returns [(spec, direction, satisfied)]."""
    out = []
    for direction, chain, spec in relative:
        ok = True
        for i in range(len(chain) - 1):
            m = pos[chain[i][0]]
            ts = [pos[t] for t in chain[i + 1]]
            ok &= m < min(ts) if direction == "above" else m > max(ts)
        out.append((spec, direction, bool(ok)))
    return out
