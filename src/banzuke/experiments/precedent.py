"""Committee precedent queries over transitions.parquet, always with counts.

    banzuke precedent landing "K1 5-10"
    banzuke precedent landing "M1 8-7" --where "year>=2010"
    banzuke precedent landing "K1 5-10" --list    # and the rows behind the counts
    banzuke precedent pair "K 5-10" "M9-10 10-5"
    banzuke precedent cells "S 7-8"      # landing cells, counted

Spec grammar: <class>[<number>|<lo>-<hi>][E|W] <record>
    class   Y O S K M J
    record  9-6 | 5-7-3 (exact W-L[-A]) | 8+ (at least 8 wins) | kk | mk | *
Windows: all (1959+), 2004+ (modern, 42-man makuuchi), last60 (the 60 most
recent basho). `--where` is a pandas query on the transition columns.
"""
import re

import numpy as np

from banzuke.ranks import CLS_CHARS, MODERN_ERA, fmt_cell, fmt_record

SPEC = re.compile(rf"^([{CLS_CHARS}])(?:(\d+)(?:-(\d+))?)?([EW])?$")


def labeled(trans):
    """Transitions with a known next banzuke (the committee's decision)."""
    return trans[trans["class_next"].notna()].reset_index(drop=True)


def windows(t):
    bashos = sorted(t["basho"].unique())
    return (("all", bashos[0]), ("2004+", MODERN_ERA), ("last60", bashos[-60]))


def parse_spec(spec):
    """'M9-10 10-5' -> dict of column filters."""
    rank, _, rec = spec.strip().partition(" ")
    m = SPEC.match(rank.upper())
    if not m:
        raise ValueError(f"bad rank spec {rank!r}; expected like K1, M9-10, J3W")
    f = {"rank_class": CLS_CHARS.index(m[1])}
    if m[2]:
        f["num_lo"], f["num_hi"] = int(m[2]), int(m[3] or m[2])
    if m[4]:
        f["side"] = int(m[4] == "W")
    rec = rec.strip().lower()
    if rec in ("", "*"):
        pass
    elif rec == "kk":
        f["wins_min"] = 8
    elif rec == "mk":
        f["wins_max"] = 7
    elif rec.endswith("+"):
        f["wins_min"] = int(rec[:-1])
    else:
        parts = [int(x) for x in rec.split("-")]
        if len(parts) not in (2, 3):
            raise ValueError(f"bad record {rec!r}; expected like 9-6, 5-7-3, 8+, kk, mk")
        f["wins"], f["losses"] = parts[0], parts[1]
        f["absences"] = parts[2] if len(parts) == 3 else 0
    return f


def select(t, spec, where=None):
    f = parse_spec(spec) if isinstance(spec, str) else spec
    m = t["rank_class"] == f["rank_class"]
    if "num_lo" in f:
        m &= t["rank_number"].between(f["num_lo"], f["num_hi"])
    for k in ("side", "wins", "losses", "absences"):
        if k in f:
            m &= t[k] == f[k]
    if "wins_min" in f:
        m &= t["wins"] >= f["wins_min"]
    if "wins_max" in f:
        m &= t["wins"] <= f["wins_max"]
    if where:
        m &= t.eval(where).astype(bool)
    return m


def landing(t, spec, where=None):
    """Where rows matching spec landed, per window: n, class counts, delta
    quantiles (half-ranks, + = down), most common landing cells."""
    rows = t[select(t, spec, where)]
    out = {}
    for w, since in windows(t):
        r = rows[rows["basho"] >= since]
        if not len(r):
            out[w] = {"n": 0}
            continue
        cls = r["class_next"].astype(int).map(lambda c: CLS_CHARS[c]).value_counts()
        cells = (r["class_next"].astype(int).astype(str) + "/" + r["number_next"].astype(int).astype(str)
                 + "/" + r["side_next"].astype(int).astype(str)).value_counts().head(5)
        out[w] = {
            "n": len(r),
            "classes": ", ".join(f"{k} {v}" for k, v in cls.items()),
            "delta": tuple(np.nanpercentile(r["delta"], [25, 50, 75]).round(1)),
            "cells": ", ".join(f"{fmt_cell(*map(int, k.split('/')))} {v}" for k, v in cells.items()),
        }
    return out


def pairs(t, spec_a, spec_b, where=None):
    """Same-basho (a-row, b-row) pairs of two different rikishi; both sides'
    identity, cell, record and landing, with column suffixes _a and _b."""
    cols = ["basho", "rikishi_id", "shikona", "rank_class", "rank_number", "side", "wins", "losses",
            "absences", "class_next", "number_next", "side_next", "position_next"]
    a = t[select(t, spec_a, where)][cols]
    b = t[select(t, spec_b, where)][cols]
    p = a.merge(b, on="basho", suffixes=("_a", "_b"))
    return p[p["rikishi_id_a"] != p["rikishi_id_b"]]


def pair_rate(t, spec_a, spec_b, where=None):
    """Same-basho pairs (a-row, b-row): share where a landed above b, per window."""
    p = pairs(t, spec_a, spec_b, where)
    out = {}
    for w, since in windows(t):
        q = p[p["basho"] >= since]
        above = int((q["position_next_a"] < q["position_next_b"]).sum())
        out[w] = {"n": len(q), "a_above": above,
                  "rate": above / len(q) if len(q) else np.nan}
    return out


def fmt_landing(spec, res):
    lines = [f"landing of {spec}:"]
    for w, r in res.items():
        if not r["n"]:
            lines.append(f"  {w:7s} n=0")
            continue
        q = r["delta"]
        lines.append(f"  {w:7s} n={r['n']:<4d} classes: {r['classes']}; delta q25/50/75 "
                     f"{q[0]:+.1f}/{q[1]:+.1f}/{q[2]:+.1f}; cells: {r['cells']}")
    return "\n".join(lines)


def fmt_pair(spec_a, spec_b, res):
    lines = [f"{spec_a} above {spec_b}:"]
    for w, r in res.items():
        rate = f"{r['rate']:.2f}" if r["n"] else "-"
        lines.append(f"  {w:7s} {r['a_above']} of {r['n']} ({rate})")
    return "\n".join(lines)


def fmt_rows(rows, limit=0):
    """The transitions behind landing()'s counts, newest basho first and in
    sheet order within one: basho, man, cell and record, landing cell, delta."""
    rows = rows.sort_values(["basho", "position"], ascending=[False, True])
    return _listing("rows", rows, limit, lambda r: f"{r.basho} {_side(r)} {r.delta:+.0f}")


def fmt_pairs(p, limit=0):
    """The pairs behind pair_rate()'s counts, newest basho first: both men as
    in fmt_rows, and whether a landed above or below b."""
    p = p.sort_values(["basho", "position_next_a", "position_next_b"], ascending=[False, True, True])
    return _listing("pairs", p, limit, lambda r: (
        f"{r.basho} {_side(r, '_a')} {'above' if r.position_next_a < r.position_next_b else 'below'}  "
        f"{_side(r, '_b')}"))


def _listing(noun, frame, limit, line):
    """Header, one `line(row)` for each of the first `limit` rows (0 = all), how many were cut."""
    shown = frame.head(limit) if limit else frame
    lines = [f"  {noun}, newest first:", *(f"    {line(r)}".rstrip() for r in shown.itertuples())]
    if len(shown) < len(frame):
        lines.append(f"    ... {len(frame) - len(shown)} more (--limit 0 for all)")
    return "\n".join(lines)


def _side(r, suffix=""):
    """One man of a listing row: shikona, cell and record, then his landing cell."""
    def g(col):
        return getattr(r, col + suffix)

    return (f"{g('shikona'):<14} {fmt_cell(g('rank_class'), g('rank_number'), g('side')):<5} "
            f"{fmt_record(g('wins'), g('losses'), g('absences')):<7} -> "
            f"{fmt_cell(g('class_next'), g('number_next'), g('side_next')):<5}")
