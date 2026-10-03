"""The human benchmark: per-basho result lists of the "Guess the Banzuke"
game (dichne.com), read from the sumodb archive.

Every GTB entry predicts the banzuke the backtest predicts for the same
target and is scored the way `metrics.evaluate` scores a forecast: 2 points
per bullseye (exact cell), 1 per hit (right rank, wrong side). Pages are
fetched once into cache/gtb/ (git-ignored; nothing from the archive is
committed) and parsed into one row per entry, so a window of basho can be
summarized as the field's percentiles and a model's backtest rows can be
placed inside each basho's field.
"""
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from banzuke.paths import GTB_CACHE

ARCHIVE_URL = "https://sumodb.sumogames.de/gtb/GTBScoreBasho.aspx?b={basho}"
DELAY_S = 1.0
RETRIES = 3
COLUMNS = ["basho", "place", "player", "bullseyes", "hits", "points"]
# archive column -> ours; the archive's optional columns (Rank, Record) are ignored
HEADER = {"Place": "place", "Shikona": "player", "Bulls-Eye": "bullseyes", "Hits": "hits",
          "Total Points": "points", "Correct Guesses": "correct"}


class ArchiveError(RuntimeError):
    """The archive page does not look like a GTB result list."""


def _get(url) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "banzuke-gtb/1.0"})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == RETRIES:
                raise
            wait = 2**attempt
            print(f"  retry {attempt} in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def _cells(tr) -> list:
    out = []
    for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S):
        cell = re.sub(r"<br\s*/?>", " ", cell)
        out.append(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip())
    return out


def parse_page(html: str) -> pd.DataFrame:
    """One archive page -> one row per entry (COLUMNS without `basho`), best
    first. Empty when the results are not posted yet (header-only table)."""
    table = re.search(r'<table class="gtbscore".*?</table>', html, re.S)
    if not table:
        raise ArchiveError("no result table on the page")
    trs = re.findall(r"<tr[^>]*>.*?</tr>", table.group(0), re.S)
    head = _cells(trs[0]) if trs else []
    missing = [h for h in HEADER if h not in head]
    if missing:
        raise ArchiveError(f"result table lacks columns {missing}; header is {head}")
    col = {ours: head.index(theirs) for theirs, ours in HEADER.items()}
    rows = []
    for tr in trs[1:]:
        c = _cells(tr)
        if len(c) != len(head):
            raise ArchiveError(f"row with {len(c)} cells under {len(head)} headers: {c}")
        row = {k: c[i] for k, i in col.items()}
        row["player"] = row["player"].rstrip(" *")  # '*' marks a debut
        for k in ("place", "bullseyes", "hits", "points", "correct"):
            row[k] = int(row[k])
        if row["points"] != 2 * row["bullseyes"] + row["hits"] \
                or row["correct"] != row["bullseyes"] + row["hits"]:
            raise ArchiveError(f"scoring does not add up: {c}")
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS[1:])


def fetch_results(basho, fresh=False) -> pd.DataFrame:
    """The entries of one target basho, from cache/gtb/{basho}.html or the
    archive. A page without entries (results not posted yet) is not cached,
    so the next run asks again."""
    path = GTB_CACHE / f"{basho}.html"
    if path.exists() and not fresh:
        entries = parse_page(path.read_text(encoding="utf-8"))
    else:
        html = _get(ARCHIVE_URL.format(basho=basho))
        entries = parse_page(html)
        if len(entries):
            GTB_CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        time.sleep(DELAY_S)
    return entries.assign(basho=int(basho))[COLUMNS]


def load_results(targets, fresh=False, progress=True) -> pd.DataFrame:
    """Entries of every target basho with posted results, one frame; names
    the targets without any on stderr."""
    frames, empty = [], []
    for basho in targets:
        cached = (GTB_CACHE / f"{basho}.html").exists() and not fresh
        entries = fetch_results(basho, fresh)
        if not len(entries):
            empty.append(basho)
        elif progress and not cached:
            print(f"{basho}: {len(entries)} GTB entries", file=sys.stderr)
        frames.append(entries)
    if empty:
        print(f"no GTB results posted for: {', '.join(map(str, empty))}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def field(entries, percentiles=(50, 75, 100)) -> pd.DataFrame:
    """Per basho: the number of entries, the percentiles of points and of
    bullseyes (pts_p50 ..., be_p50 ...; linear interpolation, p100 = the
    winner's score) and the winner's name."""
    rows = []
    for basho, g in entries.groupby("basho"):
        row = {"basho": basho, "entries": len(g)}
        row.update({f"pts_p{q:g}": np.percentile(g["points"], q) for q in percentiles})
        row.update({f"be_p{q:g}": np.percentile(g["bullseyes"], q) for q in percentiles})
        row["winner"] = g.sort_values(["points", "place"], ascending=[False, True])["player"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("basho")


def place_in_field(entries, results, label) -> pd.DataFrame:
    """Where a model's backtest rows land in each basho's field. Seeds are
    averaged within a basho first; then per basho: the model's points and
    exact cells, `place` (1 + entries scoring more; ties share the better
    place), `pct` (share of the field scoring less, ties counted half) and
    `behind`, the winner's points minus the model's."""
    per = results.groupby("basho")[["gtb_points", "exact_n"]].mean()
    rows = []
    for basho, g in entries.groupby("basho"):
        if basho not in per.index:
            continue
        pts = per.at[basho, "gtb_points"]
        above, below = (g["points"] > pts).sum(), (g["points"] < pts).sum()
        rows.append({"basho": basho, f"{label}_pts": pts, f"{label}_be": per.at[basho, "exact_n"],
                     "place": 1 + above, "pct": (below + 0.5 * (len(g) - above - below)) / len(g),
                     "behind": g["points"].max() - pts})
    return pd.DataFrame(rows, columns=["basho", f"{label}_pts", f"{label}_be", "place", "pct",
                                       "behind"]).set_index("basho")
