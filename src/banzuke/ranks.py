"""The banzuke vocabulary every module shares: rank classes as ordinals,
their letters and names, cell and record formatting, and the basho calendar.

Classes are ordered Y < O < S < K < M < J (lower = higher rank); Y/O/S/K are
sanyaku. A cell is (class, number, side) with side 0 = East, 1 = West,
written like `M3W`; a record is W-L or W-L-A.
"""

YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO = range(6)
CLS_CHARS = "YOSKMJ"
CLS_NAMES = ("yokozuna", "ozeki", "sekiwake", "komusubi", "maegashira", "juryo")
# the API's division/rank names -> ordinals
CLASS_ORD = {"Yokozuna": YOKOZUNA, "Ozeki": OZEKI, "Sekiwake": SEKIWAKE, "Komusubi": KOMUSUBI,
             "Maegashira": MAEGASHIRA, "Juryo": JURYO}

BASHO_MONTHS = (1, 3, 5, 7, 9, 11)  # six honbasho a year; a basho id is YYYYMM

# the eras the evaluation windows are cut at (docs/EXPERIMENTS.md protocol v2):
# 2004+ is the 42-man, post-kosho committee; 2019+/2020+ the recent and confirm windows
MODERN_ERA, RECENT_ERA, CONFIRM_ERA = 200401, 201901, 202001

# E and W of one rank number with identical records are "twins"; the committee
# treats such a pair as a unit (resolver, reranker, convention audit)
TWIN_KEY = ("rank_class", "rank_number", "wins", "losses", "absences")


def next_basho(basho: int) -> int:
    """The basho that follows `basho` on the calendar (202611 -> 202701)."""
    year, month = divmod(int(basho), 100)
    nxt = BASHO_MONTHS[(BASHO_MONTHS.index(month) + 1) % 6]
    return (year + (nxt == 1)) * 100 + nxt


def in_window(basho, since=None, until=None):
    """basho (int or Series) inside since..until; None leaves an end open."""
    return (basho >= (since or 0)) & (basho <= (until or 10**8))


def fmt_rank(c, num) -> str:
    """(4, 3) -> 'M3'."""
    return f"{CLS_CHARS[int(c)]}{int(num)}"


def fmt_cell(c, num, side) -> str:
    """(4, 3, 1) -> 'M3W'."""
    return f"{fmt_rank(c, num)}{'EW'[int(side)]}"


def fmt_record(wins, losses, absences=0) -> str:
    """9-6, or 5-7-3 when there were absences."""
    return f"{int(wins)}-{int(losses)}" + (f"-{int(absences)}" if absences else "")
