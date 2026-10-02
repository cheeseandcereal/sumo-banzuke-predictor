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


def next_basho(basho: int) -> int:
    """The basho that follows `basho` on the calendar (202611 -> 202701)."""
    year, month = divmod(int(basho), 100)
    nxt = BASHO_MONTHS[(BASHO_MONTHS.index(month) + 1) % 6]
    return (year + (nxt == 1)) * 100 + nxt


def fmt_cell(c, num, side) -> str:
    """(4, 3, 1) -> 'M3W'."""
    return f"{CLS_CHARS[int(c)]}{int(num)}{'EW'[int(side)]}"


def fmt_record(wins, losses, absences=0) -> str:
    """9-6, or 5-7-3 when there were absences."""
    return f"{int(wins)}-{int(losses)}" + (f"-{int(absences)}" if absences else "")
