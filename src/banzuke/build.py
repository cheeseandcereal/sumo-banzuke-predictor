"""Parse raw API JSON into tidy parquet datasets.

Outputs to data/processed/:
- tidy.parquet: one row per rikishi per basho (rank, results, prizes)
- bouts.parquet: competitive head-to-head wins (fusen excluded)
- transitions.parquet: tidy + history features + next-basho targets

Normally run via `banzuke data update` (fetch, then build) or
`banzuke data build` (rebuild from the committed raw JSON).
"""
import json

import pandas as pd

from banzuke.paths import PROCESSED, RAW_BANZUKE, RAW_BASHO

# Ordinal rank classes: lower = higher rank. Y/O/S/K = sanyaku-and-above.
CLASS_ORD = {"Yokozuna": 0, "Ozeki": 1, "Sekiwake": 2, "Komusubi": 3, "Maegashira": 4, "Juryo": 5}
YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO = range(6)

# Raw records where a scheduled bout has `result: ""`, so the API's own
# wins/losses are short by one. (basho, rikishiID) -> {bout index: result}.
# 202507 day 15: the torikumi (/api/basho/202507/torikumi/Juryo/15) has
# Nishikigi over Fujiseiun by kotenage; a 2026-10 re-fetch of the banzuke
# still returns the blank. Nishikigi J1E 7-7 -> 8-7, Fujiseiun J8W 9-5 -> 9-6.
CORRECTIONS = {
    (202507, 16): {14: "win"},
    (202507, 82): {14: "loss"},
}


def _correct(basho: int, r: dict, record: list) -> None:
    """Apply CORRECTIONS in place, then require every scheduled bout (an
    opponent is set) to carry a result; the only blank-with-opponent entries
    in the raw data are the corrected ones."""
    for i, result in CORRECTIONS.get((basho, r["rikishiID"]), {}).items():
        if record[i]["result"] != "":
            raise ValueError(f"{basho} {r['shikonaEn']} bout {i} is not blank; drop the correction")
        record[i]["result"] = result
        key = "wins" if result == "win" else "losses"
        r[key] = r.get(key, 0) + 1
    blank = [i for i, b in enumerate(record) if b["result"] == "" and b.get("opponentID")]
    if blank:
        raise ValueError(f"{basho} {r['shikonaEn']}: scheduled bout(s) {blank} without a result; "
                         "check the torikumi and add to CORRECTIONS")


def load_tidy() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bouts = [], []
    for path in sorted(RAW_BANZUKE.glob("*.json")):
        d = json.loads(path.read_text())
        basho = int(d["bashoId"])
        division = 0 if d["division"] == "Makuuchi" else 1
        for side_key in ("east", "west"):
            for r in d[side_key] or []:
                cls, num, side = r["rank"].split()
                record = r.get("record") or []
                _correct(basho, r, record)
                # competitive head-to-head wins (fusen excluded)
                bouts.extend(
                    {"basho": basho, "winner": r["rikishiID"], "loser": b["opponentID"]}
                    for b in record if b["result"] == "win"
                )
                rows.append({
                    "basho": basho,
                    "rikishi_id": r["rikishiID"],
                    "shikona": r["shikonaEn"],
                    "division": division,
                    "rank_class": CLASS_ORD[cls],
                    "rank_number": int(num),
                    "side": 0 if side == "East" else 1,
                    "wins": r.get("wins", 0),
                    "losses": r.get("losses", 0),
                    "absences": r.get("absences", 0),
                    "fusen_l": sum(1 for b in record if b["result"] == "fusen loss"),
                })
    df = pd.DataFrame(rows)

    # joint position within each basho: 0 = Y1E, sorted through juryo
    df = df.sort_values(
        ["basho", "rank_class", "rank_number", "side", "rikishi_id"], kind="stable"
    ).reset_index(drop=True)
    df["position"] = df.groupby("basho").cumcount()

    # yusho / special prizes from basho metadata
    yusho, prizes = set(), []
    for path in sorted(RAW_BASHO.glob("*.json")):
        d = json.loads(path.read_text())
        b = int(d["date"])
        for y in d.get("yusho") or []:
            if y["type"] in ("Makuuchi", "Juryo"):
                yusho.add((b, y["rikishiId"]))
        for p in d.get("specialPrizes") or []:
            prizes.append((b, p["rikishiId"]))
    key = list(zip(df["basho"], df["rikishi_id"]))
    df["yusho"] = [k in yusho for k in key]
    n_prizes = pd.Series(prizes).value_counts() if prizes else pd.Series(dtype=int)
    df["sansho"] = [n_prizes.get(k, 0) for k in key]

    # jun-yusho: best score among non-winners in the division
    wins_rest = df["wins"].where(~df["yusho"])
    best_rest = wins_rest.groupby([df["basho"], df["division"]]).transform("max")
    df["junyusho"] = (~df["yusho"]) & (df["wins"] == best_rest) & (df["wins"] > 0)

    # division sizes
    sizes = df.pivot_table(index="basho", columns="division", values="rikishi_id", aggfunc="count")
    df["mak_size"] = df["basho"].map(sizes[0])
    df["jur_size"] = df["basho"].map(sizes[1])

    for col in ("yusho", "junyusho"):
        df[col] = df[col].astype(int)
    return df, pd.DataFrame(bouts).drop_duplicates()


def main():
    """Rebuild every processed dataset from the raw JSON under data/."""
    from banzuke.features import build_transitions

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tidy, bouts = load_tidy()
    tidy.to_parquet(PROCESSED / "tidy.parquet", index=False)
    bouts.to_parquet(PROCESSED / "bouts.parquet", index=False)
    print(f"tidy: {len(tidy)} rows, {tidy['basho'].nunique()} basho; {len(bouts)} bouts")

    trans = build_transitions(tidy, bouts)
    trans.to_parquet(PROCESSED / "transitions.parquet", index=False)
    n_target = trans["position_next"].notna().sum()
    print(f"transitions: {len(trans)} rows, {n_target} with targets")
