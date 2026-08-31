"""Parse raw API JSON into tidy parquet datasets.

Outputs to data/processed/:
- tidy.parquet: one row per rikishi per basho (rank, results, prizes)
- transitions.parquet: tidy + history features + next-basho targets

Normally run via: uv run python update_data.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_BANZUKE = ROOT / "data" / "banzuke"
RAW_BASHO = ROOT / "data" / "basho"
PROCESSED = ROOT / "data" / "processed"

# Ordinal rank classes: lower = higher rank. Y/O/S/K = sanyaku-and-above.
CLASS_ORD = {"Yokozuna": 0, "Ozeki": 1, "Sekiwake": 2, "Komusubi": 3, "Maegashira": 4, "Juryo": 5}
YOKOZUNA, OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO = range(6)


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


if __name__ == "__main__":
    main()
