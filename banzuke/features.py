"""History features and next-basho targets for each rikishi-basho row.

All features at basho N use only information available at N (results of N
included, since the banzuke for N+1 is made after N ends). Lag features
are NaN when the rikishi was not on the makuuchi/juryo banzuke of the
immediately preceding basho.
"""
import numpy as np
import pandas as pd

from banzuke.build import OZEKI, SEKIWAKE, KOMUSUBI, MAEGASHIRA

# model input columns (all numeric; NaN allowed, tree models handle natively)
FEATURES = [
    "rank_class", "rank_number", "side", "position", "division",
    "wins", "losses", "absences", "fusen_l", "kk", "win8",
    "yusho", "junyusho", "sansho", "sansho_career",
    "w1", "w2", "pos1", "roll3", "traj3",
    "kk_streak", "sanyaku_tenure", "career_high", "n_basho",
    "kadoban", "demoted_ozeki", "ozeki_run3", "sanyaku3",
    "yusho1", "junyusho1",
    "year", "kosho", "mak_size", "jur_size", "boundary_dist",
]

# computed into the dataset but excluded from model inputs: joi/schedule
# awareness tested as a null result, see docs/EXPERIMENTS.md E8
SCHEDULE_FEATURES = ["opp_pos_mean", "n_joi_opp", "wins_vs_joi", "kinboshi"]

JOI = 16  # top-of-banzuke group that shares the toughest schedule


def schedule_features(tidy: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    """Realized strength-of-schedule per rikishi-basho, from actual bouts.
    The joi boundary is dynamic (absences pull lower maegashira up), so the
    fought schedule carries information the banzuke position does not."""
    a = bouts.rename(columns={"winner": "rikishi_id", "loser": "opp"}).assign(win=1)
    b = bouts.rename(columns={"loser": "rikishi_id", "winner": "opp"}).assign(win=0)
    ab = pd.concat([a, b], ignore_index=True)
    opp = tidy[["basho", "rikishi_id", "position", "rank_class"]].rename(columns={
        "rikishi_id": "opp", "position": "opp_pos", "rank_class": "opp_class"})
    ab = ab.merge(opp, on=["basho", "opp"])  # drops sub-juryo opponents
    ab["joi_opp"] = ab["opp_pos"] < JOI
    ab["win_joi"] = ab["win"] * ab["joi_opp"]
    ab["beat_yokozuna"] = ab["win"] * (ab["opp_class"] == 0)
    return ab.groupby(["basho", "rikishi_id"]).agg(
        opp_pos_mean=("opp_pos", "mean"),
        n_joi_opp=("joi_opp", "sum"),
        wins_vs_joi=("win_joi", "sum"),
        beat_yokozuna=("beat_yokozuna", "sum"),
    ).reset_index()


def build_transitions(tidy: pd.DataFrame, bouts: pd.DataFrame) -> pd.DataFrame:
    bashos = sorted(tidy["basho"].unique())
    bidx_map = {b: i for i, b in enumerate(bashos)}
    next_map = {b: n for b, n in zip(bashos, bashos[1:])}

    df = tidy.merge(schedule_features(tidy, bouts), on=["basho", "rikishi_id"], how="left")
    df[["n_joi_opp", "wins_vs_joi", "beat_yokozuna"]] = (
        df[["n_joi_opp", "wins_vs_joi", "beat_yokozuna"]].fillna(0)
    )
    df = df.sort_values(["rikishi_id", "basho"], kind="stable").reset_index(drop=True)
    df["bidx"] = df["basho"].map(bidx_map)
    df["kk"] = (df["wins"] >= 8).astype(int)
    df["win8"] = df["wins"] - 8

    g = df.groupby("rikishi_id")
    contig1 = g["bidx"].diff(1) == 1
    contig2 = g["bidx"].diff(2) == 2

    def lag(col, n, contig):
        return g[col].shift(n).where(contig)

    df["w1"] = lag("wins", 1, contig1)
    df["w2"] = lag("wins", 2, contig2)
    df["pos1"] = lag("position", 1, contig1)
    df["yusho1"] = lag("yusho", 1, contig1)
    df["junyusho1"] = lag("junyusho", 1, contig1)
    df["class1"] = lag("rank_class", 1, contig1)
    df["class2"] = lag("rank_class", 2, contig2)
    # lagged rank numbers: resolver inputs (Y/O promotion rules), not features
    df["num1"] = lag("rank_number", 1, contig1)
    df["num2"] = lag("rank_number", 2, contig2)
    df["roll3"] = df["wins"] + df["w1"] + df["w2"]
    df["traj3"] = lag("position", 2, contig2) - df["position"]

    df["demoted_ozeki"] = ((df["class1"] == OZEKI) & (df["rank_class"] == SEKIWAKE)).astype(int)
    sanyaku = df["rank_class"] <= KOMUSUBI
    df["sanyaku3"] = (
        sanyaku.astype(int)
        + (df["class1"] <= KOMUSUBI).fillna(False).astype(int)
        + (df["class2"] <= KOMUSUBI).fillna(False).astype(int)
    )
    sk_now = df["rank_class"].isin([SEKIWAKE, KOMUSUBI])
    sk1 = df["class1"].isin([SEKIWAKE, KOMUSUBI])
    sk2 = df["class2"].isin([SEKIWAKE, KOMUSUBI])
    df["ozeki_run3"] = df["roll3"].where(sk_now & sk1 & sk2)

    # single pass per-rikishi state machine for the iterative features
    kk_streak = np.zeros(len(df), dtype=int)
    tenure = np.zeros(len(df), dtype=int)
    kadoban = np.zeros(len(df), dtype=int)
    career_high = np.zeros(len(df), dtype=float)
    n_basho = np.zeros(len(df), dtype=int)
    sansho_career = np.zeros(len(df), dtype=int)

    rid = df["rikishi_id"].to_numpy()
    bidx = df["bidx"].to_numpy()
    kk_arr = df["kk"].to_numpy()
    cls = df["rank_class"].to_numpy()
    rnum = df["rank_number"].to_numpy()
    sansho = df["sansho"].to_numpy()

    prev_i = None
    for i in range(len(df)):
        new_rikishi = prev_i is None or rid[i] != rid[prev_i]
        contig = not new_rikishi and bidx[i] - bidx[prev_i] == 1
        rank_key = cls[i] * 100 + rnum[i]  # lower = better
        if new_rikishi:
            career_high[i] = rank_key
            n_basho[i] = 1
            sansho_career[i] = sansho[i]
        else:
            career_high[i] = min(career_high[prev_i], rank_key)
            n_basho[i] = n_basho[prev_i] + 1
            sansho_career[i] = sansho_career[prev_i] + sansho[i]
        kk_streak[i] = kk_arr[i] * ((kk_streak[prev_i] if contig else 0) + 1)
        tenure[i] = (cls[i] <= KOMUSUBI) * ((tenure[prev_i] if contig else 0) + 1)
        # ozeki who MK'd last basho while not kadoban is now kadoban
        kadoban[i] = int(
            cls[i] == OZEKI
            and contig
            and cls[prev_i] == OZEKI
            and kk_arr[prev_i] == 0
            and kadoban[prev_i] == 0
        )
        prev_i = i

    df["kk_streak"] = kk_streak
    df["sanyaku_tenure"] = tenure
    df["kadoban"] = kadoban
    df["career_high"] = career_high
    df["n_basho"] = n_basho
    df["sansho_career"] = sansho_career

    df["year"] = df["basho"] // 100
    df["kosho"] = ((df["basho"] >= 197201) & (df["basho"] <= 200311)).astype(int)
    # signed distance to the makuuchi/juryo boundary (negative = inside makuuchi)
    df["boundary_dist"] = df["position"] - df["mak_size"]
    # kinboshi: a maegashira defeating a yokozuna, explicitly rewarded
    df["kinboshi"] = df["beat_yokozuna"] * (df["rank_class"] == MAEGASHIRA)

    # targets: the same rikishi's row at the next calendar basho
    df["next_basho"] = df["basho"].map(next_map)
    nxt = df[["basho", "rikishi_id", "position", "division",
              "rank_class", "rank_number", "side"]].rename(columns={
        "basho": "next_basho", "position": "position_next", "division": "division_next",
        "rank_class": "class_next", "rank_number": "number_next", "side": "side_next",
    })
    df = df.merge(nxt, on=["next_basho", "rikishi_id"], how="left")
    df["dropped"] = df["next_basho"].notna() & df["position_next"].isna()
    df["delta"] = df["position_next"] - df["position"]

    return df.drop(columns=["bidx", "beat_yokozuna"])
