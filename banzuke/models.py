"""Ordering models. Shared interface:

    Model(seed=0, n_seeds=1, base=None, pair=None, **options)
    fit(train)    train: transition rows with position_next targets
    score(cands)  -> np.ndarray, lower = ranked higher

All models receive the same FEATURES; they differ in objective, which is
the experiment axis (see docs/EXPERIMENTS.md). Hyperparameters live in
BASE_PARAMS (regressors/ranker) and PAIR_PARAMS (pair classifiers). The
constructor merges per-instance overrides, so backtests and predict.py
share one configuration path and nothing mutates module state.

Seeds: `seed=k` with `n_seeds=1` sets LightGBM's random_state=k. With
`n_seeds=m > 1` the instance is a bag of m models on seeds m*k .. m*k+m-1
(so bag replicates k=0,1,... are disjoint) whose predictions are averaged.
`seed=None` omits random_state entirely, reproducing LightGBM's library
defaults (the pre-2026-09 configuration); it cannot be bagged.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from banzuke.build import KOMUSUBI, MAEGASHIRA
from banzuke.features import FEATURES

BASE_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=1, verbose=-1,
)
PAIR_PARAMS = {**BASE_PARAMS, "n_estimators": 400}


def _seeds(seed, n_seeds):
    if seed is None:
        if n_seeds != 1:
            raise ValueError("seed=None (library defaults) cannot be bagged")
        return [None]
    return list(range(seed * n_seeds, (seed + 1) * n_seeds))


def _seeded(params, seed):
    p = dict(params)
    if seed is None:
        p.pop("random_state", None)
    else:
        p["random_state"] = seed
    return p


class _Model:
    """Shared constructor: seeds plus per-stage parameter overrides and
    model-specific options (declared per class in OPTIONS with defaults)."""

    OPTIONS: dict = {}

    def __init__(self, seed=0, n_seeds=1, base=None, pair=None, **options):
        self.seeds = _seeds(seed, n_seeds)
        self.base_params = {**BASE_PARAMS, **(base or {})}
        self.pair_params = {**PAIR_PARAMS, **(pair or {})}
        unknown = set(options) - set(self.OPTIONS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown options {sorted(unknown)}; "
                            f"available: {sorted(self.OPTIONS)}")
        for k, v in self.OPTIONS.items():
            setattr(self, k, options.get(k, v))

    def score(self, cands, k=None):
        """k: index into self.seeds to score with one bag member only."""
        raise NotImplementedError


class RulesBaseline(_Model):
    """Fitted movement formula: delta ~ a + b*(wins-8) + c*absences per rank
    zone, coefficients from the most recent `recent` basho of training data."""

    name = "R"
    OPTIONS = {"recent": 60}

    @staticmethod
    def _zone(df):
        cls, num = df["rank_class"], df["rank_number"]
        return np.select(
            [cls <= KOMUSUBI, (cls == MAEGASHIRA) & (num <= 5),
             (cls == MAEGASHIRA) & (num <= 11), cls == MAEGASHIRA,
             num <= 5],
            [0, 1, 2, 3, 4], default=5,
        )

    def fit(self, train):
        recent = sorted(train["basho"].unique())[-self.recent:]
        t = train[train["basho"].isin(recent)]
        zones = self._zone(t)
        self.coefs = {}
        for z in range(6):
            rows = t[zones == z]
            X = np.column_stack([np.ones(len(rows)), rows["win8"], rows["absences"]])
            self.coefs[z], *_ = np.linalg.lstsq(X, rows["delta"], rcond=None)

    def score(self, cands, k=None):
        zones = self._zone(cands)
        X = np.column_stack([np.ones(len(cands)), cands["win8"], cands["absences"]])
        delta = np.array([X[i] @ self.coefs[z] for i, z in enumerate(zones)])
        return cands["position"].to_numpy() + delta


class LinearModel(_Model):
    name = "L"
    OPTIONS = {"alpha": 1.0}

    def fit(self, train):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha)
        )
        self.pipe.fit(train[FEATURES], train["delta"])

    def score(self, cands, k=None):
        return cands["position"].to_numpy() + self.pipe.predict(cands[FEATURES])


class GBMRegression(_Model):
    """A: gradient-boosted regression on movement delta (L2).
    half_life: optional exponential recency weight (in basho) on training rows."""

    name = "A"
    objective = "regression"
    OPTIONS = {"half_life": None}

    def _weights(self, train):
        if not self.half_life:
            return None
        order = {b: i for i, b in enumerate(sorted(train["basho"].unique()))}
        age = train["basho"].map(order).max() - train["basho"].map(order)
        return (0.5 ** (age / self.half_life)).to_numpy()

    def fit(self, train):
        X, y, w = train[FEATURES], train["delta"], self._weights(train)
        self.ms = [
            LGBMRegressor(objective=self.objective, **_seeded(self.base_params, s))
            .fit(X, y, sample_weight=w)
            for s in self.seeds
        ]

    def delta(self, cands, k=None):
        ms = self.ms if k is None else [self.ms[k]]
        return np.mean([m.predict(cands[FEATURES]) for m in ms], axis=0)

    def score(self, cands, k=None):
        return cands["position"].to_numpy() + self.delta(cands, k)


class GBMMedian(GBMRegression):
    """Aq: L1 objective (median regression), less shrinkage on big movers."""

    name = "Aq"
    objective = "regression_l1"


class GBMRecency(GBMRegression):
    """Aw: L2 GBM with exponential recency weights (half-life 60 basho, ~10y),
    biasing training toward the modern committee's behavior."""

    name = "Aw"
    OPTIONS = {"half_life": 60}


class GBMRanker(_Model):
    """B: LambdaRank on the next-basho order within each transition.
    top: relevance levels (focuses ranking quality on the top of the order)."""

    name = "B"
    OPTIONS = {"top": 60}

    def fit(self, train):
        t = train.sort_values(["basho", "position"], kind="stable")
        label = np.clip(self.top - t["position_next"], 0, self.top).astype(int)
        groups = t.groupby("basho", sort=True).size().to_numpy()
        self.ms = [
            LGBMRanker(objective="lambdarank", label_gain=list(range(self.top + 1)),
                       **_seeded(self.base_params, s)).fit(t[FEATURES], label, group=groups)
            for s in self.seeds
        ]

    def score(self, cands, k=None):
        ms = self.ms if k is None else [self.ms[k]]
        return -np.mean([m.predict(cands[FEATURES]) for m in ms], axis=0)


# --- pairwise stage -------------------------------------------------------

_H2H: set | None = None  # {(basho, winner_id, loser_id)}


def _h2h_wins():
    global _H2H
    if _H2H is None:
        from banzuke.build import PROCESSED

        b = pd.read_parquet(PROCESSED / "bouts.parquet")
        _H2H = set(zip(b["basho"], b["winner"], b["loser"]))
    return _H2H


def _window_pairs(n, window):
    """Index pairs (i, j) with i < j <= i + window over n rows in current order."""
    j_grid = np.arange(n)[:, None] + np.arange(1, window + 1)[None, :]
    mask = j_grid < n
    i_arr = np.broadcast_to(np.arange(n)[:, None], j_grid.shape)[mask]
    return i_arr, j_grid[mask]


def _pair_matrix(df, i_arr, j_arr, h2h=False):
    """Feature rows for index pairs of one basho's rows, oriented i = currently
    higher ranked: feature differences, optionally the pair's head-to-head
    bout this basho (+1 i won, -1 i lost, 0)."""
    X = df[FEATURES].to_numpy(dtype=float)
    out = X[i_arr] - X[j_arr]
    if h2h:
        wins = _h2h_wins()
        rid = df["rikishi_id"].to_numpy()
        basho = int(df["basho"].iloc[0])
        col = np.array(
            [(basho, rid[i], rid[j]) in wins for i, j in zip(i_arr, j_arr)], dtype=float
        ) - np.array(
            [(basho, rid[j], rid[i]) in wins for i, j in zip(i_arr, j_arr)], dtype=float
        )
        out = np.column_stack([out, col])
    return out


def _fit_pair_classifiers(train, seeds, params, window, h2h=False):
    """One classifier per seed on all pairs within `window` of each other in
    the current order of every training basho; label = i stays above j."""
    Xs, ys = [], []
    for _, grp in train.groupby("basho"):
        df = grp.sort_values("position", kind="stable")
        i_arr, j_arr = _window_pairs(len(df), window)
        nxt = df["position_next"].to_numpy()
        Xs.append(_pair_matrix(df, i_arr, j_arr, h2h))
        ys.append((nxt[i_arr] < nxt[j_arr]).astype(int))
    X, y = np.vstack(Xs), np.concatenate(ys)
    return [LGBMClassifier(**_seeded(params, s)).fit(X, y) for s in seeds]


def _pair_proba(models, X, k=None):
    ms = models if k is None else [models[k]]
    return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)


class PairwiseBT(_Model):
    """C: classifier on nearby candidate pairs (feature diffs), aggregated
    into local movement from the current order (approximate Bradley-Terry)."""

    name = "C"
    OPTIONS = {"pair_window": 12, "scale": 2.0, "h2h": False}

    def fit(self, train):
        self.pair = _fit_pair_classifiers(train, self.seeds, self.pair_params,
                                          self.pair_window, self.h2h)

    def score(self, cands, k=None):
        df = cands.sort_values("position", kind="stable")
        i_arr, j_arr = _window_pairs(len(df), self.pair_window)
        p = _pair_proba(self.pair, _pair_matrix(df, i_arr, j_arr, self.h2h), k)
        movement = np.zeros(len(df))
        np.add.at(movement, i_arr, p - 0.5)
        np.add.at(movement, j_arr, 0.5 - p)
        score = df["position"].to_numpy() - self.scale * movement
        return pd.Series(score, index=df.index).reindex(cands.index).to_numpy()


class GBMRerank(GBMMedian):
    """Ar: Aq's global order + learned local reranking of near-tie clusters.
    Residual analysis showed 28% of misses are E/W flips and 34% are within
    one position; the committee resolves these by wins, then prior order.
    A pairwise classifier learns that resolution; it may only reorder within
    clusters of nearly-equal base scores, so global placement stays Aq's.

    pair_window: training pairs within this current-position distance
    gap: cluster break when consecutive base scores differ by more
    cluster_max: largest cluster the reranker may reorder
    h2h: include the pair's head-to-head bout this basho as a pair feature
    """

    name = "Ar"
    OPTIONS = {**GBMMedian.OPTIONS, "pair_window": 6, "gap": 0.5, "cluster_max": 4,
               "h2h": False}

    def fit(self, train):
        super().fit(train)
        self.pair = _fit_pair_classifiers(train, self.seeds, self.pair_params,
                                          self.pair_window, self.h2h)

    def _clusters(self, base, pos):
        order = np.lexsort((pos, base))
        clusters, cur = [], [order[0]]
        for prev, i in zip(order, order[1:]):
            if base[i] - base[prev] <= self.gap and len(cur) < self.cluster_max:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        return clusters

    def score(self, cands, k=None):
        base = super().score(cands, k)
        pos = cands["position"].to_numpy()
        clusters = self._clusters(base, pos)
        # every within-cluster pair, oriented by current position to match training
        pairs = [(i, j) if pos[i] <= pos[j] else (j, i)
                 for cl in clusters for i, j in combinations(cl, 2)]
        borda = np.zeros(len(cands))
        if pairs:
            i_arr, j_arr = np.array(pairs).T
            probs = _pair_proba(self.pair, _pair_matrix(cands, i_arr, j_arr, self.h2h), k)
            for i, j, p in zip(i_arr, j_arr, probs):
                borda[i] += p
                borda[j] += 1 - p
        final = []
        for cl in clusters:
            final.extend(sorted(cl, key=lambda i: (-borda[i], base[i])))
        out = np.empty(len(cands))
        out[final] = np.arange(len(final))
        return out


class GBMRerankH2H(GBMRerank):
    """Ah: Ar + the pair's head-to-head bout result as a rerank feature.
    The committee reportedly breaks near-ties by who beat whom."""

    name = "Ah"
    OPTIONS = {**GBMRerank.OPTIONS, "h2h": True}


MODELS = {m.name: m for m in
          (RulesBaseline, LinearModel, GBMRegression, GBMMedian, GBMRecency,
           GBMRanker, PairwiseBT, GBMRerank, GBMRerankH2H)}
