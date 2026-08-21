"""Ordering models. Shared interface:

    fit(train)    train: transition rows with position_next targets
    score(cands)  -> np.ndarray, lower = ranked higher

All models receive the same FEATURES; they differ in objective, which is
the experiment axis (see docs/EXPERIMENTS.md).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from banzuke.build import KOMUSUBI, MAEGASHIRA
from banzuke.features import FEATURES

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, verbose=-1,
)


class RulesBaseline:
    """Fitted movement formula: delta ~ a + b*(wins-8) + c*absences per rank
    zone, coefficients from the most recent 60 basho of training data."""

    name = "R"
    RECENT = 60

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
        recent = sorted(train["basho"].unique())[-self.RECENT:]
        t = train[train["basho"].isin(recent)]
        zones = self._zone(t)
        self.coefs = {}
        for z in range(6):
            rows = t[zones == z]
            X = np.column_stack([np.ones(len(rows)), rows["win8"], rows["absences"]])
            self.coefs[z], *_ = np.linalg.lstsq(X, rows["delta"], rcond=None)

    def score(self, cands):
        zones = self._zone(cands)
        X = np.column_stack([np.ones(len(cands)), cands["win8"], cands["absences"]])
        delta = np.array([X[i] @ self.coefs[z] for i, z in enumerate(zones)])
        return cands["position"].to_numpy() + delta


class LinearModel:
    name = "L"

    def fit(self, train):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        )
        self.pipe.fit(train[FEATURES], train["delta"])

    def score(self, cands):
        return cands["position"].to_numpy() + self.pipe.predict(cands[FEATURES])


class GBMRegression:
    """A: gradient-boosted regression on movement delta (L2)."""

    name = "A"
    objective = "regression"

    def fit(self, train):
        self.m = LGBMRegressor(objective=self.objective, **LGB_PARAMS)
        self.m.fit(train[FEATURES], train["delta"])

    def score(self, cands):
        return cands["position"].to_numpy() + self.m.predict(cands[FEATURES])


class GBMMedian(GBMRegression):
    """A': L1 objective (median regression), less shrinkage on big movers."""

    name = "Aq"
    objective = "regression_l1"


class GBMRecency(GBMRegression):
    """Aw: L2 GBM with exponential recency weights (half-life 60 basho, ~10y),
    biasing training toward the modern committee's behavior."""

    name = "Aw"
    HALF_LIFE = 60

    def fit(self, train):
        order = {b: i for i, b in enumerate(sorted(train["basho"].unique()))}
        age = train["basho"].map(order).max() - train["basho"].map(order)
        self.m = LGBMRegressor(objective=self.objective, **LGB_PARAMS)
        self.m.fit(train[FEATURES], train["delta"],
                   sample_weight=0.5 ** (age / self.HALF_LIFE))


class GBMRanker:
    """B: LambdaRank on the next-basho order within each transition."""

    name = "B"
    TOP = 60  # relevance levels, focuses ranking quality on the top of the order

    def fit(self, train):
        t = train.sort_values(["basho", "position"], kind="stable")
        label = np.clip(self.TOP - t["position_next"], 0, self.TOP).astype(int)
        groups = t.groupby("basho", sort=True).size().to_numpy()
        self.m = LGBMRanker(
            objective="lambdarank", label_gain=list(range(self.TOP + 1)), **LGB_PARAMS
        )
        self.m.fit(t[FEATURES], label, group=groups)

    def score(self, cands):
        return -self.m.predict(cands[FEATURES])


_PAIR_CACHE: dict[tuple, tuple] = {}  # training pairs per (basho, window) never change


def _pair_diffs(df, window):
    """Feature diffs for pairs within `window` of each other in the current
    order, oriented i = currently higher ranked."""
    df = df.sort_values("position", kind="stable")
    X = df[FEATURES].to_numpy(dtype=float)
    n = len(df)
    j_grid = np.arange(n)[:, None] + np.arange(1, window + 1)[None, :]
    mask = j_grid < n
    i_arr = np.broadcast_to(np.arange(n)[:, None], j_grid.shape)[mask]
    j_arr = j_grid[mask]
    return X[i_arr] - X[j_arr], i_arr, j_arr, df


def _fit_pair_classifier(train, window, n_estimators=400):
    Xs, ys = [], []
    for b, grp in train.groupby("basho"):
        key = (b, window)
        if key not in _PAIR_CACHE:
            X, i_arr, j_arr, df = _pair_diffs(grp, window)
            nxt = df["position_next"].to_numpy()
            _PAIR_CACHE[key] = (X, (nxt[i_arr] < nxt[j_arr]).astype(int))
        Xs.append(_PAIR_CACHE[key][0])
        ys.append(_PAIR_CACHE[key][1])
    m = LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_estimators})
    m.fit(np.vstack(Xs), np.concatenate(ys))
    return m


class PairwiseBT:
    """C: classifier on nearby candidate pairs (feature diffs), aggregated
    into local movement from the current order (approximate Bradley-Terry)."""

    name = "C"
    WINDOW = 12
    SCALE = 2.0

    def fit(self, train):
        self.m = _fit_pair_classifier(train, self.WINDOW)

    def score(self, cands):
        X, i_arr, j_arr, df = _pair_diffs(cands, self.WINDOW)
        p = self.m.predict_proba(X)[:, 1]  # P(i stays above j)
        movement = np.zeros(len(df))
        np.add.at(movement, i_arr, p - 0.5)
        np.add.at(movement, j_arr, 0.5 - p)
        score = df["position"].to_numpy() - self.SCALE * movement
        # map back to cands row order
        out = pd.Series(score, index=df.index).reindex(cands.index)
        return out.to_numpy()


class GBMRerank(GBMMedian):
    """Ar: Aq's global order + learned local reranking of near-tie clusters.
    Residual analysis showed 28% of misses are E/W flips and 34% are within
    one position; the committee resolves these by wins, then prior order.
    A pairwise classifier learns that resolution; it may only reorder within
    clusters of nearly-equal base scores, so global placement stays Aq's."""

    name = "Ar"
    PAIR_WINDOW = 6   # training pairs within this current-position distance
    GAP = 0.5         # cluster break when consecutive base scores differ more
    CLUSTER_MAX = 4

    def fit(self, train):
        super().fit(train)
        self.pair = _fit_pair_classifier(train, self.PAIR_WINDOW)

    def score(self, cands):
        base = super().score(cands)
        order = np.lexsort((cands["position"].to_numpy(), base))
        X = cands[FEATURES].to_numpy(dtype=float)
        pos = cands["position"].to_numpy()

        clusters, cur = [], [order[0]]
        for prev, i in zip(order, order[1:]):
            if base[i] - base[prev] <= self.GAP and len(cur) < self.CLUSTER_MAX:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)

        final = []
        for cl in clusters:
            if len(cl) > 1:
                borda = {i: 0.0 for i in cl}
                pairs = [(i, j) for a, i in enumerate(cl) for j in cl[a + 1:]]
                # orient by current position to match training
                oriented = [(i, j) if pos[i] <= pos[j] else (j, i) for i, j in pairs]
                probs = self.pair.predict_proba(
                    np.array([X[i] - X[j] for i, j in oriented]))[:, 1]
                for (i, j), p in zip(oriented, probs):
                    borda[i] += p
                    borda[j] += 1 - p
                cl = sorted(cl, key=lambda i: (-borda[i], base[i]))
            final.extend(cl)
        out = np.empty(len(cands))
        out[final] = np.arange(len(final))
        return out


MODELS = {m.name: m for m in
          (RulesBaseline, LinearModel, GBMRegression, GBMMedian, GBMRecency,
           GBMRanker, PairwiseBT, GBMRerank)}
