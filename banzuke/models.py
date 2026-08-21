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


class PairwiseBT:
    """C: classifier on nearby candidate pairs (feature diffs), aggregated
    into local movement from the current order (approximate Bradley-Terry)."""

    name = "C"
    WINDOW = 12
    SCALE = 2.0

    def _pairs(self, df):
        df = df.sort_values("position", kind="stable")
        X = df[FEATURES].to_numpy(dtype=float)
        idx_pairs = [
            (i, j)
            for i in range(len(df))
            for j in range(i + 1, min(i + 1 + self.WINDOW, len(df)))
        ]
        i_arr = np.array([p[0] for p in idx_pairs])
        j_arr = np.array([p[1] for p in idx_pairs])
        return X[i_arr] - X[j_arr], i_arr, j_arr, df

    def fit(self, train):
        Xs, ys = [], []
        for _, grp in train.groupby("basho"):
            X, i_arr, j_arr, df = self._pairs(grp)
            nxt = df["position_next"].to_numpy()
            Xs.append(X)
            ys.append((nxt[i_arr] < nxt[j_arr]).astype(int))
        self.m = LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 400})
        self.m.fit(np.vstack(Xs), np.concatenate(ys))

    def score(self, cands):
        X, i_arr, j_arr, df = self._pairs(cands)
        p = self.m.predict_proba(X)[:, 1]  # P(i stays above j)
        movement = np.zeros(len(df))
        np.add.at(movement, i_arr, p - 0.5)
        np.add.at(movement, j_arr, 0.5 - p)
        score = df["position"].to_numpy() - self.SCALE * movement
        # map back to cands row order
        out = pd.Series(score, index=df.index).reindex(cands.index)
        return out.to_numpy()


MODELS = {m.name: m for m in
          (RulesBaseline, LinearModel, GBMRegression, GBMMedian, GBMRanker, PairwiseBT)}
