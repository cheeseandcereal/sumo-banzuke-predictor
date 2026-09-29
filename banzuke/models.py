"""Ordering models. Shared interface:

    Model(seed=0, n_seeds=None, base=None, pair=None, **options)
    fit(train)    train: transition rows with position_next targets
    score(cands)  -> np.ndarray, lower = ranked higher

All models receive the same FEATURES; they differ in objective, which is
the experiment axis (see docs/EXPERIMENTS.md). Hyperparameters live in
BASE_PARAMS (regressors/ranker) and PAIR_PARAMS (pair classifiers). The
constructor merges per-instance overrides, so backtests and predict.py
share one configuration path and nothing mutates module state.

Seeds: `seed=k` with `n_seeds=1` sets LightGBM's random_state=k. With
`n_seeds=m > 1` the instance is a bag of m models on seeds m*k .. m*k+m-1
(so bag replicates k=0,1,... are disjoint) whose predictions are averaged;
the GBM classes default to a bag of 5 (E11). `seed=None` omits
random_state entirely, reproducing LightGBM's library defaults (the
pre-2026-09 configuration); it cannot be bagged.

Defaults below are the E11-E14 selection (docs/EXPERIMENTS.md, 2026-09).
"""
import sys
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
    n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=1, verbose=-1,
)
PAIR_PARAMS = {**BASE_PARAMS, "n_estimators": 150}


def _seeds(seed, n_seeds):
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
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
    model-specific options (declared per class in OPTIONS with defaults).
    n_seeds=None uses the class default N_SEEDS (1 for deterministic models,
    a bag of 5 for the gradient-boosted ones)."""

    OPTIONS: dict = {}
    N_SEEDS = 1

    def __init__(self, seed=0, n_seeds=None, base=None, pair=None, **options):
        self.seeds = _seeds(seed, self.N_SEEDS if n_seeds is None else n_seeds)
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
    half_life: optional exponential recency weight (in basho) on training
    rows, normalized to mean 1 so regularization strength is unchanged.
    blend_l2: weight in [0, 1] of an additional L2-objective bag mixed into
    the movement prediction (0 = this class's objective only)."""

    name = "A"
    objective = "regression"
    N_SEEDS = 5
    OPTIONS = {"half_life": None, "blend_l2": 0.0}

    def _weights(self, train):
        if not self.half_life:
            return None
        order = {b: i for i, b in enumerate(sorted(train["basho"].unique()))}
        age = train["basho"].map(order).max() - train["basho"].map(order)
        w = (0.5 ** (age / self.half_life)).to_numpy()
        return w / w.mean()

    def fit(self, train):
        X, y, w = train[FEATURES], train["delta"], self._weights(train)

        def bag(objective):
            return [LGBMRegressor(objective=objective, **_seeded(self.base_params, s))
                    .fit(X, y, sample_weight=w) for s in self.seeds]

        self.ms = bag(self.objective)
        self.ms2 = bag("regression") if self.blend_l2 else []

    def delta(self, cands, k=None):
        def mean(models):
            ms = models if k is None else [models[k]]
            return np.mean([m.predict(cands[FEATURES]) for m in ms], axis=0)

        d = mean(self.ms)
        if self.blend_l2:
            d = (1 - self.blend_l2) * d + self.blend_l2 * mean(self.ms2)
        return d

    def base_score(self, cands, k=None):
        return cands["position"].to_numpy() + self.delta(cands, k)

    # the regression stage's predicted next position; for Ar this is the base
    # order its reranker works within (confidence signals need it)
    score = base_score


class GBMMedian(GBMRegression):
    """Aq: L1 objective (median regression), less shrinkage on big movers."""

    name = "Aq"
    objective = "regression_l1"


class GBMRecency(GBMRegression):
    """Aw: L2 GBM with exponential recency weights (half-life 60 basho, ~10y),
    biasing training toward the modern committee's behavior."""

    name = "Aw"
    OPTIONS = {**GBMRegression.OPTIONS, "half_life": 60}


class GBMRanker(_Model):
    """B: LambdaRank on the next-basho order within each transition.
    top: relevance levels (focuses ranking quality on the top of the order).
    truncation: lambdarank_truncation_level (LightGBM default 30; the
    makuuchi boundary sits near position 42)."""

    name = "B"
    N_SEEDS = 5
    OPTIONS = {"top": 60, "truncation": 60}

    def fit(self, train):
        t = train.sort_values(["basho", "position"], kind="stable")
        label = np.clip(self.top - t["position_next"], 0, self.top).astype(int)
        groups = t.groupby("basho", sort=True).size().to_numpy()
        extra = {"lambdarank_truncation_level": self.truncation} if self.truncation else {}
        self.ms = [
            LGBMRanker(objective="lambdarank", label_gain=list(range(self.top + 1)),
                       **_seeded(self.base_params, s), **extra)
            .fit(t[FEATURES], label, group=groups)
            for s in self.seeds
        ]

    def score(self, cands, k=None):
        ms = self.ms if k is None else [self.ms[k]]
        return -np.mean([m.predict(cands[FEATURES]) for m in ms], axis=0)


# --- pairwise stage -------------------------------------------------------

_H2H: set | None = None  # {(basho, winner_id, loser_id)}

# context appended to pair feature differences when context=True: columns
# that differencing zeroes out (era, division sizes), the pair's absolute
# location (means), and each endpoint's class/division
CONTEXT_SHARED = ["year", "kosho", "mak_size", "jur_size"]
CONTEXT_MEAN = ["position", "rank_number", "wins", "boundary_dist"]
CONTEXT_ENDS = ["rank_class", "division"]


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


def _gap_pairs(scores, pos, gap):
    """All index pairs whose scores differ by <= gap (near-ties under a base
    ordering), oriented i = currently higher ranked."""
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ii, jj = [], []
    for a in range(len(s)):
        b = a + 1
        while b < len(s) and s[b] - s[a] <= gap:
            ii.append(order[a])
            jj.append(order[b])
            b += 1
    i_arr, j_arr = np.array(ii, dtype=int), np.array(jj, dtype=int)
    swap = pos[i_arr] > pos[j_arr]
    return np.where(swap, j_arr, i_arr), np.where(swap, i_arr, j_arr)


def _pair_matrix(df, i_arr, j_arr, h2h=False, context=False, base=None):
    """Feature rows for index pairs of one basho's rows, oriented i = currently
    higher ranked: feature differences, then optionally the pair's
    head-to-head bout this basho (+1 i won, -1 i lost, 0), context columns,
    and the base-score gap base[i] - base[j]."""
    X = df[FEATURES].to_numpy(dtype=float)
    cols = [X[i_arr] - X[j_arr]]
    if h2h:
        wins = _h2h_wins()
        rid = df["rikishi_id"].to_numpy()
        basho = int(df["basho"].iloc[0])
        col = np.array(
            [(basho, rid[i], rid[j]) in wins for i, j in zip(i_arr, j_arr)], dtype=float
        ) - np.array(
            [(basho, rid[j], rid[i]) in wins for i, j in zip(i_arr, j_arr)], dtype=float
        )
        cols.append(col[:, None])
    if context:
        shared = df[CONTEXT_SHARED].to_numpy(dtype=float)[i_arr]
        means = df[CONTEXT_MEAN].to_numpy(dtype=float)
        ends = df[CONTEXT_ENDS].to_numpy(dtype=float)
        cols += [shared, (means[i_arr] + means[j_arr]) / 2, ends[i_arr], ends[j_arr]]
    if base is not None:
        cols.append((base[i_arr] - base[j_arr])[:, None])
    return np.column_stack(cols) if len(cols) > 1 else cols[0]


class _PairStage:
    """Pair classifier bag shared by C and Ar.

    pairs: how training pairs are chosen per training basho
      'window' - all pairs within `window` rows of each other in the current
                 order (label: i stays above j)
      'oof'    - pairs whose rolling out-of-fold base scores differ by at most
                 `oof_gap`: the near-ties the reranker actually adjudicates
      'mixed'  - both (a near-tie that is also a local pair appears twice,
                 i.e. carries double weight; this is the tuned/confirmed form)
    With 'oof'/'mixed' the base-score gap is a pair feature, so `oof` (a
    frame basho, rikishi_id, oof from oof_base_scores) is required."""

    def __init__(self, seeds, params, window, h2h=False, context=False,
                 pairs="window", oof=None, oof_gap=1.0):
        if pairs not in ("window", "oof", "mixed"):
            raise ValueError(f"pairs must be window|oof|mixed, got {pairs!r}")
        if pairs != "window" and oof is None:
            raise ValueError("pairs='oof'/'mixed' need rolling OOF base scores "
                             "(call Model.prepare first)")
        self.seeds, self.params, self.window, self.h2h = seeds, params, window, h2h
        self.context, self.pairs, self.oof_gap = context, pairs, oof_gap
        self.use_base = pairs != "window"
        self.oof = None if oof is None else oof.set_index(["basho", "rikishi_id"])["oof"]

    def _oof_for(self, df):
        key = pd.MultiIndex.from_arrays([df["basho"], df["rikishi_id"]])
        s = self.oof.reindex(key).to_numpy(dtype=float)
        return None if np.isnan(s).all() else s

    def fit(self, train):
        Xs, ys = [], []
        for _, grp in train.groupby("basho"):
            df = grp.sort_values("position", kind="stable")
            pos, nxt = df["position"].to_numpy(), df["position_next"].to_numpy()
            base = self._oof_for(df) if self.use_base else None
            parts = []
            if self.pairs in ("window", "mixed"):
                parts.append(_window_pairs(len(df), self.window))
            if self.pairs in ("oof", "mixed") and base is not None:
                parts.append(_gap_pairs(base, pos, self.oof_gap))
            if self.use_base and base is None:
                base = np.full(len(df), np.nan)  # before the OOF history floor
            for i_arr, j_arr in parts:
                if len(i_arr) == 0:
                    continue
                Xs.append(_pair_matrix(df, i_arr, j_arr, self.h2h, self.context,
                                       base if self.use_base else None))
                ys.append((nxt[i_arr] < nxt[j_arr]).astype(int))
        if not ys:
            raise ValueError(f"no training pairs (pairs={self.pairs!r}); with 'oof' the "
                             "OOF table must cover training basho")
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.n_train_pairs = len(y)
        self.ms = [LGBMClassifier(**_seeded(self.params, s)).fit(X, y) for s in self.seeds]
        return self

    def proba(self, df, i_arr, j_arr, base=None, k=None):
        """P(i stays above j) for index pairs of df; base = df's base scores."""
        X = _pair_matrix(df, i_arr, j_arr, self.h2h, self.context,
                         base if self.use_base else None)
        ms = self.ms if k is None else [self.ms[k]]
        return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)


def oof_base_scores(trans, base_params, objective, seeds, min_history=60, workers=1,
                    cache_dir=None, half_life=None, blend_l2=0.0, train_start=None):
    """Rolling out-of-fold base scores for every labeled transition: rows of
    source basho b are scored by a base bag trained on transitions labeled at
    or before b (next_basho <= b), i.e. the backtest's own base prediction for
    target next(b), so no row is scored by a model that saw its label. The
    base bag is built exactly as the model's own (params, objective, seeds,
    recency weights, L2 blend, training cutoff). Basho with fewer than
    `min_history` labeled predecessors get no score. Cached on disk under a
    key of the configuration, this code and the data."""
    import hashlib
    import inspect
    import json
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from pathlib import Path

    from banzuke.build import PROCESSED

    lab = trans[trans["position_next"].notna()]
    spec = {"base": base_params, "objective": objective, "seeds": list(seeds),
            "half_life": half_life, "blend_l2": blend_l2, "train_start": train_start}
    key_src = json.dumps({**spec, "min_history": min_history, "features": FEATURES},
                         sort_keys=True) + inspect.getsource(_oof_one)
    data_hash = pd.util.hash_pandas_object(
        lab[["basho", "rikishi_id", "position_next"] + FEATURES], index=False).to_numpy()
    key = hashlib.sha256(key_src.encode() + data_hash.tobytes()).hexdigest()[:16]
    cache_dir = Path(cache_dir or PROCESSED.parent.parent / "results" / "scratch" / "oof")
    path = cache_dir / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    bashos = sorted(lab["basho"].unique())
    todo = bashos[min_history:]
    print(f"computing rolling out-of-fold base scores for {len(todo)} basho x "
          f"{len(seeds)} seed(s) ({workers} workers; cached afterwards)...",
          file=sys.stderr, flush=True)
    args = [(b, spec) for b in todo]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                 initializer=_oof_init, initargs=(lab,)) as ex:
            parts = list(ex.map(_oof_one, args, chunksize=4))
    else:
        _oof_init(lab, limit_threads=False)
        parts = [_oof_one(a) for a in args]
    out = pd.concat(parts, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out


_OOF_LAB = None


def _oof_init(lab, limit_threads=True):
    global _OOF_LAB
    _OOF_LAB = lab
    if limit_threads:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(1)
        except ImportError:
            pass


def _oof_one(args):
    b, spec = args
    lab = _OOF_LAB
    train = lab[lab["next_basho"] <= b]
    if spec["train_start"]:
        train = train[train["basho"] >= spec["train_start"]]
    rows = lab[lab["basho"] == b]
    base = type("_OOFBase", (GBMRegression,), {"objective": spec["objective"]})(
        seed=0, n_seeds=len(spec["seeds"]), base=spec["base"],
        half_life=spec["half_life"], blend_l2=spec["blend_l2"])
    base.seeds = list(spec["seeds"])
    base.fit(train)
    return pd.DataFrame({"basho": rows["basho"].to_numpy(),
                         "rikishi_id": rows["rikishi_id"].to_numpy(),
                         "oof": base.score(rows)})


class PairwiseBT(_Model):
    """C: classifier on nearby candidate pairs (feature diffs), aggregated
    into local movement from the current order (approximate Bradley-Terry)."""

    name = "C"
    N_SEEDS = 5
    OPTIONS = {"pair_window": 12, "scale": 2.0, "h2h": False, "context": False}

    def fit(self, train):
        self.pair = _PairStage(self.seeds, self.pair_params, self.pair_window,
                               self.h2h, self.context).fit(train)

    def score(self, cands, k=None):
        df = cands.sort_values("position", kind="stable")
        i_arr, j_arr = _window_pairs(len(df), self.pair_window)
        p = self.pair.proba(df, i_arr, j_arr, k=k)
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
    context: append absolute/era context to the pair feature differences
    pairs, oof_gap, oof: training-pair selection, see _PairStage; 'oof' and
        'mixed' need Model.prepare(kwargs, trans) to supply rolling OOF scores
    oof_min_history: labeled basho required before a basho gets an OOF score
    After score(), `last_pairs` holds (i, j, p, in_window) for the pairs the
    reranker adjudicated, for diagnostics.
    """

    name = "Ar"
    OPTIONS = {**GBMMedian.OPTIONS, "pair_window": 6, "gap": 1.0, "cluster_max": 6,
               "h2h": False, "context": True, "pairs": "mixed", "oof_gap": 2.0,
               "oof": None, "oof_min_history": 60}

    @classmethod
    def prepare(cls, kwargs, trans, workers=1, train_start=None):
        """Target-independent inputs computed once per configuration: rolling
        OOF base scores when training pairs are selected by base near-ties.
        Returns kwargs extended with them. Uses the configuration's seed-0
        (replicate 0) bag; later replicates reuse the same OOF table."""
        if kwargs.get("pairs", cls.OPTIONS["pairs"]) == "window" or kwargs.get("oof") is not None:
            return kwargs
        proto = cls(**{k: v for k, v in kwargs.items() if k != "seed"})
        oof = oof_base_scores(trans, proto.base_params, proto.objective, proto.seeds,
                              proto.oof_min_history, workers, half_life=proto.half_life,
                              blend_l2=proto.blend_l2, train_start=train_start)
        return {**kwargs, "oof": oof}

    def fit(self, train):
        super().fit(train)
        self.pair = _PairStage(self.seeds, self.pair_params, self.pair_window, self.h2h,
                               self.context, self.pairs, self.oof, self.oof_gap).fit(train)

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
        base = self.base_score(cands, k)
        pos = cands["position"].to_numpy()
        clusters = self._clusters(base, pos)
        # every within-cluster pair, oriented by current position to match training
        pairs = [(i, j) if pos[i] <= pos[j] else (j, i)
                 for cl in clusters for i, j in combinations(cl, 2)]
        borda = np.zeros(len(cands))
        if pairs:
            i_arr, j_arr = np.array(pairs).T
            probs = self.pair.proba(cands, i_arr, j_arr, base, k)
            for i, j, p in zip(i_arr, j_arr, probs):
                borda[i] += p
                borda[j] += 1 - p
            rank_now = np.argsort(np.argsort(pos, kind="stable"), kind="stable")
            self.last_pairs = (i_arr, j_arr, probs,
                               rank_now[j_arr] - rank_now[i_arr] <= self.pair_window)
        else:
            self.last_pairs = None
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
