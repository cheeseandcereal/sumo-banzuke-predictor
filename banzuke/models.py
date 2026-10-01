"""Ordering models. Shared interface:

    Model(seed=0, n_seeds=None, base=None, pair=None, **options)
    fit(train)       train: transition rows with position_next targets
    score(cands, k)  -> np.ndarray, lower = ranked higher; k scores with one
                        bag member only

All models receive the same FEATURES; they differ in objective, which is
the experiment axis (see docs/EXPERIMENTS.md). base/pair override
BASE_PARAMS/PAIR_PARAMS. n_seeds > 1 bags LightGBM seeds
n_seeds*seed .. n_seeds*(seed+1)-1 and averages (the GBMs default to 5);
seed=None uses LightGBM's library defaults.
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

from banzuke.build import SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO
from banzuke.features import FEATURES

BASE_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=30,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=1, verbose=-1,
)
PAIR_PARAMS = {**BASE_PARAMS, "n_estimators": 150}
OOF_MIN_HISTORY = 60  # labeled basho required before a basho gets an out-of-fold score


def _seeds(seed, n_seeds):
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    if seed is None:
        if n_seeds != 1:
            raise ValueError("seed=None (library defaults) cannot be bagged")
        return [None]
    return list(range(seed * n_seeds, (seed + 1) * n_seeds))


def _cols(spec):
    """Column list from an option value: a list, or a comma-separated string
    (what `--set extra=rank_protected,mk_joi` parses to); empty -> []."""
    if not spec:
        return []
    return spec.split(",") if isinstance(spec, str) else list(spec)


def _seeded(params, seed):
    return dict(params) if seed is None else {**params, "random_state": seed}


class _Model:
    """Options are declared per class in OPTIONS with their defaults; n_seeds
    defaults to the class N_SEEDS."""

    OPTIONS: dict = {}
    N_SEEDS = 1

    def __init__(self, seed=0, n_seeds=None, base=None, pair=None, **options):
        self.seeds = _seeds(seed, self.N_SEEDS if n_seeds is None else n_seeds)
        self.base_params = {**BASE_PARAMS, **(base or {})}
        self.pair_params = {**PAIR_PARAMS, **(pair or {})}
        unknown = set(options) - set(self.OPTIONS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown options {sorted(unknown)}")
        for k, v in self.OPTIONS.items():
            setattr(self, k, options.get(k, v))

    @classmethod
    def prepare(cls, kwargs, trans, workers=1, train_start=None):
        """Target-independent inputs a configuration needs, computed once."""
        return kwargs


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
    """A: gradient-boosted regression on movement delta (L2). half_life:
    optional exponential recency weight in basho, normalized to mean 1.
    extra: dataset columns appended to FEATURES, for measuring a candidate
    input without a code change (`--set extra=kinboshi`)."""

    name = "A"
    objective = "regression"
    N_SEEDS = 5
    OPTIONS = {"half_life": None, "extra": ()}

    @property
    def features(self):
        return FEATURES + _cols(self.extra)

    def fit(self, train):
        w = None
        if self.half_life:
            order = {b: i for i, b in enumerate(sorted(train["basho"].unique()))}
            age = train["basho"].map(order).max() - train["basho"].map(order)
            w = (0.5 ** (age / self.half_life)).to_numpy()
            w = w / w.mean()
        self.ms = [
            LGBMRegressor(objective=self.objective, **_seeded(self.base_params, s))
            .fit(train[self.features], train["delta"], sample_weight=w)
            for s in self.seeds
        ]

    def base_score(self, cands, k=None):
        ms = self.ms if k is None else [self.ms[k]]
        delta = np.mean([m.predict(cands[self.features]) for m in ms], axis=0)
        return cands["position"].to_numpy() + delta

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
    """B: LambdaRank on the next-basho order within each transition. top:
    relevance levels; truncation: lambdarank_truncation_level, raised from
    LightGBM's 30 so the juryo boundary (~position 42) is trained on."""

    name = "B"
    N_SEEDS = 5
    OPTIONS = {"top": 60, "truncation": 60}

    def fit(self, train):
        t = train.sort_values(["basho", "position"], kind="stable")
        label = np.clip(self.top - t["position_next"], 0, self.top).astype(int)
        groups = t.groupby("basho", sort=True).size().to_numpy()
        self.ms = [
            LGBMRanker(objective="lambdarank", label_gain=list(range(self.top + 1)),
                       lambdarank_truncation_level=self.truncation,
                       **_seeded(self.base_params, s))
            .fit(t[FEATURES], label, group=groups)
            for s in self.seeds
        ]

    def score(self, cands, k=None):
        ms = self.ms if k is None else [self.ms[k]]
        return -np.mean([m.predict(cands[FEATURES]) for m in ms], axis=0)


# --- pairwise stage -------------------------------------------------------

_H2H: set | None = None  # {(basho, winner_id, loser_id)}

# context appended to pair feature differences: columns differencing zeroes
# (era, division sizes), the pair's location (means), each endpoint's class
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
    """Index pairs whose scores differ by at most gap, oriented i = currently
    higher ranked (NaN scores pair with nothing)."""
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


def _pair_matrix(df, i_arr, j_arr, h2h=False, context=False, base=None, extra=()):
    """Feature rows for index pairs of one basho's rows (i = currently higher
    ranked): feature differences (FEATURES + extra), then optionally the
    pair's head-to-head bout this basho (+1 i won, -1 i lost), context
    columns, base-score gap."""
    X = df[FEATURES + list(extra)].to_numpy(dtype=float)
    cols = [X[i_arr] - X[j_arr]]
    if h2h:
        wins = _h2h_wins()
        rid, basho = df["rikishi_id"].to_numpy(), int(df["basho"].iloc[0])
        won = [(basho, rid[i], rid[j]) in wins for i, j in zip(i_arr, j_arr)]
        lost = [(basho, rid[j], rid[i]) in wins for i, j in zip(i_arr, j_arr)]
        cols.append((np.array(won, dtype=float) - np.array(lost, dtype=float))[:, None])
    if context:
        means = df[CONTEXT_MEAN].to_numpy(dtype=float)
        ends = df[CONTEXT_ENDS].to_numpy(dtype=float)
        cols += [df[CONTEXT_SHARED].to_numpy(dtype=float)[i_arr],
                 (means[i_arr] + means[j_arr]) / 2, ends[i_arr], ends[j_arr]]
    if base is not None:
        cols.append((base[i_arr] - base[j_arr])[:, None])
    return np.column_stack(cols)


class _PairStage:
    """Pair classifier bag (label: i stays above j). Training pairs are all
    pairs within `window` rows of each other in the current order and, when
    rolling out-of-fold base scores are given, every pair whose OOF scores
    differ by at most `oof_gap` (the near-ties the reranker adjudicates, with
    the base-score gap as a feature)."""

    def __init__(self, seeds, params, window, h2h=False, context=False, oof=None,
                 oof_gap=2.0, extra=()):
        self.seeds, self.params, self.window = seeds, params, window
        self.h2h, self.context, self.oof_gap = h2h, context, oof_gap
        self.extra = _cols(extra)
        self.oof = None if oof is None else oof.set_index(["basho", "rikishi_id"])["oof"]

    def _matrix(self, df, i_arr, j_arr, base):
        return _pair_matrix(df, i_arr, j_arr, self.h2h, self.context, base, self.extra)

    def _base(self, df):
        if self.oof is None:
            return None
        key = pd.MultiIndex.from_arrays([df["basho"], df["rikishi_id"]])
        return self.oof.reindex(key).to_numpy(dtype=float)

    def fit(self, train):
        Xs, ys = [], []
        for _, grp in train.groupby("basho"):
            df = grp.sort_values("position", kind="stable")
            pos, nxt = df["position"].to_numpy(), df["position_next"].to_numpy()
            base = self._base(df)
            pairs = [_window_pairs(len(df), self.window)]
            if base is not None and not np.isnan(base).all():
                pairs.append(_gap_pairs(base, pos, self.oof_gap))
            for i_arr, j_arr in pairs:
                Xs.append(self._matrix(df, i_arr, j_arr, base))
                ys.append((nxt[i_arr] < nxt[j_arr]).astype(int))
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.ms = [LGBMClassifier(**_seeded(self.params, s)).fit(X, y) for s in self.seeds]
        return self

    def proba(self, df, i_arr, j_arr, base=None, k=None):
        """P(i stays above j) for index pairs of df; base = df's base scores."""
        X = self._matrix(df, i_arr, j_arr, base if self.oof is not None else None)
        ms = self.ms if k is None else [self.ms[k]]
        return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)


def oof_base_scores(trans, cls, kwargs, min_history=OOF_MIN_HISTORY, workers=1,
                    cache_dir=None, train_start=None):
    """Rolling out-of-fold base scores: rows of source basho b are scored by
    cls(**kwargs)'s base stage fitted on transitions labeled by b
    (next_basho <= b), i.e. the backtest's own base prediction for target
    next(b), so no row is scored by a model that saw its label. Cached on
    disk under a key of the configuration, the base-stage code and the data."""
    import hashlib
    import inspect
    import json
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from pathlib import Path

    from banzuke.build import PROCESSED

    lab = trans[trans["position_next"].notna()]
    kwargs = {k: v for k, v in kwargs.items() if k not in ("oof", "seed")}
    base_kw = {k: kwargs[k] for k in ("n_seeds", "base", "half_life", "extra") if k in kwargs}
    base_spec = [cls.objective, base_kw, BASE_PARAMS, min_history, train_start]
    key = hashlib.sha256(json.dumps(base_spec, sort_keys=True).encode())
    key.update((inspect.getsource(GBMRegression) + inspect.getsource(_oof_one)).encode())
    cols = FEATURES + _cols(kwargs.get("extra"))
    key.update(pd.util.hash_pandas_object(lab[["basho", "rikishi_id", "position_next"] + cols],
                                          index=False).to_numpy().tobytes())
    cache_dir = Path(cache_dir or PROCESSED.parent.parent / "results" / "scratch" / "oof")
    path = cache_dir / f"{key.hexdigest()[:16]}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    todo = sorted(lab["basho"].unique())[min_history:]
    print(f"computing out-of-fold base scores for {len(todo)} basho ({workers} workers; "
          "cached afterwards)...", file=sys.stderr, flush=True)
    args = [(b, cls, kwargs, train_start) for b in todo]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                 initializer=_oof_init, initargs=(lab,)) as ex:
            parts = list(ex.map(_oof_one, args, chunksize=4))
    else:
        _oof_init(lab)
        parts = [_oof_one(a) for a in args]
    out = pd.concat(parts, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


_OOF_LAB = None


def _oof_init(lab):
    global _OOF_LAB
    _OOF_LAB = lab


def _oof_one(args):
    b, cls, kwargs, train_start = args
    train = _OOF_LAB[_OOF_LAB["next_basho"] <= b]
    if train_start:
        train = train[train["basho"] >= train_start]
    rows = _OOF_LAB[_OOF_LAB["basho"] == b]
    m = cls(**kwargs)
    GBMRegression.fit(m, train)  # base stage only
    return pd.DataFrame({"basho": rows["basho"].to_numpy(),
                         "rikishi_id": rows["rikishi_id"].to_numpy(),
                         "oof": GBMRegression.score(m, rows)})


class PairwiseBT(_Model):
    """C: classifier on nearby candidate pairs (feature diffs), aggregated
    into local movement from the current order (approximate Bradley-Terry)."""

    name = "C"
    N_SEEDS = 5
    OPTIONS = {"pair_window": 12, "scale": 2.0}

    def fit(self, train):
        self.pair = _PairStage(self.seeds, self.pair_params, self.pair_window).fit(train)

    def score(self, cands, k=None):
        df = cands.sort_values("position", kind="stable")
        i_arr, j_arr = _window_pairs(len(df), self.pair_window)
        p = self.pair.proba(df, i_arr, j_arr, k=k)
        movement = np.zeros(len(df))
        np.add.at(movement, i_arr, p - 0.5)
        np.add.at(movement, j_arr, 0.5 - p)
        score = df["position"].to_numpy() - self.scale * movement
        return pd.Series(score, index=df.index).reindex(cands.index).to_numpy()


TWIN_SCOPE = {"sk": (SEKIWAKE, KOMUSUBI), "all": (SEKIWAKE, KOMUSUBI, MAEGASHIRA, JURYO)}


def _twin_units(cands, scope):
    """[e, w] row-index pairs of E/W twins: one rank number, identical W-L-A
    record, class within `scope` ("sk" or "all" = S/K/M/J)."""
    if scope not in TWIN_SCOPE:
        raise ValueError(f"twin_unit must be one of {sorted(TWIN_SCOPE)} or empty, got {scope!r}")
    cols = [cands[c].to_numpy() for c in ("rank_class", "rank_number", "wins", "losses", "absences")]
    side = cands["side"].to_numpy()
    groups: dict[tuple, list] = {}
    for i in np.flatnonzero(np.isin(cols[0], TWIN_SCOPE[scope])):
        groups.setdefault(tuple(int(c[i]) for c in cols), []).append(int(i))
    return [sorted(g, key=lambda i: side[i]) for g in groups.values()
            if len(g) == 2 and {side[i] for i in g} == {0, 1}]


class GBMRerank(GBMMedian):
    """Ar: Aq's global order, with a pair classifier reordering clusters of
    near-equal base scores (E/W flips and off-by-one placements are most of
    the misses; the committee resolves them by wins, then prior order).

    pair_window: training pairs within this current-position distance
    near_ties: also train on near-ties under rolling out-of-fold base scores
        (within oof_gap), the pairs the reranker actually adjudicates; needs
        the OOF table `oof`, which prepare() supplies
    context: append absolute/era context to the pair feature differences
    gap, cluster_max: cluster break when consecutive base scores differ by
        more than gap; largest cluster the reranker may reorder
    h2h: include the pair's head-to-head bout this basho as a pair feature
    twin_unit: "" / "sk" / "all": E and W of one rank number with identical
        records (of S/K, or S/K/M/J) form one unit in the reranker: clustered
        at their mean base score, compared with a rival by the mean of the
        members' pair probabilities, expanded E then W. Borda otherwise
        separates such a pair by a full point (their mutual comparison is
        certain) and drops any rival the classifier is unsure about between
        them. The committee never splits S/K twins, so "sk" is the default;
        "all" was neutral-to-worse (E17)
    """

    name = "Ar"
    OPTIONS = {**GBMMedian.OPTIONS, "pair_window": 6, "near_ties": True, "oof_gap": 2.0,
               "oof": None, "context": True, "gap": 1.0, "cluster_max": 6, "h2h": False,
               "twin_unit": "sk"}

    @classmethod
    def prepare(cls, kwargs, trans, workers=1, train_start=None):
        if not kwargs.get("near_ties", cls.OPTIONS["near_ties"]) or kwargs.get("oof") is not None:
            return kwargs
        return {**kwargs, "oof": oof_base_scores(trans, cls, kwargs, workers=workers,
                                                  train_start=train_start)}

    def fit(self, train):
        if self.near_ties and self.oof is None:
            raise ValueError("near_ties needs rolling OOF base scores: call prepare() first")
        super().fit(train)
        self.pair = _PairStage(self.seeds, self.pair_params, self.pair_window, self.h2h,
                               self.context, self.oof if self.near_ties else None,
                               self.oof_gap, self.extra).fit(train)

    def score(self, cands, k=None):
        base = self.base_score(cands, k)
        pos = cands["position"].to_numpy()
        units = [[i] for i in range(len(cands))]
        if self.twin_unit:
            for e, w in _twin_units(cands, self.twin_unit):
                units[e], units[w] = [e, w], []
        units = [u for u in units if u]
        ubase = np.array([base[u].mean() for u in units])
        upos = np.array([pos[u].min() for u in units])
        order = np.lexsort((upos, ubase))
        clusters, cur = [], [order[0]]
        for prev, i in zip(order, order[1:]):
            if ubase[i] - ubase[prev] <= self.gap and len(cur) < self.cluster_max:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        # every within-cluster pair of units, oriented by current position to match
        # training; a unit pair's probability is the mean over its member pairs
        upairs = [(a, b) if upos[a] <= upos[b] else (b, a)
                  for cl in clusters for a, b in combinations(cl, 2)]
        borda = np.zeros(len(units))
        if upairs:
            i_arr, j_arr, owner = map(np.array, zip(*[
                (i, j, q) for q, (a, b) in enumerate(upairs) for i in units[a] for j in units[b]]))
            p = self.pair.proba(cands, i_arr, j_arr, base, k)
            for (a, b), pu in zip(upairs, np.bincount(owner, p) / np.bincount(owner)):
                borda[a] += pu
                borda[b] += 1 - pu
        final = [i for cl in clusters for u in sorted(cl, key=lambda u: (-borda[u], ubase[u]))
                 for i in sorted(units[u], key=lambda i: pos[i])]
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
