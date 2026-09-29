# Sumo Banzuke Predictor

Predicts the next makuuchi banzuke (grand sumo top-division rankings)
from the results of the previous basho.

Multiple ordering models (rules formula, linear, gradient-boosted
regression/ranking/pairwise variants) competed on a shared
rolling-origin backtest covering every banzuke transition since 1959.
The promoted default, `Ar`, is a 5-seed bag of L1 gradient-boosted
movement models whose near-tie clusters are reordered by a pairwise
classifier that learned the committee's conflict-resolution habits
(trained on the near-ties it actually adjudicates, via rolling
out-of-fold base scores), feeding a resolver that applies the
near-inviolable structure (Y/O conventions, sanyaku minimums,
empirically-derived E/W layout, no promotion after make-koshi for
S/K/M). See `docs/EXPERIMENTS.md` for the full experiment log and
selection rationale.

The model trains on historical transitions between consecutive basho,
using rank, win-loss results, prizes, recent form, career context, and
era features to predict each wrestler's movement. Evaluation uses a
rolling-origin backtest that retrains on strictly earlier tournaments,
so each prediction only uses information available at the time.

Backtested performance of the current default (two independent 5-seed
bags, averaged per basho; +- is the standard error over basho):

| window | basho | exact slots /42 | GTB points | within 1 | MAE (half-ranks) | promo / demo F1 |
|---|---:|---:|---:|---:|---:|---:|
| 2004-2019 (tuning) | 95 | **20.2** +- 0.5 | 46.8 | 81% | 0.84 | .94 / .91 |
| 2020-2026 (confirmation) | 40 | **17.9** +- 1.0 | 43.0 | 80% | 0.94 | .91 / .88 |
| 2024-2026 | 17 | 20.8 +- 1.6 | 48.2 | 84% | 0.71 | .95 / .91 |

GTB points are 2 per exact slot and 1 per right-rank-wrong-side. The
2020s are harder than the 2000s-2010s for every model tried (the
committee has created extra sanyaku slots twice as often since 2020).
The 2020-2026 figures were used once to choose among four finalist
configurations, so they carry some selection optimism; the first
untouched test is the 202611 banzuke. The previous default (a single
seed, 600/400 rounds, window-trained reranker) scored 19.2 / 16.6 /
18.4 on the same windows.

For scale, in the long-running human "Guess the Banzuke" game
([dichne.com](https://www.dichne.com/Guess.htm), scored the same way:
a "bullseye" is an exact slot), the all-time top-10 players average
25-27 bullseyes (~53-55 points) per basho, and the single best entry
out of ~470 each basho lands around 33-36. Much of the model's
remaining gap is near-tie resolution: pure E/W flips and placements
off by one position, both zones where the committee's choice is close
to a coin flip on paper. See the experiment log for what has been tried.

## Data

All data comes from the excellent [sumo-api.com](https://www.sumo-api.com)
(banzuke, results, yusho, and special prizes for every basho since
1959). Raw API responses and processed Parquet datasets are committed
under `data/`, so a clone is ready to use without fetching or rebuilding.
If you find the API useful, consider [supporting its server
costs](https://ko-fi.com/sumoapi).

## Usage

Predict the upcoming banzuke from the latest fetched results:

```sh
uv run python predict.py
uv run python predict.py --retired Shikona1,Shikona2   # announced retirees
```

Disagree with a placement? Overrides constrain the assignment while the
model fills everything else (scores are computed once; there is no
conditional re-inference):

```sh
uv run python predict.py --above "Takayasu > Hakunofuji"  # A rises above B
uv run python predict.py --below "Oho < Ura"              # A drops below B
uv run python predict.py --class Aonishiki=O --count S=3  # structural beliefs
uv run python predict.py --pin Wakatakakage=M8E           # exact cell
uv run python predict.py --interactive                    # train once, iterate
```

Output annotates pinned cells, marks every wrestler your override moved
(`<-M3E`), reports whether each relation survived the structural stage,
and warns when an override breaks a banzuke convention (demoting a
yokozuna, dropping a kachi-koshi sanyaku incumbent, etc.).

Fetch available data and rebuild the processed datasets after a banzuke
release or completed basho (completed tournaments are skipped):

```sh
uv run python update_data.py
uv run python update_data.py --fetch-only  # update raw JSON only
uv run python update_data.py --build-only  # rebuild from existing JSON only
```

Re-run the model bake-off / evaluation:

```sh
uv run python backtest.py --out results/all    # 2004 through latest basho
uv run python backtest.py --end 202311         # original dev window only
uv run python analyze.py --model Ar            # residual analysis
```

Backtests run one target per worker process (`--workers`, default 16)
and report paired comparisons against a baseline you name: mean
differences in exact slots and MAE with block-bootstrap confidence
intervals and Wilcoxon p-values, seeds averaged within each basho.
Parameter experiments need no code changes:

```sh
uv run python backtest.py --models Ar --seeds 0-1 --start 200401 --end 201911 \
    --set base.n_estimators=600 --set gap=0.5 --name old_gate --baseline Ar
uv run python backtest.py --models Ar --config sweep.json --baseline Ar:bag5
```

`--set stage.key=value` overrides LightGBM parameters of the movement
(`base.*`) or pair (`pair.*`) stage; bare keys are model options
(`n_seeds`, `gap`, `cluster_max`, `pairs`, `context`, `half_life`, ...).
Results are cached per (configuration, model, seed, basho) in
`results/scratch/`; the cache key hashes the processed dataset, all
result-affecting source files, the lockfile and the resolved
configuration, so edits, data rebuilds and parameter changes
invalidate it automatically (`--fresh` to force).

Model training itself is never persisted: `predict.py` retrains on
every invocation (~30 s for the 5-seed bag; `--seeds 1` for a single
seed). The reranker's out-of-fold training scores are computed once per
data update (about two minutes, parallel) and cached in
`results/scratch/oof/`.

Both CLIs accept `--train-start BASHO` to restrict training to newer
transitions. Tested and neutral-to-worse (docs/EXPERIMENTS.md E9, E14):
the era features already let the models specialize to the modern
regime, so full history remains the default.

Run the tests with `uv run pytest -q` (about 10 s).

## Layout

- `update_data.py`: incremental fetcher and processed-data builder
- `data/banzuke/`, `data/basho/`: committed raw API responses
- `data/processed/`: committed, reproducible Parquet datasets
- `banzuke/`: dataset build, features, models, resolver, backtest harness
- `predict.py`, `backtest.py`, `analyze.py`: CLIs
- `tests/`: pytest suite (resolver conventions, feature chronology,
  model semantics, harness determinism)
- `results/`: generated backtest reports and caches (git-ignored)
- `docs/EXPERIMENTS.md`: experiment log, tuning protocol and
  model-selection rationale
