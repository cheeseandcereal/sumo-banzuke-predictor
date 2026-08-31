# Sumo Banzuke Predictor

Predicts the next makuuchi banzuke (grand sumo top-division rankings)
from the results of the previous basho.

Multiple ordering models (rules formula, linear, gradient-boosted
regression/ranking/pairwise variants) competed on a shared
rolling-origin backtest covering every banzuke transition since 1959.
The promoted default, `Ar`, is an L1 gradient-boosted movement model
whose near-tie clusters are reordered by a pairwise classifier that
learned the committee's conflict-resolution habits, feeding a resolver
that applies the near-inviolable structure (Y/O conventions, sanyaku
minimums, empirically-derived E/W layout). See `docs/EXPERIMENTS.md`
for the full experiment log and selection rationale.

The model trains on historical transitions between consecutive basho,
using rank, win-loss results, prizes, recent form, career context, and
era features to predict each wrestler's movement. Evaluation uses a
rolling-origin backtest that retrains on strictly earlier tournaments,
so each prediction only uses information available at the time.

Held-out performance (2024-2026, never touched during development):
**17.1 of 42 slots exactly right** per basho on average, **80% of
wrestlers placed within one position**, mean absolute error 0.85
half-ranks, juryo promotion/demotion F1 0.96/0.87, GTB score 42
points/basho (2 per exact slot, 1 per right-rank-wrong-side).

For scale, in the long-running human "Guess the Banzuke" game
([dichne.com](https://www.dichne.com/Guess.htm), scored the same way:
a "bullseye" is an exact slot), the all-time top-10 players average
25-27 bullseyes (~53-55 points) per basho, and the single best entry
out of ~470 each basho lands around 33-36. Most of the model's
remaining gap is near-tie resolution: of its ~24 misses per basho,
~6.5 are pure E/W flips and ~8 are off by one position, both zones
where the committee's choice is close to a coin flip on paper. Perfect
near-tie resolution would score ~32/42. See the experiment log for
what has been tried.

## Data

All data comes from the excellent [sumo-api.com](https://www.sumo-api.com)
(banzuke, results, yusho, and special prizes for every basho since
1959). Raw API responses are committed under `data/` so cloning this
repo does not require re-scraping. If you find the API useful, consider
[supporting its server costs](https://ko-fi.com/sumoapi).

## Usage

Build the processed dataset after cloning (it is generated from the
committed raw data and is not tracked by git):

```sh
uv run python -m banzuke.build
```

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

Update and rebuild data after a basho ends (fetching is incremental and
skips existing files):

```sh
uv run python fetch_banzuke.py && uv run python fetch_basho.py
uv run python -m banzuke.build
```

Re-run the model bake-off / evaluation:

```sh
uv run python backtest.py --out results/all    # 2004 through latest basho
uv run python backtest.py --end 202311         # original dev window only
uv run python analyze.py --model Ar            # residual analysis
```

Backtest results are cached per (model, basho) in `results/scratch/`;
re-runs with the same code and data reuse them. The cache key hashes
the processed dataset and all result-affecting source files, so edits
and data rebuilds invalidate it automatically (`--fresh` to force).
Model training itself is never persisted: `predict.py` retrains on
every invocation (seconds; `--seeds 1` for the fastest run).

Both CLIs accept `--train-start BASHO` to restrict training to newer
transitions. Tested and neutral-to-worse (docs/EXPERIMENTS.md E9):
the era features already let the models specialize to the modern
regime, so full history remains the default.

## Layout

- `fetch_banzuke.py`, `fetch_basho.py`: incremental scrapers
- `data/banzuke/`, `data/basho/`: raw API responses (committed)
- `data/processed/`: rebuildable parquet dataset (gitignored)
- `banzuke/`: dataset build, features, models, resolver, backtest harness
- `predict.py`, `backtest.py`, `analyze.py`: CLIs
- `results/`: backtest metric summaries (committed)
- `docs/EXPERIMENTS.md`: experiment log and model-selection rationale
