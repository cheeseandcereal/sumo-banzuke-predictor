# sumo-banzuke

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

Predict the upcoming banzuke from the latest fetched results:

```sh
uv run python predict.py
uv run python predict.py --retired Shikona1,Shikona2   # announced retirees
```

Update data after a basho ends (incremental, skips existing files):

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

## Layout

- `fetch_banzuke.py`, `fetch_basho.py`: incremental scrapers
- `data/banzuke/`, `data/basho/`: raw API responses (committed)
- `data/processed/`: rebuildable parquet dataset (gitignored)
- `banzuke/`: dataset build, features, models, resolver, backtest harness
- `predict.py`, `backtest.py`, `analyze.py`: CLIs
- `results/`: backtest metric summaries (committed)
- `docs/EXPERIMENTS.md`: experiment log and model-selection rationale
