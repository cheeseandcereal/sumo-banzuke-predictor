# sumo-banzuke

Predicts the next makuuchi banzuke (grand sumo top-division rankings)
from the results of the previous basho.

Multiple ordering models (rules formula, linear, gradient-boosted
regression/ranking variants) compete on a shared rolling-origin backtest
covering every banzuke transition since 1958; the best performer on
modern-era data is promoted as the default predictor. See
`docs/EXPERIMENTS.md` for what was tried and how it scored.

## Data

All data comes from the excellent [sumo-api.com](https://www.sumo-api.com)
(banzuke, results, yusho, and special prizes for every basho since 1958).
Raw API responses are committed under `data/` so cloning this repo does
not require re-scraping. If you find the API useful, consider
[supporting its server costs](https://ko-fi.com/sumoapi).

To update after a new basho:

```sh
uv run python fetch_banzuke.py   # incremental, skips existing files
uv run python fetch_basho.py
uv run python -m banzuke.build   # rebuild processed dataset
```

## Usage

```sh
uv run python predict.py 202609            # predict an upcoming banzuke
uv run python backtest.py --help           # run/inspect the model bake-off
```

## Layout

- `fetch_banzuke.py`, `fetch_basho.py`: incremental scrapers
- `data/banzuke/`, `data/basho/`: raw API responses (committed)
- `data/processed/`: rebuildable parquet dataset (gitignored)
- `banzuke/`: dataset build, features, models, resolver, backtest harness
- `results/`: backtest metric summaries (committed)
- `docs/EXPERIMENTS.md`: experiment log and model-selection rationale
