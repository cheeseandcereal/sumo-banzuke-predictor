# Sumo Banzuke Predictor

Predicts the next makuuchi banzuke (grand sumo top-division rankings)
from the results of the previous basho.

Multiple ordering models (rules formula, linear, gradient-boosted
regression/ranking/pairwise variants) competed on a shared
rolling-origin backtest covering every banzuke transition since 1959.
The promoted default, `Ar`, is a seed-bagged L1 gradient-boosted
movement model whose near-tie clusters are reordered by a pairwise
classifier that learned the committee's conflict-resolution habits
(identical-record sanyaku E/W pairs move as one unit),
feeding a resolver that applies the near-inviolable structure (Y/O
conventions, sanyaku minimums, empirically-derived E/W layout, no
promotion after make-koshi for S/K/M). See `docs/EXPERIMENTS.md` for
the full experiment log and selection rationale.

The model trains on historical transitions between consecutive basho,
using rank, win-loss results, prizes, recent form, career context, and
era features to predict each wrestler's movement. Evaluation uses a
rolling-origin backtest that retrains on strictly earlier tournaments,
so each prediction only uses information available at the time.

Backtested performance (202001-202609, 40 basho): **17.9 of 42 slots
exactly right** per basho on average (+-1.0 standard error), **80% of
wrestlers placed within one position**, mean absolute error 0.94
half-ranks, juryo promotion/demotion F1 0.91/0.88, GTB score 43
points/basho (2 per exact slot, 1 per right-rank-wrong-side). Over the
2004-2019 window used for tuning it scores 20.3 exact slots; the 2020s
are harder for every model tried. These figures were used to pick the
current configuration, so the first untouched test is the 202611
banzuke.

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

A marker column before each name says where to look: `!` occupies an
S/K slot the rules created for a promotion claim, `?` low and `??` very
low confidence (scored from tight ordering calls the model made itself
and from seed disagreement), `~` a big move whose landing spot is noisy. A
`review` list under the sheet names the decisions behind them, least
confident first, with the override that tests the alternative and, for
created slots, how often the committee honoured such claims and how many
cells shift without one. Backtested on 2024-2026: unmarked cells are
exact 60% of the time, `?` 35%, `??` 25%; `uv run python analyze.py`
prints the calibration.

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
uv run python -m banzuke.conventions           # resolver rules vs committee history
uv run python -m experiments.explain sheet 202309   # one historical forecast next to the real banzuke (after `build`)
uv run python -m experiments.precedent landing "K1 5-10"   # where did the committee put that record
```

The resolver's hard-coded conventions (yokozuna never demoted, sanyaku
minimums, forced claims, make-koshi ceiling, E/W layout, twin order) are
empirical regularities read off history once. `banzuke.conventions`
recounts each over the full record and the last 60/30 basho and names
its last violation, so a committee that changes its habits shows up
in the table after the next data update rather than silently costing
slots; `analyze.py` prints the same table.

Parameter experiments need no code changes: `--set base.n_estimators=600`
overrides a LightGBM parameter of the movement (`base.*`) or pair
(`pair.*`) stage or a model option, and `--baseline Ar` pairs every
result against the defaults per basho (mean difference, bootstrap
interval, Wilcoxon p). Results are cached per (configuration, model,
seed, basho) in `results/scratch/`; the cache key hashes the processed
dataset, all result-affecting source files and the configuration, so
edits, data rebuilds and parameter changes invalidate it automatically
(`--fresh` to force). Model training itself is never persisted:
`predict.py` retrains on every invocation (about 30 s; `--seeds 1` for
the fastest run), plus a one-off two minutes after each data update for
the reranker's out-of-fold training scores.

`predict.py` and `backtest.py` accept `--train-start BASHO` to restrict
training to newer transitions. Tested and neutral-to-worse (docs/EXPERIMENTS.md E9):
the era features already let the models specialize to the modern
regime, so full history remains the default.

`uv run pytest -q` runs the regression tests (a few seconds).

## Layout

- `update_data.py`: incremental fetcher and processed-data builder
- `data/banzuke/`, `data/basho/`: committed raw API responses
- `data/processed/`: committed, reproducible Parquet datasets
- `banzuke/`: dataset build, features, models, resolver, backtest harness
- `predict.py`, `backtest.py`, `analyze.py`: CLIs
- `tests/`: regression tests
- `results/`: generated backtest reports and caches
- `docs/EXPERIMENTS.md`: experiment log and model-selection rationale
