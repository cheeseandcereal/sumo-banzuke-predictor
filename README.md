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
promotion rules and order, make-koshi sanyaku exits, sanyaku minimums,
empirically-derived E/W layout, no promotion after make-koshi for S/K/M).
`docs/MODEL.md` walks through that pipeline stage by stage (every
dataset column, both model stages, the resolver's rules with their
precedent, what `banzuke predict` prints); `docs/EXPERIMENTS.md` is the full
experiment log and selection rationale.

The model trains on historical transitions between consecutive basho,
using rank, win-loss results, prizes, recent form, career context, and
era features to predict each wrestler's movement. Evaluation uses a
rolling-origin backtest that retrains on strictly earlier tournaments,
so each prediction only uses information available at the time.

Backtested performance (202001-202609, 40 basho, two bag replicates):
**21.1 of 42 slots exactly right** per basho on average, **83% of
wrestlers placed within one position**, mean absolute error 0.75
half-ranks, juryo promotion/demotion F1 0.92/0.94, GTB score 49
points/basho (2 per exact slot, 1 per right-rank-wrong-side).

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
under `data/`, including the reranker's out-of-fold training scores, so
a clone is ready to use without fetching, rebuilding or any one-off
computation.
If you find the API useful, consider [supporting its server
costs](https://ko-fi.com/sumoapi).

## Usage

The project is a Python package with one command, `banzuke`. From a
clone:

```sh
uv sync                      # creates .venv with the package installed
uv run banzuke --help        # every command; `banzuke COMMAND --help` for its options
```

(`uv run banzuke ...` throughout; with the virtualenv activated, plain
`banzuke ...` works.)

Predict the upcoming banzuke from the latest fetched results:

```sh
uv run banzuke predict
uv run banzuke predict --retired Shikona1,Shikona2   # announced retirees
uv run banzuke predict --protected Shikona           # a full absence the JSA exempted (rank frozen)
```

Disagree with a placement? Overrides constrain the assignment while the
model fills everything else (scores are computed once; there is no
conditional re-inference):

```sh
uv run banzuke predict --above "Takayasu > Hakunofuji"  # A rises above B
uv run banzuke predict --below "Oho < Ura"              # A drops below B
uv run banzuke predict --class Aonishiki=O --count S=3  # structural beliefs
uv run banzuke predict --pin Wakatakakage=M8E           # exact cell
uv run banzuke predict --interactive                    # train once, iterate
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
exact 60% of the time, `?` 35%, `??` 25%; `banzuke analyze` prints the
calibration.

Fetch available data and rebuild the processed datasets after a banzuke
release or completed basho (completed tournaments are skipped). The
build ends with the reranker's out-of-fold table, `oof.parquet` (about
two minutes with many workers; a no-op when the committed one still
matches; `--skip-oof` leaves it alone); commit it with the other files:

```sh
uv run banzuke data update        # fetch what is new, then rebuild
uv run banzuke data fetch         # update the raw JSON only
uv run banzuke data build         # rebuild from the existing JSON only
```

Re-run the model bake-off / evaluation:

```sh
uv run banzuke backtest --out cache/all       # 2004 through latest basho
uv run banzuke backtest --end 202311          # original dev window only
uv run banzuke analyze --model Ar             # residual analysis
uv run banzuke conventions                    # resolver rules vs committee history
uv run banzuke explain build --start 201901   # cache the frames the research tools read
uv run banzuke explain sheet 202309           # one historical forecast next to the real banzuke
uv run banzuke precedent landing "K1 5-10"    # where did the committee put that record
```

The resolver's hard-coded conventions (yokozuna never demoted, Y/O order
by wins, make-koshi sekiwake and komusubi exits, the demoted ozeki at the
bottom of the sekiwake, sanyaku minimums, forced claims, make-koshi
ceiling, E/W layout, twin order) are empirical regularities read off
history once. `banzuke conventions` recounts each over the full record
and the last 60/30 basho and names its last violation, so a committee
that changes its habits shows up in the table after the next data update
rather than silently costing slots; `banzuke analyze` prints the same
table.

Parameter experiments need no code changes: `--set base.n_estimators=600`
overrides a LightGBM parameter of the movement (`base.*`) or pair
(`pair.*`) stage or a model option, and `--baseline Ar` pairs every
result against the defaults per basho (mean difference, bootstrap
interval, Wilcoxon p). Results are cached per (configuration, model,
seed, basho) in `cache/`; the cache key hashes the processed
dataset, all result-affecting source files and the configuration, so
edits, data rebuilds and parameter changes invalidate it automatically
(`--fresh` to force). Model training itself is never persisted:
`banzuke predict` retrains on every invocation (about 30 s; `--seeds 1`
for the fastest run). The one training input derived from a model fit,
the reranker's out-of-fold base scores, is committed as
`data/processed/oof.parquet` with the key of the data and base stage it
came from; a command that finds it stale rebuilds it in place (about two
minutes), and a configuration with a different base stage (`--seeds`,
`--set base.*`, `--train-start`) computes its own under `cache/oof/`.

`banzuke predict` and `banzuke backtest` accept `--train-start BASHO` to
restrict training to newer transitions. Tested and neutral-to-worse
(docs/EXPERIMENTS.md E9): the era features already let the models
specialize to the modern regime, so full history remains the default.

`uv run pytest -q` runs the regression tests (a few seconds).

## Layout

- `src/banzuke/`: the package
  - `cli/`: the `banzuke` command, one module per subcommand (options and
    printing only)
  - `paths.py`, `ranks.py`: where the files are; the rank vocabulary
    (class ordinals, cell and record formatting, basho calendar)
  - `fetch.py`, `build.py`, `features.py`: raw API fetcher, dataset
    build, history features and targets
  - `models.py`, `resolver.py`, `overrides.py`, `confidence.py`,
    `forecast.py`: ordering models, the structural resolver, human
    overrides, confidence markers, forecast assembly
  - `harness.py`, `metrics.py`, `analysis.py`, `conventions.py`:
    backtest harness and cache, scoring, residual analysis, convention
    audit
  - `experiments/`: tools for finding and testing committee conventions
    without retraining (`explain`, `precedent`, `rules`)
- `data/banzuke/`, `data/basho/`: committed raw API responses
- `data/processed/`: committed, reproducible Parquet datasets and the
  default model's out-of-fold table
- `tests/`: regression tests (library invariants, command line)
- `cache/`: git-ignored; every regenerable file (keyed backtest, OOF and
  explain caches, per-run dumps, `--out` reports) lands here
- `docs/MODEL.md`: how the default model works, stage by stage
- `docs/EXPERIMENTS.md`: experiment log and model-selection rationale
