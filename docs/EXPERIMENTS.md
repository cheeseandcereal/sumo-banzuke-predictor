# Experiment log

Goal: predict the next makuuchi banzuke from the previous basho's
results. Architecture: per-rikishi ordering model + structure resolver
(Y/O membership rules, sanyaku minimums, empirical E/W layout). The
ordering objective is the experiment axis; all contenders share the
same features, resolver, and backtest.

## Protocol

- Rolling-origin backtest: predict each target basho using only
  strictly-prior transitions; models retrained per target.
- Dev window 2004-2023 (42-man makuuchi, post-kosho era).
  Held-out confirmation window: 2024-2026, evaluated once at the end.
- Committee behavior drifts, so summaries are reported for full dev,
  2014+, and 2020+ windows; recent windows carry the most weight in
  model selection.
- Primary metric: exact slots /42 (GTB-style). Secondary: half-rank MAE,
  Kendall tau, juryo-boundary promotion/demotion F1, sanyaku accuracy.
  Paired sign test on exact slots vs the leader; simpler model wins ties.
- Transitions 201101->201105 and 202003->202007 span cancelled basho and
  behave as normal single transitions (the cancelled banzuke were reissued).
- Backtest candidates exclude rikishi absent from the next banzuke
  (retirees, juryo->makushita drops). This is mild structural peeking,
  equivalent to a GTB player knowing announced retirements; it does not
  leak results.

## E1: E/W layout conventions (2026-08, resolver rules)

Derived empirically from all 399 banzuke:

- Maegashira: strict E,W alternation in 389/399; the 10 exceptions are
  mid-banzuke gaps from post-meeting retirements. Encoded as strict.
- Y/O block: strict alternation per class, but an odd-sized O block
  sends its last member to the lighter column of the combined Y+O block
  (e.g. Y1 O3 lays out Y1E / O1E O1W O2W). Majority pattern in every
  (nY, nO) combination observed.
- Extra S/K slot (odd count): goes to the lighter column counting the
  blocks above it: 56/62 for S, 35/38 for K (~90%).
- Numbering counts per side (E,W,W -> 1E, 1W, 2W).

Impact (2023 smoke test, model A): exact slots 8.8/42 -> 12.5/42. A
single structure error shifts every slot below it, so layout and
sanyaku-count correctness dominate exact-slot scoring.

## E2: forced sekiwake for 11+ win komusubi (2026-08, resolver rule)

202309 case: Kotonowaka (K1E, 11-4) received a created S2E slot while
both S incumbents kept kachi-koshi. With rigid max(2, forced) counts the
whole maegashira block shifted one position (~30 slots lost). Komusubi
with 11+ wins have historically always been promoted, so this is
encoded as a forced slot. Remaining sanyaku-count fuzziness (extra K
slots, strong M1 cases) left to residual analysis.

## E3: main bake-off (2026-08, running)

Contenders, all sharing FEATURES (rank, results, prizes, momentum
history, kadoban/ozeki-run flags, era):

| ID | Architecture | Hypothesis tested |
|----|--------------|-------------------|
| R  | per-zone fitted movement formula (last 60 basho) | mostly mechanical |
| L  | ridge regression on delta | linear is enough |
| A  | LightGBM regression on delta (L2) | zone-aware nonlinear, marginal signal |
| Aq | LightGBM regression, L1 objective | L2 shrinkage hurts big movers |
| Aw | A + exponential recency weights (half-life 60 basho) | committee drift matters in training |
| B  | LightGBM LambdaRank on next-basho order | relative/resolution signal matters |
| C  | pairwise classifier on nearby pairs (feature diffs), local Bradley-Terry aggregation | pair-asymmetric resolution bias matters |

Results: pending.
