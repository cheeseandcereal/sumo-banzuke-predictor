# Experiment log

Goal: predict the next makuuchi banzuke from the previous basho's
results. Architecture: per-rikishi ordering model + structure resolver
(Y/O membership rules, sanyaku minimums, empirical E/W layout, S/K/M
make-koshi ceiling). The ordering objective is the experiment axis;
all contenders share the same features, resolver, and backtest.

## Protocol

- Rolling-origin backtest: predict each target basho using only
  strictly-prior transitions; models retrained per target.
- Dev window 2004-2023 (42-man makuuchi, post-kosho era).
  Held-out confirmation window: 2024-2026, evaluated once at the end.
- Committee behavior drifts, so summaries are reported for full dev,
  2014+, and 2020+ windows; recent windows carry the most weight in
  model selection.
- Primary metric: exact slots /42 (GTB-style). Secondary: half-rank MAE,
  within-1-position fraction, GTB points (2x exact + 1x right rank
  wrong side), Kendall tau, juryo-boundary promotion/demotion F1,
  sanyaku accuracy. Paired sign test on exact slots vs the leader;
  simpler model wins ties. Note the models themselves train on
  placement distance (L1 on movement delta), never on exact slots;
  exact is only the selection criterion.
- Transitions 201101->201105 and 202003->202007 span cancelled basho and
  behave as normal single transitions (the cancelled banzuke were reissued).
- Backtest candidates exclude rikishi absent from the next banzuke
  (retirees, juryo->makushita drops). This is mild structural peeking,
  equivalent to a GTB player knowing announced retirements; it does not
  leak results.

## Protocol v2 (2026-09, parameter tuning round)

Written before the sweeps below were run. Motivation: a seed study of
the E10 configuration (5 seeds, 118 dev basho) found a seed-to-seed
SD of the 118-basho mean of 0.14 exact slots, a per-basho paired SD
between two configurations of 2.2 slots (the same as pure seed noise:
the resolver turns tiny score changes into block shifts), and hence a
paired SE of ~0.25 slots on 95 basho. Every hyperparameter effect
measured so far is 0.2-0.5 slots. Exact slots cannot resolve such
effects on the data that exists; half-rank MAE can (paired SE ~0.01
against effects of 0.02-0.03), and across configurations its mean
change correlates -0.96 with the change in exact slots.

- Every basho through 202609 has been seen by some earlier decision
  (E6/E9/E10 reused the 2024+ holdout). Split: **screen** 2004-2019
  (95 targets), **confirm** 2020-2026 (40 targets). The confirm window
  is evaluated once per experiment block and never iterated on; the
  first untouched test is the 202611 banzuke.
- Baseline: `Ar`, explicit seed 0, 600/400 rounds, everything else as
  in E10. LightGBM's unset seed (used before this round) is a different
  model; backtest and predict.py now share one seeded configuration.
- Information policy: the forecast is sized like the previous banzuke
  (`mak_size_policy=prior`; constant 42 since 2004, -0.02 slots vs
  reading the target's size). Candidate exclusion of announced
  departures is unchanged.
- Seeds are averaged within a basho before any comparison; the paired
  unit is the basho, never rows, pairs or seeds. Bagged configurations
  are compared as two disjoint bag replicates (seeds 0-4, 5-9).
- Decision metric: paired **MAE** (Wilcoxon signed-rank, 6-basho
  block-bootstrap 95% CI). Secondary: GTB points (per-basho correlation
  with exact slots .93). Exact slots remain the headline and a
  guardrail, not the selector. Reranker experiments also report
  within-cluster pair accuracy against the prior-order baseline.
- Search: staged coordinate sweeps of ~40 configurations in small
  families, each a one-dimensional decision; at most 4 nested finalist
  bundles go to the confirm window. Losers are listed.
- Acceptance: confirm dMAE < 0 with CI upper bound < +0.01; screen
  dMAE < -0.01 with the same sign on both screen halves (2004-2011 /
  2012-2019); guardrails dExact >= -0.2, promotion/demotion F1 and
  sanyaku-set exactness >= -0.01. Ties go to the simpler configuration.
  p-values after a search are exploratory and reported as such.
- Reproduction: every row below has a `backtest.py --set` command;
  per-basho CSVs live in the git-ignored `results/scratch/`.

### Baseline under protocol v2 (Ar, seeds 0-4 averaged)

| window | n | exact/42 | seed SD | basho SE | MAE | seed SD | GTB | promo/demo F1 | sanyaku sets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| screen 2004-2019 | 95 | 19.19 | 0.10 | 0.45 | 0.920 | 0.008 | 44.6 | .934/.900 | .60 |
| confirm 2020-2026 | 40 | 16.55 | 0.22 | 0.81 | 1.030 | 0.009 | 40.6 | .919/.873 | .49 |
| 2024-2026 (old holdout) | 17 | 18.41 | 0.54 | 1.23 | 0.834 | 0.020 | 44.3 | .953/.895 | .44 |

`uv run python backtest.py --models Ar --seeds 0-4 --start 200401`.
The E10 headline (18.2 on 17 basho, one unset-seed draw) sits inside
this distribution. 2020+ is genuinely harder: the committee created an
extra S/K slot in 37% of basho since 2020 vs 18% over 2004-2019.

### Where the misses are (analyze.py, Ar seeds 0-2, 2004-2026)

- Of 5,662 makuuchi slots, 44.3% exact; misses: 27.6% pure E/W flips,
  33.3% off by one position, 39.1% farther.
- Sanyaku block-size errors in 16 of 135 basho (22.5% since 2020),
  costing 6.8 exact slots each: 0.81 slots/basho over the window, 1.53
  since 2020 (an upper bound on a perfect count model). 9 are
  over-creation, all komusubi, six of them M1 8-7/9-6 claims the
  committee declined when the zone was full; 7 are under-creation, three
  of them make-koshi S incumbents kept in sanyaku, which no win
  threshold reaches. Lowering the claim thresholds one win would fix 5
  cases and create 17 new errors: not warranted.
- Within predicted S/K blocks whose membership is right, 40 of 184 are
  misordered (86 slots, 0.64/basho); 11 of those are forced claimants
  placed ahead of a higher-scored fill by the resolver (22 slots, the
  ceiling on that ordering rule).
- Seed spread is informative: slots where three seeds agree (71%) are
  exact 51% of the time, spread 1 (22%) 31%, spread 2 (5.5%) 19%
  (Spearman rho with |error| .26). It is a sensitivity signal, not an
  interval.

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

## E3: main bake-off (2026-08)

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

Exact slots /42, dev window (118 basho): Aq 16.75, A 16.18 (p=.12),
B 15.68 (p=.009), Aw 15.58 (p=.016), R 13.09, L 11.71, C 9.82.
2020+ slice: Aq 12.91, B 12.70, Aw 12.04, A 11.87 (p=.001).

Findings: L1 beats L2 as hypothesized, and the gap widens in the
chaotic recent era. LambdaRank is competitive on ordering but much
worse at the juryo boundary (promo F1 .72 vs .93): its scores have no
positional anchor. Recency-weighted training does not pay for its
reduced effective sample (the era features already carry drift). C's
current-position anchoring cannot express big moves; standalone it is
the worst contender, but see E4 for where its machinery works.

## E4: residual analysis -> boundary_dist + local rerank Ar

Residual analysis of Aq (analyze.py): 28% of misses were pure E/W
flips, 34% within one position; juryo promotees under-promoted by
+1.10 positions; in model-inverted adjacent pairs the committee
favored the climber (prior order held only 44% vs 63% baseline), and
with tied wins prior order held 94%. So: near-tie resolution is
learnable signal, and the model was too anchored, not too little.

Changes:
- boundary_dist feature (signed distance to the makuuchi/juryo line)
- Ar = Aq global order + pairwise classifier reranking only within
  near-tie clusters (consecutive base scores within GAP=0.5, clusters
  capped at 4). GAP=1.5 was tried first and clustered 80% of pairs,
  degrading results; 0.5 clusters ~39%.

Dev: Ar 18.07 vs Aq 16.96 (sign 32-72, p<.001). Promotee bias fell to
+0.78, big-winner bias +0.29 -> +0.06, sanyaku-zone exact .65 -> .72.

## E5: forced komusubi claims

All 10 dev-window sanyaku-count misses were under-created slots for
strong upper maegashira. Empirical claim precision (1990+):
M1 8+ wins .85-.90, M2 11+ 1.00, M3 10+ .91, M4 12+ 1.00, M5 13+ 1.00;
stable across eras. Encoded as forced K claims. M1 12+ jumping straight
to S was left out (2 of 4 historically).

Dev: sanyaku_exact .55 -> .60, 2020+ exact_n 13.74 -> 14.13 (Ar);
full-window exact_n flat (18.07 -> 18.01), consistent with the
committee creating slots more liberally in recent years. Kept per the
recency-weighted selection rule.

## E6: held-out confirmation 2024-2026 (one shot)

16 basho, all contenders, evaluated once: Ar 17.13 exact (40.8%),
MAE 0.854, promo F1 .96; A 16.69, Aq 16.63, Aw 15.81, B 15.44,
R 15.44, L 9.44, C 8.31. Sign tests are underpowered at n=16, but Ar
leads every ordering metric and the dev-window ordering reproduces.

**Decision: Ar promoted as the default predictor** (predict.py).
Runner-up Aq kept as the simpler fallback.

## E7: head-to-head bout results in the reranker

Hypothesis (sumo-fan folklore): the committee breaks near-ties by who
beat whom during the basho. Bout winners extracted from the record
arrays (191k bouts, fusen excluded) and appended as a pair feature
(+1/-1/0) in Ar's rerank stage (model Ah).

Result: null. Full dev 18.01 -> 17.97 (sign 19-18, p=1.0);
2014+ 17.42 -> 17.51 (p=.36); 2020+ 14.13 -> 13.91. Either the
committee does not actually use it, or the effect is already carried
by the wins/prior-rank features. Ar remains the default; Ah kept
in-tree for future re-evaluation.

## E8: joi / strength-of-schedule awareness

Hypothesis: the committee favors the joi (the top ~16 who share the
toughest schedule; its boundary shifts with absences), and an 8-7
against the joi schedule outranks an 8-7 against mid-maegashira. A
static in-joi flag would be redundant (trees already split on
position), so the realized schedule was computed from the 191k bout
records: mean opponent position, joi opponents faced, wins over joi
opponents, and kinboshi (maegashira defeating a yokozuna). Added to
FEATURES for both the base model and the rerank pair diffs.

Result: null. Aq full 16.60 -> 16.62 (sign 54-54); Ar full
18.01 -> 17.71 (50-54, p=.77), 2014+ 17.42 -> 17.56 (p=.41),
2020+ 14.13 -> 13.74. Joi membership is almost perfectly implied by
rank position, which the model already has; the dynamic residue is too
rare to move a 42-slot metric. Columns remain in the dataset
(SCHEDULE_FEATURES) but are excluded from model inputs.

## Benchmark calibration: human "Guess the Banzuke" players

Verified from dichne.com (2026-08). GTB "bullseye" = our exact-slot
metric; a "hit" is the right rank on the wrong side. Numbers:

- all-time top-10 GTB players average 25-27 bullseyes per basho
- the per-basho winning entry (best of ~470 correlated entries, a
  strong selection effect) lands roughly 33-36 in predictable basho
- this model's held-out average is 17.1, with 6.5 E/W flips and 7.8
  off-by-one misses per basho; resolving every near-tie perfectly
  would yield ~32/42

So the gap to the best individual humans is ~8 slots and lives almost
entirely in near-tie resolution and sanyaku-count calls. Note also
that GTB entries close ~4 weeks after the basho: players see announced
yokozuna/ozeki promotions, retirements, and the shin-juryo
announcement (which pins boundary exchange counts); the backtest gets
none of these.

## E9: hard training cutoff at 2010 (--train-start)

Hypothesis: old-regime transitions (1959+) mislead the model about the
modern committee, so train only on 2010+ (~6.5k rows vs 25k). This is
the hard-cutoff version of E3's Aw (recency weights), which already
lost. Result, paired on identical targets vs full-history training:

- dev 2014-2023: Aq 16.41 -> 15.73 (sign 24-32), Ar 17.42 -> 16.69
  (18-28, p=.18); MAE worsens 0.98 -> ~1.09 for both
- holdout 2024+: Aq 16.62 -> 15.94; Ar 17.12 -> 17.38 (sign 6-6,
  noise) with MAE still worse (0.854 -> 0.885)

Neutral-to-worse everywhere that has statistical power: the era
features (year, kosho, mak_size) already let trees specialize to the
modern regime, so a cutoff only shrinks the sample; rare big-move
patterns thin out first. Full history stays the default;
`--train-start` remains available on backtest.py and predict.py for
future re-testing (the result cache keys on it).

## E10: no make-koshi promotion for S/K/M (2026-09)

Hypothesis: enforcing no promotion after make-koshi improves slotting.
Trigger: Takayasu (M2E, 7-8 in 202609) predicted at M1W. Historical
audit found no S/K exceptions since 2004 and only two M exceptions,
both in 201107; Y/O within-class rises still occur after losing records.

Change: S/K/M without KK cannot rise above their current rank label,
including west->east. Retention is allowed; whole-banzuke position can
still improve when sanyaku shrinks. Enforced in class/slot assignment.

Paired on identical Ar scores:
- dev 2004-2023 (118 basho): exact 18.01 -> 18.58 (27-5, p=.00011),
  MAE .993 -> .961
- dev 2020-2023: exact 14.13 -> 15.26 (11-0, p=.00098)
- confirmation 202401-202609 (17 basho, reused holdout plus one target):
  exact 17.35 -> 18.18 (7-0, p=.016), MAE .854 -> .829

Non-KK promotions 109->0. Full-dev promotion/demotion F1 dips
.932/.900 -> .930/.893, chiefly from the 201107 exception; accuracy
still improves excluding it. Confirmation boundary F1 unchanged.
**Kept in the resolver**: consistent gains, strongest recently.
Takayasu now lands at M2E for 202611, whose actual banzuke is not yet
available.

## Overrides (predict.py, 2026-08)

Human-in-the-loop overrides constrain the assignment, not the model:
scores are computed once; `--above`/`--below` splice the model's
ordering (the named wrestler moves the minimal distance to clear its
target, applied sequentially, no-op if already satisfied); `--class`,
`--count`, and `--pin` steer the resolver's structural stage. There is
no conditional re-inference: the model does not revise its opinion of
C because you moved A above B; the sheet shifts structurally around
the override.

Precedence: overrides outrank the resolver's near-inviolable
conventions, and each broken convention is reported (demoted yokozuna,
retained kadoban ozeki, dropped kachi-koshi incumbent, cancelled rule
promotion). A relation that the ordering satisfies can still be
inverted in the final labels by a forced convention (e.g. the
forced-komusubi claim); this is reported as VIOLATED rather than
silently overriding the convention. The resolver refactor that made
slots addressable (block_slots) reproduced the committed holdout
results exactly, 144/144 (model, basho) rows.

## Known limitations / future leads

- Juryo promotee placement still +0.78 under-promoted; bottom-of-sheet
  slotting is genuinely noisy ("banzuke luck").
- Ozeki/yokozuna promotion thresholds are hardcoded conventions in the
  resolver; borderline cases (32-win runs with a yusho) surface in
  predict.py notes rather than being decided statistically.
- COVID-era kadoban exemptions (Mitakeumi 2022) are not modeled.
- Sanyaku count is right in ~89% of basho; when wrong it still costs
  ~7 slots. A learned count model is the next candidate experiment.
- Untried ideas for the near-tie gap: per-slot majority ensembling
  across models/seeds; committee-regime features (banzuke committee
  membership changes); for live use, feeding announced Y/O promotions
  and shin-juryo counts into the resolver as constraints (information
  GTB players have); scraping GTB archives for a paired per-basho
  model-vs-human comparison on identical targets.
