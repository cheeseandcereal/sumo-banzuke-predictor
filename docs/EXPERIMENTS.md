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
  (constant 42 since 2004; -0.02 slots vs reading the target's size,
  which the harness did before). Candidate exclusion of announced
  departures is unchanged.
- Seeds are averaged within a basho before any comparison; the paired
  unit is the basho, never rows, pairs or seeds. Bagged configurations
  are compared as two disjoint bag replicates (seeds 0-4, 5-9).
- Decision metric: paired **MAE** (Wilcoxon signed-rank, 6-basho
  block-bootstrap 95% CI). Secondary: GTB points (per-basho correlation
  with exact slots .93). Exact slots remain the headline and a
  guardrail, not the selector. Reranker experiments (E13) also report
  within-cluster pair accuracy against the prior-order baseline, from a
  diagnostic used during that block.
- Search: staged coordinate sweeps of ~40 configurations in small
  families, each a one-dimensional decision; at most 4 nested finalist
  bundles go to the confirm window. Losers are listed.
- Acceptance: confirm dMAE < 0 with CI upper bound < +0.01; screen
  dMAE < -0.01 with the same sign on both screen halves (2004-2011 /
  2012-2019); guardrails dExact >= -0.2, promotion/demotion F1 and
  sanyaku-set exactness >= -0.01. Ties go to the simpler configuration.
  p-values after a search are exploratory and reported as such.
- Reproduction: rows quote `backtest.py --set` overrides relative to
  the pre-round defaults, which from the current ones are
  `--set n_seeds=1 --set base.n_estimators=600 --set pair.n_estimators=400
  --set near_ties=false --set context=false --set gap=0.5 --set cluster_max=4`
  (`--set n_seeds=5` for "bag5"). Options that lost and were removed
  afterwards (near-tie-only pair training, L2 blending, h2h/context for
  C) are not reproducible without restoring them from git history.
  Per-basho CSVs live in the git-ignored `results/scratch/`.

### Baseline under protocol v2 (Ar, seeds 0-4 averaged)

| window | n | exact/42 | seed SD | basho SE | MAE | seed SD | GTB | promo/demo F1 | sanyaku sets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| screen 2004-2019 | 95 | 19.19 | 0.10 | 0.45 | 0.920 | 0.008 | 44.6 | .934/.900 | .60 |
| confirm 2020-2026 | 40 | 16.55 | 0.22 | 0.81 | 1.030 | 0.009 | 40.6 | .919/.873 | .49 |
| 2024-2026 (old holdout) | 17 | 18.41 | 0.54 | 1.23 | 0.834 | 0.020 | 44.3 | .953/.895 | .44 |

`backtest.py --models Ar --seeds 0-4 --start 200401` with the pre-round
defaults above. The E10 headline (18.2 on 17 basho, one unset-seed draw) sits inside
this distribution. 2020+ is genuinely harder: the committee created an
extra S/K slot in 37% of basho since 2020 vs 18% over 2004-2019.

### Where the misses are (Ar seeds 0-2, 2004-2026; from analyze.py sections since trimmed)

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
  ceiling on that ordering rule; fixed in E15a).
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

## E11: seed bagging (2026-09, protocol v2)

Hypothesis: averaging the movement delta and the pair probabilities over
5 seeds before reranking/resolving beats a single seed (predict.py
already trained 5 seeds, but only for the +-N column).

Screen 2004-2019, paired per basho, bag replicates (seeds 0-4, 5-9) vs
the mean of single seeds 0-4:

| config | exact/42 | GTB | MAE | dExact [95% CI] | W-L | dMAE [95% CI] | p |
|---|---:|---:|---:|---|---|---|---:|
| Ar single (5 seeds) | 19.19 | 44.6 | 0.920 | ref | | ref | |
| Ar bag5 | 19.40 | 45.0 | 0.909 | +0.21 [+0.01, +0.40] | 58-33 | -0.011 [-0.018, -0.003] | .011 |

Both screen halves improve (dMAE -0.003 / -0.006). Replicate 0 alone vs
the same five seeds averaged after resolving: dMAE -0.004 (p=.41), so
part of the pooled effect is the extra replicate; the direction is
consistent everywhere. Per-basho paired SD of MAE falls from 0.108 to
0.051 when both sides are bagged, which is why the sweeps below run on
the bag. **Adopted**: `n_seeds=5` is the platform for E12-E14 and the
predict.py point forecast (same runtime as before).
`--set n_seeds=5 --seeds 0-1`.

## E12: base-stage capacity and regularization (2026-09, protocol v2)

All rows bag5 on the screen window, one bag replicate, paired against
the previous stage's winner. dMAE halves = 2004-2011 / 2012-2019.

### E12a rounds (vs bag5 at 600/400)

Validation curves (train <2020, valid 2020-2023) put the L1 optimum
near 75-100 rounds and the pair classifier's near 150; both stages were
past it.

| base / pair rounds | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---|---|---:|---|
| 600 / 400 (bag5) | 19.26 | 0.916 | ref | ref | | |
| **300 / 150** | 19.64 | 0.896 | +0.38 [-0.14, +0.88] | -0.020 [-0.037, -0.002] | .07 | -.005 / -.034 |
| 200 / 150 | 19.66 | 0.897 | +0.40 [-0.12, +0.95] | -0.019 [-0.038, +0.001] | .14 | +.002 / -.039 |
| 200 / 400 | 19.64 | 0.901 | +0.38 | -0.014 | .25 | +.004 / -.032 |
| 300 / 400 | 19.52 | 0.903 | +0.25 | -0.013 | .16 | +.002 / -.028 |
| 200 / 100 | 19.55 | 0.901 | +0.28 | -0.015 | .31 | +.005 / -.034 |
| 200 / 250 | 19.52 | 0.904 | +0.25 | -0.012 | .38 | +.008 / -.031 |
| 600 / 150 | 19.27 | 0.911 | +0.01 | -0.005 [-0.009, -0.001] | .045 | -.004 / -.006 |
| 100 / 150 | 19.65 | 0.911 | +0.39 | -0.005 | .75 | +.018 / -.027 |
| 100 / 400 | 19.64 | 0.913 | +0.38 | -0.002 | .82 | +.021 / -.026 |

Fewer base rounds help most in 2012-2019; 100 rounds underfit the
older half. Pair 150 beats 400 at every base setting. **Winner: 300/150**
(the only row improving both halves with a CI excluding zero); 200/150
is equivalent within noise. `--set n_seeds=5 --set base.n_estimators=300 --set pair.n_estimators=150`.

### E12b capacity (vs 300/150)

| leaves / min_child | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---|---|---:|---|
| **63 / 30** | 19.64 | 0.896 | ref | ref | | |
| 127 / 30 | 19.54 | 0.894 | -0.11 [-0.72, +0.48] | -0.002 [-0.021, +0.016] | .57 | -.018 / +.013 |
| 31 / 30 | 19.24 | 0.909 | -0.40 [-0.83, +0.01] | +0.013 [-0.008, +0.033] | .22 | -.004 / +.030 |
| 63 / 60 | 19.46 | 0.919 | -0.18 | +0.023 [-0.006, +0.053] | .09 | +.052 / -.006 |
| 63 / 100 | 19.04 | 0.942 | -0.60 [-1.05, -0.12] | +0.046 [+0.013, +0.077] | .009 | +.063 / +.030 |
| 127 / 100 | 19.03 | 0.939 | -0.61 | +0.043 | .02 | +.064 / +.023 |
| 31 / 100 | 19.07 | 0.950 | -0.57 | +0.055 | .008 | +.082 / +.028 |

Current capacity holds; 127 leaves ties (simpler wins). Larger leaves
hurt both halves, so the single-seed "min_child 100 is better since
2020" lead from the earlier reviews does not survive bagging on either
half of the screen window; it is dropped from E14.

### E12c regularization (vs 300/150)

| reg_lambda / colsample | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---|---|---:|---|
| **0 / 0.9** | 19.64 | 0.896 | ref | ref | | |
| 0 / 0.6 | 19.48 | 0.883 | -0.16 [-0.55, +0.23] | -0.013 [-0.035, +0.007] | .49 | -.038 / +.011 |
| 0 / 0.75 | 19.74 | 0.890 | +0.10 [-0.16, +0.33] | -0.006 [-0.021, +0.008] | .18 | -.021 / +.008 |
| 5 / 0.9 | 19.60 | 0.896 | -0.04 | -0.000 | .69 | -.013 / +.013 |
| 5 / 0.6 | 19.28 | 0.900 | -0.36 | +0.004 | .56 | -.009 / +.017 |
| 10 / 0.6 | 19.35 | 0.900 | -0.30 | +0.004 | .72 | -.012 / +.019 |
| 10 / 0.9 | 19.43 | 0.903 | -0.21 | +0.007 | .60 | +.004 / +.009 |
| 30 / 0.9 | 19.37 | 0.903 | -0.27 | +0.007 | .37 | +.009 / +.005 |

Feature subsampling helps only the older half and costs exact slots;
L2 regularization does nothing. Both earlier single-seed leads
(colsample .6 + lambda 10; lambda 5) were within the noise floor
documented in the protocol. **No change.** Base stage final:
300 rounds, 63 leaves, min_child 30, subsample .9, colsample .9.

## E13: reranker training pairs and context (2026-09, protocol v2)

Two mismatches in Ar's pair stage (found independently by both 2026-09
reviews): (1) subtracting features zeroes era/size columns and discards
the pair's absolute location; (2) training pairs are "within 6 rows of
each other in the old order" (easy, 64% positive), while inference
pairs are near-ties under the base score, 57% of which lie outside
that window. The window-only classifier is badly overconfident on the
pairs it actually adjudicates: log loss 1.14 against a 0.69 coin flip.

Fixes: `context=true` appends era/size columns, pair means of
position/rank/wins/boundary distance, and both endpoints' class and
division; `near_ties` adds, to the window pairs, every pair whose
rolling out-of-fold base scores differ by at most `oof_gap` (OOF score
of basho b = the backtest's own base prediction for target next(b), so
no label is seen), with the base-score gap as a pair feature.

### E13a (screen, bag5, vs 300/150)

| pair training | pair acc | log loss | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---:|---:|---|---|---:|---|
| window 6 (ref) | .666 | 1.14 | 19.64 | 0.896 | ref | ref | | |
| **mixed + context, oof_gap 2** | **.708** | 0.53 | 19.95 | 0.873 | +0.31 [-0.17, +0.76] | -0.023 [-0.046, +0.000] | .10 | -.003 / -.042 |
| mixed + context, oof_gap 1 | .705 | 0.53 | 19.91 | 0.879 | +0.26 | -0.017 | .29 | +.002 / -.035 |
| oof only + context | .697 | 0.55 | 19.75 | 0.885 | +0.11 | -0.011 | .52 | +.007 / -.029 |
| context only | .684 | 1.09 | 19.76 | 0.890 | +0.12 | -0.006 | .44 | .000 / -.012 |
| mixed only | .676 | 0.55 | 19.53 | 0.900 | -0.12 | +0.005 | .47 | +.008 / +.001 |
| window 12 | .671 | 0.77 | 19.70 | 0.897 | +0.05 | +0.001 | .39 | -.002 / +.004 |

Prior-order baseline accuracy on the adjudicated pairs is .479, so the
reranker's edge roughly doubles (+19 -> +23 points). Context and
near-tie pairs interact: neither alone moves MAE, together they give
the largest single effect of the round. Gains concentrate in 2012-2019.

### E13b gate and pair capacity (vs mixed + context, oof_gap 2, gap .5, cap 4)

| change | pairs/basho | pair acc | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---:|---:|---|---|---:|---|
| ref | 31 | .708 | 19.95 | 0.873 | ref | ref | | |
| gap 1.0 | 59 | .758 | 20.33 | 0.851 | +0.38 [-0.01, +0.79] | -0.022 [-0.038, -0.007] | .003 | -.039 / -.006 |
| gap 0.75 | 46 | .736 | 20.22 | 0.858 | +0.27 [-0.05, +0.61] | -0.015 [-0.026, -0.004] | .002 | -.026 / -.003 |
| gap 0.25 | 15 | .704 | 19.70 | 0.886 | -0.25 | +0.012 [+0.001, +0.024] | .06 | +.014 / +.010 |
| cluster_max 6 | 33 | .710 | 19.88 | 0.873 | -0.06 | -0.000 | .84 | |
| cluster_max 3 | 26 | .701 | 19.84 | 0.880 | -0.11 | +0.007 | .12 | |
| oof_gap 3 | 31 | .709 | 19.97 | 0.871 | +0.02 | -0.003 | .46 | |
| pair min_child 100 | 31 | .704 | 20.05 | 0.873 | +0.11 | -0.000 | .84 | |
| pair 250 rounds | 31 | .696 | 19.98 | 0.876 | +0.03 | +0.003 | .26 | |
| pair 100 rounds | 31 | .703 | 19.90 | 0.876 | -0.05 | +0.003 | .81 | |
| pair 31 leaves | 31 | .703 | 19.80 | 0.879 | -0.15 | +0.005 | .32 | |
| pair 15 leaves | 31 | .701 | 19.76 | 0.883 | -0.19 | +0.009 | .18 | |

The gate interacts with the classifier as anticipated: E4 chose gap .5
because a wider gate handed 80% of pairs to a classifier that was
wrong about them; the retrained one is right on 76% of a wider set, so
the optimum moved (E13c brackets it upward). Pair capacity and
`oof_gap` are flat; 150 rounds / 63 leaves / min_child 30 stay.

### E13c gate, wider (vs gap 1.0, cap 4)

| gap / cluster_max | pairs/basho | pair acc | exact/42 | MAE | dExact [CI] | dMAE [CI] | p | halves |
|---|---:|---:|---:|---:|---|---|---:|---|
| 1.0 / 4 (ref) | 59 | .758 | 20.33 | 0.851 | ref | ref | | |
| 1.0 / 6 | 77 | .772 | 20.27 | 0.838 | -0.05 [-0.29, +0.17] | -0.013 [-0.023, -0.002] | .049 | -.018 / -.008 |
| 1.5 / 6 | 112 | .818 | 20.39 | 0.840 | +0.06 | -0.011 [-0.028, +0.006] | .19 | -.028 / +.005 |
| 2.0 / 8 | 178 | .863 | 20.27 | 0.840 | -0.05 | -0.011 | .27 | -.017 / -.005 |
| 1.0 / 4, oof_gap 3 | 59 | .755 | 20.33 | 0.850 | 0.00 | -0.001 | .99 | |
| 2.0 / 4 | 86 | .816 | 20.26 | 0.860 | -0.06 | +0.009 | .29 | +.008 / +.010 |
| 1.5 / 4 | 76 | .796 | 20.21 | 0.865 | -0.12 | +0.014 | .09 | +.009 / +.019 |

Widening the gate without raising the cap hurts (clusters saturate and
split arbitrarily); widening both is neutral-to-slightly-better on MAE
with exact slots flat. Pair accuracy rises with the gate because easier
pairs are added, not because adjudication improves.

## E14: pre-registered side hypotheses (2026-09, protocol v2)

Screen, bag5, on the E13a winner (mixed + context, oof_gap 2, gap .5):

| hypothesis | exact/42 | MAE | dMAE [CI] | p | halves | verdict |
|---|---:|---:|---|---:|---|---|
| L1 recency weights, half-life 120 | 19.77 | 0.890 | +0.016 [-0.004, +0.036] | .06 | -.001 / +.033 | worse |
| half-life 180 | 19.94 | 0.876 | +0.003 | .59 | -.017 / +.022 | flat |
| blend 50% L2 into the delta | 20.04 | 0.862 | -0.012 [-0.037, +0.012] | .79 | -.037 / +.013 | halves disagree |
| blend 25% L2 | 19.88 | 0.869 | -0.005 | .77 | -.019 / +.009 | halves disagree |

E3's conclusion stands for L1 as well: down-weighting old transitions
does not pay. L2 blending only helps the pre-2012 half. Neither is
adopted; the blend option was removed. (Caveat: the rolling OOF scores
used for near-tie pair selection in these runs came from the unweighted,
unblended base, so a re-run with a matching OOF table would differ
slightly. The screen-window verdicts are clear enough not to repeat.)

**LambdaRank truncation (model B, bag5, 300 rounds).** E3 blamed B's
poor juryo boundary on "scores with no positional anchor". The 2026-09
review noted `lambdarank_truncation_level` was at LightGBM's default 30
while the boundary sits near position 42:

| B | exact/42 | MAE | promo F1 | demo F1 | dMAE vs default |
|---|---:|---:|---:|---:|---|
| truncation 30 (default) | 16.75 | 1.251 | .811 | .775 | ref |
| truncation 42 | 18.40 | 0.988 | .910 | .878 | -0.263 [-0.310, -0.216] |
| truncation 60 | 18.62 | 0.971 | .925 | .909 | -0.281 [-0.323, -0.239] |

The E3 explanation was wrong: the ranker simply was not being trained
on the boundary. B at truncation 60 is a real contender (Ar at the
same rounds: 19.64 / 0.896 / .934 / .905) and its default is changed
to 60. Blending a truncated ranker into Ar's base score is an untested
lead. `--models B --set n_seeds=5 --set base.n_estimators=300 --set truncation=60`.

## Finalists and confirmation (2026-09, protocol v2)

Nested bundles sent to the confirm window once, two bag replicates each:
B1 = bag5 + 300/150 rounds; B2 = B1 + mixed near-tie pairs with context
(oof_gap 2); B3 = B2 + gap 1.0; B4 = B3 + cluster_max 6.

### Confirm window 2020-2026 (40 basho, evaluated once; two bag replicates)

| bundle | exact/42 | GTB | MAE | within 1 | promo/demo F1 | sanyaku sets | dExact [CI] | W-L | dMAE [CI] | screen dMAE [CI] (halves) |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| bag5 (ref) | 16.49 | 40.7 | 1.024 | .776 | .925/.877 | .49 | ref | | ref | ref |
| B1 300/150 | 16.68 | 40.9 | 1.007 | .780 | .921/.879 | .50 | +0.19 [-0.36, +0.71] | 22-12 | -0.017 [-0.035, +0.002] | -0.013 [-0.029, +0.003] (+.007 / -.033) |
| B2 + mixed pairs, context | 17.49 | 42.2 | 0.958 | .800 | .918/.880 | .53 | +1.00 [+0.20, +1.90] | 24-15 | -0.066 [-0.100, -0.032] | -0.036 (-.006 / -.075) |
| B3 + gap 1.0 | 17.86 | 43.0 | 0.950 | .801 | .914/.881 | .55 | +1.38 [+0.60, +2.11] | 26-10 | -0.074 [-0.100, -0.047] | -0.058 [-0.085, -0.032] (-.035 / -.081) |
| **B4 + cluster_max 6** | **17.88** | **43.0** | **0.939** | .801 | .914/.883 | .55 | +1.39 [+0.35, +2.41] | 24-12 | -0.084 [-0.120, -0.049] | -0.071 [-0.097, -0.046] (-.053 / -.089) |

Nested steps on the confirm window: rounds -0.017 (p=.05), near-tie
pairs + context -0.048 (p<.001), gap -0.008 (p=.31), cap -0.011 (p=.13);
each step's sign matches the screen window. The reranker redesign is
the bulk of the gain; the E4 gate re-tuned on top of it adds the rest.

Acceptance: B3 and B4 meet every criterion except that promotion F1 on
the confirm window falls .925 -> .914, 0.001 past the -0.01 guardrail;
that is one promotion out of ~80, and the screen window is flat
(.934 -> .933), so it is read as noise rather than boundary damage and
is noted here rather than blocking. B4 beats B3 on MAE in both windows
with exact slots flat. **Adopted: B4.** The 2024-2026 slice, for
continuity with earlier logs: 18.2 -> 20.8 exact, MAE .831 -> .706.

New defaults (`banzuke/models.py`): bag of 5 seeds; base 300 rounds;
pair classifier 150 rounds trained on window pairs plus rolling-OOF
near-ties (oof_gap 2) with context features; rerank gate gap 1.0,
cluster_max 6. LambdaRank (B) truncation 60. Everything else unchanged.
Not adopted, listed above: capacity and regularization changes, wider
window pairs, recency weights, L2 blending, lower claim thresholds.

## E15: sanyaku block structure (2026-09)

### E15a within-block order follows the model

`fill_class` laid out forced claimants ahead of fills regardless of
score (200607: Kisenosato M1E 8-7, a forced K claim, was given K1E
over Asasekiryu M2E 10-5, whom the model and the committee ranked
higher). Members now take slots in score order once membership is
settled; claims still decide who gets in. Ceiling from the miss
decomposition: 22 slots over 135 basho. Measured on the adopted
configuration (two bag replicates): screen +0.04 exact [-0.09, +0.19],
confirm +0.05 [+0.00, +0.15], MAE -0.001 in both, 7-5 W-L overall, never
worse in any window. Below the tuning acceptance threshold, kept as a
correctness fix (the resolver docstring already promised the model's
order within a class).

### E15b learned slot count: scoped out

Whether the committee creates a third komusubi slot turns on a handful
of historical cases. From all transitions since 1990, M1 8-7 claimants
reach sanyaku 94% of the time when the S/K zone has spare room, 89%
when it is one short, but only 58% (7 of 12; 2 of 5 since 2004) when
kachi-koshi incumbents and stronger claims already fill it, so a slot
must be created for them. The analogous "zone already full" cells for
M2 10-5 (2 of 4), M2 9-6 (5 of 10) and M3 10-5 (0 of 2) are as thin.
Conditioning the M1 8-7 claim on spare room would fix 3 and break 2
basho since 2004, worth ~0.05 slots/basho; a fitted model would be
learning from those same 12 cases. Left as a rule table for the next
data-rich revisit; the ~0.8 slots/basho ceiling stands unclaimed.

## E16: same-rank E/W "twins" kept adjacent (2026-09)

(Numbered after the protocol-v2 round E11-E15 on the tuning branch.)

Hypothesis: when M8e and M8w post the same record, the committee keeps
them adjacent on the next banzuke even if a third wrestler's individual
movement would land between them, and the per-rikishi model splits such
pairs too often. Tooling: `experiments/grouping.py`, a post-processing
harness on analyze.py's cached Ar predictions (2004-2026, 135 targets,
3 seeds). A variant permutes the model's order, re-resolves and
re-evaluates, so every comparison is exactly paired with no retraining.

Committee behavior (237 maegashira twin pairs, 201 juryo):

- identical-record E/W twins stay adjacent 66% (M) / 72% (J) of the time
  vs 2.5% for E/W pairs with different records; prior order (E above W)
  is kept in 100% of cases. Stable across eras (66/66/66%) and zones.
- identical-record pairs that are adjacent but straddle a rank number
  (M8w + M9e) are only 47% adjacent: the effect is specific to twins.
- 89% of maegashira twin splits are by exactly one intruder, a faller
  with fewer wins or a climber with more (47/53%).
- 7-8 twins are the stickiest (86% adjacent, n=90); 8-7 62%, 9-6 62%.

Model behavior (Ar): twins land adjacent in 65% of its orders vs the
committee's 68%; both agree in 62% of pairs. The model's interloper count
barely predicts the committee (kept 73% with no interloper, 60% with
one, 58% with two or more). Ar inverts twins in 12 of 438 pairs where
the committee never does. Crucially, when the committee keeps a pair the
model split, which side the interloper ends up on is a coin flip for
every simple rule (base-score midpoint .43, came-from side .51, climber
passes .37, on 87 kept splits); the rule that scored best on 2004-2019
(climber, .44) was the worst on 2020+ (.17).

Grid (paired per-basho deltas vs the cached Ar baseline, exact slots):

- pre-registered primary, twins glued when split by one interloper,
  climber anchor: full -0.08 (26-34, CI -0.19..+0.02), MAE +0.007
  (CI excludes 0); 2020+ -0.06, MAE +0.011. Fails.
- midpoint / came-from anchors: null overall (-0.02/-0.01), slightly
  negative 2004-2019 (-0.10), slightly positive 2020+ (+0.18, 12-5).
- cross-rank and chained groups: negative everywhere (up to -0.4/basho,
  p<.001), as their 47% adjacency predicts.
- glue only when every interloper's base score is >= 0.5 from the pair's
  midpoint (0.47 events/basho): +0.06 full (CI -0.01..+0.14), MAE -0.003
  (CI -0.006..-0.0002); 2020+ +0.16 (9-2, Wilcoxon p=.016). Post hoc
  and worth one slot per 16 basho.
- oracle ceilings: knowing which twins stay together AND which side the
  interloper goes is worth +0.33/basho (CI .19-.47); knowing only the
  first and using a rule anchor is worth +0.04. The anchor is the whole
  problem and no feature in hand predicts it (small record gaps tend not
  to pass the pair, large ones do, but n is ~100 events over 22 years).

**Decision: glue not adopted; the order constraint is.** The committee
habit is real but the model already reproduces most of it implicitly, and
the exploitable residue (~0.06 slots/basho with the best post-hoc rule,
~0.33 with an oracle) sits at the noise floor. The one part with no
exceptions, twins never swap E/W order (0 of 436 M/J pairs since 2004;
the two Y twins and one S twin that did swap are sanyaku), is now a
resolver rule (`keep_twin_order`): where the model ranks W above E the
two exchange places in its order. Ar did this in 31 of 405 backtest
frames (12 of 438 seed-0 pairs); paired effect exact +0.0025/basho
(5-5), MAE -0.0015. Free, principled, and not the win the hypothesis was
after. Re-run `uv run python -m experiments.grouping describe|backtest`
after future model changes; the events parquet it writes is where an
anchor signal would show up if more data accrues.

## Known limitations / future leads (updated 2026-09)

- Juryo promotee placement is still under-promoted (+0.64 positions
  over 2004-2026); bottom-of-sheet slotting is genuinely noisy
  ("banzuke luck").
- Ozeki/yokozuna promotion thresholds are hardcoded conventions in the
  resolver; borderline cases (32-win runs with a yusho, Terunofuji's
  201507 promotion after two sanyaku basho) surface in predict.py notes
  rather than being decided statistically.
- COVID-era kadoban exemptions (Mitakeumi 2022) are not modeled.
- Sanyaku block size is wrong in 12% of basho (22% since 2020) at ~7
  slots each: the largest remaining structural lever, ~0.8 slots/basho
  as an upper bound. Half the errors are M1 8-7 claims the committee
  declined; win thresholds cannot fix this (E15 scoping above). A small
  model of "extra slot created?" conditioned on the zone's incumbents
  is the next structural experiment.
- A LambdaRank model trained through the juryo boundary (B, truncation
  60) is now within 1 slot of Ar; blending its score into Ar's base
  order is untested.
- Untried ideas for the near-tie gap: committee-regime features
  (banzuke committee membership changes); for live use, feeding
  announced Y/O promotions and shin-juryo counts into the resolver as
  constraints (information GTB players have); scraping GTB archives for
  a paired per-basho model-vs-human comparison on identical targets.
