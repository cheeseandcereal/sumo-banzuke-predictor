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

### E16 on the tuned Ar (this branch, bag of 5, 3 seeds)

Same harness, same cache window, baseline 19.61 exact / MAE .867. The
tuned reranker behaves differently around twins:

- it keeps identical-record twins adjacent in only 46% of its orders vs
  the committee's 69% (main's Ar: 65%); one-interloper splits 170 vs 97.
  It inverts twins once in 438, so `keep_twin_order` touches 1 frame
  (-1 slot, noise) here.
- the anchor is still a coin flip: on 142 kept splits the interloper
  went where the base-midpoint rule says 54%, the came-from rule 50%.
- glue twins whatever the interloper count, midpoint anchor: full +0.17
  exact/basho (50-32, Wilcoxon p=.036, CI -0.00..+0.33), screen
  2004-2019 +0.19, 2014+ +0.14, confirm 2020+ +0.11 (13-10, CI
  -0.20..+0.43); MAE +.004 full, .000 confirm. One-interloper only:
  +0.14 / +0.18 / +0.11 / +0.07 with MAE neutral everywhere.
- the pre-registered primary here (one interloper, all-below anchor,
  q=.50 on the screen) made +0.17 on the full window (Wilcoxon p=.024)
  but -0.03 on the confirm window with MAE +.004: fails the bar.
- per event (162 split twin pairs touching makuuchi, glued in
  isolation): +0.19 slots each under the midpoint anchor (72-44-46, sign
  p=.012), and positive whether the committee kept the pair (+0.20) or
  split it (+0.18). The tuned reranker's placement of the interloper is
  itself noisy, so consolidating the trio helps either way.
- oracle ceiling is larger than on main: +0.50/basho for twins, +0.71
  for identical-record chains.

Status: suggestive, not confirmed. The mechanism is significant, the
global effect (+0.1 to +0.17/basho, one slot every 6-10 basho) has a
confirm-window CI that includes zero, and the variant was one of a
grid. Pursued as E17: the split is a Borda artifact, and the
S/K-scoped unit became the default.

## E17: twins as one reranker unit (2026-09, protocol v2)

Follow-up to E16, prompted by the 202611 forecast: Ar put Ura and
Fujiseiun between Hakunofuji (K1E 5-10) and Daieisho (K1W 5-10). The
committee has never done that: identical-record S/K twins stayed
adjacent in 26 of 26 cases since 2004 (92 of 98 since 1959, the last
split in 1993), while both branches' Ar split them in ~28% of backtest
frames.

Diagnosis: an aggregation artifact, not a classifier error. Within a
cluster {E, W, X} the twins' mutual comparison is certain (P = .999),
so Borda separates them by a full point, and any rival X the
classifier is honestly unsure about (roughly .33 < P < .67 against
each twin) lands between them. The calibrated reranker (E13) is
*more* exposed than main's overconfident one: it correctly rates a K
5-10 against an M9 10-5 near .5 (the committee: 56% since 2004), and
Borda converts "unsure" into "between", the one outcome the committee
never chooses. No pair feature can fix this; the algebra holds with
identical probabilities.

Change: option `twin_unit` ("sk" or "all"). E and W of one rank number
with identical W-L-A records are one unit in the reranker: clustered at
their mean base score, compared with a rival by the mean of the
members' pair probabilities, expanded E then W. Adjacency is assumed,
the rival's side is learned, so the classifier still decides where the
unit goes. predict.py gains `--set` so options can be used live.

Screen and confirm, two bag replicates, paired vs the current default:

| config | window | exact/42 | MAE | dExact [CI] | W-L | dMAE [CI] | p |
|---|---|---:|---:|---|---|---|---:|
| default | screen | 20.28 | 0.840 | ref | | ref | |
| twin_unit=sk | screen | 20.30 | 0.840 | +0.02 [-0.05, +0.08] | 4-3 | +0.000 [-0.002, +0.002] | 1.0 |
| twin_unit=all | screen | 20.46 | 0.849 | +0.18 [-0.11, +0.47] | 36-25 | +0.009 [-0.002, +0.021] | .10 |
| default | confirm | 17.93 | 0.938 | ref | | ref | |
| twin_unit=sk | confirm | 17.93 | 0.937 | +0.00 [-0.05, +0.06] | 1-2 | -0.001 [-0.002, +0.000] | .16 |
| twin_unit=all | confirm | 17.94 | 0.936 | +0.01 [-0.30, +0.38] | 10-14 | -0.002 [-0.013, +0.009] | .79 |

"sk" touches ~0.2 pairs per basho and is metric-neutral everywhere
(never worse in any window; GTB +0.03 on confirm). "all" gains exact
slots on the screen (+0.18, 36-25) while worsening MAE there (+0.009,
p=.10), and is flat on the confirm window: the M/J twin split is a
real committee habit a third of the time, and forcing adjacency trades
E/W-flip-sized wins for occasional two-slot losses. Fails the bar.

**Adopted: `twin_unit="sk"` as the default**, on the E15a precedent
(a convention with no exceptions in the modern record, neutral on the
metrics, kept as a correctness fix). The convention audit
(`banzuke.conventions`, main) tracks S/K twin adjacency so the first
counter-example will be visible. The 202611 forecast now reads
Takayasu / Ura, Hakunofuji / Daieisho, Fujiseiun / Kotoshoho at M2-M4;
the committee's choice among those three rows is historically a coin
flip, the pair staying together is not.

## E18: mechanical-formula feature for promotee ordering (2026-09, protocol v2)

Trigger: the 202611 forecast placed Kyokukaiyu (J1W 8-7) above
Kitanowaka (J6E 11-4). The committee orders juryo promotees almost
mechanically: `juryo position - 4 x wins` reproduces the relative order
of 94.2% of 976 promotee pairs since 2004, and at a margin of 3
half-ranks (this pair) the higher juryo rank lands higher only 4% of
the time. Ar's base model and reranker match the committee's exchange
rate (gap slope .51 vs .53 half-ranks per margin unit) and order
promotee pairs at 93.6%; the pair above was a 1-in-20 near-tie miss
(pair classifier .551), not a bias.

Hypothesis tested anyway: giving the GBMs the formula directly (model
R's per-zone linear fit of delta on win8 and absences, refit for every
basho on the labels known at the time, as `mech_delta` / `mech_pos`
inputs to both stages, option `mech`) would let trees use the linear
exchange rate instead of approximating it with splits.

Screen and confirm, two bag replicates, paired vs the default:

| config | window | exact/42 | MAE | dExact [CI] | W-L | dMAE [CI] | p |
|---|---|---:|---:|---|---|---|---:|
| mech=true | screen | 20.06 | 0.867 | -0.24 [-0.61, +0.14] | 37-50 | **+0.027 [+0.012, +0.041]** | .022 |
| mech=true | confirm | 18.00 | 0.919 | +0.08 [-0.50, +0.55] | 17-18 | -0.018 [-0.047, +0.010] | .19 |

Worse on the screen window in both halves (+.019 / +.035), with the
base stage's own promotee ordering falling from .944 to .903 on a
screen-window sample: the formula over-spreads promotees (its gap
slope is .89, 168% of the committee's) and the trees trust it in
near-ties where the learned features were already right. The confirm
window's small gain does not rescue a significant screen loss. **Not
adopted**; code removed, `transitions.parquet` unchanged. With `mech`
the 202611 sheet fixed Kitanowaka / Kyokukaiyu and broke Daiseizan /
Arashifuji (margin -1, the lower-ranked lands higher 83%): one
near-tie traded for another, which is what a null looks like on one
sheet.

Lesson for reading forecasts: a single inverted promotee pair with a
3-half-rank margin is a known 5% event, and the confidence markers
(`?`/`??` on both wrestlers) already flag it; `--above "Kitanowaka >
Kyokukaiyu"` is the right tool for a user who wants the historical
favorite.

## E19: miss-driven case analysis of the 2019+ forecasts (2026-10)

Goal: re-create every forecast from 201901 to 202609 as it would have been
made at the time, compare with the real banzuke, and study the misses one
case at a time with the committee's precedent counted for each. Not a
tuning round: no default was changed; every candidate is measured and
logged for a later decision.

Tooling, committed under `experiments/`:

- `explain.py build` re-runs each backtest frame (train on next_basho < T,
  rolling OOF for the pair stage, asserted per frame) and records the
  reranker's units and clusters, every pair probability within 2.0 base
  points, the sheet the resolver would have produced from the base order
  alone, and a per-cell decomposition: `cell_err` (where the predicted label
  sits in the actual layout), `block_offset` (the label shift caused by a
  wrong Y/O/S/K count above), `local_err`, and the stage a miss was born
  in (cascade / structural / boundary / rerank / base / resolver). Readers:
  `sheet` (side-by-side banzuke), `explain` (per-rikishi stage table, cluster
  pair probabilities, precedent lines), `docket`, `calibration`. The cache
  (135 targets x 3 seeds, 25 min on 24 workers) reproduces the E17 confirm
  figures exactly (17.93 / .937).
- `precedent.py`: "where did K1 5-10 land" / "who was higher, K 5-10 or
  M9-10 10-5" with counts for all history, 2004+ and the last 60 basho.
- `rules.py`: candidate rules applied to the cached orders (order edits,
  class assertions, or flipping the inputs of a resolver rule), re-resolved
  and re-evaluated: exactly paired, three seeds averaged, block-bootstrap
  CIs, no retraining (E16's harness generalised).

Where the 2019+ misses are (seed 0, 46 targets, 1,093 missed cells):
base order 660 (14.3/basho; 467 off by one), cascade of a structural error
208 (4.5/basho, 17 frames), reranker introduced or worsened 143 (3.1; it
also fixed 2.9/basho), structural 49, boundary 23, resolver 10. The
structural frames are the lever: 11 of the 17 are Y/O rule errors (132
cascade cells), 6 are S/K counts (76 cells). Pair calibration is good
overall (log loss .37 screen / .40 recent, confident pairs 94% right); the
misses concentrate in lopsided situations the committee decides 90-100%
one way while the model sits near .5 or on the wrong side.

Verdicts over the 568 notable cells (structural, boundary, |local err| >= 2,
confident near misses): lost coin flip 246, miscalibration 234, flow
artifact 62, missing principled rule 16, unknowable 9, new precedent 1;
the 208 cascade cells are charged to their structural cause. Full docket,
sheet reviews and the triage are under `results/scratch/explain/`
(git-ignored; `review/adhoc.md` is the pattern index).

Regularities established (counts 2004+ unless stated), all now rows in the
convention audit as "watch":

- Y who stay and O who stay are ordered by wins, then the yusho winner,
  then a man who fought above a full-kyujo man, then prior position: all
  139 Y pairs and 554 O pairs since 2004. Ar put
  Kisenosato 0-5-10 above two 0-0-15 yokozuna with p = 1.00 the wrong way
  (201901) and called two make-koshi ozeki at .51 (202309).
- A demoted ozeki is the bottom sekiwake: never S1E (0/24 since 1990), never
  above a kachi-koshi S incumbent (0/22), below a komusubi promoted to S
  (0/5). Ar's reranker lifted him to S1E/S2E in six 2019+ frames (p .76-.85).
- Make-koshi sekiwake: 7 wins -> komusubi 31/32 (the exception, Goeido 201305,
  had nobody to take the slot), <= 6 wins -> maegashira 26/26; the falling
  S outranks weak M1 claims for the K slot. Make-koshi komusubi with <= 6
  wins leaves sanyaku 0/47 (one exception since: Takayasu 202507). Ar kept
  S 7-8 at S (202201, 202203), put S 6-9 at K (202609), kept K 6-9 (202403).
- Yokozuna promotion: the previous yusho or jun-yusho must have been fought
  as ozeki (0/3 when at sekiwake; 6/8 when at ozeki). Ozeki promotion: a
  33+ run with 12+ now is accepted when one of the three basho was at M1-M3
  (3/3 before 202605, 4/5 since 2004), where `ozeki_run3` is NaN.
- Full or heavy kyujo (0 wins, 8+ absences) at S/K/M drops +24 (full) /
  +23 (partial): non-exempt 2004+ n = 70, q25/50/75 +22/+24/+26, the same
  from any rank and era. Exceptions are the 2021-22 COVID stable
  withdrawals (9 frozen rows) and 200401. Ar drops these men 17.6 cells on
  average with seed spread >= 2 in 24% of the rows: kosho-era rows (median
  +1) and the frozen COVID rows pull it down.
- Komusubi newcomers rank below kachi-koshi komusubi incumbents (2/49).

Patterns without a rule (documented leads): zone pressure (KK risers from
M9+ are lifted 4 cells more when 8+ of M1-M6 are make-koshi; Ar's signed
error moves 1.4 cells across those buckets, Spearman .13); drift the year
feature mostly tracks (M 7-8 keeps its cell 9% in 2004-11 vs 37% in
2024-26, committee drop 1.9 -> 1.1 cells, Ar 2.1 -> 1.4; joi make-koshi
excess drop 5.7 -> 3.2, Ar 5.2 -> 3.7); promotees now land M13 or lower
93/94 when they have 10+ wins (66% before 2020) and the old promotee bias
is gone (+1.15 -> +0.21 cells); Borda cycles that override a correct
direct pair (202411, 202501, 202601, 202109); cluster cuts that separate
men who had to be compared (202201, 202309); a twin unit that included a
rule-promoted twin (201911); the make-koshi ceiling leaving a 6-9 man flat
(3/593 historically, five frames). Unknowables: COVID freezes (202103,
202111, 202203, 202207, 202209), the 41-cell makuuchi of 202111,
Takakeisho's 33-win decline (201903), Mitakeumi's kadoban exemption
(202209/202211, which also resets the `kadoban` flag).

Data bug found: two 202507 juryo rows (Nishikigi 7-7-0, real 8-7;
Fujiseiun 9-5-0, real 10-5) have a bout with an empty result in the raw
API record, a fusen win not counted; Nishikigi's missing win flipped a
boundary call in the 202509 forecast. Fix in `build.py` at the next
rebuild (only 2 rows since 2004).

## E20: committee rules measured on the cached orders (2026-10)

`uv run python -m experiments.rules --start 200401`. Paired exact slots per
basho vs the cached order re-resolved (V0), three seeds averaged, 135 basho;
MAE is the decision metric where quoted. Nothing adopted in this round.

| rule | precedent | full 2004-2026 dExact [CI] | 2004-2018 | 2019+ | W-L full |
|---|---|---|---:|---:|---|
| R4 demoted ozeki is the bottom sekiwake | 0/22, 0/24 | +0.15 [+0.07, +0.23], dMAE -.004 | +0.11 | +0.22 (6-0) | 12-0 |
| R14 Y/O who stay ordered by wins, yusho first, fought before idle, then prior | 693/693 pairs | +0.18 [+0.08, +0.29], dMAE -.005 | +0.20 | +0.12 | 11-0 |
| R2 MK sekiwake: 7 -> K, <= 6 -> M, S 7-8 over weak M1 claims | 31/32, 26/26 | +0.27 [-0.02, +0.66], dMAE -.005 | +0.03 | +0.73 (6-0) | 13-1 |
| R3 MK komusubi <= 6 wins -> M | 0/47 | +0.02 | +0.02 | +0.03 | 4-2 |
| R7 Y rule only after a yusho fought as ozeki | 0/3 vs 6/8 | +0.06 | +0.06 | +0.07 | 3-0 |
| R8 O rule accepts one M1-M3 basho in the run (12+ now) | 4/5 | +0.31 [0.00, +0.74] | +0.18 | +0.57 (2-0) | 4-0 |
| R12 K newcomers below KK K incumbents | 2/49 | +0.05 | +0.05 | +0.07 | 4-0 |
| R1 0 wins, 8+ absences at S/K/M lands +24 / +23 | q50 +24, n=70 | +0.20 [0.00, +0.48], dMAE -.013 | +0.06 | +0.47 (8-8) | 20-16 |
| **bundle of the eight** | | **+1.29 [+0.83, +1.83], dMAE -.032 [-.057, -.007]** | +0.82 (34-5) | +2.20 (23-6) | 57-11 |
| bundle without R1 | | +1.13 [+0.70, +1.65], dMAE -.019 [-.027, -.012] | +0.76 (24-1) | +1.84 (19-1) | 43-2 |

Also measured, not recommended: R9/R13 (weak M1 claims never create a third
K slot) +0.21 / +0.14 full but 5-2 / 6-4 with one -8 frame, the E15b
thin-cell problem; R10 (make-koshi with <= 6 wins drops >= 2 cells) 0-2;
R11 (S/K East incumbent keeps the cell unless West has 4+ more wins) +0.02
full, +0.15 on 2019+ (3-0), a 2016+ lean (1/7 passed) that is not yet a
rule. The bundle's worst frames lose 2-3 cells (COVID freezes, the 2010
suspension basho where the committee dropped the suspended men +25/+26);
five frames gain 11 or more.

Reading: R2, R3, R4, R7, R12 and R14 are 100%-precedent conventions
(E10/E15a/E17 bar) that together are worth about one exact slot per basho
on every window, never worse than 2-1 in any window; they are the
recommended next resolver change, to be confirmed by a protocol-v2
backtest after implementation (resolver rules reproduce exactly from
cached orders, so the table above is the expected result). R8 rests on
four cases and should carry an audit row. R1 is the largest single lever
on the recent window but is a fitted constant that loses in COVID frames;
the model-side fix (flag the exempt rows so the base learns the modern
drop) is preferred, with a predict.py note meanwhile. Leads needing
retraining: zone-pressure features (MK count in M1-M6, KK count in M1-M8),
the COVID flag, a formula prior for promotee pairs in the reranker,
conditioning M1 8-9 claims on zone room and adding M2 10-5 (12/16) to the
claim table.

## E21: data rebuild, the blank 202507 bout, lagged class columns (2026-10)

Change: `build.py` gains a `CORRECTIONS` table for raw records whose
scheduled bout has `result: ""`. The one case in 400 banzuke is 202507
juryo day 15: the API's torikumi has Nishikigi (J1E) over Fujiseiun (J8W)
by kotenage, the banzuke record is blank for both and a 2026-10 re-fetch
still returns the blank; the API's own totals were short by one (Nishikigi
7-7-0 -> 8-7, Fujiseiun 9-5-0 -> 9-6; E19 quoted 10-5 for Fujiseiun, which
was wrong). Any other scheduled bout without a result now fails the build.
The brief's "only 2 rows since 2004 have wins + losses + absences != 15"
was also wrong: 56 do (29 no-shows with an empty record, among them the 15
men expelled at 201105, 21 mid-basho retirements whose API `absences` is
unfilled, 2 partial records), so no blanket assertion; the check is scoped
to blank results on scheduled bouts (an opponent is set), which only the
two corrected rows have.

`features.py` keeps `class1`/`class2` and adds `num1`/`num2` (lagged rank
number, same contiguity rule as `w1`): resolver inputs for the Y/O rule
corrections of E22, not in `FEATURES`. Four transition rows change (the two
corrected records and Nishikigi's lags at 202509 and 202511), so the OOF
table recomputed (2 min) and every cache was rebuilt. `explain.py` records
pair probabilities within 5.0 base points (was 2.0; clusters span up to
4.4, so every in-cluster pair is now on disk, E24 needs that).
Worker defaults are now cpu_count - 2 everywhere and `predict.py` gained
`--workers`.

Reproduction: the E19 cache rebuilt on the fixed data (135 targets x 3
seeds, 12 min on 14 workers) gives 19.62 exact / 0.864 MAE over 2004-2026
(E19: 20.37 / 0.852 on 2004-2018, reproduced to the decimal, and 18.14 /
0.892 on 2019+, now 18.16 / 0.888), seeds 0-1 on 2020-2026 17.94 / 0.932
(E17's adopted configuration: 17.93 / 0.937 on the old data; the frames
from 202509 on differ, 202509 -2 exact and 202607 +2 among them), every
frame with `train_max < target`, `repro` 1.000.
`experiments.rules` on it reproduces the E20 table: bundle without R1
+1.13 exact per basho, 43-2 (the two fixed rows change nothing before
202509).

## E22: the E20 committee rules in the resolver (2026-10)

Hypothesis (E20): six conventions with no modern exception and one at
47/49, ported from `experiments/rules.py` into `banzuke/resolver.py`, are
worth about one exact slot per basho on every window. Each precedent count
below was recomputed from `transitions.parquet` at porting time (2004+
unless stated; `python -m banzuke.conventions` tracks them).

Change (`rule_masks()` in the resolver, shared with the convention audit;
overrides still win and each convention they break is reported):

- R14: yokozuna who stay and ozeki who stay are ordered by wins, then
  yusho, then having fought (a man with bouts above a full-kyujo man),
  then prior position: 139/139 Y pairs, 554/554 O pairs. A new yokozuna or
  ozeki still takes the lowest slot of his class (7/7 Y, 28/28 O
  promotions landed below every incumbent who stayed).
- R4: a demoted (kadoban make-koshi) ozeki is the bottom sekiwake: 0/45
  above another man landing at sekiwake since 1990, never S1E (0/24).
  rules.py only placed him below the other *claimants*; the port puts him
  last in the block (no frame differs).
- R2: a sekiwake with 7 wins takes a komusubi slot when at least two other
  sekiwake candidates exist (kachi-koshi S/K incumbents, M claimants, a
  demoted ozeki): 29/29. With fewer candidates the model decides (6 cases:
  4 went to komusubi, Goeido stayed S1W in 201207 and 201305). M1 claims
  with 8-9 wins never create a third komusubi slot for him: the
  lowest-scored go back to maegashira while the block exceeds 2 (the
  `_trim_weak_m1` semantics of rules.py, done on the member list).
- R3: komusubi with <= 6 wins (131/132; Takayasu 202505 kept) and komusubi
  with 7 wins below K1E (24/24, rules.py said K1W; the two K2E cases also
  left) leave sanyaku; sekiwake with <= 6 wins too (69/70; Daieisho
  202207, the frozen Nagoya record). A K1E 7-8 is left to the model (5/12
  kept, all moved to K1W).
- R7: the yokozuna rule needs the earlier yusho / jun-yusho fought as
  ozeki: 6/8 promoted when it was, 0/3 when it was at sekiwake (Hakuho
  200605, Terunofuji 202105, Aonishiki 202601).
- R8: the ozeki rule also accepts 12+ wins now with a 33-win run where
  exactly one of the two earlier basho was at M1-M3 and the other in
  sanyaku: 4/4 (Terunofuji 201505, Tochinoshin 201805, Aonishiki 202511,
  Kirishima 202603); runs with two maegashira basho 0/3 (Kotooshu 200509,
  Terunofuji 202011, Onosato 202405). Four cases: an audit row carries the
  thin precedent, and predict.py notes every Y/O promotion on the sheet
  with the results behind it ("34 wins over last 3 basho, one of them at
  M1" names this path).
- R12 (own commit, adopted on measured value): komusubi newcomers rank
  below kachi-koshi komusubi incumbents, 47/49 (Takakeisho 11-4 in 201711
  and Tamawashi 13-2 in 202209 went above an 8-7 / 9-6 incumbent).

Faithfulness: `experiments.rules --cache <E19 dir>` re-resolves the old
resolver's orders with the new one. Every bundle rule is an exact no-op in
all 405 frames (`changed` 0 for R2, R3, R4, R5, R6, R7, R8, R12, R14), and
the `cached` variant (the old sheets) pairs the port against them:

| window | n | old exact / MAE | new exact / MAE | dExact [CI] | W-L | dMAE [CI] |
|---|---:|---:|---:|---|---|---|
| full 2004-2026 | 135 | 19.62 / 0.864 | 20.80 / 0.844 | +1.18 [+0.73, +1.70] | 44-2 | -0.020 [-0.029, -0.013] |
| 2004-2018 | 89 | 20.37 / 0.852 | 21.18 / 0.833 | +0.80 [+0.51, +1.13] | 24-1 | -0.019 [-0.032, -0.010] |
| 2019+ | 46 | 18.16 / 0.888 | 20.06 / 0.867 | +1.90 [+0.82, +3.17] | 20-1 | -0.021 [-0.032, -0.011] |
| confirm 2020+ | 40 | 17.94 / 0.929 | 19.94 / 0.910 | +2.00 [+0.81, +3.35] | 17-1 | -0.019 [-0.029, -0.009] |

(three seeds averaged; E20 measured +1.13 / 43-2 for the same bundle on the
same frames, the difference being R12's +0.05 and the 202509+ data fix).
With the true next order as input the resolver now reproduces 102 of the
135 banzuke since 2004 cell for cell (was 95): 200607, 201507, 201805,
201807, 202107, 202305, 202601, 202605 gained, 202507 lost (Takayasu kept
at K1W after 6-9, the one R3 exception). Y/O/S/K counts wrong in 2019+
frames (seed 0): 17 -> 12 (202107, 202201, 202305, 202601, 202605 fixed);
the twelve left are declined weak M1 claims (201911, 202105, 202505,
202603), Takakeisho's declined 33-win run (201903), Asanoyama's 32-win
promotion over the cancelled basho (202007), yusho-after-jun-yusho
promotions the rule gets both ways (202101, 202303 declined; 202109
Terunofuji promoted on a 14-1 jun-yusho after a yusho, 3/8 all-time for
that pattern), and the COVID frames (202209, 202211, 202301).

Protocol v2 (two bag replicates, full retrain on the rebuilt data, paired
against the E17 round's default rows in the backtest cache; those are the
pre-`twin_unit` rows, which the adopted `twin_unit=sk` configuration beat
by +0.02 exact on the screen and -0.001 MAE on the confirm window, E17):

| window | n | E17 default exact / MAE | this exact / MAE | dExact [CI] | W-L | dMAE [CI] | p (MAE) |
|---|---:|---:|---:|---|---|---|---:|
| screen 2004-2019 | 95 | 20.28 / 0.840 | 21.13 / 0.820 | +0.84 [+0.52, +1.21] | 27-3 | -0.020 [-0.033, -0.011] | <.001 |
| screen halves | | | | +0.74 / +0.94 | 13-1 / 14-2 | -0.015 / -0.025 | .004 / .001 |
| confirm 2020-2026 (soft) | 40 | 17.93 / 0.938 | 19.94 / 0.913 | +2.01 [+0.82, +3.36] | 18-3 | -0.025 [-0.036, -0.015] | .005 |
| full | 135 | 19.59 / 0.869 | 20.77 / 0.847 | +1.19 [+0.74, +1.72] | 45-6 | -0.022 [-0.031, -0.014] | <.001 |

Guardrails: promotion/demotion F1 .935/.905 -> .935/.905 on the screen,
.914/.883 -> .917/.881 on the confirm window; sanyaku-set exactness .647 ->
.753 and .550 -> .612. **Adopted** (all seven rules; R12 in its own
commit). On the cached orders no frame loses more than one cell (201307,
202507); against the retrained E17 default the worst frame is 202509 at
-2.0 and four basho gain 11 or more (201807, 202211, 202305, 202605).
Tests: the eight regained banzuke are reproduced from the true order,
one fixture per rule on a reversed order, the override path still wins and
warns. The convention audit's seven watch rows are resolver rows now (the
7-win sekiwake row counts the guarded cases, 0/29) plus rows for the two
Y/O corrections and the K1W exit; the kyujo row stays a watch.

Not adopted, measured on the same frames: R9/R13 (weak M1 claims never
create a third slot) +0.26 / +0.18 full, 4-1 / 5-3 with a -10 frame
(202303); R10
(make-koshi <= 6 wins drops >= 2 cells) 0-2; R11 (East incumbent keeps the
cell unless West has 4+ more wins) +0.03 full, +0.17 on 2019+ (3-0), a
2016+ lean. Leads seen while porting: a yusho after a 12-3 jun-yusho was
promoted 1/3 since 2004 (Kisenosato 14-1 yes, Takakeisho 13-2 and 12-3
no) and a jun-yusho after a yusho, both as ozeki, 1/2 (Terunofuji 14-1 in
202107 yes, Hakuho 13-2 in 200607 no): no rule there yet.

## E23: `rank_protected`, the kyujo training-data fix (2026-10, protocol v2)

Hypothesis (E19, E20): the committee drops a full kyujo (0 wins, 8+
absences) at S/K/M by +24 cells from any rank in any modern era (non-exempt
2004+ n = 104, q25/50/75 +22/+24/+25), and Ar under-drops these men by about
6 cells because the labels are contaminated: kosho-era rows (1972-2003,
median +1 when kosho was granted) and the 2020-2022 COVID exemptions teach
"full absence, small drop". The inputs are there; a flag for the exempt
rows should let the base learn both regimes. E20's placement rule R1 is the
oracle (+0.47 exact on 2019+, 8-8 there because it loses in the COVID
frames).

Change: `rank_protected` in `transitions.parquet` (not in `FEATURES`): 1 on
a full-kyujo row at S/K/M/J whose rank was frozen: kosho granted in
1972-2003 (read off the frozen rank, `delta <= 3`; the decision was public
before the banzuke; 137 rows, the other 85 full absences of that era
dropped 10+) or one of 26 listed 2004+ exemptions (last kosho cases 200401,
Tamanoi 202009, the Hatsu 2021 withdrawals, Miyagino 202109, Tagonoura and
Nishikido 202201, Ichinojo 202205, Takayasu 202207; every listed row
dropped <= 3, every unlisted full kyujo since 2004 dropped 17+). The 202207
partial records (Tamawashi 5-8-2 at +2 and the like) stay unflagged:
documented unknowables. Models take it through the new `extra` option
(`--set extra=rank_protected`: appended to the base inputs and the pair
differences, part of the OOF cache key); `predict.py --protected Name`
sets it live.

Two bag replicates, paired against the E22 default on the same data:

| window | n | exact/42 | MAE | dExact [CI] | W-L | dMAE [CI] | p | promo/demo F1 | sanyaku sets |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| screen 2004-2019 | 95 | 21.13 -> 21.05 | 0.820 -> 0.799 | -0.08 [-0.34, +0.17] | 35-41 | -0.021 [-0.042, -0.002] | .17 | .935/.905 -> .937/.920 | .753 -> .747 |
| half 2004-2011 | 47 | 19.73 -> 19.79 | 0.962 -> 0.914 | +0.05 | 19-20 | -0.048 [-0.076, -0.024] | .003 | | |
| half 2012-2019 | 48 | 22.49 -> 22.28 | 0.680 -> 0.686 | -0.21 [-0.59, +0.15] | 16-21 | +0.006 [-0.011, +0.021] | .16 | | |
| confirm 2020-2026 (soft) | 40 | 19.94 -> 21.05 | 0.913 -> 0.746 | +1.11 [+0.15, +2.14] | 19-18 | -0.167 [-0.278, -0.076] | .001 | .917/.881 -> .923/.936 | .612 -> .625 |
| 2019+ | 46 | 20.06 -> 21.00 | 0.869 -> 0.726 | +0.94 [+0.05, +1.88] | 22-20 | -0.144 [-0.248, -0.056] | .003 | | |
| 2024-2026 | 17 | 22.79 -> 24.21 | 0.682 -> 0.592 | +1.41 [+0.38, +2.97] | 11-5 | -0.090 [-0.176, -0.033] | .02 | | |

The gain sits exactly where the diagnosis put the damage: 2004-2011
(kosho-era contamination of the training labels) and 2020+ (the COVID
rows); 2012-2019, where Ar already dropped these men about right, is flat
(dMAE +0.006, CI through zero), so the screen halves do not share a sign.
Read as the pre-stated mechanism rather than a tuning effect with a split
verdict. The confirm window clears the bar by a wide margin (dMAE -0.167,
CI upper -0.076; exact +1.11; demotion F1 .881 -> .936, within-1 .806 ->
.839) and beats R1's oracle on 2019+ (+0.94 vs +0.47 exact, -0.144 vs
-0.006 MAE), because it also fixes the frames where the rule lost. **Adopted:
`rank_protected` joins FEATURES** (appended last, so the measured
configuration is reproduced column for column). Diagnostic on the explain
caches (135 targets x 3 seeds; 99 unprotected full kyujo at S/K/M, actual
drop 23.7): predicted drop 17.9 -> 22.2, signed error -5.8 -> -1.5, mean
absolute error 6.3 -> 2.3, rows with seed spread >= 2 25% -> 20%; by era
2004-11 -9.9 -> -2.9, 2012-19 -0.8 -> -0.6, 2020-23 -6.7 -> -1.3, 2024-26
-6.0 -> -0.5. The 10 protected makuuchi rows (actual drop 0.8) went from a
predicted 13.1 to 1.5.

Folded into the same final rebuild: the `kadoban` feature no longer resets
after an exempted second make-koshi (Mitakeumi 202209; 18 kosho-era rows
before 2004 also flip, all consistent with their outcome), so a third one
demotes him in the resolver (+1 exact, -0.09 MAE in the 202211 frame on
the cached orders; Y/O/S/K counts wrong in 2019+ frames 12 -> 11). Against
the measured `extra=rank_protected`
configuration the final default is neutral (full +0.13 exact, MAE +0.000;
confirm +0.04 / +0.004). Final default against the E17 default, two bag
replicates:

| window | n | E17 exact / MAE | final exact / MAE | dExact [CI] | W-L | dMAE [CI] | p (MAE) | promo/demo F1 | sanyaku sets |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| screen 2004-2019 | 95 | 20.28 / 0.840 | 21.22 / 0.797 | +0.93 [+0.57, +1.32] | 51-32 | -0.042 [-0.062, -0.025] | <.001 | .935/.905 -> .938/.916 | .647 -> .763 |
| half 2004-2011 | 47 | 18.99 / 0.977 | 19.85 / 0.919 | +0.86 [+0.48, +1.31] | 27-14 | -0.058 [-0.083, -0.035] | <.001 | | |
| half 2012-2019 | 48 | 21.55 / 0.705 | 22.55 / 0.678 | +1.00 [+0.42, +1.65] | 24-18 | -0.027 [-0.051, -0.007] | .16 | | |
| confirm 2020-2026 (soft) | 40 | 17.93 / 0.938 | 21.09 / 0.750 | +3.16 [+1.60, +4.62] | 25-11 | -0.188 [-0.305, -0.089] | <.001 | .914/.883 -> .920/.935 | .550 -> .662 |
| 2024-2026 | 17 | 20.82 / 0.706 | 23.85 / 0.600 | +3.03 [+0.59, +4.97] | 13-3 | -0.106 [-0.193, -0.037] | .008 | | |
| full | 135 | 19.59 / 0.869 | 21.18 / 0.783 | +1.59 [+0.98, +2.29] | 76-43 | -0.086 [-0.133, -0.048] | <.001 | | |

The 202611 banzuke, the first basho untouched by any decision, was not
available when this round closed.

## E24: reranker aggregation (2026-10, offline)

Hypothesis (E19 5.3): Borda within a cluster overrode a correct direct pair
through a cycle in four frames (202411, 202501, 202601, 202109), a twin unit
swallowed a rule-promoted twin (201911), the reranker lifted a demoted ozeki
(202607). Tool: `experiments/rerank_offline.py` rebuilds the reranker from
a cached frame's base scores, twin units and pair probabilities (all
in-cluster pairs are on disk since E21), checks it against the cached
order (405/405 frames reproduce), then re-aggregates, re-resolves and
re-evaluates: exactly paired, three seeds, no model needed. On the E22
resolver:

| variant | full dExact [CI] | W-L | full dMAE | 2004-2018 | 2019+ | confirm 2020+ |
|---|---|---|---|---|---|---|
| exclude rule-decided men (Y/O incumbents, rule promotions and returns, kadoban make-koshi) from the clusters | +0.01 [-0.00, +0.03] | 3-1 | -0.000 | +0.02 (3-0) | -0.01 (0-1) | 0.00 (4 frames touched in 405) |
| Kemeny within clusters, Borda tiebreak | +0.23 [-0.01, +0.47] | 65-49 | +0.005 [-0.003, +0.014] | +0.31 (47-28, p .03), MAE +.003 | +0.06 (18-21), MAE +.009 | 0.00 (15-19), MAE +.005 [-.009, +.018] |
| Copeland, Borda tiebreak | +0.22 [-0.01, +0.46] | 64-44 | +0.004 | +0.34 (47-26), MAE +.001 | 0.00 (17-18), MAE +.009 | -0.04 (14-17), MAE +.009 |

Exclusion is a no-op in practice: the E22 rules already decide those men's
class and order, so the only effect left is on cluster-mates whose cluster
they chained (4 frames). Kemeny and Copeland trade E/W flips for larger
misses: exact slots up on 2004-2018, MAE up everywhere, flat on 2019+ and
the confirm window. **Not adopted** (the bar was neutral-or-better on both
windows). Lead kept: Kemeny's +0.31 on the old window says the cycles are
real; a better-calibrated pair stage might turn it into a MAE gain too.

## E25: zone-pressure (supply) features (2026-10, protocol v2)

Hypothesis (E19 5.4): the base model scores men independently and cannot
see the field; kachi-koshi risers from M9+ with 9-11 wins land 4 cells
higher when 8+ of M1-M6 are make-koshi (actual rise -8.7 with <= 5 vs
-13.3 with 10+ on the E19 frames; Ar's signed error -0.63 -> +0.86 across
those buckets, Spearman .14). Per-basho counts of the source banzuke
(`features.supply_features`): `mk_joi` (make-koshi in M1-M6), `kk_upper`
(kachi-koshi in M1-M8), `strong_below` (M9+ with 10+ wins), `kk_sanyaku`
(kachi-koshi S/K), `n_absent` (men with an absence; 14 at 202207 against a
median of 3). Taken through `--set extra_shared=...`: appended to the base
inputs and to the pair stage's context columns.

Two bag replicates, paired against the E22 default (same data):

| config | window | exact/42 | MAE | dExact [CI] | W-L | dMAE [CI] | p |
|---|---|---:|---:|---|---|---|---:|
| all five | screen 2004-2019 | 21.13 -> 20.94 | 0.820 -> 0.823 | -0.18 [-0.72, +0.38] | 44-43 | +0.003 [-0.021, +0.026] | .80 |
| all five | halves | | | +0.13 / -0.49 | | -0.016 / +0.022 | |
| all five | confirm 2020-2026 (soft) | 19.94 -> 20.00 | 0.913 -> 0.887 | +0.06 [-0.55, +0.68] | 18-19 | -0.026 [-0.053, +0.004] | .28 |
| all five | full | 20.77 -> 20.66 | 0.847 -> 0.842 | -0.11 | 62-62 | -0.006 [-0.025, +0.013] | |
| `mk_joi,kk_upper` (vs the final default) | screen 2004-2019 | 21.22 -> 21.22 | 0.797 -> 0.797 | 0.00 [-0.51, +0.52] | 43-44 | -0.000 [-0.022, +0.020] | .81 |
| `mk_joi,kk_upper` | halves | | | +0.19 / -0.19 | | -0.003 / +0.003 | |
| `mk_joi,kk_upper` | confirm 2020-2026 (soft) | 21.09 -> 21.35 | 0.750 -> 0.726 | +0.26 [-0.39, +0.85] | 16-18 | -0.024 [-0.067, +0.015] | .19 |
| `mk_joi,kk_upper` | full | 21.18 -> 21.26 | 0.783 -> 0.776 | +0.08 | 59-62 | -0.007 [-0.028, +0.012] | .37 |

**Not adopted.** Flat on the screen with the halves disagreeing, a
suggestive but insignificant MAE gain on the confirm window that is not
concentrated in the COVID frames (without 202207/202209 it is -0.029).
The columns stay in the dataset (`SUPPLY_FEATURES`) for a later angle: a
pair-stage-only form, or the interaction with the riser's own record.
Diagnostic (seed-0 bag, KK risers from M9+ with 9-11 wins, signed error
by `mk_joi` bucket <= 5 / 6-7 / 8-9 / 10+): default -0.65 / -0.38 / +0.16 /
+0.67 (Spearman with `mk_joi` .13), with `mk_joi,kk_upper` -0.33 / -0.15 /
+0.10 / -0.03 (Spearman .00). The model learns the field; the correction
is worth about half a cell on 4-5 men per basho and does not reach the
slot or MAE metrics.

## Reading a forecast: reviewer checklist (from E19, updated E22-E25)

Each item is a situation the committee decides lopsidedly; the ones marked
"enforced" the resolver now applies (E22; an override that breaks one is
reported as a broken convention, and every Y/O promotion is listed under
`notes:` with the results behind it, or "by override" when no convention
explains it). The rest are still the reviewer's job. Counts are 2004+
unless stated.

- Full or heavy kyujo (0 wins, 8+ absences) at S/K/M: the committee drops
  him about 24 cells from any rank (q25/50/75 +22/+24/+25, n=104 incl.
  juryo); since E23 the model learns this (`rank_protected` marks the
  exempted rows). If the JSA has exempted the man (stable-wide COVID-style
  withdrawal), pass `--protected Name`.
- Enforced: make-koshi sekiwake with 7 wins goes to komusubi when two other
  sekiwake candidates exist (29/29), 6 or fewer to maegashira (69/70);
  make-koshi komusubi with 6 or fewer wins (131/132) or 7 wins below K1E
  (24/24) leaves sanyaku. A K1E 7-8 is the model's call (5/12 kept, always
  at K1W). `--class Name=K` / `--class Name=M` to disagree.
- Enforced: a demoted ozeki is the bottom sekiwake (0/45 above another
  sekiwake since 1990, 0/24 at S1E).
- Enforced: yokozuna who stay and ozeki who stay are ordered by wins, then
  yusho, then having fought, then prior position (693/693 pairs).
- Enforced: the yokozuna rule needs the earlier yusho / jun-yusho fought as
  ozeki (0/3 otherwise); the ozeki rule accepts a 33-win run with one M1-M3
  basho when the current record is 12+ (4/4, two maegashira basho 0/3).
  Still the reviewer's: a yusho with 13 or fewer wins after a 12-3
  jun-yusho was declined 0/2 (Takakeisho 202011, 202301) while 14-1 after
  12-3 was promoted (Kisenosato 201701); a jun-yusho after a yusho, both as
  ozeki, was promoted 1/2 (Terunofuji 14-1 in 202107 yes, Hakuho 13-2 in
  200607 no). `--class Name=Y`.
- Enforced: komusubi newcomers rank below kachi-koshi komusubi incumbents
  (47/49; the two exceptions had 11-4 and 13-2).
- Created third komusubi slot for an M1 claim with 8-9 wins when both K
  incumbents are kachi-koshi: honoured 58% historically, 0/3 for an M1W
  behind an M1E with the same record; the `!` marker and its footprint
  line already show this. A falling 7-win sekiwake takes precedence over
  such a claim (enforced).
- Promotees: since 2020 a promotee with 10+ wins lands M13 or lower
  (93/94); the order among promotees follows juryo position - 4 x wins
  (94%); a J1 8-7 above a J5-J7 11-4 is 2/9.
- Zone pressure: when 8+ of M1-M6 are make-koshi, risers from M9+ with 9-11
  wins land about 4 cells higher than otherwise; when the top is crowded
  with kachi-koshi, risers are compressed and make-koshi men get their
  standard drops. Supply features did not help the model (E25); still a
  reading aid.
- S/K East incumbent vs a West incumbent with more wins, both staying:
  since 2016 the West passed 1/7; S1W 8-7 stayed put under a K1E 10+ 3/3
  since 2020. Not a rule yet; treat a predicted swap as `?`.
- A 6-9 maegashira predicted at his own cell (3/593 historically) means
  the men above him were placed wrong, not that he stays.
- Reranker pairs at .4-.6 inside a cluster are honest coin flips; a Borda
  order that contradicts the direct pair (cycle) is worth testing with
  `--above` (Kemeny aggregation gained exact slots but lost MAE, E24).

## Known limitations / future leads (updated 2026-10, after E25)

- Juryo promotee placement: the under-promotion bias has faded (+1.15
  cells in 2004-11, +0.21 in 2024-26); since 2020 promotees with 10+ wins
  land M13 or lower 93/94, and the order among them follows juryo
  position - 4 x wins, which the reranker occasionally inverts (E19).
- Full or heavy kyujo at S/K/M: fixed on the model side (E23,
  `rank_protected`); the 202207 partial records frozen "on the record
  known at withdrawal" are not flagged and remain unknowables. A new JSA
  exemption needs a `PROTECTED` entry (or `--protected` live).
- Zone pressure: the regularity is real (risers from M9+ land about 4
  cells higher when 8+ of M1-M6 are make-koshi) but per-basho supply
  columns did not move the metrics (E25). A pair-stage-only or
  interaction form is untested.
- Ozeki/yokozuna promotion thresholds are hardcoded conventions in the
  resolver (E22 added the fought-as-ozeki and one-maegashira-basho
  conditions); the thin cases left to the reviewer are a yusho after a
  12-3 jun-yusho (1/3 since 2004), a jun-yusho after a yusho (1/2),
  Asanoyama's 32-win promotion over the cancelled 202005 and Takakeisho's
  declined 33 (201903); predict.py lists every Y/O promotion under `notes:`
  with the results behind it, so the call is visible.
- COVID-era kadoban exemptions (Mitakeumi 2022) are not modeled beyond the
  kadoban flag surviving the exempted basho (E22 note).
- Structural errors on 2019+ (E22): Y/O/S/K counts wrong in 12 of 46
  frames (was 17); the twelve left are declined weak M1 claims (58%
  honoured, 0/3 for an M1W behind a same-record M1E), slots created for men
  the claim table does not know (M2 10-5 12/16; S+K >= 6 when Y+O <= 3),
  the Y/O cases above and the COVID frames.
- Reranker aggregation: Kemeny within clusters gained +0.31 exact per
  basho on 2004-2018 but cost MAE everywhere (E24); worth a retest if the
  pair stage's calibration improves.
- A LambdaRank model trained through the juryo boundary (B, truncation
  60) is within 1 slot of Ar; blending its score into Ar's base order is
  untested.
- Untried ideas for the near-tie gap: committee-regime features
  (banzuke committee membership changes); for live use, feeding
  announced Y/O promotions and shin-juryo counts into the resolver as
  constraints (information GTB players have); scraping GTB archives for
  a paired per-basho model-vs-human comparison on identical targets.
