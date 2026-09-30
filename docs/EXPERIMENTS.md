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
