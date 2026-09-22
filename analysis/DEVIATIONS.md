# Deviations from analysis/PREDICTIONS.md

Each entry was written and committed before the experiment it affects was run.

## D5 — P1.2's matched null is infeasible; match on outcome only (2026-09-22)

P1.2 pre-registered a null that removes rows "with the same (source, HeartDisease) counts"
from paired rows whose ST_Slope was not filled. That draw is impossible without
replacement. The filled rows include 155 Hungarian patients without disease, and only 31
unfilled Hungarian patients without disease exist (VA: 36 filled vs 15 available).

**Replacement:** each null draw removes 191 rows without disease and 111 with disease
(the filled rows' outcome counts), drawn without replacement from all paired rows whose
ST_Slope was not filled, across all sources. Everything else in P1.2 is unchanged. The
null matches prevalence but not hospital mix: the filled rows are Hungary (188), VA (98)
and Switzerland (16). The report states this.

Found while writing the unit test for the draw, before any T1 experiment was run.

## D6 — LOSO discrimination is the n-weighted mean of within-site AUCs (2026-09-22)

**When:** after the teardown ran, before any honest-baseline model was fitted. Found by the
spec's shuffled-label gate (`tests/test_methodology.py`) on its first run.

The spec says two things about the leave-one-source-out estimate: "outer AUC is computed on
pooled out-of-fold predictions" and "aggregation across the four folds is an n-weighted
mean". Over 50 label shuffles, the pooled out-of-fold AUC of the honest LR pipeline averaged
**0.4886** (range 0.432–0.544), and its 95% interval contained 0.5 in only **42/50** runs
(the gate needs ≥ 44). The n-weighted mean of within-site AUCs averaged **0.4954**. Pooling
mixes four models whose baselines differ. After a shuffle, a held-out site that happens to
have more positives leaves fewer in training, so its predictions shift down. That
anti-correlation pushes pooled AUC below 0.5 when there is no signal.

**Decision:** for LOSO, the primary discrimination estimate is the n-weighted mean of the
four within-site AUCs. Its interval comes from a bootstrap stratified by (site, class). Pooled
LOSO AUC is still reported, labelled as biased under the null. Repeated 10-fold CV keeps pooled
out-of-fold AUC per repeat: its folds are stratified, so training prevalence barely varies.

**Effect on T7 (P7.1):** P7.1 was registered on pooled AUC and is reported exactly as
registered. The report adds that the pooled LOSO estimator is biased low by about 0.011 under
the null, and gives the within-site comparison next to it.

## Notes on reading the teardown results (no deviation)

- **P1.2's null is uninterpretable.** Matching the filled rows' outcome counts (D5) forces
  each null draw to remove 191 of the 218 unfilled healthy rows. The remaining healthy class
  is then dominated by rows whose ST_Slope was filled with `Up`, which P1.1 shows is decided
  by the label. That is why null accuracy rises to about 0.94. The verdict is reported as
  registered. Only the full-vs-filled-removed comparison is interpreted.
- **T6 sanity check failed** (tree prediction identity 99.65% DT, 99.78% RF, against a 99.9%
  threshold). An exactly representable rescaling (×4) gives 0/36,800 differing tree
  predictions, while standardisation gives 128/36,800. The differences come from
  floating-point rounding in the affine rescale flipping tied splits (the one-hot encoding has
  two complementary column pairs), not from test-row information.

**Reproduced by committed code** (`scripts/shuffled_label_gate.py` →
`results/shuffled_label_gate.json`): pooled 0.4886, 42/50; within-site 0.4954, 44/50. The
within-site estimator passes the gate with no margin: 44 is exactly the minimum.
