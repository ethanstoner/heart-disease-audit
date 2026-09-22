# Pre-registered predictions for the teardown (T1–T7) and the honest baseline

Written and committed 2026-09-22, before any experiment below was run. Every choice an
analyst could make after seeing results is fixed here. Outcomes are reported whichever way
they fall.

## What the author had already seen

- The control's per-model accuracy distributions over its 1,000 split seeds
  (`results/control_seeds.csv`: mean RF 0.869, SVM 0.865, LR 0.861, DT 0.793). No per-seed
  ranking, paired difference or bootstrap.
- The per-column fill counts from the relaxed matcher (commit e32f576): ST_Slope 302,
  FastingBS 89 (75 filled as `1`), Oldpeak 60, RestingBP 57, MaxHR 53, ExerciseAngina 53
  (33 `Y`), Cholesterol 28, RestingECG 2. No disease rate of any filled group.
- The spec's figures: per-site prevalence 45.9 / 36.1 / 93.5 / 74.5%; pooled disease rate
  by ST_Slope in the Kaggle CSV (Up 19.75%, Flat 82.83%); `Cholesterol == 0` rows 88.4%
  diseased vs 47.7%.
- That all 270 Statlog rows duplicate Cleveland rows, and that UCI's four files hold 2 exact
  duplicates (1190 − 272 = 918 = 920 − 2).
- Arithmetic from those counts: in-sample source-only AUC 0.720; `Cholesterol == 0` AUC
  0.625. These are stated as arithmetic, not predictions.

## Common definitions

- **Matcher:** the relaxed one-to-one matcher (`provenance.py`, commit e32f576). It
  supersedes the strict matcher of plan 1 (207 filled slopes), whose '?'→0 assumption was
  false.
- **Filled cell:** a cell that is '?' in the paired UCI row and concrete in Kaggle
  (`filled_cells`). The 6 unpaired Kaggle rows, and filled cells in them, are not
  recoverable and stay as published in every arm.
- **Control:** `run_conventional` (random forest headline) with its defaults, including
  `scale_scope="full"`.
- **Seeds:** split seeds are `derive("control", 1000)`. Other experiments use
  `derive("<experiment id>", n)` with the ids named below. Percentiles use `np.percentile`
  (linear interpolation).
- **Paired difference:** computed per split seed (same test rows in both arms). The median
  over seeds is reported, with the central 95% of the per-seed differences.

## T1 — Undisclosed imputation

**P1.1 (filled values carry target information).** Within Hungary + VA only (Cleveland has
no filled slopes; Switzerland is reported separately):
`gap_filled = P(disease | filled = a) − P(disease | filled = b)` and
`gap_observed = P(disease | observed = a) − P(disease | observed = b)`, with
(a, b) = (`Flat`, `Up`) for ST_Slope and (`1`, `0`) for FastingBS.
Prediction: `gap_filled > gap_observed` for each column. *Falsified* for a column if the
point estimate of `gap_filled − gap_observed` is ≤ 0. Reported with a stratified bootstrap
95% CI (B = 2000, id `t1_gap`). Only a CI excluding 0 counts as strong support; groups with
fewer than 20 rows are flagged.

**P1.2 (spec prediction 4).** Arm F removes the Kaggle rows whose ST_Slope was filled.
The null removes the same number of rows drawn from paired rows whose ST_Slope was *not*
filled, stratified to the same (source, HeartDisease) counts; 50 draws (id `t1_removal`).
Each arm runs the control on 1,000 seeds (for the null, 1,000 seeds per draw; the draw's
statistic is its median). The statistic is median RF accuracy. Accuracy minus the test set's
majority-class rate is reported alongside. Prediction: arm F's median is below the full-data
median, and below the 2.5th percentile of the 50 null medians. *Falsified* if either fails.

**P1.3 (undoing the fill).** Both arms run the control with `impute_scope="train"`, which
imputes every canonically missing value (NaN, and 0 in Cholesterol / RestingBP) with the
training-row median (numeric) or mode (categorical), applied to the raw frame before
encoding. Arm P: the published CSV. Arm R: the published CSV with every filled cell reverted
to NaN. Prediction: the median paired difference R − P in RF accuracy is < 0.
*Falsified* if it is ≥ 0.

## T2 — Provenance leak

**P2.1.** On the 920-row UCI frame, logistic regression on one-hot `source` only, with
stratified 10-fold CV (shuffle, id `t2_cv`) and pooled out-of-fold scores, reaches AUC within
0.02 of the in-sample 0.720. *Falsified* if the gap exceeds 0.02.

**P2.2.** On the UCI frame, logistic regression on the 11 canonical missingness indicators
only (no clinical values, no source), under the same CV, reaches AUC ≥ 0.65. *Falsified* if
below 0.65. On the Kaggle CSV the only visible proxy is `Cholesterol == 0`, whose AUC (0.625)
is arithmetic.

## T3 — Duplicates (counterfactual; no defect is claimed for the published CSV)

**P3.1.** On the 1,190-row merge (UCI 920 + Statlog 270), duplicate groups are rows equal
on all 11 features and the target. Over 1,000 seeds, the control's median RF accuracy under
a random 80/20 split exceeds its median under a split that keeps duplicate groups together
(`GroupShuffleSplit`, test_size 0.2, same seeds). *Falsified* if not higher. The share of
random-split test rows with a duplicate in training is reported as description.

## T4 — Split noise

**P4.1.** Over the control's 1,000 seeds, random forest's share of "most accurate of the
four models" wins, with ties splitting the credit 1/k, is below 50%. *Falsified* if ≥ 50%.

**P4.2.** The standard deviation across seeds of the paired RF − LR accuracy difference is
smaller than the standard deviation across seeds of RF accuracy. *Falsified* if not smaller.

**P4.3.** An exact McNemar test of RF vs LR on each seed's test set rejects at α = 0.05 on
fewer than 20% of seeds. *Falsified* if ≥ 20%.

## T5 — Winner's curse (applies to the 10/30 notebooks that report a best-of-N)

**P5.1.** For each of the 1,000 control seeds, bootstrap the 184 test rows B = 2000 times
(id `t5`). In each resample, pick the model with the highest in-bag accuracy (ties: the mean
over the tied models), and record its in-bag accuracy minus its out-of-bag accuracy. The
bias is the mean over resamples and then over seeds. Prediction: bias is in [0.5, 3.0]
percentage points. *Falsified* if outside.

## T6 — Fitting before the split

**Sanity check, not a prediction:** RF and DT test predictions are identical between
`scale_scope="full"` and `"train"` on at least 99.9% of (seed, row) pairs.

**P6.1.** The median paired difference (full − train scaler) is below 0.5 pp in absolute
value for both logistic regression and SVM. *Falsified* if either is ≥ 0.5 pp.

**P6.2.** With `impute_scope="full"` vs `"train"` (median of non-zero values, which also
covers the single RestingBP zero), the median paired difference in RF accuracy is below
0.5 pp in absolute value. *Falsified* if ≥ 0.5 pp. XGBoost (4/30 notebooks) is also run with
native NaN handling, and with full-row and train-row imputation. This is reported without
a prediction.

## T7 — Fold structure

**P7.1 (spec prediction 2).** On the 920-row UCI frame with the 11 features (no source,
no missingness indicators), a logistic-regression pipeline (train-fold median/mode
imputation, one-hot, standard scaling) produces out-of-fold scores under two schemes:
leave-one-source-out, and stratified 10-fold (shuffle, id `t7_cv`). The statistic is the
difference in AUC of pooled out-of-fold scores (LOSO − 10-fold), with a paired stratified
bootstrap over patients (B = 2000, id `t7_boot`), percentile 95% CI. *Falsified* unless the
CI's upper bound is < 0. Random forest is reported the same way, without a prediction.

## Honest baseline (plan 4)

**P8.1 (spec prediction 3).** In the 10×5 repeated stratified CV control arm, logistic
regression is not significantly worse than XGBoost: the Nadeau–Bengio corrected resampled
t-test, Holm-corrected across the three model-vs-LR comparisons, gives p ≥ 0.05 for XGBoost
vs LR. *Falsified* if corrected p < 0.05 with XGBoost better.
