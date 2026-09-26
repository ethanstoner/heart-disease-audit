# Heart Disease Prediction: A Reproducibility Audit

[![CI](https://github.com/ethanstoner/heart-disease-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/ethanstoner/heart-disease-audit/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

I traced the Kaggle "Heart Failure Prediction" CSV, one of the most-used tabular ML datasets on
GitHub, back to its four UCI hospital sources, and found that its filled-in missing values
encode the diagnosis.

![The filled-in values encode the diagnosis](images/fill_rule.png)

### Highlights

- **The filled values encode the label.** Every filled ST_Slope is `Flat` for a patient with
  heart disease (111 of 111) and `Up` for a patient without (191 of 191). The dataset's
  description never mentions missing values or filling.
- **It inflates reported accuracy.** Reverting the 644 recovered filled cells to missing drops
  the typical published pipeline's median random-forest accuracy from **86.4% to 79.3%** over
  1,000 paired train/test splits.
- **Nobody noticed.** A pre-registered survey of 30 public notebooks found **0 of 30** mention
  any imputation at the source, and 27 of 30 report a single split, where the split alone moves
  accuracy by **9 points**.
- **Honest estimate.** Trained on three hospitals and tested on the fourth, logistic regression
  reaches an AUC of **0.83**, and no model beats it. **14 of 15** pre-registered predictions held.

**Python · pandas · scikit-learn · XGBoost · SciPy · pytest · GitHub Actions**

## Overview

The deliverable is a set of findings, not a better accuracy score. The question: when hundreds
of notebooks report roughly 86% accuracy on this CSV, how much of that comes from the model,
and how much from how the dataset was built, how the notebooks were evaluated, and which
hospital each patient came from? A leaked label in a popular teaching dataset trains people to
trust numbers that won't hold on new patients.

Every number below comes from committed code that was run, and every experiment's prediction
was committed before the experiment ran.

## Method

```
UCI files (4 hospitals, 920 rows) --+
                                    +--> provenance matcher --> 644 filled cells recovered
Kaggle CSV (918 rows) --------------+                              |
                                                                   v
survey of 30 notebooks --> "typical pipeline" --> 1,000-split control and teardown (T1 to T7)
                                                                   |
                                                                   v
               honest baseline: leave-one-hospital-out, nested CV, imputation inside the pipeline
```

1. **Provenance.** Pin and hash-verify the raw files, then pair each published row with the UCI
   row it came from, matching only on fields the UCI row actually recorded.
2. **Survey.** Code 30 public notebooks against a fixed rubric, pre-registered in
   `survey/PROTOCOL.md`, and reduce them to one modal pipeline.
3. **Control and teardown.** Run that pipeline over 1,000 seeded splits, then turn each
   practice on and off in paired experiments (T1 to T7).
4. **Honest baseline.** Train on three hospitals, test on the fourth, with tuning and imputation
   kept inside the training hospitals.

## Engineering Highlights

- **Built a provenance matcher** that pairs 912 of 918 published rows one-to-one with their UCI
  source. It never sees ST_Slope, and agrees with UCI on it 610 of 610 times.
- **Ruled out a matching artifact:** with the diagnosis also left out of the match key, 904 rows
  still pair up, their diagnoses agree on every one, and all 296 filled slopes follow the rule
  (`tests/test_provenance.py`).
- **Pre-registered 15 predictions** (`analysis/PREDICTIONS.md`) and logged every deviation
  before the affected experiment ran. 14 held, 1 was falsified, and it is reported as such.
- **Implemented the statistics and tested them against known answers:** bootstrap, DeLong,
  McNemar, Nadeau-Bengio and Holm, each checked against a hand-computed or brute-force result,
  plus interval-coverage, Type-I error and shuffled-label tests of the estimators themselves.
- **Caught a flaw in my own spec:** the shuffled-label gate showed the planned pooled AUC
  estimator is biased below 0.5 under the null, so the primary estimate was switched before any
  baseline model was fitted (D6).
- **Made it reproducible:** raw data pinned by SHA-256 / md5 and verified on every load, all
  seeds derived from one master seed, pinned dependency versions, and a paired multi-seed runner
  parallelised across CPU cores. Re-running the 1,000-split teardown reproduces its results.
- **CI:** the full test suite and all five notebooks execute end to end on every push.

## Findings

### 1. The published CSV's filled-in values encode the label

- The CSV (918 rows) is the four UCI heart-disease files (920 rows) minus their 2 exact
  duplicates. The author's documented arithmetic (1,190 - 272 = 918) adds Statlog, and all 270
  Statlog rows are copies of Cleveland rows.
- The UCI files have 662 missing values in the 11 columns the CSV keeps; the published CSV has
  none. The description never mentions missing values, `?`, imputation or filling (checked
  through Kaggle's metadata API on 2026-09-22, `results/kaggle_description_check.json`).
- Filled FastingBS (89 of 89, 74 of them from Switzerland) and ExerciseAngina (53 of 53) follow
  the same rule as ST_Slope. Filled Oldpeak values don't overlap between the classes (-0.1 to
  0.4 without disease, 0.7 to 1.9 with). Together these four columns hold 504 of the 644 filled
  cells. The filled RestingBP, MaxHR and Cholesterol values show no clean rule.
- Reverting the fill costs the typical pipeline a median paired **7.1 points** (central 95%
  over 1,000 splits: 2.2 to 12.5). This is a paired difference for one pipeline, not a new
  "true" accuracy; finding 4 gives the honest estimate.

This shows what the filled values do. It does not identify how they were produced.

### 2. What published notebooks actually do

![Survey of 30 notebooks](images/survey_practices.png)

- 27 of 30 evaluate on a single train/test split, and 19 of 30 fit a transform on rows that
  include the test set.
- The median reported headline accuracy is 86.1%. The survey's typical pipeline, run over
  1,000 random splits of the same data, lands in the same place (central 95%: 82.1% to 91.3%).

![Control over 1,000 splits](images/control_seed_distribution.png)

### 3. Which practices matter here, and which don't

![Effect of each practice](images/effect_summary.png)

- The label-encoding fill is the largest effect in the published CSV.
- Reporting the best of four models on one test set (10 of 30 notebooks) overstates accuracy
  by 1.5 points on average (winner's-curse bootstrap).
- Fitting the scaler before the split, the leak textbooks warn about most, changes nothing
  measurable here, and neither does median-imputing `Cholesterol = 0` on all rows.
- Keeping duplicates would have added about 8 points. The published CSV does not have that
  defect: its author removed all 272.
- On a single split, random forest is the most accurate of the four models only 49.6% of the
  time, and an exact McNemar test separates it from logistic regression on 3% of splits.

### 4. An honest estimate

![Honest estimates](images/honest_estimates.png)

- Logistic regression reaches a within-hospital AUC of **0.832 [0.796, 0.865]** with
  provenance proxies (arm A) and **0.822 [0.783, 0.857]** without them (arm B).
- Random 10-fold x 5 CV reports 0.875 to 0.890 for the same model, but its own within-hospital
  AUC is about the same as the held-out-hospital estimate. The extra discrimination comes from
  ranking patients across hospitals whose disease rates run from 36% to 93%, not from
  diagnosing better.
- Calibration fails at an unseen hospital: the model badly under-predicts risk in Switzerland
  (calibration intercept +2.65) and over-predicts it in Hungary (-0.84). AUC does not show this.
- No model is significantly better than logistic regression after Holm correction (corrected
  resampled t-test on the 10x5 random-CV control). In arm A, XGBoost and the MLP are
  significantly worse.

### Pre-registration scorecard

Predictions are in `analysis/PREDICTIONS.md` and the survey protocol; deviations in
`analysis/DEVIATIONS.md` (D5, D6) and `survey/DEVIATIONS.md` (D1 to D4). Counting P1.1 once per
column, 14 of 15 predictions were supported and 1 was falsified.

| Prediction | Result |
|---|---|
| P1.1 Filled ST_Slope / FastingBS carry target information | Supported (disease-rate gap 1.00 for filled values vs 0.34 / 0.30 for recorded values) |
| P1.2 Removing filled-slope rows lowers accuracy | Supported (87.0% to 81.5%); its matched null is uninterpretable, see D5 |
| P1.3 Undoing the fill lowers accuracy | Supported (-7.1 points) |
| P2.1 Hospital-only CV AUC within 0.02 of in-sample 0.720 | **Falsified** (0.697, a gap of 0.022) |
| P2.2 Missingness indicators alone reach AUC >= 0.65 | Supported (0.688) |
| P3.1 Duplicates inflate accuracy in the 1,190-row merge | Supported (87.0% vs 78.9% duplicate-aware) |
| P4.1 Random forest wins on < 50% of splits | Supported (49.6%) |
| P4.2 Paired RF - LR difference varies less than RF alone | Supported (SD 1.7 vs 2.3 points) |
| P4.3 McNemar RF vs LR significant on < 20% of splits | Supported (3%) |
| P5.1 Best-of-4 bias between 0.5 and 3 points | Supported (1.5) |
| P6.1 Scaler before the split: < 0.5 points for LR and SVM | Supported (median 0) |
| P6.2 Cholesterol imputation before the split: < 0.5 points | Supported (median 0) |
| P7.1 Held-out-hospital AUC below random 10-fold | Supported (-0.040 [-0.050, -0.031]; the pooled estimator is biased low, see D6) |
| P8.1 Logistic regression not beaten by XGBoost | Supported (XGBoost significantly worse in arm A) |

Two checks that failed, and why:

- **Tree sanity check.** RF and decision-tree predictions were identical under the two scaler
  scopes on 99.78% and 99.65% of rows, against a 99.9% threshold. An exactly representable
  rescaling changes no prediction, so the differences come from floating-point rounding, not
  leakage (notebook 04).
- **Shuffled-label gate.** With labels shuffled, pooled out-of-fold AUC across held-out
  hospitals averages 0.489 and its interval contains 0.5 in only 42 of 50 runs (44 required).
  The n-weighted mean of within-hospital AUCs averages 0.495 and passes with no margin, 44 of
  50, so it is the primary estimate (D6, `results/shuffled_label_gate.json`).

### Limitations

- **External validity.** All four cohorts are patients referred for coronary angiography, a
  selected high-risk population with strong spectrum and verification bias. Nothing here
  generalises to screening the general population.
- With four hospitals there is no interval for "performance at a new hospital". Switzerland
  has only 8 patients without disease, so its AUC is flagged as underpowered wherever it
  appears.
- The survey's coders were two instances of the same language model; their 249/250 agreement
  measures how clear the rubric is, not agreement between independent judges. GitHub code
  search caps results at 1,000 and skips large files, so the sample is not uniform.
- The Kaggle CSV was retrieved from two public GitHub mirrors pinned by commit, verified by an
  md5 shared across five independent copies, not from the Kaggle API.

## Reproduce

Python 3.12, no GPU needed.

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt && venv/bin/pip install -e .
venv/bin/pytest -q                            # fetches and hash-verifies the data
venv/bin/python scripts/reproduce_control.py  # the control over 1,000 splits
venv/bin/python scripts/run_teardown.py       # T1 to T7
venv/bin/python scripts/run_baseline.py       # the honest baseline
venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
```

The survey's search step (`scripts/snapshot_search.py`) cannot be reproduced, which is why its
result is frozen in `survey/snapshot.json`. Everything after it is deterministic.

## Testing

```bash
venv/bin/pytest -q
```

77 tests: unit tests for data loading, the provenance matcher, the conventional pipeline, the
teardown experiments and the evaluation statistics, plus methodology tests of the statistics
themselves (bootstrap interval coverage, Type-I error rate, the shuffled-label gate). All 77
pass locally in about 80 seconds on a 32-core machine; CI runs the suite and executes every notebook on each push.

## Project Structure

| Path | Contents |
|---|---|
| `src/heart_audit/` | data, provenance matcher, the conventional pipeline, teardown experiments, evaluation statistics, honest baseline, figures |
| `tests/` | unit tests, plus coverage, Type-I and shuffled-label tests of the statistics |
| `survey/` | protocol, frozen search, screening and coding of 30 notebooks, deviations |
| `analysis/` | pre-registered predictions and deviations |
| `results/` | every number the notebooks report |
| `notebooks/` | 01 survey, 02 reproduction, 03 data integrity, 04 evaluation, 05 honest baseline |

## What I Learned

- **Check the data before the model.** My own design spec assumed Kaggle had recoded UCI's `?`
  to 0. The matcher showed every `?` had become a concrete value decided by the label, and that
  one fact outweighs every modelling choice the notebooks argue about.
- **Test the estimator, not just the code.** The pooled held-out-hospital AUC the spec called
  for sits below 0.5 when there is no signal, because each held-out hospital shifts the training
  prevalence. Only the shuffled-label gate caught it.
- **The famous leak wasn't the one that mattered.** Fitting the scaler before the split, the
  mistake textbooks warn about most, changed nothing measurable here; the undocumented fill cost
  7.1 points.

## Data

- UCI Heart Disease: Janosi, Steinbrunn, Pfisterer and Detrano (1988). Detrano et al.,
  "International application of a new probability algorithm for the diagnosis of coronary
  artery disease", *American Journal of Cardiology* 64 (1989).
- UCI Statlog (Heart).
- Heart Failure Prediction Dataset, Kaggle (2021):
  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The survey critiques practices across the population of notebooks. It names no notebook
author; repository URLs and commit SHAs are in `survey/screening.csv`.

## License

MIT (code). The data remains under its original terms.
