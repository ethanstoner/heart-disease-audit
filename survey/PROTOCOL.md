# Survey protocol (pre-registered 2026-09-22)

This file was committed before the search snapshot was taken. Its commit precedes
`survey/snapshot.json` in history.

## Population
Public GitHub Jupyter notebooks that train a classifier on the Kaggle "Heart Failure
Prediction" CSV (918 rows, 12 columns; columns include `ST_Slope`, `ChestPainType`,
`HeartDisease`).

## Search
`gh api search/code`, query `ST_Slope ChestPainType HeartDisease extension:ipynb`,
100 per page, pages 1–10 (the API maximum of 1000 results), run once. The raw results,
including each file's commit ref, are frozen in `survey/snapshot.json`. GitHub search is not
reproducible; the snapshot is the reproducible starting point.

Known sampling biases, stated in advance: GitHub code search indexes only files under roughly
384 KB, so notebooks with large embedded outputs are under-represented; results are ranked
by relevance and capped at 1000, so the snapshot is not a uniform sample of all matches.

## Ordering (mechanical, before any notebook is opened)
1. Drop results whose repository is a fork.
2. Group the remaining notebooks by repository.
3. Order repositories by a seeded shuffle: `numpy.random.default_rng(20260922)` permutation
   of the repository names sorted lexicographically.
4. Within a repository, notebooks are screened in lexicographic path order, and at most the
   first eligible one is included.

## Screening (in that order, until 30 notebooks are included)
Each notebook is opened at the commit ref recorded in the snapshot. It is **included** if all hold:
- E0. It is fetchable at that ref and parses as notebook JSON.
- E1. It uses the 918-row Kaggle CSV: a printed shape of (918, 12), or it loads a file with
  exactly the 12 Kaggle columns and nothing in the notebook shows a different row count.
  Merged or other heart datasets are excluded.
- E2. It fits at least one supervised classifier predicting `HeartDisease`.
- E3. It shows at least one numeric evaluation metric for that classifier on data not used to
  fit it (a test split or cross-validation), visible in a cell output or in markdown. Code
  that computes a metric but never shows it does not qualify.
- E4. Its code cells are not a near-verbatim copy of a notebook already included (identical
  code apart from whitespace, comments and variable names). Course-exercise solutions are
  eligible; a second copy of the same solution is not.

Each exclusion records the first failing criterion. Screening stops at the 30th inclusion.
All screened notebooks, included or not, are listed in `survey/screening.csv`.

## Headline model and value
**Headline model**, first rule that applies: (a) the model named in the final markdown
conclusion; (b) the model selected or saved last in code; (c) the model with the best value
of the primary metric.

**Primary metric**: accuracy if reported for the headline model, otherwise the metric named
in the conclusion.

**Value**: if several values are shown for the headline model, the held-out test-set value of
the last-executed configuration. A cross-validation mean is used only if no test-set value
exists. Training-set scores are never used. Percentages are divided by 100.

## Rubric (coded for each included notebook)

| field | values | definition |
|---|---|---|
| `metric_name` | accuracy / auc / f1 / other | primary metric (above) |
| `metric_value` | float in [0,1] | headline value (above) |
| `accuracy_value` | float in [0,1] or blank | headline model's held-out accuracy if shown anywhere, same selection rule; else blank |
| `value_source` | output / text | where `metric_value` was read |
| `eval_design` | single_split / cv / both | how the headline value was produced |
| `test_size` | float or blank | test fraction of the headline split (sklearn default 0.25 if unspecified) |
| `stratify` | yes / no / na | headline split stratified on the target |
| `seed_fixed` | yes / no | `random_state` fixed on the headline split |
| `models` | `;`-separated, closed list | model types evaluated: logistic_regression, knn, svm, decision_tree, random_forest, gradient_boosting, xgboost, lightgbm, catboost, naive_bayes, mlp, other |
| `headline_model` | one value from the same list | headline model (above) |
| `n_models` | int | distinct model types evaluated on the same test set; tuned and untuned versions of one type count once |
| `best_of_n` | yes / no | n_models ≥ 2 and the headline value is the maximum (ties count as maximum) over models on one shared test set |
| `tuning` | none / grid / random / other | hyperparameter search for the headline model |
| `hyperparams` | default / set | headline model's hyperparameters left at defaults or set by hand/tuning |
| `fit_before_split` | yes / no / na | any fitted transform or statistic computed on rows that include the test rows and then used by the model: scaler, imputer, SMOTE, PCA, feature selection (including by feature importance or correlation), or anything fit before `cross_val_score` outside a Pipeline. One-hot/label encoding does not count. na if none is used. |
| `scaling` | standard / minmax / none / other | feature scaling |
| `encoding` | onehot / label / mixed / none | categorical encoding |
| `chol_zero` | ignored / imputed / dropped / other | treatment of `Cholesterol == 0` |
| `chol_zero_before_split` | yes / no / na | if imputed with a data-derived statistic, whether that statistic included test rows; na for constants or when not imputed |
| `outlier_removal` | yes / no | rows removed as outliers |
| `outlier_before_split` | yes / no / na | if removed, whether before the split |
| `resampling` | none / smote / other | class rebalancing |
| `features_dropped` | `;`-separated or blank | input columns removed before modelling |
| `dedup` | dropped / checked / none | duplicates dropped, only checked (the CSV has no exact duplicates), or not addressed |
| `source_imputation_mentioned` | yes / no | any statement that `ST_Slope` or other values were missing or imputed in the source data |

## Reliability
10 included notebooks (`default_rng(20260922).choice(30, 10, replace=False)` over inclusion
order) are coded a second time, independently, without sight of the first coding. Per-field
percent agreement is reported; fields below 80% agreement are flagged in the report.
Disagreements are resolved by the researcher re-reading the notebook, with both codings
visible, and this is stated. The resolved value is used.

## Modal pipeline
For each field, the most frequent value across the 30 included notebooks (blank values
ignored). **Ties** go to whichever tied value appears first in screening order, and are
reported. **Models**: those appearing in at least 50% of notebooks; if fewer than 2 qualify,
the 4 most frequent, including any tied for 4th. The modal pipeline combines per-field modes
and need not match any single notebook; the report says so. It defines the control in plan
2b. If the mode is not what the spec assumed, the control follows the mode.

## Reporting
No author or repository is named in narrative text. URLs and commit SHAs appear only in the
appendix (`screening.csv`).
