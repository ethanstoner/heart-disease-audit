# Deviations from the pre-registered protocol

Each entry was written and committed before the data it affects was collected or analysed,
unless stated otherwise.

## D1 — E4 applied as functional equivalence (2026-09-22)

**When:** after screening repositories 1–50 (31 inclusions), before screening any further
repository and before any coded field was aggregated.

**Problem:** E4's aim is "a second copy of the same solution is not [eligible]", but its
parenthetical test, "identical code apart from whitespace, comments and variable names",
does not exclude the copies actually found. Five included notebooks (screening positions 5,
10, 16, 35, 37) are the same course lab. Positions 10, 16 and 37 differ from 5 only by:
`help(train_test_split)`, a plot-style line, print formatting, `early_stopping_rounds`
moved from the XGBoost constructor to `fit()` (the same behaviour), and in 37 one `max_depth`
value (3 → 4). Position 35 is the same lab reformatted, with an identical split, seed and
models. Read literally, one lab would supply 5 of 30 notebooks, which is the
over-representation E4 was written to prevent.

**Primary analysis:** a notebook fails E4 if it is the same solution as an earlier inclusion
up to non-functional changes: whitespace, comments, variable names, help/print/plot-style
lines, argument order, and equivalent API forms. A change to a single hyperparameter value
inside otherwise identical lab code does not make it a different solution. Positions 10,
16, 35 and 37 are excluded as copies of 5. Screening continues past repository 50 until 30
inclusions are reached.

**Sensitivity analysis:** the literal reading (no E4 exclusions among 1–50) gives 31
inclusions in repositories 1–50. Stopping at the 30th drops position 49. The modal pipeline
and the headline-accuracy distribution are reported under both readings.

## D2 — PCA-exercise value rule: pre-registered reading kept, alternative reported

**When:** same point as D1.

**Observation:** several included notebooks (positions 3, 6, 22, 28) evaluate each model
twice, before and after PCA. The value rule ("the last-executed configuration") selects the
post-PCA value, which is usually lower (for example 0.8611 rather than 0.9444).

**Decision:** no deviation. The pre-registered reading is primary. The alternative, the
headline model's best shown test value, is recorded in each notebook's `notes` and reported
as a sensitivity analysis of the reproduction gate.

**Applied again after screening repositories 51–62:** position 61 is the same lab as position
5 and differs only in the ways listed above, so it is excluded under D1. Positions 22 and 54
(the same PCA exercise) differ functionally: one ordinal-encodes three columns and the other
one-hot encodes all of them. Both are retained.

## D3 — Consistency correction to `stratify` for cross-validated notebooks (2026-09-22)

**When:** while resolving the double-coding disagreements, before any field was aggregated.

The only double-coding disagreement (position 41, `stratify`: `yes` vs `na`) exposed an
inconsistency. All three notebooks whose headline value comes from cross-validation
(positions 31, 41, 47) use `cross_val_score(..., cv=<int>)` on a classifier, which sklearn
runs as `StratifiedKFold`. Positions 41 and 47 were coded `yes`; position 31 was coded `na`
by both coders. The factually correct value, `yes`, is applied to all three. Position 31 is
changed even though its two codings agreed, and this is outside the protocol's
disagreement-resolution step. Every override is listed in `RESOLUTIONS` in
`scripts/assemble_survey.py`.

**Caveat on reliability:** both coders are instances of the same language model working from
the same protocol. The 99.6% agreement (249/250 field values) measures how clear the rubric is
and how consistently it is applied. It is not agreement between independent human judges.
