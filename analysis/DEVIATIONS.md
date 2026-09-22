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
