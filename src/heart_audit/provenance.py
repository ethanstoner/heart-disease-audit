"""Match rows of the published Kaggle CSV back to the UCI source rows they came from.

A UCI row and a Kaggle row are compatible when they agree on every key field the UCI row
actually recorded ('?' fields are skipped, because Kaggle filled them). A pair is accepted
only when each row is the other's single compatible candidate. ST_Slope is excluded from
the key, so agreement on it validates the matcher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from heart_audit.data import FEATURES, TARGET

MATCH_KEY = [f for f in FEATURES if f != "ST_Slope"] + [TARGET]
_CODES = {"M": 1, "F": 0, "TA": 1, "ATA": 2, "NAP": 3, "ASY": 4,
          "Normal": 0, "ST": 1, "LVH": 2, "Y": 1, "N": 0}


def _numeric_key(df: pd.DataFrame, key: list[str]) -> np.ndarray:
    cols = []
    for col in key:
        s = df[col]
        s = s.astype(float) if s.dtype.kind in "biuf" else s.map(_CODES).astype(float)
        cols.append(s.round(1).to_numpy())
    return np.column_stack(cols)


def _compatible(kaggle: pd.DataFrame, uci: pd.DataFrame, key: list[str]) -> np.ndarray:
    """Boolean matrix [uci row, kaggle row]."""
    u, k = _numeric_key(uci, key), _numeric_key(kaggle, key)
    ok = np.ones((len(u), len(k)), dtype=bool)
    for j in range(u.shape[1]):
        recorded = ~np.isnan(u[:, j])
        ok &= ~recorded[:, None] | (u[:, j][:, None] == k[:, j][None, :])
    return ok


def match_sources(kaggle: pd.DataFrame, uci: pd.DataFrame, key: list[str] = MATCH_KEY) -> pd.DataFrame:
    """One row per Kaggle row, in order.

    Columns: n_candidates (compatible UCI rows), source ('unknown' when there is no candidate
    or candidates span sources), uci_row and uci_slope (set only for one-to-one pairs),
    kaggle_slope.
    """
    ok = _compatible(kaggle, uci, key)
    n_per_kaggle, n_per_uci = ok.sum(axis=0), ok.sum(axis=1)
    sources = uci["source"].to_numpy()
    uci_slope = uci["ST_Slope"].to_numpy()
    rows = []
    for j in range(len(kaggle)):
        cand = np.flatnonzero(ok[:, j])
        cand_sources = set(sources[cand])
        paired = len(cand) == 1 and n_per_uci[cand[0]] == 1
        rows.append({
            "n_candidates": len(cand),
            "source": cand_sources.pop() if len(cand_sources) == 1 else "unknown",
            "uci_row": cand[0] if paired else pd.NA,
            "uci_slope": uci_slope[cand[0]] if paired else np.nan,
        })
    out = pd.DataFrame(rows)
    out["uci_row"] = out["uci_row"].astype("Int64")
    out["kaggle_slope"] = kaggle["ST_Slope"].to_numpy()
    return out


def slope_disagreements(matches: pd.DataFrame) -> int:
    """Paired rows whose UCI slope was recorded and differs from Kaggle's. Must be 0."""
    both = matches["uci_row"].notna() & matches["uci_slope"].notna()
    return int((matches.loc[both, "uci_slope"] != matches.loc[both, "kaggle_slope"]).sum())


def filled_slopes(matches: pd.DataFrame) -> pd.DataFrame:
    """Paired rows where UCI slope was '?' but Kaggle has a value."""
    return matches[matches["uci_row"].notna() & matches["uci_slope"].isna()]


def filled_cells(kaggle: pd.DataFrame, uci: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Every cell that is '?' in the paired UCI row and concrete in Kaggle.

    Columns: kaggle_row, column, kaggle_value, source. A lower bound on the fill-in, since
    unpaired rows are not examined.
    """
    paired = matches.index[matches["uci_row"].notna()]
    rows = []
    for j in paired:
        i = int(matches.at[j, "uci_row"])
        for col in FEATURES:
            if pd.isna(uci.iloc[i][col]):
                rows.append({"kaggle_row": int(j), "column": col,
                             "kaggle_value": kaggle.iloc[j][col], "source": uci.iloc[i]["source"]})
    return pd.DataFrame(rows, columns=["kaggle_row", "column", "kaggle_value", "source"])
