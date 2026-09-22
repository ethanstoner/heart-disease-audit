"""Match rows of the published Kaggle CSV back to the UCI source rows they came from."""
from __future__ import annotations

import numpy as np
import pandas as pd

from heart_audit.data import FEATURES, TARGET, ZERO_MEANS_MISSING

MATCH_KEY = [f for f in FEATURES if f != "ST_Slope"] + [TARGET]
_NUMERIC_KEY = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", TARGET]


def _key(df: pd.DataFrame) -> pd.DataFrame:
    key = df[MATCH_KEY].copy()
    for col in ZERO_MEANS_MISSING:
        key[col] = key[col].fillna(0)
    for col in MATCH_KEY:
        if col in _NUMERIC_KEY:
            key[col] = key[col].astype("Float64").round(1)
        else:
            key[col] = key[col].astype("string")
    return key


def match_sources(kaggle: pd.DataFrame, uci: pd.DataFrame) -> pd.DataFrame:
    """One row per Kaggle row, in order.

    Columns: n_candidates, source ('unknown' when unmatched or when candidates span sources),
    uci_row and uci_slope (set only for a unique candidate), kaggle_slope.
    """
    k = _key(kaggle).assign(krow=np.arange(len(kaggle)))
    u = _key(uci).assign(
        urow=np.arange(len(uci)),
        source=uci["source"].to_numpy(),
        uci_slope=uci["ST_Slope"].to_numpy(),
    )
    groups = k.merge(u, on=MATCH_KEY, how="left").groupby("krow")
    first = groups.first()
    n_candidates = groups["urow"].count()
    one_source = groups["source"].nunique().eq(1)
    unique = n_candidates.eq(1)
    return pd.DataFrame({
        "n_candidates": n_candidates.to_numpy(),
        "source": np.where(one_source, first["source"], "unknown"),
        "uci_row": first["urow"].where(unique).astype("Int64").to_numpy(),
        "uci_slope": first["uci_slope"].where(unique).to_numpy(),
        "kaggle_slope": kaggle["ST_Slope"].to_numpy(),
    })


def slope_disagreements(matches: pd.DataFrame) -> int:
    """Uniquely matched rows whose UCI slope was recorded and differs from Kaggle's. Must be 0."""
    both = matches["uci_row"].notna() & matches["uci_slope"].notna()
    return int((matches.loc[both, "uci_slope"] != matches.loc[both, "kaggle_slope"]).sum())


def filled_slopes(matches: pd.DataFrame) -> pd.DataFrame:
    """Uniquely matched rows where UCI slope was '?' but Kaggle has a value. A lower bound on the fill-in."""
    return matches[matches["uci_row"].notna() & matches["uci_slope"].isna()]
