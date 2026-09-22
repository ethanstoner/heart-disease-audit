"""Pre-registered notebook survey: screening order and modal pipeline. See survey/PROTOCOL.md."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

SEED = 20260922


def screening_order(items: list[dict], seed: int = SEED) -> list[tuple[str, list[dict]]]:
    """Non-fork results grouped by repository, repositories in seeded-shuffle order,
    each repository's notebooks in path order with exact duplicate results removed."""
    by_repo: dict[str, dict[str, dict]] = {}
    for item in items:
        if not item["fork"]:
            by_repo.setdefault(item["repo"], {})[item["path"]] = item
    repos = sorted(by_repo)
    perm = np.random.default_rng(seed).permutation(len(repos))
    return [(repos[k], [by_repo[repos[k]][p] for p in sorted(by_repo[repos[k]])]) for k in perm]


def modal_pipeline(coding: pd.DataFrame) -> tuple[dict, list[str]]:
    """Per-field mode over included notebooks (rows in screening order), and the fields that tied.

    Ties go to the tied value screened first. `models`: types in at least half the notebooks;
    if fewer than 2 qualify, the 4 most frequent including any tied for 4th.
    """
    modal, ties = {}, []
    for col in coding.columns:
        if col == "models":
            counts = Counter(m for cell in coding[col] for m in set(cell.split(";")))
            chosen = [m for m, c in counts.items() if c >= len(coding) / 2]
            if len(chosen) < 2:
                cutoff = sorted(counts.values(), reverse=True)[:4][-1]
                chosen = [m for m, c in counts.items() if c >= cutoff]
            modal[col] = sorted(chosen)
            continue
        values = coding[col].dropna()
        counts = values.value_counts()
        top = counts[counts == counts.max()].index
        if len(top) > 1:
            ties.append(col)
        modal[col] = next(v for v in values if v in top)
    return modal, ties
