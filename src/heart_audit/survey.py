"""Pre-registered notebook survey: mechanical pre-filter and seeded screening order. See survey/PROTOCOL.md."""
from __future__ import annotations

import numpy as np

SEED = 20260922


def prefilter(items: list[dict]) -> list[dict]:
    """Drop forks; keep the lexicographically first notebook path per repository."""
    best: dict[str, dict] = {}
    for item in items:
        if item["fork"]:
            continue
        current = best.get(item["repo"])
        if current is None or item["path"] < current["path"]:
            best[item["repo"]] = item
    return sorted(best.values(), key=lambda i: (i["repo"], i["path"]))


def screening_order(items: list[dict], seed: int = SEED) -> list[dict]:
    ordered = sorted(items, key=lambda i: (i["repo"], i["path"]))
    perm = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[k] for k in perm]
