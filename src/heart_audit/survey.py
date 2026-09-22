"""Pre-registered notebook survey: mechanical screening order. See survey/PROTOCOL.md."""
from __future__ import annotations

import numpy as np

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
