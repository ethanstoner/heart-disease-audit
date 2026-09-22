"""Global seed policy: one master seed, per-experiment seeds derived from it by name."""
from __future__ import annotations

import zlib

import numpy as np

MASTER_SEED = 20260922


def derive(name: str, n: int) -> list[int]:
    """n seeds for the experiment `name`; a prefix of derive(name, m) for any m >= n."""
    seq = np.random.SeedSequence([MASTER_SEED, zlib.crc32(name.encode())])
    return [int(s) for s in seq.generate_state(n, dtype=np.uint32) >> 1]
