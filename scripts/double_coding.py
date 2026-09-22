"""Per-field agreement between first and second coding. See survey/PROTOCOL.md "Reliability".

Usage: python scripts/double_coding.py SECOND_CODING_JSON
Writes survey/double_coding.csv (one row per notebook x field) and prints per-field agreement.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

from heart_audit.survey import RUBRIC

SURVEY = Path(__file__).resolve().parents[1] / "survey"


def norm(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str) and ";" in v:
        return ";".join(sorted(v.split(";")))
    return str(v)


def main() -> None:
    first = {r["file"]: r for r in json.loads((SURVEY / "coder_output.json").read_text(encoding="utf-8"))}
    second = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = []
    for r in second:
        f = r["file"].removesuffix(".txt")
        a, b = first[f]["coding"], r["coding"] or {}
        for field in RUBRIC:
            rows.append({"file": f, "field": field, "first": norm(a.get(field)), "second": norm(b.get(field)),
                         "agree": norm(a.get(field)) == norm(b.get(field)), "resolved": "", "resolution_note": ""})
    df = pd.DataFrame(rows)
    df.to_csv(SURVEY / "double_coding.csv", index=False)
    agreement = df.groupby("field", sort=False)["agree"].mean().mul(100).round(0)
    print(agreement.to_string())
    print(f"overall {df.agree.mean() * 100:.1f}%  fields < 80%: {list(agreement[agreement < 80].index)}")


if __name__ == "__main__":
    main()
