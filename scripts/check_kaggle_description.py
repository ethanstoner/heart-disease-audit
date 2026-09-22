"""Check whether the Kaggle dataset description discloses missing values or imputation.

Uses Kaggle's public metadata endpoint (no API key needed). Writes results/kaggle_description_check.json.
"""
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/view/fedesoriano/heart-failure-prediction"
TERMS = ["missing", "imput", "fill", "?", "NaN", "null", "zero", "duplicat", "1190", "918"]
OUT = Path(__file__).resolve().parents[1] / "results" / "kaggle_description_check.json"


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "heart-disease-audit"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    desc = meta["descriptionNullable"]
    counts = {t: len(re.findall(re.escape(t), desc, flags=re.I)) for t in TERMS}
    out = {"checked_at": datetime.now(timezone.utc).isoformat(), "url": URL,
           "description_chars": len(desc), "dataset_last_updated": meta.get("lastUpdated"),
           "term_counts": counts,
           "discloses_missing_or_imputation": any(counts[t] for t in ["missing", "imput", "fill", "?", "NaN", "null"])}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
