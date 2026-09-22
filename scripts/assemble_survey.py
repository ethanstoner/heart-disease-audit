"""Build screening.csv and coding.csv from the coders' output. See survey/PROTOCOL.md and DEVIATIONS.md.

Usage: python scripts/assemble_survey.py BATCH_JSON [BATCH_JSON ...]
Writes survey/coder_output.json (all coder records), survey/screening.csv, survey/coding.csv
(primary, deviation D1) and survey/coding_literal.csv (E4 read literally).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from heart_audit.survey import RUBRIC, rubric_violations, screening_order

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "survey"
N_TARGET = 30

# Deviation D1: functional copies of an earlier inclusion (file -> the inclusion it copies).
E4_COPIES = {"010-1": "005-1", "016-1": "005-1", "035-1": "005-1", "037-1": "005-1", "061-1": "005-1"}

# Researcher resolutions, applied over the first coding: (file, field) -> (value, reason).
RESOLUTIONS = {
    ("041-1", "stratify"): ("yes", "double-coding disagreement: cv=5 on a classifier is StratifiedKFold"),
    ("031-1", "stratify"): ("yes", "consistency (D3): cv=5 on a classifier is StratifiedKFold"),
}


def load(paths: list[str]) -> list[dict]:
    records = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))]
    records.sort(key=lambda r: r["file"])
    for r in records:
        r["file"] = r["file"].removesuffix(".txt")
    return records


def resolve(records: list[dict]) -> None:
    by_file = {r["file"]: r for r in records}
    for (f, field), (value, _) in RESOLUTIONS.items():
        by_file[f]["coding"][field] = value


def screen(records: list[dict], e4: dict[str, str]) -> pd.DataFrame:
    snapshot = json.loads((SURVEY / "snapshot.json").read_text(encoding="utf-8"))["items"]
    order = screening_order(snapshot)
    rows, included = [], 0
    for r in records:
        pos, k = (int(x) for x in r["file"].split("-"))
        repo, notebooks = order[pos - 1]
        item = notebooks[k - 1]
        # Identity comes from the snapshot; the coder's transcription is only cross-checked.
        assert item["path"] == r["path"], r["file"]
        if repo != r["repo"]:
            print(f"{r['file']}: coder wrote repo {r['repo']!r}, snapshot has {repo!r}")
        decision, criterion, evidence = r["decision"], r["criterion"], r["evidence"]
        if decision == "include" and r["file"] in e4:
            decision, criterion, evidence = "exclude", "E4", f"functional copy of {e4[r['file']]} (deviation D1)"
        rows.append({
            "position": pos, "file": r["file"], "repo": repo, "path": item["path"], "ref": item["ref"],
            "html_url": item["html_url"], "decision": decision, "criterion": criterion, "evidence": evidence,
            **({f: r["coding"][f] for f in RUBRIC} if decision == "include" else {}),
        })
        included += decision == "include"
        if included == N_TARGET:
            break
    if included < N_TARGET:
        raise SystemExit(f"only {included} inclusions: screen more repositories")
    return pd.DataFrame(rows)


def main() -> None:
    records = load(sys.argv[1:])
    for r in records:
        if r["decision"] == "include" and (bad := rubric_violations(r["coding"])):
            raise SystemExit(f"{r['file']}: {bad}")
    (SURVEY / "coder_output.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    resolve(records)

    primary = screen(records, E4_COPIES)
    primary[["position", "file", "repo", "path", "ref", "html_url", "decision", "criterion", "evidence"]].to_csv(
        SURVEY / "screening.csv", index=False)
    coding_cols = ["position", "file", *RUBRIC]
    primary.loc[primary.decision == "include", coding_cols].to_csv(SURVEY / "coding.csv", index=False)

    literal = screen(records, {})
    literal.loc[literal.decision == "include", coding_cols].to_csv(SURVEY / "coding_literal.csv", index=False)
    print(f"screened {len(primary)} (primary), {len(literal)} (literal); 30 included in each")


if __name__ == "__main__":
    main()
