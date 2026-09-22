"""Fetch notebooks in screening order at their snapshot refs and flatten them to text.

Output goes to a directory outside the repository: notebook contents are never committed.
Usage: python scripts/fetch_notebooks.py OUT_DIR FIRST_REPO LAST_REPO   (1-based, inclusive)
"""
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

from heart_audit.survey import screening_order

SNAPSHOT = Path(__file__).resolve().parents[1] / "survey" / "snapshot.json"
MAX_OUTPUT_CHARS = 3000


def flatten(nb: dict) -> str:
    parts = []
    for n, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        parts.append(f"### [{n}] {cell.get('cell_type')}\n{src}")
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain") or ""
            text = "".join(text)
            if text:
                parts.append(f"--- output\n{text[:MAX_OUTPUT_CHARS]}")
    return "\n".join(parts)


def main() -> None:
    out_dir, first, last = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    order = screening_order(json.loads(SNAPSHOT.read_text(encoding="utf-8"))["items"])
    for pos in range(first, last + 1):
        repo, notebooks = order[pos - 1]
        for k, item in enumerate(notebooks, 1):
            name = f"{pos:03d}-{k}.txt"
            url = f"https://raw.githubusercontent.com/{repo}/{item['ref']}/{quote(item['path'])}"
            header = f"repo: {repo}\npath: {item['path']}\nref: {item['ref']}\nurl: {item['html_url']}\n\n"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "heart-disease-audit"}), timeout=60) as r:
                    body = flatten(json.loads(r.read().decode("utf-8")))
            except Exception as exc:  # E0: record and move on
                body = f"E0 FETCH_OR_PARSE_FAILED: {type(exc).__name__}: {exc}"
            (out_dir / name).write_text(header + body, encoding="utf-8")
            print(name, repo, item["path"], len(body))


if __name__ == "__main__":
    main()
