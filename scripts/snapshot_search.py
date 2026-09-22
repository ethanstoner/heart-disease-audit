"""Run the pre-registered search once and freeze the raw results. See survey/PROTOCOL.md."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

QUERY = "ST_Slope ChestPainType HeartDisease extension:ipynb"
OUT = Path(__file__).resolve().parents[1] / "survey" / "snapshot.json"


def page(n: int) -> dict:
    out = subprocess.run(
        ["gh", "api", "-X", "GET", "search/code", "-f", f"q={QUERY}", "-f", "per_page=100", "-f", f"page={n}"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(out.stdout)


def main() -> None:
    items, total = [], None
    for n in range(1, 11):
        data = page(n)
        total = data["total_count"]
        for it in data["items"]:
            items.append({
                "repo": it["repository"]["full_name"],
                "fork": it["repository"]["fork"],
                "path": it["path"],
                "sha": it["sha"],
                "ref": parse_qs(urlparse(it["url"]).query)["ref"][0],
                "html_url": it["html_url"],
            })
        if len(data["items"]) < 100:
            break
        time.sleep(7)  # code search allows 10 requests/minute
    OUT.write_text(json.dumps({
        "query": QUERY,
        "taken_at": datetime.now(timezone.utc).isoformat(),
        "total_count": total,
        "items": items,
    }, indent=1), encoding="utf-8")
    print(f"total_count={total} saved={len(items)}")


if __name__ == "__main__":
    main()
