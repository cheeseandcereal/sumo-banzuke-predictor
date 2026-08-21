#!/usr/bin/env python3
"""Download basho metadata (dates, yusho, special prizes) from sumo-api.com.

Saves raw JSON to data/basho/{bashoId}.json. Re-runnable: skips existing
files. Only fetches basho for which we already have banzuke data.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://www.sumo-api.com/api/basho/{basho}"
BANZUKE_DIR = Path(__file__).parent / "data" / "banzuke"
OUT_DIR = Path(__file__).parent / "data" / "basho"
DELAY_S = 0.15
RETRIES = 3


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "banzuke-fetch/1.0"})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == RETRIES:
                raise
            wait = 2**attempt
            print(f"  retry {attempt} in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bashos = sorted({p.name.split("_")[0] for p in BANZUKE_DIR.glob("*.json")})
    fetched, skipped, empty = 0, 0, []
    for basho in bashos:
        path = OUT_DIR / f"{basho}.json"
        if path.exists():
            skipped += 1
            continue
        data = fetch(BASE_URL.format(basho=basho))
        if not data.get("date"):
            empty.append(basho)
            continue
        path.write_text(json.dumps(data, ensure_ascii=False))
        fetched += 1
        time.sleep(DELAY_S)
    print(f"Done. Fetched {fetched}, skipped {skipped} existing.")
    if empty:
        print(f"No data for: {', '.join(empty)}")


if __name__ == "__main__":
    main()
