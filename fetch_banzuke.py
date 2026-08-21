#!/usr/bin/env python3
"""Download all banzuke data (Makuuchi + Juryo) from sumo-api.com.

Saves raw JSON per basho/division to data/banzuke/{bashoId}_{division}.json.
Re-runnable: skips files that already exist, so run it again after each
basho to pick up the newest data.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

BASE_URL = "https://www.sumo-api.com/api/basho/{basho}/banzuke/{division}"
OUT_DIR = Path(__file__).parent / "data" / "banzuke"
DIVISIONS = ("Makuuchi", "Juryo")
FIRST_BASHO = "195911"  # earliest basho with yusho + special prize data
BASHO_MONTHS = (1, 3, 5, 7, 9, 11)
DELAY_S = 0.15
RETRIES = 3


def basho_ids():
    """All basho IDs from FIRST_BASHO through the current month."""
    today = date.today()
    for year in range(int(FIRST_BASHO[:4]), today.year + 1):
        for month in BASHO_MONTHS:
            if (year, month) > (today.year, today.month):
                return
            basho = f"{year}{month:02d}"
            if basho >= FIRST_BASHO:
                yield basho


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
    fetched, skipped, empty = 0, 0, []
    for basho in basho_ids():
        for division in DIVISIONS:
            path = OUT_DIR / f"{basho}_{division}.json"
            if path.exists():
                skipped += 1
                continue
            data = fetch(BASE_URL.format(basho=basho, division=division))
            if not data.get("bashoId"):  # cancelled/missing basho
                empty.append(f"{basho} {division}")
                continue
            path.write_text(json.dumps(data, ensure_ascii=False))
            fetched += 1
            print(f"{basho} {division}: {len(data.get('east') or []) + len(data.get('west') or [])} rikishi")
            time.sleep(DELAY_S)
    print(f"\nDone. Fetched {fetched}, skipped {skipped} existing.")
    if empty:
        print(f"No data for: {', '.join(empty)}")


if __name__ == "__main__":
    main()
