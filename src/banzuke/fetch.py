"""Incremental fetcher for the raw sumo-api.com responses under data/.

Banzuke files (one per basho and division) and basho files (yusho, special
prizes) are refetched until the basho is complete: a banzuke with results
for a basho whose yusho is known is skipped. The next scheduled basho is
always tried, so a freshly released banzuke is picked up.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date

from banzuke.paths import RAW_BANZUKE, RAW_BASHO

BANZUKE_URL = "https://www.sumo-api.com/api/basho/{basho}/banzuke/{division}"
BASHO_URL = "https://www.sumo-api.com/api/basho/{basho}"
DIVISIONS = ("Makuuchi", "Juryo")
FIRST_BASHO = "195911"  # earliest basho with yusho + special prize data
BASHO_MONTHS = (1, 3, 5, 7, 9, 11)
DELAY_S = 0.15
RETRIES = 3


def basho_ids():
    """All basho IDs through the next scheduled basho."""
    today = date.today()
    for year in range(int(FIRST_BASHO[:4]), today.year + 2):
        for month in BASHO_MONTHS:
            basho = f"{year}{month:02d}"
            if basho >= FIRST_BASHO:
                yield basho
            if (year, month) > (today.year, today.month):
                return


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


def fetch_banzuke():
    RAW_BANZUKE.mkdir(parents=True, exist_ok=True)
    fetched, skipped, empty = 0, 0, []
    for basho in basho_ids():
        basho_path = RAW_BASHO / f"{basho}.json"
        basho_data = json.loads(basho_path.read_text()) if basho_path.exists() else {}
        for division in DIVISIONS:
            path = RAW_BANZUKE / f"{basho}_{division}.json"
            if path.exists():
                existing = json.loads(path.read_text())
                rikishi = (existing.get("east") or []) + (existing.get("west") or [])
                if any(r.get("record") for r in rikishi) and basho_data.get("yusho"):
                    skipped += 1
                    continue
            data = fetch(BANZUKE_URL.format(basho=basho, division=division))
            if not data.get("bashoId"):
                empty.append(f"{basho} {division}")
                continue
            path.write_text(json.dumps(data, ensure_ascii=False))
            fetched += 1
            count = len(data.get("east") or []) + len(data.get("west") or [])
            print(f"{basho} {division}: {count} rikishi")
            time.sleep(DELAY_S)
    print(f"Banzuke: fetched {fetched}, skipped {skipped} completed.")
    if empty:
        print(f"No banzuke data for: {', '.join(empty)}")


def fetch_basho():
    RAW_BASHO.mkdir(parents=True, exist_ok=True)
    bashos = sorted({p.name.split("_")[0] for p in RAW_BANZUKE.glob("*.json")})
    fetched, skipped, empty = 0, 0, []
    for basho in bashos:
        path = RAW_BASHO / f"{basho}.json"
        if path.exists() and json.loads(path.read_text()).get("yusho"):
            skipped += 1
            continue
        data = fetch(BASHO_URL.format(basho=basho))
        if not data.get("date"):
            empty.append(basho)
            continue
        path.write_text(json.dumps(data, ensure_ascii=False))
        fetched += 1
        time.sleep(DELAY_S)
    print(f"Basho: fetched {fetched}, skipped {skipped} completed.")
    if empty:
        print(f"No basho data for: {', '.join(empty)}")


def fetch_all():
    fetch_banzuke()
    fetch_basho()
