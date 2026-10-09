"""Download a manga subset from the Jikan API (unofficial MyAnimeList API).

What it does
  1. Walks /top/manga (MAL's ranking, ordered by score) page by page and keeps
     entries whose type is in --types until it has --n titles.
  2. For each kept title, fetches /manga/{id}/relations. A relation of kind
     "Adaptation" pointing to an entry of type "anime" is how MAL records
     "this manga got an anime". That becomes our label later.
  3. Optionally (--anime-dates) fetches /anime/{id} for every linked anime so we
     know *when* the adaptation aired. Used only to pick the cutoff year.

Every HTTP response is cached as a JSON file under --cache-dir, keyed by URL,
so you can Ctrl-C at any point and re-run: cached URLs are not requested again.

Rate limits: Jikan allows roughly 3 requests/second and 60 requests/minute.
We stay under both by spacing requests at least --min-interval seconds apart
(default 1.1s) and backing off exponentially on 429 / 5xx responses.

Budget: ~120 list pages + 1 relations call per title. For --n 3000 that is
about 3,100 requests, roughly an hour. --anime-dates adds roughly one call per
linked anime (cached, so shared anime are only fetched once).

Usage
  python scripts/fetch_jikan.py --n 3000
  python scripts/fetch_jikan.py --n 3000 --anime-dates
Output: data/raw/manga_raw.jsonl (one JSON object per manga)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.jikan.moe/v4"


class JikanClient:
    """Tiny GET client with an on-disk cache, request spacing, and retries."""

    def __init__(self, cache_dir, min_interval=1.1, max_retries=6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "manga2anime-predictor (student project)"
        self._last_request = 0.0
        self.n_network = 0

    def _cache_path(self, path, params):
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        readable = key.strip("/").replace("/", "_").replace("?", "__").replace("&", "_").replace("=", "-")
        return self.cache_dir / f"{readable[:80]}__{digest}.json"

    def _wait_for_slot(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def get(self, path, params=None):
        """Return parsed JSON for BASE_URL+path, or None if the resource is 404."""
        cache_file = self._cache_path(path, params)
        if cache_file.exists():
            return json.loads(cache_file.read_text())["body"]

        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot()
            try:
                resp = self.session.get(BASE_URL + path, params=params, timeout=30)
            except requests.RequestException as exc:
                print(f"  network error on {path} ({exc}); retry {attempt} in {backoff:.0f}s", file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue
            self.n_network += 1

            if resp.status_code == 200:
                body = resp.json()
                break
            if resp.status_code == 404:
                body = None  # cache the miss too, so we don't keep asking
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                print(f"  HTTP {resp.status_code} on {path}; retry {attempt} in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
        else:
            raise RuntimeError(f"Giving up on {path} after {self.max_retries} attempts")

        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"path": path, "params": params, "body": body}))
        tmp.replace(cache_file)  # atomic, so Ctrl-C never leaves a half-written cache file
        return body


def fetch_top_manga(client, n, types):
    """Walk the score-ordered top list, keeping only the requested types."""
    kept, seen, page = [], set(), 1
    while len(kept) < n:
        body = client.get("/top/manga", {"page": page})
        if body is None or not body.get("data"):
            print(f"Top list ran out at page {page}; got {len(kept)} titles.")
            break
        for m in body["data"]:
            # The ranking can shift between page requests; skip duplicates.
            if m["mal_id"] in seen or m.get("type") not in types:
                continue
            seen.add(m["mal_id"])
            kept.append(m)
            if len(kept) == n:
                break
        if page % 10 == 0:
            print(f"  top list page {page}: {len(kept)}/{n} kept")
        if not body.get("pagination", {}).get("has_next_page", False):
            break
        page += 1
    return kept


def adaptation_anime_ids(relations_body):
    """mal_ids of anime linked with relation 'Adaptation'."""
    ids = []
    for rel in (relations_body or {}).get("data", []):
        if rel.get("relation") != "Adaptation":
            continue
        ids.extend(e["mal_id"] for e in rel.get("entry", []) if e.get("type") == "anime")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000, help="number of titles to collect")
    ap.add_argument("--types", default="Manga,Manhwa,Manhua",
                    help="comma-separated MAL types to keep (others, e.g. Light Novel, are skipped)")
    ap.add_argument("--anime-dates", action="store_true",
                    help="also fetch each adaptation's air date (for the cutoff-year analysis)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default="data/raw/manga_raw.jsonl")
    ap.add_argument("--min-interval", type=float, default=1.1, help="seconds between requests")
    args = ap.parse_args()

    client = JikanClient(args.cache_dir, min_interval=args.min_interval)
    types = set(args.types.split(","))

    print(f"Step 1/3: top list (types={sorted(types)})")
    manga = fetch_top_manga(client, args.n, types)

    print(f"Step 2/3: relations for {len(manga)} titles")
    for i, m in enumerate(manga, 1):
        m["relations"] = (client.get(f"/manga/{m['mal_id']}/relations") or {}).get("data", [])
        if i % 100 == 0:
            print(f"  {i}/{len(manga)}  (network requests so far: {client.n_network})")

    if args.anime_dates:
        print("Step 3/3: air dates of adaptations")
        for i, m in enumerate(manga, 1):
            m["adaptation_anime"] = []
            for aid in adaptation_anime_ids({"data": m["relations"]}):
                a = (client.get(f"/anime/{aid}") or {}).get("data")
                if a:
                    m["adaptation_anime"].append({
                        "mal_id": aid,
                        "type": a.get("type"),
                        "aired_from": (a.get("aired") or {}).get("from"),
                    })
            if i % 100 == 0:
                print(f"  {i}/{len(manga)}  (network requests so far: {client.n_network})")
    else:
        print("Step 3/3: skipped (pass --anime-dates to fetch adaptation air dates)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for m in manga:
            m.pop("images", None)  # big and useless for us
            f.write(json.dumps(m) + "\n")
    print(f"Wrote {len(manga)} titles to {out} ({client.n_network} network requests this run)")


if __name__ == "__main__":
    main()
