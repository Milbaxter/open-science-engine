"""Rate-limited, disk-cached client for the ScienceMadeReadable search API.

Public limit is 30 requests/min/IP (a topic map costs 5). Every response is
cached in data/cache.sqlite so reruns and graders are free.
"""

import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

BASE = "https://api.sciencemadereadable.com/api/data"
DATA = Path(__file__).resolve().parents[2] / "data"
MIN_INTERVAL = 2.1  # seconds per request unit -> ~28/min


class Api:
    def __init__(self, cache_path: Path = DATA / "cache.sqlite"):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self.http = httpx.Client(timeout=60, headers={"User-Agent": "open-science-engine/0.1"})
        self._next = 0.0
        self.calls = 0

    def get(self, path: str, cost: int = 1, **params) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        key = path + "?" + urlencode(sorted(params.items()))
        row = self.db.execute("select v from c where k=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        for attempt in range(8):
            wait = self._next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next = time.monotonic() + MIN_INTERVAL * cost
            r = self.http.get(BASE + path, params=params)
            self.calls += 1
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10)) + 1)
                continue
            if r.status_code in (502, 503, 504):
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            data = r.json()
            self.db.execute("insert or replace into c values (?,?)", (key, json.dumps(data)))
            self.db.commit()
            return data
        raise RuntimeError(f"API kept failing: {key}")

    # thin wrappers -------------------------------------------------------
    def search(self, q, **kw):
        return self.get("/search", q=q, **kw)

    def similar(self, ref, **kw):
        return self.get(f"/similar/{ref}", **kw)

    def works(self, ref):
        return self.get(f"/works/{ref}")

    def topics(self, q, **kw):
        return self.get("/topics", cost=5, q=q, **kw)

    def count(self, query: str, **kw) -> int:
        """Exact keyword match count (BM25 mode reports `matches`)."""
        d = self.search(query, mode="keyword", k=1, abstracts="false", **kw)
        return int(d.get("matches") or 0)
