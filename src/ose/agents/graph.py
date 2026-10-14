"""Shared helpers over the pre-cutoff entity co-occurrence graph."""

from itertools import combinations

import numpy as np

from ..entities import load_entities
from ..sandbox import Sandbox


class Graph:
    def __init__(self, task, top: int = 500, recent_from: int | None = None):
        ents = load_entities(task)
        recent_from = recent_from or task.last_visible_year - 4
        df_all, df_recent = {}, {}
        for d in ents["docs"]:
            for e in d["e"]:
                df_all[e] = df_all.get(e, 0) + 1
                if (d["year"] or 0) >= recent_from:
                    df_recent[e] = df_recent.get(e, 0) + 1
        self.names = sorted(df_all, key=lambda e: -df_all[e])[:top]
        idx = {e: i for i, e in enumerate(self.names)}
        n = len(self.names)
        self.adj = np.zeros((n, n), dtype=np.float32)
        for d in ents["docs"]:
            ii = [idx[e] for e in d["e"] if e in idx]
            for i, j in combinations(ii, 2):
                self.adj[i, j] = self.adj[j, i] = 1
        self.df = np.array([df_all[e] for e in self.names], dtype=np.float32)
        self.growth = np.array([(df_recent.get(e, 0) + 1) / (df_all[e] + 1) for e in self.names],
                               dtype=np.float32)


def take_novel(sb: Sandbox, pairs, k: int, conf=lambda i: 0.1, relation="is linked to",
               max_checks: int = 400):
    """Walk ranked (a, b) pairs, keep the first k never co-mentioned before the cutoff."""
    out = []
    for n, (a, b) in enumerate(pairs):
        if n >= max_checks:
            break
        if sb.count(a, b) == 0:
            out.append({"a": a, "b": b, "relation": relation, "mechanism": "",
                        "support_ids": [], "confidence": conf(len(out))})
            if len(out) == k:
                break
    return out
