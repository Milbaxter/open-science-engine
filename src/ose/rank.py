"""Ranking benchmark (Science4Cast-style): every agent scores the SAME pool of
entity pairs that were never co-mentioned with the disease before the cutoff;
we measure how well the scores separate pairs that DID get co-mentioned after.

Far more statistical power than "propose 20": hundreds of labelled pairs, ~20%
positives, AUC with a bootstrap CI.
"""

import json
from datetime import datetime, timezone

import numpy as np

from .api import Api
from .sandbox import Sandbox, boolean_query
from .task import ROOT, Task


def pool_path(task: Task):
    return task.data_dir / "pool.json"


def build_pool(task: Task, n: int = 500, top: int = 300, seed: int = 0, max_checks: int = 8000):
    """Sample locally-unlinked pairs among the `top` most frequent entities, keep the
    ones with zero pre-cutoff co-mentions, label each by post-cutoff co-mentions.
    Deterministic + API-cached, so an interrupted build resumes for free."""
    from .agents.graph import Graph

    api = Api()
    sb = Sandbox(task, api)
    g = Graph(task, top=top)
    rng = np.random.default_rng(seed)
    seen, pairs, checks = set(), [], 0
    while len(pairs) < n and checks < max_checks:
        i, j = sorted(rng.choice(len(g.names), 2, replace=False))
        if g.adj[i, j] or (i, j) in seen:
            continue
        seen.add((i, j))
        checks += 1
        a, b = g.names[i], g.names[j]
        if sb.count(a, b) == 0:
            post = api.count(boolean_query(task, [a, b]), year_from=task.cutoff_year)
            pairs.append({"a": a, "b": b, "post": post})
        if checks % 100 == 0:
            pos = sum(p["post"] > 0 for p in pairs)
            print(f"  pool: {checks} checked, {len(pairs)} novel, {pos} emerged", flush=True)
    out = {"task": task.id, "top": top, "seed": seed, "checked": checks, "pairs": pairs}
    pool_path(task).write_text(json.dumps(out))
    print(f"pool: {len(pairs)} pairs, {sum(p['post'] > 0 for p in pairs)} positives -> {pool_path(task)}")
    return out


# -- metrics --------------------------------------------------------------------
def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ranks = np.empty(len(s))
    order = s.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def metrics(scores, labels, seed: int = 0, boot: int = 1000) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    top = np.argsort(-s, kind="mergesort")
    hits = y[top]
    ap = float((np.cumsum(hits) / np.arange(1, len(y) + 1))[hits].mean()) if hits.any() else 0.0
    rng = np.random.default_rng(seed)
    bs = [auc(s[idx], y[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"auc": round(auc(s, y), 4), "auc_ci": [round(float(lo), 4), round(float(hi), 4)],
            "ap": round(ap, 4), "base_rate": round(float(y.mean()), 4),
            "p_at_50": round(float(hits[:50].mean()), 4), "p_at_100": round(float(hits[:100].mean()), 4)}


# -- run ------------------------------------------------------------------------
def run_rank(task: Task, agent: str, seed: int = 0) -> dict:
    from . import rankers
    from .llm import TOTAL_COST

    pool = json.loads(pool_path(task).read_text())
    pairs = [(p["a"], p["b"]) for p in pool["pairs"]]  # labels never leave this function
    labels = [p["post"] > 0 for p in pool["pairs"]]
    scores = rankers.get(agent)(task, pairs, seed)
    assert len(scores) == len(pairs)
    res = {"task": task.id, "agent": agent, "seed": seed, "n": len(pairs),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "usd": round(TOTAL_COST["usd"], 4), **metrics(scores, labels, seed),
           "top10": [pairs[i] for i in np.argsort(-np.asarray(scores, float), kind="mergesort")[:10]],
           "scores": [round(float(x), 5) for x in scores]}
    d = ROOT / "results" / task.id / "rank"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{agent}-s{seed}.json").write_text(json.dumps(res, indent=1))
    print(f"rank {agent} s{seed}: AUC {res['auc']} {res['auc_ci']}  AP {res['ap']} (base {res['base_rate']})  "
          f"P@50 {res['p_at_50']}  ${res['usd']}")
    return res
