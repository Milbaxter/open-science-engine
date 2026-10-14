"""Baseline: link prediction on the co-occurrence graph.

Adamic-Adar over shared neighbours, boosted by how fast each entity is
growing in recent years. Classic Science4Cast-style predictor; the bar an
LLM agent has to clear.
"""

import numpy as np

from .graph import Graph, take_novel


def propose(task, sb, seed=0):
    g = Graph(task)
    deg = g.adj.sum(1)
    inv_log = np.where(deg > 1, 1 / np.log(np.maximum(deg, 2)), 0)
    aa = (g.adj * inv_log) @ g.adj
    score = aa * np.sqrt(np.outer(g.growth, g.growth))
    score[g.adj > 0] = -1
    np.fill_diagonal(score, -1)
    iu = np.triu_indices_from(score, 1)
    order = np.argsort(-score[iu])
    ranked = ((g.names[iu[0][o]], g.names[iu[1][o]]) for o in order[:2000])
    return take_novel(sb, ranked, task.k, conf=lambda i: round(0.3 - 0.01 * i, 2))
