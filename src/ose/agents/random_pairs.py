"""Baseline: random novel pairs of corpus entities."""

import numpy as np

from .graph import Graph, take_novel


def propose(task, sb, seed=0):
    g = Graph(task)
    rng = np.random.default_rng(seed)
    n = len(g.names)

    def pairs():
        while True:
            i, j = rng.choice(n, 2, replace=False)
            if not g.adj[i, j]:
                yield g.names[i], g.names[j]

    return take_novel(sb, pairs(), task.k, conf=lambda i: 0.05)
