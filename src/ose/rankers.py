"""Rankers score a fixed list of (a, b) pairs; higher = more likely to be
co-mentioned after the cutoff. They see only pre-cutoff data, never labels.
Add one by writing `def name(task, pairs, seed) -> list[float]` and registering it.
"""

import numpy as np

from .llm import LLMError, complete_json


def _graph(task):
    from .agents.graph import Graph
    g = Graph(task, top=300)
    return g, {e: i for i, e in enumerate(g.names)}


def random(task, pairs, seed=0):
    return list(np.random.default_rng(seed).random(len(pairs)))


def popularity(task, pairs, seed=0):
    """Preferential attachment: frequent entities get linked first."""
    g, idx = _graph(task)
    return [float(np.log(g.df[idx[a]]) + np.log(g.df[idx[b]])) for a, b in pairs]


def trend(task, pairs, seed=0):
    """Adamic-Adar over shared neighbours x recent growth of both entities."""
    g, idx = _graph(task)
    deg = g.adj.sum(1)
    inv_log = np.where(deg > 1, 1 / np.log(np.maximum(deg, 2)), 0)
    aa = (g.adj * inv_log) @ g.adj
    return [float(aa[idx[a], idx[b]] * np.sqrt(g.growth[idx[a]] * g.growth[idx[b]])) for a, b in pairs]


LLM_PROMPT = """These entity pairs have NEVER been mentioned together in a {disease} paper
published before {cutoff}. For each pair, estimate the probability (0-100) that at
least one {disease} paper published in the following ~10 months mentions both terms
in its title or abstract.

Think about which research directions are heating up, which entities are central
enough to collide, and which pairings are implausible or semantically odd.

{lines}

Return a JSON list with one object per pair: {{"i": <id>, "p": <0-100>}}"""


def llm(task, pairs, seed=0, batch=50):
    order = np.random.default_rng(seed).permutation(len(pairs))  # avoid position effects
    scores = [0.0] * len(pairs)
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        lines = "\n".join(f"{i}. {pairs[i][0]}  +  {pairs[i][1]}" for i in chunk)
        prompt = LLM_PROMPT.format(disease=task.disease, cutoff=task.cutoff, lines=lines)
        for attempt in range(3):
            try:
                got = {int(x["i"]): float(x["p"]) for x in complete_json(prompt, task.model)}
                break
            except (LLMError, ValueError, KeyError, TypeError) as e:
                print(f"  llm ranker retry {attempt}: {str(e)[:120]}")
                got = {}
        for i in chunk:
            scores[i] = got.get(int(i), 0.0)
        print(f"  llm ranker {min(start + batch, len(order))}/{len(order)}", flush=True)
    return scores


RANKERS = {"random": random, "popularity": popularity, "trend": trend, "llm": llm}


def get(name):
    return RANKERS[name]
