"""Agents take (task, sandbox, seed) and return task.k hypotheses:
{"a", "b", "relation", "mechanism", "support_ids", "confidence"}.
Names may list synonyms as "name1 | name2". Baselines live here too.
"""

from importlib import import_module

AGENTS = {"random": "random_pairs", "trend": "trend", "plain_llm": "plain_llm", "bridge": "bridge"}


def get(name: str):
    return import_module(f".{AGENTS[name]}", __name__).propose
