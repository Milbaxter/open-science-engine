"""Baseline: one prompt, no tools, no literature access."""

from ..llm import complete_json

PROMPT = """Propose {k} novel, specific, testable hypotheses about {disease}.
Each links two named entities (a drug, gene, protein, pathway, cell type,
exposure, biomarker, phenotype...) that, to your knowledge, have NOT yet been
reported together in the {disease} literature, but that you predict will be.

Use the names researchers would write in an abstract. You may give synonyms
as "name1 | name2". Prefer concrete entities over broad processes.

Return a JSON list of {k} objects:
{{"a": str, "b": str, "relation": str, "mechanism": str (one sentence), "confidence": 0-1}}"""


def propose(task, sb, seed=0):
    hyps = complete_json(PROMPT.format(k=task.k, disease=task.disease), task.model)
    for h in hyps:
        h.setdefault("support_ids", [])
    return hyps[: task.k]
