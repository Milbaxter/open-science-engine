"""Baseline: one prompt, no tools, no literature access."""

from ..llm import complete_json

# Grading is exact keyword search, so entity names must be the bare terms authors write.
NAMING = """Naming rules for "a" and "b" (they are matched verbatim against abstracts):
- a bare entity name of 1-3 words: "SGLT2 inhibitors", "PINK1", "Akkermansia muciniphila", "ferroptosis"
- no disease name, no qualifiers or processes glued on: write "stress granules", not "TIA1-positive stress granule clearance in ALS"
- list common synonyms/abbreviations with " | ": "nicotinamide mononucleotide | NMN"
- put the mechanism and direction in "relation" and "mechanism", never in the names"""

PROMPT = """Propose {k} novel, specific, testable hypotheses about {disease}.
Each links two named entities (a drug, gene, protein, pathway, cell type,
exposure, biomarker, phenotype...) that, to your knowledge, have NOT yet been
reported together in the {disease} literature, but that you predict will be.
Prefer concrete entities over broad processes.

""" + NAMING + """

Return a JSON list of {k} objects:
{{"a": str, "b": str, "relation": str, "mechanism": str (one sentence), "confidence": 0-1}}"""


def propose(task, sb, seed=0):
    hyps = complete_json(PROMPT.format(k=task.k, disease=task.disease), task.model)
    for h in hyps:
        h.setdefault("support_ids", [])
    return hyps[: task.k]
