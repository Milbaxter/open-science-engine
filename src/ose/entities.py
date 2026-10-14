"""Entity vocabulary + per-paper mentions from the pre-cutoff corpus.

Candidate names come from MeSH headings; a name counts as an
entity only if it literally appears in >=3 titles/abstracts, because grading
is done with exact keyword search. Output: data/<task>/entities.json
"""

import json
import re
from collections import Counter

from .corpus import load
from .task import Task

GENERIC = {
    "humans", "animals", "male", "female", "adult", "aged", "middle aged", "child", "adolescent",
    "young adult", "infant", "mice", "rats", "mice, transgenic", "disease models, animal",
    "retrospective studies", "prospective studies", "cohort studies", "case-control studies",
    "cross-sectional studies", "follow-up studies", "time factors", "risk factors", "treatment outcome",
    "surveys and questionnaires", "incidence", "prevalence", "prognosis", "disease progression",
    "severity of illness index", "cells, cultured", "mutation", "neurons", "motor neurons", "brain",
    "spinal cord", "neurodegenerative diseases", "disease", "medicine", "biology", "chemistry",
    "computer science", "psychology", "physics", "pathology", "internal medicine", "neuroscience",
    "gene", "genetics", "cell", "protein", "patients", "phenotype", "pathogenesis", "in vivo",
    "in vitro", "cell biology", "biochemistry", "cancer research", "immunology", "pharmacology",
    "family", "physical therapy", "pediatrics", "surgery", "cohort", "population", "therapy",
    "clinical trial", "randomized controlled trial", "meta-analysis", "review", "survival",
    "amyotrophic lateral sclerosis", "motor neuron disease", "als", "motor neurone disease",
    "sclerosis", "lateral sclerosis", "amyotrophic", "disease model", "animal model", "mechanism",
    # MeSH headings that are also everyday words, so literal text matches are noise
    "research", "risk", "lead", "light", "knowledge", "attention", "communication", "behavior",
    "language", "environment", "memory", "mortality", "syndrome", "mutant", "genes", "oxygen",
    "acids", "viral", "membrane", "nuclear", "blood", "plasma", "serum", "muscle", "muscles", "aging",
    "metabolism", "homeostasis", "rna", "dna", "genome", "cytoplasm", "organelles", "microscopy",
    "sarcoma", "germany", "united kingdom", "united states", "plant", "reflex", "colon", "causality",
    "uncertainty", "nervous system", "central nervous system", "mitochondrial", "atrophy", "paralysis",
    "dna-binding", "rna-binding", "gene expression", "cell death", "quality of life", "stem cells",
    "antibodies", "electrophysiology", "locomotion", "vitamins", "cough", "pneumonia", "dyspnea",
    "extracellular space", "bone marrow", "biomarkers", "temporality", "taxonomy", "syllable",
    "telecommunications", "veterinary medicine", "virulence", "warmblood", "treatment failure",
}
MIN_DF, MAX_DF_FRAC, MAX_NGRAM = 3, 0.15, 5


def norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "").lower()
    s = re.sub(r"[^a-z0-9\-\+]+", " ", s)
    return " ".join(s.split())


def ngrams(text: str) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i:i + n]) for n in range(1, MAX_NGRAM + 1) for i in range(len(toks) - n + 1)}


def candidate_names(p: dict) -> set[str]:
    names = set()
    for field in ("mesh",):
        for raw in (p.get(field) or "").split("|"):
            raw = re.sub(r"\(.*?\)", "", raw).strip()
            raw = re.sub(r",\s*human$", "", raw, flags=re.I)
            raw = re.sub(r"\s+proteins?$", "", raw, flags=re.I) if len(raw.split()) > 1 else raw
            n = norm(raw)
            if n and "," not in raw and n not in GENERIC and len(n) > 2:
                names.add(n)
    return names


def build(task: Task) -> dict:
    papers = load(task)
    cands = Counter()
    for p in papers:
        cands.update(candidate_names(p))
    vocab = {n for n, c in cands.items() if c >= 2}

    grams = [ngrams(norm((p.get("title") or "") + " " + (p.get("abstract") or ""))) for p in papers]
    df = Counter()
    for g in grams:
        df.update(vocab & g)
    n_docs = len(papers)
    ents = {n for n, c in df.items() if c >= MIN_DF and c / n_docs <= MAX_DF_FRAC}

    docs = []
    for p, g in zip(papers, grams):
        hits = sorted(ents & g)
        if hits:
            docs.append({"id": p["id"], "year": p.get("year"), "e": hits})
    out = {"n_docs": n_docs, "df": {e: df[e] for e in sorted(ents)}, "docs": docs}
    (task.data_dir / "entities.json").write_text(json.dumps(out))
    print(f"entities: {len(ents)} entities over {n_docs} papers ({len(docs)} with mentions)")
    return out


def load_entities(task: Task) -> dict:
    return json.loads((task.data_dir / "entities.json").read_text())
