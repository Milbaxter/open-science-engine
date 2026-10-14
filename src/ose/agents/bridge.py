"""Bridge agent: find research groups that sit close in meaning but rarely
cite each other, and ask the LLM what links them (Swanson-style A–B–C).

1. topic maps for each seed query (pre-cutoff) -> research groups
2. score group pairs: near on the science map, few citations between them
3. LLM reads both groups' core/rising/newest papers, proposes cross-group pairs
4. keep candidates nobody has co-mentioned before the cutoff, top-k by confidence
"""

import math

from ..llm import LLMError, complete_json

PAIRS_PER_RUN = 12
PROMPT = """You are hunting for undiscovered connections in {disease} research.
Literature visible to you ends {cutoff}. Two research communities below work on
nearby problems but almost never cite each other.

GROUP A: {name_a}
{papers_a}

GROUP B: {name_b}
{papers_b}

Propose {n} hypotheses, each linking one specific entity typical of group A
(drug, gene, protein, pathway, cell type, exposure, biomarker...) to one specific
entity typical of group B, which you predict upcoming {disease} papers will
connect. Use names as written in abstracts; synonyms as "name1 | name2".
Avoid links that are already well known.

Return a JSON list: {{"a": str, "b": str, "relation": str, "mechanism": str, "support_ids": [paper ids], "confidence": 0-1}}"""


def _papers(group: dict, per_lens: int = 4) -> str:
    seen, lines = set(), []
    for lens in ("core", "rising", "newest", "bridges"):
        for p in (group.get("lenses") or {}).get(lens, [])[:per_lens]:
            if p["id"] not in seen:
                seen.add(p["id"])
                lines.append(f"- [{p['id']}] ({p.get('year')}, {lens}) {p.get('title')}")
    return "\n".join(lines)


def bridge_pairs(maps: list[dict]) -> list[tuple[float, dict, dict]]:
    out = []
    for m in maps:
        groups = m.get("groups", [])
        for i, ga in enumerate(groups):
            for gb in groups[i + 1:]:
                ca, cb = ga.get("centre") or {}, gb.get("centre") or {}
                if not ca or not cb:
                    continue
                dist = math.dist((ca["x"], ca["y"]), (cb["x"], cb["y"]))
                cites = (ga.get("cites_groups") or {}).get(str(gb["id"]), 0) + \
                        (gb.get("cites_groups") or {}).get(str(ga["id"]), 0)
                density = cites / math.sqrt(ga["size"] * gb["size"])
                out.append(((1 - dist) / (1 + density), ga, gb))
    out.sort(key=lambda t: -t[0])
    seen, uniq = set(), []
    for s, a, b in out:
        key = frozenset((a["name"], b["name"]))
        if key not in seen:
            seen.add(key)
            uniq.append((s, a, b))
    return uniq


def propose(task, sb, seed=0):
    maps = [sb.topics(q, papers=1000, abstracts="none") for q in task.seed_queries]
    cands = []
    for score, ga, gb in bridge_pairs(maps)[seed * PAIRS_PER_RUN:(seed + 1) * PAIRS_PER_RUN]:
        prompt = PROMPT.format(disease=task.disease, cutoff=task.cutoff, n=4,
                               name_a=ga["name"], papers_a=_papers(ga),
                               name_b=gb["name"], papers_b=_papers(gb))
        try:
            hyps = complete_json(prompt, task.model)
        except (LLMError, ValueError) as e:
            print(f"  bridge: skipped pair ({e})")
            continue
        for h in hyps:
            h["bridge"] = [ga["name"], gb["name"], round(score, 3)]
        cands += hyps

    cands.sort(key=lambda h: -float(h.get("confidence", 0)))
    out, seen = [], set()
    for h in cands:
        key = frozenset((h["a"].lower(), h["b"].lower()))
        if key in seen:
            continue
        seen.add(key)
        if sb.count(h["a"], h["b"]) == 0:  # agent-side novelty check, pre-cutoff only
            out.append(h)
        if len(out) == task.k:
            break
    return out
