"""Grade hypotheses against the hidden post-cutoff literature.

Per hypothesis (A, relation, B):
  valid     A and B each appear verbatim in some pre-cutoff paper  (else score 0)
  novel     no pre-cutoff paper mentions disease + A + B           (else score 0)
  emerged   >=1 post-cutoff paper mentions disease + A + B          (mechanical signal)
  confirmed two LLM judges read those papers: +1 supported, -1 contradicted, 0 untested
  surprise  -log10 of how likely A and B co-occur by chance before the cutoff
score = novel * (confirmed or emerged) * surprise, averaged over task.k slots.
"""

import math

from .api import Api
from .llm import LLMError, complete_json
from .sandbox import boolean_query, term_clause
from .task import Task

JUDGE_PROMPT = """A research hypothesis was made before {cutoff} about {disease}:

  {a}  --[{relation}]-->  {b}
  Mechanism claimed: {mechanism}

Below are papers published AFTER {cutoff} that mention both entities.
Decide whether, taken together, they provide evidence about THIS SPECIFIC claim
(same direction of effect, same relation). Mere co-mention, speculation, or a
review restating old knowledge is "untested".

{papers}

Return {{"verdict": "supported" | "contradicted" | "untested", "paper_id": "<best id or null>", "why": "<one sentence>"}}"""


def _post_papers(api: Api, task: Task, q: str, n: int = 8) -> list[dict]:
    d = api.search(q, mode="keyword", k=n, year_from=task.cutoff_year)
    return [r for r in d.get("results", []) if str(r.get("date") or "9999") >= task.cutoff]


def judge(api: Api, task: Task, h: dict, q: str) -> dict:
    papers = _post_papers(api, task, q)
    if not papers:
        return {"confirmed": 0, "votes": []}
    text = "\n\n".join(f"[{p['id']}] ({p.get('date')}) {p.get('title')}\n{(p.get('abstract') or '')[:1800]}"
                       for p in papers)
    prompt = JUDGE_PROMPT.format(cutoff=task.cutoff, disease=task.disease, papers=text,
                                 a=h["a"], b=h["b"], relation=h.get("relation", "is linked to"),
                                 mechanism=h.get("mechanism", "-"))
    votes = []
    for m in task.judge_models:
        try:
            votes.append({"model": m, **complete_json(prompt, m)})
        except (LLMError, ValueError) as e:
            votes.append({"model": m, "verdict": "error", "why": str(e)[:200]})
    v = {x.get("verdict") for x in votes}
    confirmed = 1 if v == {"supported"} else -1 if v == {"contradicted"} else 0
    return {"confirmed": confirmed, "votes": votes}


def grade(task: Task, hyps: list[dict], use_judge: bool = True, api: Api | None = None) -> dict:
    assert task.cutoff.endswith("-01-01"), "M1 grader assumes a Jan-1 cutoff (exact year filters)"
    api = api or Api()
    pre_kw = {"year_to": task.last_visible_year}
    post_kw = {"year_from": task.cutoff_year}
    n_all = api.count(task.disease_query, **pre_kw)

    rows = []
    for h in hyps[: task.k]:
        q = boolean_query(task, [h["a"], h["b"]])
        pre, post = api.count(q, **pre_kw), api.count(q, **post_kw)
        na = api.count(boolean_query(task, [h["a"]]), **pre_kw)
        nb = api.count(boolean_query(task, [h["b"]]), **pre_kw)
        p = (na + 1) / (n_all + 1) * (nb + 1) / (n_all + 1)
        surprise = min(8.0, -math.log10(p))
        # a term nobody ever wrote can't be confirmed by keyword search, and would make
        # "novel" trivially true; such hypotheses are invalid and score 0
        valid = all(api.count(term_clause(t), **pre_kw) > 0 for t in (h["a"], h["b"]))
        novel, emerged = valid and pre == 0, post > 0
        row = {**h, "pre": pre, "post": post, "n_a": na, "n_b": nb, "valid": valid, "novel": novel,
               "emerged": emerged, "surprise": round(surprise, 3)}
        row["mech_score"] = round(novel * emerged * surprise, 3)
        if use_judge and novel and emerged:
            row.update(judge(api, task, h, q))
        row["score"] = round(novel * row.get("confirmed", 0) * surprise, 3) if use_judge else None
        rows.append(row)

    k = task.k
    target = [(r.get("confirmed", 0) == 1) if use_judge else (r["novel"] and r["emerged"]) for r in rows]
    brier = sum((float(r.get("confidence", 0.5)) - t) ** 2 for r, t in zip(rows, target)) / max(1, len(rows))
    return {
        "n": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "novel": sum(r["novel"] for r in rows),
        "emerged": sum(r["novel"] and r["emerged"] for r in rows),
        "confirmed": sum(r.get("confirmed", 0) == 1 for r in rows),
        "contradicted": sum(r.get("confirmed", 0) == -1 for r in rows),
        "mech_score": round(sum(r["mech_score"] for r in rows) / k, 4),
        "score": round(sum(r["score"] for r in rows) / k, 4) if use_judge else None,
        "brier": round(brier, 4),
        "rows": rows,
    }
