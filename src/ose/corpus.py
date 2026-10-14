"""Build a pre-cutoff disease corpus from the public API.

Union of semantic seed queries sliced by year plus a keyword sweep per year.
Output: data/<task>/corpus.jsonl (one paper per line, abstracts included).
"""

import json

from .sandbox import Sandbox
from .task import Task

SLICES = [(1990, 2004), (2005, 2009), (2010, 2013), (2014, 2016), (2017, 2018),
          (2019, 2019), (2020, 2020), (2021, 2021), (2022, 2022), (2023, 2023),
          (2024, 2024), (2025, 2025)]
KEEP = ("id", "doi", "title", "date", "year", "type", "venue", "citations",
        "keywords", "mesh", "abstract", "junk")


def build(task: Task, sb: Sandbox | None = None) -> list[dict]:
    sb = sb or Sandbox(task)
    out = task.data_dir / "corpus.jsonl"
    papers: dict[str, dict] = {}
    if out.exists():
        for line in out.open():
            p = json.loads(line)
            papers[p["id"]] = p

    def add(results):
        for r in results:
            if r.get("id") and (r.get("junk") or 0) < 0.5:
                papers[r["id"]] = {k: r.get(k) for k in KEEP}

    last = task.last_visible_year
    jobs = [(q, a, b, "semantic") for q in task.seed_queries for a, b in SLICES if a <= last]
    jobs += [(task.disease_query, y, y, "keyword") for y in range(1990, last + 1)]
    for i, (q, a, b, mode) in enumerate(jobs):
        res = sb.api.search(q, k=100, mode=mode, year_from=a, year_to=min(b, last))
        add(sb._scrub(res.get("results", [])))
        if i % 20 == 0:
            print(f"  corpus {i}/{len(jobs)} queries, {len(papers)} papers, {sb.api.calls} api calls", flush=True)

    with out.open("w") as f:
        for p in sorted(papers.values(), key=lambda p: p.get("date") or ""):
            f.write(json.dumps(p) + "\n")
    print(f"corpus: {len(papers)} papers -> {out}")
    return list(papers.values())


def load(task: Task) -> list[dict]:
    return [json.loads(l) for l in (task.data_dir / "corpus.jsonl").open()]
