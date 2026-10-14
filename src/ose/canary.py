"""Leakage canary: post-cutoff papers must never reach an agent."""

from .api import Api
from .sandbox import Sandbox
from .task import Task


def check(task: Task) -> bool:
    api = Api()
    sb = Sandbox(task, api)
    # find real post-cutoff papers, then try every sandbox door to reach them
    post = [r for r in api.search(task.disease, k=5, year_from=task.cutoff_year).get("results", [])
            if str(r.get("date")) >= task.cutoff]
    assert post, "no post-cutoff papers to test with"
    leaks = []
    for p in post:
        hits = sb.search(p["title"], k=20) + sb.search(f'"{p["title"]}"', k=20, mode="keyword")
        if any(h.get("id") == p["id"] or str(h.get("date")) >= task.cutoff for h in hits):
            leaks.append(("search", p["id"]))
        if sb.work(p["id"]) is not None:
            leaks.append(("work", p["id"]))
    print(f"canary: probed {len(post)} post-cutoff papers -> {'LEAK ' + str(leaks) if leaks else 'no leaks'}")
    return not leaks
