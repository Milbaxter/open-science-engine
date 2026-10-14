"""Time-locked view of the literature: everything an agent may touch.

Every call is clamped to papers published before task.cutoff, both via the
API's year filters and by scrubbing any dated record that slips through.
"""

from .api import Api
from .task import Task


class Sandbox:
    def __init__(self, task: Task, api: Api | None = None):
        self.task = task
        self.api = api or Api()

    # -- leakage guard ------------------------------------------------------
    def _visible(self, rec: dict) -> bool:
        date, year = rec.get("date"), rec.get("year")
        if date:
            return str(date) < self.task.cutoff
        if year is not None:
            return int(year) <= self.task.last_visible_year
        return True

    def _scrub(self, obj):
        if isinstance(obj, list):
            return [self._scrub(x) for x in obj if not (isinstance(x, dict) and not self._visible(x))]
        if isinstance(obj, dict):
            return {k: self._scrub(v) for k, v in obj.items()}
        return obj

    def _years(self, year_from=None):
        return {"year_from": year_from, "year_to": self.task.last_visible_year}

    # -- tools --------------------------------------------------------------
    def search(self, q: str, k: int = 20, mode: str = "auto", year_from: int | None = None,
               abstracts: bool = True) -> list[dict]:
        d = self.api.search(q, k=k, mode=mode, abstracts=str(abstracts).lower(), **self._years(year_from))
        return self._scrub(d.get("results", []))

    def similar(self, ref: str, k: int = 20) -> list[dict]:
        d = self.api.similar(ref, k=k, **self._years())
        return self._scrub(d.get("results", []))

    def work(self, ref: str) -> dict | None:
        d = self.api.works(ref)
        return d if d and self._visible(d) else None

    def topics(self, q: str, papers: int = 1000, **kw) -> dict:
        return self._scrub(self.api.topics(q, papers=papers, **self._years(), **kw))

    def count(self, *terms: str, in_disease: bool = True) -> int:
        """Pre-cutoff number of papers mentioning all terms (and the disease)."""
        return self.api.count(boolean_query(self.task, terms, in_disease), **self._years())


def term_clause(term: str) -> str:
    variants = [v.strip().replace('"', "") for v in term.split("|") if v.strip()]
    inner = " OR ".join(f'"{v}"' for v in variants)
    return f"({inner})" if len(variants) > 1 else inner


def boolean_query(task: Task, terms, in_disease: bool = True) -> str:
    parts = ([task.disease_query] if in_disease else []) + [term_clause(t) for t in terms]
    return " AND ".join(parts)
