"""Expose the time-locked sandbox as MCP tools, so any agent harness can compete.

    claude mcp add ose -- uv run --directory <repo> ose mcp --task als-2026-01
"""

from mcp.server.fastmcp import FastMCP

from .sandbox import Sandbox
from .task import Task


def serve(task: Task):
    sb = Sandbox(task)
    app = FastMCP(f"ose-{task.id}", instructions=(
        f"Literature on {task.disease} published before {task.cutoff}. Nothing later is visible. "
        f"Goal: propose {task.k} novel hypotheses (entity A, relation, entity B) that later papers will confirm."))

    @app.tool()
    def search(q: str, k: int = 20, mode: str = "auto", year_from: int | None = None) -> list[dict]:
        """Search papers by meaning (semantic) or exact words (keyword)."""
        return sb.search(q, k=min(k, 100), mode=mode, year_from=year_from)

    @app.tool()
    def similar(paper_id: str, k: int = 20) -> list[dict]:
        """Papers closest in meaning to one paper (W-number or DOI)."""
        return sb.similar(paper_id, k=min(k, 100))

    @app.tool()
    def work(paper_id: str) -> dict | None:
        """Full record for one paper."""
        return sb.work(paper_id)

    @app.tool()
    def topic_map(q: str, papers: int = 1000) -> dict:
        """Research groups inside a topic, with reading lists (expensive)."""
        return sb.topics(q, papers=papers, abstracts="short")

    @app.tool()
    def count(a: str, b: str = "") -> int:
        """How many pre-cutoff disease papers mention a (and b). 0 means the pair is novel."""
        return sb.count(*[t for t in (a, b) if t])

    app.run()
