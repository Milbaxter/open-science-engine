import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="ose", description="Open Science Engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="build the pre-cutoff disease corpus")
    c.add_argument("--task", default="als-2026-01")

    e = sub.add_parser("entities", help="build entity vocab + co-occurrence graph")
    e.add_argument("--task", default="als-2026-01")

    r = sub.add_parser("run", help="run an agent and grade it")
    r.add_argument("--agent", required=True, help="random | trend | plain_llm | bridge")
    r.add_argument("--task", default="als-2026-01")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-judge", action="store_true", help="mechanical grading only")

    sub.add_parser("board", help="render leaderboard/index.html")

    k = sub.add_parser("canary", help="leakage canary check")
    k.add_argument("--task", default="als-2026-01")

    m = sub.add_parser("mcp", help="serve sandbox tools over MCP (stdio)")
    m.add_argument("--task", default="als-2026-01")

    pl = sub.add_parser("pool", help="build the shared candidate-pair pool for ranking")
    pl.add_argument("--task", default="als-2026-01")
    pl.add_argument("--n", type=int, default=500)

    rk = sub.add_parser("rank", help="score the pool with a ranker and compute AUC")
    rk.add_argument("--agent", required=True, help="random | popularity | trend | llm")
    rk.add_argument("--task", default="als-2026-01")
    rk.add_argument("--seed", type=int, default=0)

    sub.add_parser("label", help="export judged items for hand-labelling")
    sub.add_parser("agree", help="judge vs human agreement")

    a = p.parse_args()
    if a.cmd in ("label", "agree"):
        from . import label
        return label.export() if a.cmd == "label" else label.agree()
    from .task import load_task

    if a.cmd == "pool":
        from .rank import build_pool
        build_pool(load_task(a.task), n=a.n)
    elif a.cmd == "rank":
        from .rank import run_rank
        run_rank(load_task(a.task), a.agent, a.seed)
    elif a.cmd == "corpus":
        from .corpus import build
        build(load_task(a.task))
    elif a.cmd == "entities":
        from .entities import build
        build(load_task(a.task))
    elif a.cmd == "run":
        from .run import run
        run(load_task(a.task), a.agent, a.seed, judge=not a.no_judge)
    elif a.cmd == "board":
        from .board import render
        render()
    elif a.cmd == "canary":
        from .canary import check
        check(load_task(a.task))
    elif a.cmd == "mcp":
        from .mcp_server import serve
        serve(load_task(a.task))
