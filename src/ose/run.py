import json
import time
from datetime import datetime, timezone

from . import agents
from .api import Api
from .grader import grade
from .llm import TOTAL_COST
from .sandbox import Sandbox
from .task import ROOT, Task


def run(task: Task, agent: str, seed: int = 0, judge: bool = True) -> dict:
    api = Api()
    sb = Sandbox(task, api)
    t0 = time.time()
    hyps = agents.get(agent)(task, sb, seed)
    agent_cost, agent_calls = TOTAL_COST["usd"], api.calls
    print(f"{agent}: {len(hyps)} hypotheses in {time.time() - t0:.0f}s, ${agent_cost:.2f}")

    res = grade(task, hyps, use_judge=judge, api=api)
    out = {
        "task": task.id, "agent": agent, "seed": seed, "model": task.model if agent in ("plain_llm", "bridge") else None,
        "judged": judge, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_usd": round(agent_cost, 4), "judge_usd": round(TOTAL_COST["usd"] - agent_cost, 4),
        "agent_api_calls": agent_calls, **res,
    }
    d = ROOT / "results" / task.id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{agent}-s{seed}{'' if judge else '-mech'}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"novel {res['novel']}/{res['n']}  emerged {res['emerged']}  confirmed {res['confirmed']}  "
          f"mech {res['mech_score']}  score {res['score']}  brier {res['brier']}  -> {path.name}")
    return out
