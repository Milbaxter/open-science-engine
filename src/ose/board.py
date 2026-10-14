"""Render results/*/*.json into leaderboard/index.html."""

import html
import json
from collections import defaultdict
from statistics import mean, pstdev

from .task import ROOT

BASELINES = ("random", "trend")


def _agg(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0) if vals else (None, None)


def render():
    by_task = defaultdict(lambda: defaultdict(list))
    for f in sorted((ROOT / "results").glob("*/*.json")):
        r = json.loads(f.read_text())
        by_task[r["task"]][(r["agent"], r["judged"])].append(r)

    sections = []
    for task, groups in by_task.items():
        rows = []
        base = {a: _agg(groups.get((a, False), []), "mech_score")[0] for a in BASELINES}
        for (agent, judged), runs in sorted(groups.items(), key=lambda kv: -(_agg(kv[1], "mech_score")[0] or 0)):
            m, sd = _agg(runs, "mech_score")
            s, _ = _agg(runs, "score")
            usd = mean(r["agent_usd"] + r["judge_usd"] for r in runs)
            lift = " / ".join(f"{m / base[b]:.1f}×" if base[b] else "–" for b in BASELINES)
            rows.append(
                f"<tr><td>{html.escape(agent)}</td><td>{'judged' if judged else 'mechanical'}</td>"
                f"<td>{len(runs)}</td><td>{mean(r['novel'] for r in runs):.1f}</td>"
                f"<td>{mean(r['emerged'] for r in runs):.1f}</td>"
                f"<td>{'–' if s is None else f'{mean(r['confirmed'] for r in runs):.1f}'}</td>"
                f"<td><b>{m:.3f}</b> ±{sd:.3f}</td><td>{'–' if s is None else f'{s:.3f}'}</td>"
                f"<td>{lift}</td><td>${usd:.2f}</td></tr>")
        hyps = []
        for (agent, judged), runs in groups.items():
            for row in runs[0]["rows"]:
                if row["novel"] and row["emerged"]:
                    hyps.append(f"<li><b>{html.escape(agent)}</b>: {html.escape(row['a'])} — "
                                f"{html.escape(row.get('relation', ''))} — {html.escape(row['b'])} "
                                f"<span class=m>(post-cutoff papers: {row['post']}, surprise {row['surprise']}"
                                f"{', judged ' + str(row.get('confirmed')) if 'confirmed' in row else ''})</span></li>")
        sections.append(f"""<h2>{html.escape(task)}</h2>
<table><tr><th>agent</th><th>grading</th><th>runs</th><th>novel /20</th><th>emerged</th><th>confirmed</th>
<th>mech score</th><th>judged score</th><th>lift vs random / trend</th><th>cost</th></tr>{''.join(rows)}</table>
<h3>Hypotheses that came true (first run per agent)</h3><ul>{''.join(hyps) or '<li>none yet</li>'}</ul>""")

    out = ROOT / "leaderboard" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Open Science Engine</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--line:#e4e1d8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--fg:#ecebe6;--mute:#9a988f;--line:#2e2d29}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:32px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{color:var(--mute);font-weight:500;font-size:13px}} .m{{color:var(--mute);font-size:13px}} p{{color:var(--mute)}}
</style>
<h1>Open Science Engine — backtest leaderboard</h1>
<p>Agents see only papers published before the cutoff and propose 20 hypotheses (A — relation — B).
<b>Novel</b>: no earlier paper mentions the disease with both A and B. <b>Emerged</b>: a later paper does.
<b>Mech score</b> = mean of novel × emerged × surprise (−log₁₀ chance co-occurrence). <b>Judged</b> replaces “emerged” with two LLM judges reading the later papers.</p>
{''.join(sections)}""")
    print(f"leaderboard -> {out}")
