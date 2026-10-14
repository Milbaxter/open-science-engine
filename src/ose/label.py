"""Judge validation: hand-label judged hypotheses, then measure judge–human agreement.

`ose label` writes label/index.html from every judged row in results/.
Open it, mark each item, click "Download labels" and save as label/labels.json.
`ose agree` prints agreement; scores aren't trusted below 85%.
"""

import html
import json

from .api import Api
from .grader import _post_papers
from .sandbox import boolean_query
from .task import ROOT, load_task

VERDICT = {1: "supported", -1: "contradicted", 0: "untested"}


def export(limit: int = 200):
    api, items = Api(), []
    for f in sorted((ROOT / "results").glob("*/*.json")):
        r = json.loads(f.read_text())
        task = load_task(r["task"])
        for row in r["rows"]:
            if "votes" in row and len(items) < limit:
                papers = _post_papers(api, task, boolean_query(task, [row["a"], row["b"]]))
                items.append({"key": f"{f.stem}:{row['a']}|{row['b']}", "judge": VERDICT[row["confirmed"]],
                              "claim": f"{row['a']} —{row.get('relation', '')}→ {row['b']}",
                              "mechanism": row.get("mechanism", ""),
                              "papers": [{"id": p["id"], "title": p.get("title"),
                                          "abstract": (p.get("abstract") or "")[:1800]} for p in papers]})
    cards = "".join(
        f"<section data-key='{html.escape(it['key'], quote=True)}'><h3>{i + 1}. {html.escape(it['claim'])}</h3>"
        f"<p class=m>{html.escape(it['mechanism'])}</p>"
        + "".join(f"<details><summary>{html.escape(p['title'] or '')}</summary><p>{html.escape(p['abstract'])}</p></details>"
                  for p in it["papers"])
        + "".join(f"<label><input type=radio name=q{i} value={v}> {v}</label> " for v in VERDICT.values())
        + "</section>" for i, it in enumerate(items))
    out = ROOT / "label" / "index.html"
    out.parent.mkdir(exist_ok=True)
    (out.parent / "items.json").write_text(json.dumps(items, indent=1))
    out.write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Judge labels</title><style>body{{font:15px/1.5 system-ui;max-width:860px;margin:24px auto;padding:0 16px}}
section{{border-bottom:1px solid #ddd;padding:12px 0}}.m{{color:#666}}summary{{cursor:pointer}}</style>
<h1>Label: do the later papers support the claim?</h1>
<p class=m>Supported = evidence for this specific claim and direction. Mere co-mention or a review restating it = untested.</p>
{cards}<p><button onclick="dl()">Download labels</button></p>
<script>function dl(){{const o={{}};document.querySelectorAll('section').forEach((s,i)=>{{const c=s.querySelector('input:checked');if(c)o[s.dataset.key]=c.value}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)]));a.download='labels.json';a.click()}}</script>""")
    print(f"label: {len(items)} items -> {out}")


def agree():
    items = {it["key"]: it["judge"] for it in json.loads((ROOT / "label" / "items.json").read_text())}
    labels = json.loads((ROOT / "label" / "labels.json").read_text())
    both = [k for k in labels if k in items]
    hits = sum(items[k] == labels[k] for k in both)
    rate = hits / len(both) if both else 0
    print(f"judge–human agreement: {hits}/{len(both)} = {rate:.0%} ({'OK' if rate >= 0.85 else 'below 85% gate'})")
