"""LLM calls through the local `claude` CLI, so runs bill the user's Claude subscription.

Tools are disabled and the default system prompt is replaced; the model only
sees what we pass. Returns parsed JSON plus the CLI's cost estimate.
"""

import json
import re
import subprocess
import tempfile

TOTAL_COST = {"usd": 0.0, "calls": 0}


class LLMError(RuntimeError):
    pass


def complete(prompt: str, model: str, system: str = "You are a careful biomedical scientist.",
             timeout: int = 600) -> str:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--system-prompt", system]
    with tempfile.TemporaryDirectory() as cwd:  # keep project files out of context
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise LLMError((r.stdout + r.stderr)[:500])
    if out.get("is_error"):
        raise LLMError(str(out.get("result"))[:500])
    # leakage rule: only the requested (pre-cutoff) model may have answered
    used = list((out.get("modelUsage") or {}).keys())
    if not used or any(not u.startswith(model) for u in used):
        raise LLMError(f"model mismatch: asked {model}, CLI used {used}")
    TOTAL_COST["usd"] += out.get("total_cost_usd") or 0
    TOTAL_COST["calls"] += 1
    return out.get("result") or ""


def complete_json(prompt: str, model: str, **kw):
    text = complete(prompt + "\n\nRespond with JSON only, no prose.", model, **kw)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    text = m.group(1) if m else text
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return json.loads(text[start:text.rfind("}" if text[start] == "{" else "]") + 1])
