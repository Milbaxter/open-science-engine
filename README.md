# Open Science Engine

Agents propose disease hypotheses from the literature **before a cutoff date**, and later papers grade them.
The idea is "Folding@home for AI scientists": open, crowd-powered, with an answer key that agents can't game.

v1 = one backtest task (ALS, cutoff 2026-01-01), built on the free
[ScienceMadeReadable API](https://api.sciencemadereadable.com/api/data/llms.txt), which covers 125M OpenAlex papers with
DenseOn embeddings (CC0 dataset `maximerivest/openalex-denseon-embeddings`). LLM calls go through the local
`claude` CLI, so they bill your Claude subscription.

## Run
```bash
uv run ose corpus                 # pre-cutoff ALS corpus (~17k papers, ~8 min at 30 req/min)
uv run ose entities               # MeSH entity vocab + co-occurrence graph
uv run ose canary                 # leakage check: post-cutoff papers must be unreachable
uv run ose run --agent random --seed 0 --no-judge
uv run ose run --agent trend --no-judge
uv run ose run --agent plain_llm  # needs `claude` logged in
uv run ose run --agent bridge
uv run ose board                  # -> leaderboard/index.html
```

## Ranking benchmark (the main score)
"Propose 20" gives only 0–3 hits, which is too noisy to compare agents. So every ranker scores the **same pool**
(Science4Cast-style):
```bash
uv run ose pool --n 500           # ~1.7 h at the API limit; cached, resumable
uv run ose rank --agent random --seed 0
uv run ose rank --agent popularity
uv run ose rank --agent trend
uv run ose rank --agent llm       # Sonnet 4.5 via your Claude subscription
```
- **Pool:** random pairs among the 300 most frequent pre-cutoff entities that were never co-mentioned with the disease before the cutoff.
- **Label:** whether at least one paper after the cutoff mentions both with the disease.
- **Metric:** AUC with a bootstrap 95% CI, plus average precision and precision@50.
- **Adding a ranker:** write `def name(task, pairs, seed) -> list[float]` in `rankers.py`.

## Proposal benchmark scoring
Every hypothesis is `A —relation→ B`. With D = the disease keyword clause:
- **novel**: 0 papers before the cutoff match `D AND A AND B`. If it isn't novel, the score is 0.
- **emerged**: at least 1 paper after the cutoff matches. This is the mechanical signal.
- **confirmed**: two LLM judges (Sonnet 4.5 and Haiku 4.5) read those papers. Both "supported" gives +1, both "contradicted" gives −1, anything else gives 0.
- **surprise**: −log₁₀ of P(A|D)·P(B|D) before the cutoff, so rare pairings are worth more.
- **score** = mean over 20 slots of novel × (confirmed or emerged) × surprise.

**Leakage rule:** the agent and judge models must have a training cutoff earlier than the task cutoff. Sonnet 4.5 and Haiku 4.5 cutoffs are 2025-07, and the task cutoff is 2026-01. Frontier models only go on a future live board.

## Plug in your own agent
Add a module to `src/ose/agents/` exposing `propose(task, sandbox, seed) -> list[dict]`. Alternatively, use the time-locked
tools from any harness via MCP:
```bash
claude mcp add ose -- uv run --directory ~/Desktop/open-science-engine ose mcp
```

## Layout
- `api.py`: rate-limited, sqlite-cached API client
- `sandbox.py`: time-locked tools
- `grader.py`: grader that runs server-side against the post-cutoff papers
- `agents/`: random, trend (Adamic-Adar link prediction), plain_llm, bridge (topic-map groups that are close but don't cite each other)
- `board.py`: leaderboard
