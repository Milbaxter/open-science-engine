import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Task:
    id: str
    disease: str
    disease_query: str
    cutoff: str  # ISO date; agents see strictly earlier papers
    window_end: str
    k: int
    model: str
    judge_models: list[str]
    seed_queries: list[str] = field(default_factory=list)

    @property
    def cutoff_year(self) -> int:
        return int(self.cutoff[:4])

    @property
    def last_visible_year(self) -> int:
        # a Jan-1 cutoff makes the year filter exact; otherwise also filter by date
        return self.cutoff_year - 1 if self.cutoff[5:] == "01-01" else self.cutoff_year

    @property
    def data_dir(self) -> Path:
        d = ROOT / "data" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_task(name: str) -> Task:
    path = Path(name) if name.endswith(".toml") else ROOT / "tasks" / f"{name}.toml"
    return Task(**tomllib.loads(path.read_text()))
