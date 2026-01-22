from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw_csv: Path = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    figures_dir: Path = PROJECT_ROOT / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    test_size: float = 0.2


PATHS = Paths()
SETTINGS = Settings()
