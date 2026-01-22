from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from churnpulse.config import PATHS, SETTINGS
from churnpulse.data import clean_boston, load_csv
from churnpulse.pipeline import build_pipeline, split_xy
from churnpulse.report import save_json, write_markdown_report


def _ensure_dirs() -> None:
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(df: pd.DataFrame) -> Tuple[object, Dict[str, float]]:
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SETTINGS.test_size, random_state=SETTINGS.seed
    )

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return pipe, metrics


def run_training(raw_csv: Path | None = None) -> None:
    _ensure_dirs()
    csv_path = raw_csv or PATHS.raw_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {csv_path}. Place your CSV at {PATHS.raw_csv}"
        )

    df = load_csv(str(csv_path))
    df = clean_boston(df)

    model, metrics = train_and_evaluate(df)

    model_path = PATHS.artifacts_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SETTINGS.seed,
        "test_size": SETTINGS.test_size,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target": "MEDV",
        "model_artifact": str(model_path.name),
    }
    save_json(metadata, PATHS.artifacts_dir / "metadata.json")
    save_json(metrics, PATHS.artifacts_dir / "metrics.json")

    figures_rel = {}
    write_markdown_report(metrics, figures_rel, PATHS.reports_dir / "report.md")

    print("✅ Training complete")
    print(f"- Model: {model_path}")
    print(f"- Metrics: {PATHS.artifacts_dir / 'metrics.json'}")
    print(f"- Report: {PATHS.reports_dir / 'report.md'}")
