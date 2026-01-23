
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _infer_target_column(df: pd.DataFrame, target: Optional[str] = None) -> str:
    if target and target in df.columns:
        return target
    for col in ["Churn", "churn", "target", "label"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not infer target column. Pass --target explicitly or ensure a 'Churn' column exists."
    )


def _to_binary(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        return (y.astype(float) > 0).astype(int)

    s = y.astype(str).str.strip().str.lower()
    pos = {"yes", "y", "true", "1", "churn", "churned", "cancel", "cancelled"}
    neg = {"no", "n", "false", "0", "stay", "active", "not churn", "not_churn"}

    mapped = s.map(lambda v: 1 if v in pos else (0 if v in neg else np.nan))
    if mapped.isna().any():
        bad = sorted(set(s[mapped.isna()].unique()))
        raise ValueError(f"Unrecognized target values in label column: {bad[:20]}")
    return mapped.astype(int)


def train(
    csv_path: str,
    target: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(csv_path)

    target_col = _infer_target_column(df, target=target)
    y = _to_binary(df[target_col])
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )

    clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    cm = confusion_matrix(y_test, pred).tolist()

    artifacts = Path(artifacts_dir)
    reports = Path(reports_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, artifacts / "model.joblib")

    with open(artifacts / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_text = f"""# ChurnPulse Report

## Model
- Logistic Regression (scikit-learn)

## Metrics (holdout set)
- **ROC-AUC:** {metrics["roc_auc"]:.3f}
- **Precision:** {metrics["precision"]:.3f}
- **Recall:** {metrics["recall"]:.3f}
- **F1:** {metrics["f1"]:.3f}

## Confusion matrix (threshold=0.50)
{cm}

## Interpretation (2 lines)
- With a limited retention budget, prefer **higher precision** (raise threshold) to avoid contacting too many non-churners.
- If missing churners is costly, prefer **higher recall** (lower threshold) to catch more at-risk customers.
"""
    (reports / "report.md").write_text(report_text, encoding="utf-8")

    return metrics
