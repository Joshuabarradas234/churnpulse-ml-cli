from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def plot_roc(y_true, y_proba, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_proba)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(metrics: Dict[str, Any], figures: Dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# ChurnPulse Report\n\n")
    md.append("## Metrics\n")
    for k, v in metrics.items():
        md.append(f"- **{k}**: {v}\n")

    md.append("\n## Figures\n")
    for title, relpath in figures.items():
        md.append(f"### {title}\n")
        md.append(f"![{title}]({relpath})\n")

    md.append("\n## Notes & Limitations\n")
    md.append("- Baseline, interpretable model (logistic regression).\n")
    md.append("- Results depend on dataset version and preprocessing choices.\n")
    md.append("- Use for decision support; avoid using as sole decision-maker.\n")

    out_path.write_text("".join(md))
