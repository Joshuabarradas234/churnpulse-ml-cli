from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from churnpulse.train import train as train_model

app = typer.Typer(add_completion=False, help="ChurnPulse: train a churn classifier and write outputs.")


@app.command("train")
def train_cmd(
    csv: Path = typer.Option(
        ...,
        "--csv",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV dataset (must include churn/label column).",
    ),
    target: Optional[str] = typer.Option(
        None,
        "--target",
        help="Target/label column name (e.g., Churn). If omitted we try to auto-detect.",
    ),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir"),
    reports_dir: str = typer.Option("reports", "--reports-dir"),
):
    metrics = train_model(
        csv_path=str(csv),
        target=target,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )

    typer.echo("Training complete ✅")
    typer.echo(f"Model:   {artifacts_dir}/model.joblib")
    typer.echo(f"Metrics: {artifacts_dir}/metrics.json")
    typer.echo(f"Report:  {reports_dir}/report.md")
    typer.echo(
        f"Summary: ROC-AUC={metrics['roc_auc']:.3f}, "
        f"Precision={metrics['precision']:.3f}, "
        f"Recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )
