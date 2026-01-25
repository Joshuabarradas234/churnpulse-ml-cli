# ChurnPulse

A packaged Python **ML training CLI** that takes a tabular CSV, trains a baseline model, and outputs a **saved model + metrics + a human-readable report**.

> A lightweight, reproducible churn training pipeline you can run locally or in Docker.

---

## Quickstart (Docker) — recommended (Windows PowerShell)

```powershell
# from repo root
docker build -t churnpulse .

# run training using the included demo dataset (writes outputs to ./artifacts and ./reports)
docker run --rm -v "${PWD}:/work" -w /work churnpulse `
  --csv /work/data/raw/demo_churn.csv `
  --target Churn

# view outputs
type .\artifacts\metrics.json
type .\reports\report.md

```
## Quickstart (Local install) — Windows PowerShell

# 1) from repo root
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2) install this project (editable)
python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation

# 3) run training (demo dataset)
churnpulse --csv .\data\raw\demo_churn.csv --target Churn

# view outputs
type .\artifacts\metrics.json
type .\reports\report.md

# Outputs (generated)

After a run, the CLI writes reproducible artifacts:

artifacts/model.joblib — trained model

artifacts/metrics.json — metrics snapshot

artifacts/metadata.json — run metadata

reports/report.md — quick human-readable report

artifacts/ and reports/ are intentionally not committed (generated outputs).
```
Quickstart (Local install) — Windows PowerShell
1) from repo root
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

2) install this project (editable)
python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation

3) run training (demo dataset)
churnpulse --csv .\data\raw\demo_churn.csv --target Churn

view outputs
type .\artifacts\metrics.json
type .\reports\report.md

Quickstart (Local install) — Windows PowerShell
1) from repo root
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

2) install this project (editable)
python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation

3) run training (demo dataset)
churnpulse --csv .\data\raw\demo_churn.csv --target Churn

view outputs
type .\artifacts\metrics.json
type .\reports\report.md


# Business problem + ROI

Customer churn (customers who stop buying/cancel) is a major driver of revenue loss—acquiring new customers is typically more expensive than retaining existing ones.

This project predicts which customers are most likely to churn so teams can intervene early with targeted retention actions (e.g., support outreach, tailored offers, win-back campaigns). It helps businesses prioritize limited retention budgets by focusing effort on the highest-risk customers first instead of applying blanket discounts to everyone.

Example: if retention outreach costs £2 per customer, the model helps concentrate spend on high-risk customers rather than messaging the entire customer base.

# Results (baseline)

Example metrics from a holdout split (your numbers will vary by dataset/seed):

Metric	Value
ROC-AUC	0.992
Precision	0.941
Recall	0.960
F1	0.950

Interpretation: in production, you’d calibrate an action threshold (e.g., a “high-risk” band) based on business tolerance for false alarms vs missed churners.

# Project structure

src/churnpulse/ — CLI + training pipeline code

data/raw/ — demo dataset (includes demo_churn.csv)

tests/ — unit tests

Dockerfile — containerized quickstart

.github/workflows/ci.yml — CI workflow (builds image + smoke test + uploads outputs)

# Common issues

PowerShell volume mounts: use -v "${PWD}:/work" (CMD syntax is different).

Target column error: if your dataset doesn’t have Churn, pass the correct label column via --target <colname>.
