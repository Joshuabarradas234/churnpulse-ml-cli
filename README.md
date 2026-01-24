# ChurnPulse

A packaged Python **ML training CLI** that takes a tabular CSV, trains a baseline model, and outputs a **saved model + metrics + a human-readable report**.
##
> A production-ready machine learning CLI that helps businesses reduce revenue loss by predicting customer churn (customers likely to leave or cancel).
## Quickstart (Docker) — recommended

```powershell
# from repo root
docker build -t churnpulse .

# run training using the included demo dataset (writes outputs to ./artifacts and ./reports)
docker run --rm -v "${PWD}:/work" -w /work churnpulse --csv /work/data/raw/demo_churn.csv --target Churn

# view outputs
type .\artifacts\metrics.json
type .\reports\report.md
```
## Business problem + ROI

Customer churn (customers who stop buying/cancel) is a major driver of revenue loss—acquiring new customers is typically more expensive than retaining existing ones. This project predicts which customers are most likely to churn so teams can intervene early with targeted retention actions (e.g., support outreach, tailored offers, win-back campaigns). It helps businesses prioritize limited retention budgets by focusing effort on the highest-risk customers first instead of applying blanket discounts to everyone. Decisions enabled include: who to contact, when to contact them, what offer/service action to use, and how to measure impact over time.  
Example: if retention outreach costs **£2 per customer**, the model helps concentrate spend on high-risk customers rather than messaging the entire customer base.

## Results (baseline)

Example metrics (holdout set):

- ROC-AUC: 0.992  
- Precision: 0.941  
- Recall: 0.960  
- F1: 0.950  

(Your numbers will vary by dataset/seed.)


**Interpretation:** R² of **0.878** suggests the model explains most of the variance in the target.  
RMSE/MAE indicate the typical prediction error is ~**2–3 units** in the target’s scale. In production, you’d calibrate an action threshold (e.g., “high-risk” band) based on business tolerance for false alarms vs missed churners.


## What it does (in ~20 seconds)
- Loads a CSV dataset (you provide the path)
- Cleans/preprocesses the data
- Trains a baseline scikit-learn model
- Writes reproducible outputs you can review or reuse:
  - `artifacts/model.joblib` (trained model)
  - `artifacts/metrics.json` (metrics snapshot)
  - `artifacts/metadata.json` (run metadata)
  - `reports/report.md` (quick report for humans)

## Project structure

- `src/churnpulse/` – CLI + training pipeline code
- `artifacts/` – generated outputs (model + metrics). Not committed.
- `reports/` – generated human-readable report. Not committed.
- `tests/` – unit tests
- `Dockerfile` – containerized quickstart
 
## Quickstart (Windows / PowerShell)

```powershell
# 1) Go to project root
cd C:\Users\Master\churnpulse

# 2) Create + activate venv (first time)
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3) Install this project (editable)
python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation

# 4) Run training (use your CSV path)
churnpulse --csv ".\data\raw\telco_churn.csv"
```
## Docker Quickstart (recommended)
docker build -t churnpulse .
docker run --rm -v "${PWD}:/work" -w /work churnpulse --csv /work/data/raw/demo_churn.csv --target Churn
type .\artifacts\metrics.json
type .\reports\report.md

# Build (from repo root)
docker build -t churnpulse .

# Run (mount repo into container, run from /work)
docker run --rm -v "${PWD}:/work" -w /work churnpulse --csv "/work/data/raw/telco_churn.csv"

Training complete
- Model: artifacts/model.joblib
- Metrics: artifacts/metrics.json
- Report: reports/report.md
type .\artifacts\metrics.json
type .\reports\report.md

## Common issues

- **PowerShell path mounts:** use `-v "${PWD}:/work"` (CMD would be `-v "%cd%:/work"`).
- **Target column error:** if your dataset doesn’t have `Churn`, pass the correct label column via `--target <colname>`.

 CI: GitHub Actions builds the Docker image and runs --help.

