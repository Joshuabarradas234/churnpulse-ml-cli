# ChurnPulse

A packaged Python **ML training CLI** that takes a tabular CSV, trains a baseline model, and outputs a **saved model + metrics + a human-readable report**.
##
> A production-ready machine learning CLI that helps businesses reduce revenue loss by predicting customer churn (customers likely to leave or cancel).

## Business problem + ROI

Customer churn (customers who stop buying/cancel) is a major driver of revenue loss—acquiring new customers is typically more expensive than retaining existing ones. This project predicts which customers are most likely to churn so teams can intervene early with targeted retention actions (e.g., support outreach, tailored offers, win-back campaigns). It helps businesses prioritize limited retention budgets by focusing effort on the highest-risk customers first instead of applying blanket discounts to everyone. Decisions enabled include: who to contact, when to contact them, what offer/service action to use, and how to measure impact over time.  
Example: if retention outreach costs **£2 per customer**, the model helps concentrate spend on high-risk customers rather than messaging the entire customer base.

## Results (baseline)

Metrics are computed on a holdout split (baseline model).

| Metric | Value |
|-------:|------:|
| RMSE   | 2.988 |
| MAE    | 2.078 |
| R²     | 0.878 |

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

## Docker Quickstart (recommended)

This is the fastest way to run ChurnPulse without installing Python locally.

### Build
```bash
docker build -t churnpulse .

