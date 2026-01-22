# ChurnPulse

A packaged Python **ML training CLI** that takes a tabular CSV, trains a baseline model, and outputs a **saved model + metrics + a human-readable report**.
##
> A production-ready machine learning CLI that helps businesses reduce revenue loss by predicting customer churn (customers likely to leave or cancel).


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
