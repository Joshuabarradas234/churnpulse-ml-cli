from __future__ import annotations

from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from churnpulse.config import PATHS

app = FastAPI(title="ChurnPulse API", version="0.1.0")

MODEL = None


class ChurnRequest(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=0, ge=0)


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_label: int


@app.on_event("startup")
def _load_model() -> None:
    global MODEL
    model_path = PATHS.artifacts_dir / "model.joblib"
    if not model_path.exists():
        MODEL = None
        return
    MODEL = joblib.load(model_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest) -> ChurnResponse:
    if MODEL is None:
        raise RuntimeError("Model not loaded. Run training first to create artifacts/model.joblib")

    X = pd.DataFrame([req.model_dump()])
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)
    return ChurnResponse(churn_probability=proba, churn_label=label)
