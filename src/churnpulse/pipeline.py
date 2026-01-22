from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churnpulse.data import TARGET_COL


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = list(X.columns)

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(transformers=[("num", numeric, numeric_cols)])

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])
