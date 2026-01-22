from __future__ import annotations

import pandas as pd

# Your CSV uses PRICE as the target column (not MEDV)
TARGET_COL = "PRICE"


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_boston(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # clean column names
    df.columns = [c.strip() for c in df.columns]

    # drop "Unnamed: 0" style index columns if present
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # make everything numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' in CSV. Found columns: {list(df.columns)}"
        )

    # drop rows missing target
    df = df.dropna(subset=[TARGET_COL])

    # fill missing features with median
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df

