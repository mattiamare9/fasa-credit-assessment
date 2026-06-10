"""
Model training, evaluation, and prediction for FASA credit assessment.

Two model variants are supported:
  - "baseline"    : original features only
  - "engineered"  : original + engineered features (requires features.py)

Both share the same sklearn Pipeline structure; only the ColumnTransformer
inputs differ.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Feature column definitions                                                    #
# --------------------------------------------------------------------------- #
NUM_FEATURES = [
    "revenue_m",
    "ebitda_margin",
    "debt_ratio",
    "interest_coverage",
    "cash_ratio",
    "years_in_operation",
    "employee_count",
    "revenue_growth",
]
CAT_FEATURES = ["sector", "country"]
TEXT_FEATURE = "business_description"
TARGET = "defaulted"
RANDOM_STATE = 42

# Engineered numeric and flag columns (appended on top of NUM_FEATURES)
ENG_NUM_FEATURES = [
    "log_revenue_m",
    "log_employee_count",
    "leverage_pressure",
    "liquidity_profitability",
    "growth_profitability",
    "narrative_length",
]
ENG_FLAG_FEATURES = [
    "young_company",
    "weak_interest_coverage",
    "high_debt_ratio",
    "low_cash_ratio",
]

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"


# --------------------------------------------------------------------------- #
# Pipeline builders                                                             #
# --------------------------------------------------------------------------- #
def _numeric_transformer() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _categorical_transformer() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_pipeline(variant: str = "baseline") -> Pipeline:
    """Return an sklearn Pipeline for *variant* ∈ {"baseline", "engineered"}."""
    if variant == "baseline":
        num_cols = NUM_FEATURES
        extra_transformers = []
    elif variant == "engineered":
        num_cols = NUM_FEATURES + ENG_NUM_FEATURES
        extra_transformers = [
            ("flags", _numeric_transformer(), ENG_FLAG_FEATURES),
        ]
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Choose 'baseline' or 'engineered'.")

    preprocessor = ColumnTransformer(
        [
            ("num",  _numeric_transformer(), num_cols),
            ("cat",  _categorical_transformer(), CAT_FEATURES),
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True), TEXT_FEATURE),
        ]
        + extra_transformers
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def _feature_cols(variant: str) -> list[str]:
    if variant == "baseline":
        return NUM_FEATURES + CAT_FEATURES + [TEXT_FEATURE]
    return NUM_FEATURES + ENG_NUM_FEATURES + ENG_FLAG_FEATURES + CAT_FEATURES + [TEXT_FEATURE]


def _select(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    return df[_feature_cols(variant)].copy()


# --------------------------------------------------------------------------- #
# Training & evaluation                                                         #
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    df_train: pd.DataFrame,
    variant: str = "baseline",
) -> Tuple[Pipeline, dict]:
    """Train on 80 % split, evaluate on 20 %, save metrics + val predictions.

    Files written (variant-specific):
      outputs/metrics_{variant}.json
      outputs/validation_predictions_{variant}.csv
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    X = _select(df_train, variant)
    y = df_train[TARGET]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(variant)
    pipeline.fit(X_tr, y_tr)

    probs = pipeline.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "variant": variant,
        "roc_auc": round(float(roc_auc_score(y_val, probs)), 4),
        "average_precision": round(float(average_precision_score(y_val, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_val, probs)), 4),
        "accuracy_at_0_5": round(float(accuracy_score(y_val, preds)), 4),
        "val_size": len(y_val),
        "val_positive_rate": round(float(y_val.mean()), 4),
    }

    with open(OUTPUTS_DIR / f"metrics_{variant}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    val_df = pd.DataFrame({
        "company_id": df_train.loc[X_val.index, "company_id"].values,
        "true_label": y_val.values,
        "predicted_probability": np.round(probs, 4),
        "predicted_label": preds,
    })
    val_df.to_csv(OUTPUTS_DIR / f"validation_predictions_{variant}.csv", index=False)

    # also write un-suffixed files so downstream code using old paths still works
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    val_df.to_csv(OUTPUTS_DIR / "validation_predictions.csv", index=False)

    return pipeline, metrics


def train_final(df_train: pd.DataFrame, variant: str = "baseline") -> Pipeline:
    """Fit on the full training set."""
    pipeline = build_pipeline(variant)
    pipeline.fit(_select(df_train, variant), df_train[TARGET])
    return pipeline


def predict(pipeline: Pipeline, df_scoring: pd.DataFrame, variant: str = "baseline") -> np.ndarray:
    return pipeline.predict_proba(_select(df_scoring, variant))[:, 1]
