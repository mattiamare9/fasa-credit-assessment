"""
Model training, evaluation, and prediction for FASA credit assessment.

Two algorithm families × two feature sets = four variants:
  algorithm  ∈ {"lr", "lgbm"}
  features   ∈ {"baseline", "engineered"}

Variant key is "{algorithm}_{features}", e.g. "lgbm_engineered".
For backward compatibility, "baseline" → "lr_baseline" and
"engineered" → "lr_engineered" are also accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
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
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #
def _normalize_variant(variant: str) -> tuple[str, str]:
    """Return (algorithm, features) from a variant string.

    Accepts: "lr_baseline", "lr_engineered", "lgbm_baseline", "lgbm_engineered"
    Also accepts legacy shorthands: "baseline" → ("lr", "baseline"),
                                    "engineered" → ("lr", "engineered")
    """
    _legacy = {"baseline": ("lr", "baseline"), "engineered": ("lr", "engineered")}
    if variant in _legacy:
        return _legacy[variant]
    parts = variant.split("_", 1)
    if len(parts) != 2 or parts[0] not in ("lr", "lgbm") or parts[1] not in ("baseline", "engineered"):
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            "Use 'lr_baseline', 'lr_engineered', 'lgbm_baseline', or 'lgbm_engineered'."
        )
    return parts[0], parts[1]


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


def _preprocessor(features: str) -> ColumnTransformer:
    if features == "baseline":
        num_cols = NUM_FEATURES
        extra = []
    else:
        num_cols = NUM_FEATURES + ENG_NUM_FEATURES
        extra = [("flags", _numeric_transformer(), ENG_FLAG_FEATURES)]

    return ColumnTransformer(
        [
            ("num",  _numeric_transformer(), num_cols),
            ("cat",  _categorical_transformer(), CAT_FEATURES),
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True), TEXT_FEATURE),
        ]
        + extra
    )


# --------------------------------------------------------------------------- #
# Pipeline builders                                                             #
# --------------------------------------------------------------------------- #
def build_pipeline(variant: str = "lr_baseline") -> Pipeline:
    """Return an sklearn Pipeline for the given variant string."""
    algo, features = _normalize_variant(variant)

    if algo == "lr":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        # scale_pos_weight ≈ neg/pos ratio; 5.0 matches the ~17 % default rate
        # in the training data and is equivalent to class_weight="balanced" for LR
        classifier = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            scale_pos_weight=5.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    return Pipeline([
        ("preprocessor", _preprocessor(features)),
        ("classifier", classifier),
    ])


def _feature_cols(features: str) -> list[str]:
    if features == "baseline":
        return NUM_FEATURES + CAT_FEATURES + [TEXT_FEATURE]
    return NUM_FEATURES + ENG_NUM_FEATURES + ENG_FLAG_FEATURES + CAT_FEATURES + [TEXT_FEATURE]


def _select(df: pd.DataFrame, features: str) -> pd.DataFrame:
    return df[_feature_cols(features)].copy()


# --------------------------------------------------------------------------- #
# Training & evaluation                                                         #
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    df_train: pd.DataFrame,
    variant: str = "lr_baseline",
) -> Tuple[Pipeline, dict]:
    """Train on 80 % split, evaluate on 20 %, save metrics + val predictions."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    algo, features = _normalize_variant(variant)
    canonical = f"{algo}_{features}"

    X = _select(df_train, features)
    y = df_train[TARGET]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(canonical)
    pipeline.fit(X_tr, y_tr)

    probs = pipeline.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "variant": canonical,
        "algorithm": algo,
        "features": features,
        "roc_auc": round(float(roc_auc_score(y_val, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_val, probs)), 4),
        "f1_score": round(float(f1_score(y_val, preds)), 4),
        "brier_score": round(float(brier_score_loss(y_val, probs)), 4),
        "accuracy_at_0_5": round(float(accuracy_score(y_val, preds)), 4),
        "val_size": len(y_val),
        "val_positive_rate": round(float(y_val.mean()), 4),
    }

    with open(OUTPUTS_DIR / f"metrics_{canonical}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    val_df = pd.DataFrame({
        "company_id": df_train.loc[X_val.index, "company_id"].values,
        "true_label": y_val.values,
        "predicted_probability": np.round(probs, 4),
        "predicted_label": preds,
    })
    val_df.to_csv(OUTPUTS_DIR / f"validation_predictions_{canonical}.csv", index=False)

    # backward-compat aliases for the best-variant unsuffixed files (written by cli.py)
    return pipeline, metrics


def train_final(df_train: pd.DataFrame, variant: str = "lr_baseline") -> Pipeline:
    """Fit on the full training set."""
    algo, features = _normalize_variant(variant)
    canonical = f"{algo}_{features}"
    pipeline = build_pipeline(canonical)
    pipeline.fit(_select(df_train, features), df_train[TARGET])
    return pipeline


def predict(pipeline: Pipeline, df_scoring: pd.DataFrame, variant: str = "lr_baseline") -> np.ndarray:
    _, features = _normalize_variant(variant)
    return pipeline.predict_proba(_select(df_scoring, features))[:, 1]
