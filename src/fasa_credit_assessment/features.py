"""
Feature engineering for the FASA credit assessment pipeline.

`make_features(df, fit_stats)` is the single entry point.

On the training set call it without `fit_stats`; it returns the enriched DataFrame
**and** a dict of fitted thresholds.  Pass that dict unchanged when transforming the
scoring set to guarantee no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explain import _RISK_KEYWORDS, _POSITIVE_KEYWORDS

# --------------------------------------------------------------------------- #
# Engineered column names                                                       #
# --------------------------------------------------------------------------- #
ENG_NUMERIC = [
    "log_revenue_m",
    "log_employee_count",
    "leverage_pressure",
    "liquidity_profitability",
    "growth_profitability",
    "narrative_length",
]

ENG_FLAGS = [
    "young_company",
    "weak_interest_coverage",
    "high_debt_ratio",
    "low_cash_ratio",
]


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
def make_features(
    df: pd.DataFrame,
    fit_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Add engineered features to *df* and return (enriched_df, fit_stats).

    Parameters
    ----------
    df:
        DataFrame that must contain the raw financial columns plus
        ``business_description``.
    fit_stats:
        Pass ``None`` for the **training** set — quantile thresholds will be
        computed from *df* and returned in the second element.
        Pass the dict returned by the training call when transforming the
        **scoring** set so the same thresholds are used (no leakage).

    Returns
    -------
    enriched_df:
        Copy of *df* with ENG_NUMERIC + ENG_FLAGS columns appended.
    fit_stats:
        Dict of fitted thresholds (always returned; unchanged when input is not
        None).
    """
    df = df.copy()
    fitting = fit_stats is None

    # ------------------------------------------------------------------ #
    # 1. Log transforms                                                    #
    # ------------------------------------------------------------------ #
    df["log_revenue_m"] = np.log1p(df["revenue_m"].clip(lower=0))
    df["log_employee_count"] = np.log1p(df["employee_count"].clip(lower=0))

    # ------------------------------------------------------------------ #
    # 2. Ratio interactions                                                #
    # ------------------------------------------------------------------ #
    df["leverage_pressure"] = df["debt_ratio"] / (df["interest_coverage"] + 0.1)
    df["liquidity_profitability"] = df["cash_ratio"] * df["ebitda_margin"]
    df["growth_profitability"] = df["revenue_growth"] * df["ebitda_margin"]

    # ------------------------------------------------------------------ #
    # 3. Narrative features                                                #
    # ------------------------------------------------------------------ #
    desc = df["business_description"].fillna("").str.lower()
    df["narrative_length"] = desc.str.split().str.len().fillna(0).astype(int)
    df["risk_keyword_count"] = desc.apply(
        lambda t: sum(1 for kw in _RISK_KEYWORDS if kw in t)
    )
    df["positive_keyword_count"] = desc.apply(
        lambda t: sum(1 for kw in _POSITIVE_KEYWORDS if kw in t)
    )

    # ------------------------------------------------------------------ #
    # 4. Boolean flags                                                     #
    # ------------------------------------------------------------------ #
    df["young_company"] = (df["years_in_operation"] < 3).astype(int)
    df["weak_interest_coverage"] = (df["interest_coverage"] < 1.5).astype(int)

    # Quantile-based flags — fit on training set only
    if fitting:
        high_debt_q = float(df["debt_ratio"].quantile(0.75))
        low_cash_q = float(df["cash_ratio"].quantile(0.25))
        fit_stats = {
            "high_debt_ratio_threshold": high_debt_q,
            "low_cash_ratio_threshold": low_cash_q,
        }

    df["high_debt_ratio"] = (
        df["debt_ratio"] > fit_stats["high_debt_ratio_threshold"]
    ).astype(int)
    df["low_cash_ratio"] = (
        df["cash_ratio"] < fit_stats["low_cash_ratio_threshold"]
    ).astype(int)

    return df, fit_stats


def correlation_summary(df: pd.DataFrame, target_col: str = "defaulted") -> pd.DataFrame:
    """Return a DataFrame of Pearson correlations between numeric cols and *target_col*."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    corr = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    return corr.sort_values(ascending=False).to_frame("correlation_with_default")
