from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


def get_risk_rating(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    if prob < 0.55:
        return "Medium"
    return "High"


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, row: pd.Series, prob: float) -> str: ...

    def explain_batch(self, df: pd.DataFrame, probs) -> list[str]:
        return [
            self.explain(row, float(p))
            for (_, row), p in zip(df.iterrows(), probs)
        ]


_RISK_KEYWORDS = [
    "legacy debt",
    "customer concentration",
    "regulatory uncertainty",
    "volatile commodity prices",
    "declining demand",
    "margin pressure",
    "litigation",
    "supplier dependency",
]

_POSITIVE_KEYWORDS = [
    "resilient demand",
    "disciplined growth",
    "stable",
    "diversified",
    "strong market position",
]


class RuleBasedExplainer(BaseExplainer):
    """Deterministic rule-based explainer. Replace with EmbeddingLLMExplainer later.

    Uses both original financial ratios and pre-computed engineered features when
    present in the row (e.g. leverage_pressure, risk_keyword_count).
    """

    def explain(self, row: pd.Series, prob: float) -> str:
        rating = get_risk_rating(prob)
        concerns: list[str] = []
        mitigants: list[str] = []

        # ------------------------------------------------------------------ #
        # Original financial flags                                             #
        # ------------------------------------------------------------------ #
        if row.get("debt_ratio", 0) > 0.6:
            concerns.append("elevated leverage")
        if row.get("interest_coverage", 99) < 1.5:
            concerns.append("weak interest coverage")
        if row.get("cash_ratio", 99) < 0.05:
            concerns.append("limited liquidity")
        if row.get("revenue_growth", 0) < 0:
            concerns.append("declining revenues")
        if row.get("ebitda_margin", 99) < 0.1:
            concerns.append("thin operating margins")
        if row.get("years_in_operation", 99) < 3:
            concerns.append("short operating history")

        if row.get("cash_ratio", 0) > 0.15:
            mitigants.append("strong liquidity")
        if row.get("interest_coverage", 0) > 3:
            mitigants.append("comfortable interest coverage")
        if row.get("ebitda_margin", 0) > 0.2:
            mitigants.append("healthy EBITDA margin")
        if row.get("revenue_growth", 0) > 0.1:
            mitigants.append("positive revenue growth")
        if row.get("years_in_operation", 0) > 10:
            mitigants.append("established operating history")

        # ------------------------------------------------------------------ #
        # Engineered feature flags (only if present in the row)               #
        # ------------------------------------------------------------------ #
        leverage_pressure = row.get("leverage_pressure")
        if leverage_pressure is not None and leverage_pressure > 0.4:
            concerns.append(f"high leverage pressure ({leverage_pressure:.2f})")

        high_debt = row.get("high_debt_ratio")
        if high_debt is not None and int(high_debt) == 1:
            if "elevated leverage" not in concerns:
                concerns.append("debt ratio above 75th-percentile threshold")

        low_cash = row.get("low_cash_ratio")
        if low_cash is not None and int(low_cash) == 1:
            if "limited liquidity" not in concerns:
                concerns.append("cash ratio below 25th-percentile threshold")

        weak_ic = row.get("weak_interest_coverage")
        if weak_ic is not None and int(weak_ic) == 1:
            if "weak interest coverage" not in concerns:
                concerns.append("interest coverage below 1.5×")

        young = row.get("young_company")
        if young is not None and int(young) == 1:
            if "short operating history" not in concerns:
                concerns.append("company is less than 3 years old")

        # ------------------------------------------------------------------ #
        # Narrative signals                                                    #
        # ------------------------------------------------------------------ #
        # Prefer pre-computed counts when available; fall back to keyword scan
        risk_count = row.get("risk_keyword_count")
        pos_count  = row.get("positive_keyword_count")

        if risk_count is not None:
            if int(risk_count) > 0:
                concerns.append(f"{int(risk_count)} risk signal(s) in narrative")
        else:
            desc = str(row.get("business_description", "")).lower()
            narrative_risks = [kw for kw in _RISK_KEYWORDS if kw in desc]
            if narrative_risks:
                concerns.append(f"narrative references to {', '.join(narrative_risks)}")

        if pos_count is not None:
            if int(pos_count) > 0:
                mitigants.append(f"{int(pos_count)} positive signal(s) in narrative")
        else:
            desc = str(row.get("business_description", "")).lower()
            narrative_positives = [kw for kw in _POSITIVE_KEYWORDS if kw in desc]
            if narrative_positives:
                mitigants.append(f"narrative signals of {', '.join(narrative_positives)}")

        # ------------------------------------------------------------------ #
        # Assemble                                                             #
        # ------------------------------------------------------------------ #
        parts = [f"{rating} risk: predicted default probability is {prob:.4f}."]
        if concerns:
            parts.append(f"Main concerns: {', '.join(concerns)}.")
        if mitigants:
            parts.append(f"Mitigants: {', '.join(mitigants)}.")
        if not concerns and not mitigants:
            parts.append("No material concerns identified.")

        return " ".join(parts)
