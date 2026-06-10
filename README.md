# FASA Credit Assessment

**Fasanara AI Credit Risk Analyst Challenge** — prototype credit default prediction with structured financial data, narrative text, feature engineering, and analyst-style explanations.

---

## Problem

Given historical company financials, qualitative business descriptions, and binary default outcomes (1 000 labelled companies), estimate the probability of default for 50 unseen scoring companies and produce a concise analyst-style explanation for each prediction.

---

## Data

| File | Description |
|------|-------------|
| `data/train_companies.csv` | Structured financials for 1 000 training companies |
| `data/train_narratives.csv` | Free-text business descriptions for training companies |
| `data/train_outcomes.csv` | Binary default label (`defaulted`: 0/1) |
| `data/scoring_companies.csv` | 50 unseen companies to score (includes `business_description`) |

Dataset is fully synthetic. Default rate in training set ≈ 17.4 %.

---

## Approach

1. **Join** the three training files on `company_id` to produce a single supervised dataset.
2. **EDA** — explore distributions, correlations, default rates by sector/country, and narrative signals (see `notebooks/01_eda_and_model_story.ipynb`).
3. **Feature engineering** — add log transforms, ratio interactions, binary flags, and narrative keyword counts (see `src/fasa_credit_assessment/features.py`).
4. **Compare two models** — baseline (original features) vs. engineered (original + engineered features), both using a scikit-learn `ColumnTransformer` pipeline.
5. **Evaluate** on a held-out 20 % stratified split, saving metrics and validation predictions for both variants.
6. **Retrain** the best-performing variant on the full training set, predict default probabilities for scoring companies, and generate rule-based explanations.

---

## Features

**Numeric** (median-imputed, standard-scaled):
`revenue_m`, `ebitda_margin`, `debt_ratio`, `interest_coverage`, `cash_ratio`, `years_in_operation`, `employee_count`, `revenue_growth`

**Categorical** (most-frequent-imputed, one-hot-encoded):
`sector`, `country`

**Text** (`business_description`):
TF-IDF with unigrams + bigrams, `max_features=500`, sublinear TF scaling

---

## Exploratory Analysis and Feature Engineering

`notebooks/01_eda_and_model_story.ipynb` covers:

- Target distribution (17.4 % default rate, ~5:1 imbalance)
- Missing values check (none found in training data)
- Summary statistics and mean comparison by default class
- Boxplots for key financial variables split by default status
- Correlation matrix and correlation with target
- Default rate by sector (Logistics 23.5 %, Hospitality 22.2 % — highest risk)
- Default rate by country (Ireland 24.4 %, UK 20.8 % — highest risk)
- Narrative keyword exploration

`src/fasa_credit_assessment/features.py` adds 12 engineered features:

| Feature | Description |
|---------|-------------|
| `log_revenue_m` | Log-normalised revenue (right-skewed raw distribution) |
| `log_employee_count` | Log-normalised headcount |
| `leverage_pressure` | `debt_ratio / (interest_coverage + 0.1)` — combined stress indicator |
| `liquidity_profitability` | `cash_ratio × ebitda_margin` — jointly low = high risk |
| `growth_profitability` | `revenue_growth × ebitda_margin` |
| `young_company` | Flag: `years_in_operation < 3` |
| `weak_interest_coverage` | Flag: `interest_coverage < 1.5` |
| `high_debt_ratio` | Flag: debt ratio above training 75th percentile (≈ 0.50) |
| `low_cash_ratio` | Flag: cash ratio below training 25th percentile (≈ 0.064) |
| `risk_keyword_count` | Risk phrase count in narrative |
| `positive_keyword_count` | Positive phrase count in narrative |
| `narrative_length` | Word count of business description |

**Leakage prevention:** quantile thresholds are fitted only on the training set and passed as `fit_stats` when transforming the scoring set.

**Multicollinearity note:** engineered ratio features correlate with their source columns. Under L2 regularisation logistic regression tolerates this, but the baseline model (ROC-AUC 0.7631) slightly outperforms the engineered variant (0.7479) because the engineered features add correlated noise without new information for a linear model. The engineered features would be more valuable in a tree-based model (LightGBM / XGBoost).

---

## Model

`LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)` inside a single `sklearn.pipeline.Pipeline`. `class_weight="balanced"` corrects for the ~5:1 class imbalance without resampling.

---

## Evaluation (held-out 20 % split)

| Metric | Baseline | Engineered |
|--------|----------|------------|
| ROC-AUC | **0.7631** | 0.7479 |
| Average Precision | 0.492 | 0.4948 |
| Brier Score | 0.1806 | 0.1817 |
| Accuracy @ threshold 0.5 | 0.730 | 0.745 |

Full metrics saved to `outputs/metrics_baseline.json` and `outputs/metrics_engineered.json`.  
Validation predictions saved to `outputs/validation_predictions_baseline.csv` and `outputs/validation_predictions_engineered.csv`.

---

## Explanation Logic

Explanations are generated deterministically by `RuleBasedExplainer` in `src/fasa_credit_assessment/explain.py`.

The explainer uses both **original financial ratios** and **engineered features** when available:

**Risk signals:**
- `debt_ratio > 0.6` → elevated leverage
- `interest_coverage < 1.5` → weak interest coverage
- `cash_ratio < 0.05` → limited liquidity
- `revenue_growth < 0` → declining revenues
- `ebitda_margin < 0.1` → thin operating margins
- `years_in_operation < 3` → short operating history
- `leverage_pressure > 0.4` → high leverage pressure (engineered)
- `high_debt_ratio = 1` → debt above 75th-percentile threshold (engineered)
- `risk_keyword_count > 0` → risk signals in narrative (engineered)

**Mitigating signals:**
- `cash_ratio > 0.15`, `interest_coverage > 3`, `ebitda_margin > 0.2`, `revenue_growth > 0.1`, `years_in_operation > 10`
- `positive_keyword_count > 0` → positive signals in narrative (engineered)

The `BaseExplainer` abstract interface allows the rule-based backend to be swapped for an embedding + LLM backend without changing `cli.py` or `model.py`.

---

## Risk Rating

| Probability | Rating |
|-------------|--------|
| < 0.25 | Low |
| 0.25 – 0.54 | Medium |
| ≥ 0.55 | High |

---

## How to Run

```bash
# 1. Install dependencies
pip install uv
uv sync
uv pip install -e .

# 2. Run the full pipeline (trains both models, compares, saves all outputs)
uv run python -m fasa_credit_assessment.cli

# OR via the installed script entry point
uv run fasa-credit
```

---

## Generated Outputs

| File | Contents |
|------|----------|
| `outputs/predictions.csv` | `company_id, predicted_default_probability, risk_rating, explanation` for all 50 scoring companies |
| `outputs/metrics.json` | Best-variant metrics (ROC-AUC, average precision, Brier score, accuracy) |
| `outputs/metrics_baseline.json` | Baseline model metrics |
| `outputs/metrics_engineered.json` | Engineered model metrics |
| `outputs/validation_predictions.csv` | Best-variant validation predictions |
| `outputs/validation_predictions_baseline.csv` | Baseline held-out predictions |
| `outputs/validation_predictions_engineered.csv` | Engineered held-out predictions |

---

## Demo / Presentation Guide

Walk the hiring panel through the following in order:

1. **Notebook** (`notebooks/01_eda_and_model_story.ipynb`)  
   Open with `uv run jupyter lab` or `uv run jupyter notebook`.  
   Walk through: business objective → EDA plots → feature engineering rationale → model comparison → scoring preview.

2. **Validation metrics** (`outputs/metrics_baseline.json` and `outputs/metrics_engineered.json`)  
   Discuss ROC-AUC vs. average precision trade-off, why baseline outperforms engineered for logistic regression, and what would change with a tree model.

3. **Predictions file** (`outputs/predictions.csv`)  
   Show the 50 rows: probability, risk rating, and explanation columns. Highlight that explanations are deterministic and analyst-readable.

4. **Example explanations** — select one per risk tier from `predictions.csv`:
   - *High*: e.g. "High risk: … Main concerns: elevated leverage, 1 risk signal(s) in narrative."
   - *Medium*: e.g. "Medium risk: … Mitigants: strong liquidity, healthy EBITDA margin."
   - *Low*: e.g. "Low risk: … Mitigants: comfortable interest coverage, established operating history."

5. **Extensibility** — point to `BaseExplainer` in `explain.py` and `build_pipeline(variant=...)` in `model.py` as designed extension points.

---

## Known Limitations & Possible Improvements

- **Text features:** TF-IDF captures surface keywords but misses semantic meaning. A swap to sentence embeddings (via `sentence-transformers`) or a frozen LLM encoder would likely improve text signal quality — `BaseExplainer` / `predict` are already structured for this drop-in.
- **Model:** Logistic regression is interpretable and fast but may underfit non-linear interactions. Gradient boosting (LightGBM / XGBoost) is the natural next step, and would also benefit more from the engineered features.
- **Threshold:** 0.5 is used for binary labels; a calibrated threshold optimised for precision-recall would be more appropriate for credit risk.
- **Calibration:** Predicted probabilities are not post-hoc calibrated (e.g. Platt scaling). Calibration matters for risk rating thresholds.
- **Feature engineering:** The engineered features express domain intuition clearly. For a tree model, they would likely add +2–4 % ROC-AUC.
- **Dataset size:** 1 000 synthetic samples limits generalisation; the approach is a prototype and not production-ready.
