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

`src/fasa_credit_assessment/features.py` adds 12 engineered features, of which 10 are used in model training:

| Feature | Used in model | Description |
|---------|:---:|-------------|
| `log_revenue_m` | ✓ | Log-normalised revenue (right-skewed raw distribution) |
| `log_employee_count` | ✓ | Log-normalised headcount |
| `leverage_pressure` | ✓ | `debt_ratio / (interest_coverage + 0.1)` — combined stress indicator |
| `liquidity_profitability` | ✓ | `cash_ratio × ebitda_margin` — jointly low = high risk |
| `growth_profitability` | ✓ | `revenue_growth × ebitda_margin` |
| `narrative_length` | ✓ | Word count of business description |
| `young_company` | ✓ | Flag: `years_in_operation < 3` |
| `weak_interest_coverage` | ✓ | Flag: `interest_coverage < 1.5` |
| `high_debt_ratio` | ✓ | Flag: debt ratio above training 75th percentile (≈ 0.50) |
| `low_cash_ratio` | ✓ | Flag: cash ratio below training 25th percentile (≈ 0.064) |
| `risk_keyword_count` | — | Risk phrase count — used in explanation layer only |
| `positive_keyword_count` | — | Positive phrase count — used in explanation layer only |

**Leakage prevention:** quantile thresholds are fitted only on the training set and passed as `fit_stats` when transforming the scoring set.

---

## Models

Two algorithm families × two feature sets = four variants, all trained via a single `sklearn.pipeline.Pipeline`:

| Algorithm | Imbalance handling | Variant key |
|---|---|---|
| `LogisticRegression(max_iter=2000, random_state=42)` | `class_weight="balanced"` | `lr_baseline`, `lr_engineered` |
| `LGBMClassifier(n_estimators=400, learning_rate=0.05)` | `scale_pos_weight=5.0` | `lgbm_baseline`, `lgbm_engineered` |

---

## Evaluation (held-out 20 % split)

| Variant | ROC-AUC | PR AUC | F1 Score | Brier | Acc @0.5 |
|---------|---------|--------|----------|-------|----------|
| **LR baseline** | **0.7631** | 0.492 | **0.460** | 0.1806 | 0.730 |
| LR engineered | 0.7491 | 0.496 | 0.489 | 0.1816 | 0.760 |
| LGBM baseline | 0.7032 | 0.438 | 0.327 | 0.1483 | 0.835 |
| LGBM engineered | 0.7356 | 0.465 | 0.360 | **0.1439** | **0.840** |

**Key observations:**
- PR AUC and F1 are reported alongside ROC-AUC because the ~5:1 class imbalance makes ROC-AUC optimistic — PR AUC focuses on the minority (default) class
- LR baseline wins on ROC-AUC and F1; engineered features add collinearity noise for the linear model
- LGBM engineered beats LGBM baseline on all metrics (+0.032 ROC-AUC, +0.034 F1) — tree models exploit non-linear interactions natively
- LGBM has superior Brier scores (better probability calibration), which matters for risk rating thresholds
- On a larger real-world dataset, `lgbm_engineered` would likely become the winner

Full metrics saved to `outputs/metrics_{variant}.json` for all four variants.  
Validation predictions saved to `outputs/validation_predictions_{variant}.csv`.

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
| `outputs/metrics_lr_baseline.json` | LR baseline metrics |
| `outputs/metrics_lr_engineered.json` | LR engineered metrics |
| `outputs/metrics_lgbm_baseline.json` | LGBM baseline metrics |
| `outputs/metrics_lgbm_engineered.json` | LGBM engineered metrics |
| `outputs/validation_predictions.csv` | Best-variant validation predictions |
| `outputs/validation_predictions_lr_baseline.csv` | LR baseline held-out predictions |
| `outputs/validation_predictions_lr_engineered.csv` | LR engineered held-out predictions |
| `outputs/validation_predictions_lgbm_baseline.csv` | LGBM baseline held-out predictions |
| `outputs/validation_predictions_lgbm_engineered.csv` | LGBM engineered held-out predictions |

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
- **Model:** LightGBM is included alongside logistic regression. On 1 000 synthetic samples LR still edges LGBM on ROC-AUC; on a larger real-world dataset LGBM engineered would be expected to win. XGBoost or CatBoost could also be explored.
- **Threshold:** 0.5 is used for binary labels; a calibrated threshold optimised for precision-recall would be more appropriate for credit risk.
- **Calibration:** Predicted probabilities are not post-hoc calibrated (e.g. Platt scaling). Calibration matters for risk rating thresholds.
- **Feature engineering:** The engineered features express domain intuition clearly. For a tree model, they would likely add +2–4 % ROC-AUC.
- **Dataset size:** 1 000 synthetic samples limits generalisation; the approach is a prototype and not production-ready.
