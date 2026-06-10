from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import load_scoring, load_train
from .explain import RuleBasedExplainer, get_risk_rating
from .features import make_features
from .model import OUTPUTS_DIR, predict, train_and_evaluate, train_final


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    ROC-AUC:       {metrics['roc_auc']}")
    print(f"    PR AUC:        {metrics['pr_auc']}")
    print(f"    F1 Score:      {metrics['f1_score']}")
    print(f"    Brier Score:   {metrics['brier_score']}")
    print(f"    Accuracy @0.5: {metrics['accuracy_at_0_5']}")


def _comparison_table(label: str, m_base: dict, m_eng: dict) -> None:
    print(f"\n=== {label} comparison (held-out 20 % split) ===")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Engineered':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")
    for key in ("roc_auc", "pr_auc", "f1_score", "brier_score", "accuracy_at_0_5"):
        print(f"  {key:<22} {m_base[key]:>12} {m_eng[key]:>12}")


def main() -> None:
    print("=== FASA Credit Assessment Pipeline ===\n")

    # ------------------------------------------------------------------ #
    # Load raw data                                                         #
    # ------------------------------------------------------------------ #
    print("Loading data...")
    df_train_raw = load_train()
    df_scoring_raw = load_scoring()
    print(f"  Train: {len(df_train_raw)} companies | Scoring: {len(df_scoring_raw)} companies")

    # ------------------------------------------------------------------ #
    # Feature engineering (fit on training, apply to scoring)              #
    # ------------------------------------------------------------------ #
    print("\nApplying feature engineering...")
    df_train_eng, fit_stats = make_features(df_train_raw)
    df_scoring_eng, _ = make_features(df_scoring_raw, fit_stats=fit_stats)
    print(f"  Fitted thresholds: {fit_stats}")

    # ------------------------------------------------------------------ #
    # Logistic Regression — baseline & engineered                          #
    # ------------------------------------------------------------------ #
    print("\n--- Logistic Regression ---")
    print("Training LR baseline...")
    _, m_lr_base = train_and_evaluate(df_train_raw, variant="lr_baseline")
    _print_metrics("lr_baseline", m_lr_base)

    print("\nTraining LR engineered...")
    _, m_lr_eng = train_and_evaluate(df_train_eng, variant="lr_engineered")
    _print_metrics("lr_engineered", m_lr_eng)

    _comparison_table("Logistic Regression", m_lr_base, m_lr_eng)

    # ------------------------------------------------------------------ #
    # LightGBM — baseline & engineered                                     #
    # ------------------------------------------------------------------ #
    print("\n--- LightGBM ---")
    print("Training LGBM baseline...")
    _, m_lgbm_base = train_and_evaluate(df_train_raw, variant="lgbm_baseline")
    _print_metrics("lgbm_baseline", m_lgbm_base)

    print("\nTraining LGBM engineered...")
    _, m_lgbm_eng = train_and_evaluate(df_train_eng, variant="lgbm_engineered")
    _print_metrics("lgbm_engineered", m_lgbm_eng)

    _comparison_table("LightGBM", m_lgbm_base, m_lgbm_eng)

    # ------------------------------------------------------------------ #
    # Cross-algorithm summary                                              #
    # ------------------------------------------------------------------ #
    all_metrics = {
        "lr_baseline": m_lr_base,
        "lr_engineered": m_lr_eng,
        "lgbm_baseline": m_lgbm_base,
        "lgbm_engineered": m_lgbm_eng,
    }

    print("\n=== All-model summary ===")
    print(f"  {'Variant':<22} {'ROC-AUC':>10} {'PR AUC':>10} {'F1':>8} {'Brier':>8} {'Acc@0.5':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for name, m in all_metrics.items():
        print(f"  {name:<22} {m['roc_auc']:>10} {m['pr_auc']:>10} {m['f1_score']:>8} {m['brier_score']:>8} {m['accuracy_at_0_5']:>10}")

    best_variant = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    print(f"\n  → Best variant by ROC-AUC: '{best_variant}'")

    # Save summary metrics for best variant under the generic names
    OUTPUTS_DIR.mkdir(exist_ok=True)
    best_metrics = all_metrics[best_variant]
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # ------------------------------------------------------------------ #
    # Final models — fit each variant on full training set, score all 50  #
    # ------------------------------------------------------------------ #
    variant_dfs = {
        "lr_baseline":      (df_train_raw, df_scoring_raw),
        "lr_engineered":    (df_train_eng, df_scoring_eng),
        "lgbm_baseline":    (df_train_raw, df_scoring_raw),
        "lgbm_engineered":  (df_train_eng, df_scoring_eng),
    }

    explainer = RuleBasedExplainer()
    saved_prediction_files = []

    print("\nTraining final models and scoring companies...")
    for variant, (df_train_v, df_scoring_v) in variant_dfs.items():
        pipeline = train_final(df_train_v, variant=variant)
        probs = predict(pipeline, df_scoring_v, variant=variant)
        probs_rounded = np.round(probs, 4)
        explanations = explainer.explain_batch(df_scoring_eng, probs_rounded)

        out_path = OUTPUTS_DIR / f"predictions_{variant}.csv"
        pd.DataFrame({
            "company_id": df_scoring_raw["company_id"],
            "predicted_default_probability": probs_rounded,
            "risk_rating": [get_risk_rating(p) for p in probs_rounded],
            "explanation": explanations,
        }).to_csv(out_path, index=False)
        saved_prediction_files.append(out_path)
        print(f"  Saved {out_path.name}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Output files ===")
    for v in all_metrics:
        print(f"  {OUTPUTS_DIR / f'metrics_{v}.json'}")
        print(f"  {OUTPUTS_DIR / f'validation_predictions_{v}.csv'}")
    print(f"  {OUTPUTS_DIR / 'metrics.json'}  (best variant: {best_variant})")
    for p in saved_prediction_files:
        print(f"  {p}")


if __name__ == "__main__":
    main()
