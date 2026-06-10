from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_scoring, load_train
from .explain import RuleBasedExplainer, get_risk_rating
from .features import make_features
from .model import OUTPUTS_DIR, predict, train_and_evaluate, train_final


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    ROC-AUC:           {metrics['roc_auc']}")
    print(f"    Average Precision: {metrics['average_precision']}")
    print(f"    Brier Score:       {metrics['brier_score']}")
    print(f"    Accuracy @0.5:     {metrics['accuracy_at_0_5']}")


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
    # Model A — baseline (original features)                               #
    # ------------------------------------------------------------------ #
    print("\nTraining and evaluating — BASELINE (original features)...")
    _, metrics_base = train_and_evaluate(df_train_raw, variant="baseline")
    _print_metrics("baseline", metrics_base)

    # ------------------------------------------------------------------ #
    # Model B — engineered (original + engineered features)                #
    # ------------------------------------------------------------------ #
    print("\nTraining and evaluating — ENGINEERED (original + engineered features)...")
    _, metrics_eng = train_and_evaluate(df_train_eng, variant="engineered")
    _print_metrics("engineered", metrics_eng)

    # ------------------------------------------------------------------ #
    # Comparison table                                                      #
    # ------------------------------------------------------------------ #
    print("\n=== Model comparison (held-out 20 % split) ===")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Engineered':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}")
    for key in ("roc_auc", "average_precision", "brier_score", "accuracy_at_0_5"):
        print(f"  {key:<22} {metrics_base[key]:>12} {metrics_eng[key]:>12}")

    # pick the better variant for scoring output (by ROC-AUC)
    if metrics_eng["roc_auc"] >= metrics_base["roc_auc"]:
        best_variant = "engineered"
        df_train_best = df_train_eng
        df_scoring_best = df_scoring_eng
    else:
        best_variant = "baseline"
        df_train_best = df_train_raw
        df_scoring_best = df_scoring_raw

    print(f"\n  → Using '{best_variant}' model for final scoring output.")

    # ------------------------------------------------------------------ #
    # Final model — fit on full training set, score unseen companies       #
    # ------------------------------------------------------------------ #
    print("\nTraining final model on full training set...")
    pipeline = train_final(df_train_best, variant=best_variant)

    print("Predicting scoring companies...")
    probs = predict(pipeline, df_scoring_best, variant=best_variant)
    probs_rounded = np.round(probs, 4)

    # ------------------------------------------------------------------ #
    # Explanations (use engineered df for richer signals)                  #
    # ------------------------------------------------------------------ #
    explainer = RuleBasedExplainer()
    explanations = explainer.explain_batch(df_scoring_eng, probs_rounded)

    # ------------------------------------------------------------------ #
    # Save predictions                                                     #
    # ------------------------------------------------------------------ #
    OUTPUTS_DIR.mkdir(exist_ok=True)
    predictions = pd.DataFrame({
        "company_id": df_scoring_raw["company_id"],
        "predicted_default_probability": probs_rounded,
        "risk_rating": [get_risk_rating(p) for p in probs_rounded],
        "explanation": explanations,
    })

    out_path = OUTPUTS_DIR / "predictions.csv"
    predictions.to_csv(out_path, index=False)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n=== Output files ===")
    print(f"  {OUTPUTS_DIR / 'metrics_baseline.json'}")
    print(f"  {OUTPUTS_DIR / 'metrics_engineered.json'}")
    print(f"  {OUTPUTS_DIR / 'metrics.json'}  (best variant: {best_variant})")
    print(f"  {OUTPUTS_DIR / 'validation_predictions_baseline.csv'}")
    print(f"  {OUTPUTS_DIR / 'validation_predictions_engineered.csv'}")
    print(f"  {OUTPUTS_DIR / 'validation_predictions.csv'}  (best variant)")
    print(f"  {out_path}")

    print(f"\n=== First 10 predictions ===")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
