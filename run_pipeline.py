"""
End-to-end pipeline: load -> discretize -> mine rules -> validate with RF -> SHAP.
Prints and saves the results that back the README's "Key Results" table.

Usage:
    python run_pipeline.py
"""

import json
import os

import pandas as pd

from src.data_prep import CONDITION_LABELS, discretize, load_raw, to_onehot_basket
from src.mining import condition_specific_rules, mine_rules
from src.validation import compute_shap_values, train_and_validate

DATA_PATH = "data/engine_fault_data.csv"
REPORT_PATH = "reports/findings_summary.json"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the real dataset from "
            "https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data "
            f"and place it at {DATA_PATH} (or run "
            "`python data/generate_synthetic_data.py` for a schema-matched demo)."
        )

    print("Loading data...")
    df = load_raw(DATA_PATH)
    print(f"  {len(df)} rows loaded.")
    print(f"  Class balance: {df['Condition_Label'].value_counts(normalize=True).to_dict()}")

    print("\nDiscretizing sensors into Low/Mid/High buckets...")
    df_disc = discretize(df)

    print("\nBuilding one-hot basket and mining association rules...")
    onehot = to_onehot_basket(df_disc)
    rules = mine_rules(onehot, min_support=0.01, min_confidence=0.3, max_len=3)

    results_by_class = {}
    for label in ["Critical", "Warning", "Normal"]:
        class_rules = condition_specific_rules(rules, label=label)
        results_by_class[label] = class_rules.head(5).to_dict(orient="records")
        print(f"\nTop rules -> {label} (by lift):")
        print(class_rules.head(5).to_string(index=False) if not class_rules.empty else "  (none found at current thresholds)")

    print("\nTraining Random Forest for rule validation (3-class)...")
    clf, X_test, y_test, metrics = train_and_validate(df)
    print(f"  Accuracy:              {metrics['accuracy']:.3f}")
    print(f"  F1 (macro):            {metrics['f1_macro']:.3f}")
    print(f"  F1 (weighted):         {metrics['f1_weighted']:.3f}")
    print(f"  ROC-AUC (OvR, macro):  {metrics['roc_auc_ovr_macro']:.3f}")

    print("\nComputing SHAP values for the Critical class...")
    sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = compute_shap_values(clf, sample, class_index=2)
    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=sample.columns
    ).sort_values(ascending=False)
    print("  Mean |SHAP| per sensor (feature importance for the Critical class):")
    print(mean_abs_shap.to_string())

    os.makedirs("reports", exist_ok=True)
    summary = {
        "n_rows": len(df),
        "class_balance": df["Condition_Label"].value_counts(normalize=True).to_dict(),
        "condition_label_mapping_assumption": CONDITION_LABELS,
        "top_rules_by_class": results_by_class,
        "random_forest_metrics": metrics,
        "mean_abs_shap_critical_by_sensor": mean_abs_shap.to_dict(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full summary to {REPORT_PATH}")


if __name__ == "__main__":
    main()
