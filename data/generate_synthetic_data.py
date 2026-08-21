"""
Generates a synthetic dataset matching the *real* schema of the Kaggle
"Engine Fault Detection Data" set (11 sensors + Engine_Condition, 0/1/2),
with deliberately injected fault patterns.

Purpose: sanity-check that the pipeline (Apriori mining, RF validation,
SHAP) correctly recovers known patterns when a real signal exists. This
is NOT a substitute for the real dataset -- see README for how the real
data behaves differently (much weaker sensor-condition association).

Usage:
    python data/generate_synthetic_data.py --out data/synthetic_validation_data.csv
"""

import argparse

import numpy as np
import pandas as pd

np.random.seed(42)


def generate(n: int = 10000) -> pd.DataFrame:
    df = pd.DataFrame({
        "Vibration_Amplitude": np.random.uniform(0.1, 10, n),
        "RMS_Vibration": np.random.uniform(0.05, 5, n),
        "Vibration_Frequency": np.random.uniform(20, 2000, n),
        "Surface_Temperature": np.random.uniform(30, 150, n),
        "Exhaust_Temperature": np.random.uniform(200, 600, n),
        "Acoustic_dB": np.random.uniform(60, 120, n),
        "Acoustic_Frequency": np.random.uniform(100, 5000, n),
        "Intake_Pressure": np.random.uniform(90, 120, n),
        "Exhaust_Pressure": np.random.uniform(80, 110, n),
        "Frequency_Band_Energy": np.random.uniform(0.1, 1.0, n),
        "Amplitude_Mean": np.random.uniform(0.01, 0.5, n),
    })

    # Inject two clear fault mechanisms so Apriori/RF have real signal to
    # recover (mirrors plausible physical failure modes for a condition
    # monitoring sensor suite):
    #   Warning: high vibration amplitude + high surface temperature
    #   Critical: high RMS vibration + high acoustic dB (bearing/mechanical failure signature)
    warning_cond = (df["Vibration_Amplitude"] > 7.0) & (df["Surface_Temperature"] > 110)
    critical_cond = (df["RMS_Vibration"] > 3.8) & (df["Acoustic_dB"] > 105)

    condition = np.zeros(n, dtype=int)
    condition[warning_cond.values] = 1
    condition[critical_cond.values] = 2  # critical overrides warning if both trigger

    # Add label noise so it isn't a trivially perfect rule (realistic sensor noise)
    flip_mask = np.random.rand(n) < 0.08
    random_labels = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    condition = np.where(flip_mask, random_labels, condition)

    df["Engine_Condition"] = condition
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_validation_data.csv")
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    df = generate(args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df["Engine_Condition"].value_counts(normalize=True).rename("share"))
