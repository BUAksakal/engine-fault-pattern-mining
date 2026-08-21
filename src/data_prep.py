"""
Loads the real Kaggle "Engine Fault Detection Data" CSV
(https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data) and
discretizes each continuous sensor into Low / Mid / High buckets via
tertile binning, so Apriori can operate on categorical "items" instead of
raw floats.

Real schema (verified against the actual dataset, 10,000 rows):
  Vibration_Amplitude, RMS_Vibration, Vibration_Frequency,
  Surface_Temperature, Exhaust_Temperature, Acoustic_dB,
  Acoustic_Frequency, Intake_Pressure, Exhaust_Pressure,
  Frequency_Band_Energy, Amplitude_Mean, Engine_Condition

Engine_Condition is a 3-class label (0/1/2). The dataset card does not
document what each value means; based on the class balance
(0: 59.6%, 1: 30.4%, 2: 10.0% -- a typical severity long-tail) we treat
it as an ordinal severity scale: 0 = Normal, 1 = Warning, 2 = Critical.
This is a documented assumption, not a confirmed label mapping.
"""

import pandas as pd

SENSOR_COLUMNS = [
    "Vibration_Amplitude",
    "RMS_Vibration",
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean",
]

TARGET_COLUMN = "Engine_Condition"

CONDITION_LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}

SHORT_NAMES = {
    "Vibration_Amplitude": "vib_amp",
    "RMS_Vibration": "rms_vib",
    "Vibration_Frequency": "vib_freq",
    "Surface_Temperature": "surf_temp",
    "Exhaust_Temperature": "exhaust_temp",
    "Acoustic_dB": "acoustic_db",
    "Acoustic_Frequency": "acoustic_freq",
    "Intake_Pressure": "intake_pressure",
    "Exhaust_Pressure": "exhaust_pressure",
    "Frequency_Band_Energy": "freq_band_energy",
    "Amplitude_Mean": "amp_mean",
}


def load_raw(path: str = "data/engine_fault_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SENSOR_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df = df.dropna(subset=SENSOR_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    df["Condition_Label"] = df[TARGET_COLUMN].map(CONDITION_LABELS)
    return df


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Bins each sensor into Low/Mid/High via tertiles, returns a copy with
    both the original columns and the new discretized label columns."""
    out = df.copy()
    for col in SENSOR_COLUMNS:
        short = SHORT_NAMES[col]
        out[f"{short}_bucket"] = pd.qcut(
            out[col], q=3, labels=["Low", "Mid", "High"], duplicates="drop"
        )
    return out


def to_onehot_basket(df_discretized: pd.DataFrame) -> pd.DataFrame:
    """Turns the discretized sensor buckets + condition label into a
    one-hot 'transaction basket' matrix suitable for mlxtend's apriori()."""
    bucket_cols = [f"{SHORT_NAMES[c]}_bucket" for c in SENSOR_COLUMNS]
    basket_source = df_discretized[bucket_cols].astype(str).copy()
    basket_source["condition"] = df_discretized["Condition_Label"]

    onehot = pd.DataFrame(index=basket_source.index)
    for col in basket_source.columns:
        for val in basket_source[col].unique():
            item_name = f"{col.replace('_bucket', '')}_{val}" if col != "condition" else val
            onehot[item_name] = (basket_source[col] == val)
    return onehot
