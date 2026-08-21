"""
Interactive Streamlit dashboard for the engine fault pattern mining pipeline.
Works on the real Kaggle dataset or the schema-matched synthetic validation set.

Run with:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.data_prep import SENSOR_COLUMNS, TARGET_COLUMN, discretize, load_raw, to_onehot_basket
from src.mining import condition_specific_rules, mine_rules
from src.validation import compute_shap_values, train_and_validate

st.set_page_config(page_title="Engine Fault Pattern Mining", layout="wide")
st.title("🔧 Automotive Engine Fault Pattern Mining")
st.caption(
    "Interpretable engine condition classification via Apriori association rule "
    "mining, validated with a Random Forest classifier and explained with SHAP."
)

st.sidebar.header("Dataset")
data_choice = st.sidebar.radio(
    "Source",
    ["Real Kaggle data (data/engine_fault_data.csv)", "Synthetic validation set (data/synthetic_validation_data.csv)"],
)
DATA_PATH = (
    "data/engine_fault_data.csv"
    if data_choice.startswith("Real")
    else "data/synthetic_validation_data.csv"
)


@st.cache_data
def get_data(path):
    df = load_raw(path)
    df_disc = discretize(df)
    return df, df_disc


try:
    df, df_disc = get_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"No dataset found at `{DATA_PATH}`. Download the real dataset from "
        "[Kaggle](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data) "
        "or run `python data/generate_synthetic_data.py` for the validation set."
    )
    st.stop()

if DATA_PATH.startswith("data/synthetic"):
    st.info(
        "You're viewing the **synthetic validation set** -- used to confirm the "
        "pipeline correctly recovers known, injected fault patterns. Switch to "
        "the real Kaggle data in the sidebar to see how actual sensor data behaves."
    )

st.sidebar.header("Mining Parameters")
min_support = st.sidebar.slider("Minimum support", 0.002, 0.05, 0.01, step=0.001)
min_confidence = st.sidebar.slider("Minimum confidence", 0.1, 0.9, 0.3, step=0.05)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Critical rate", f"{(df[TARGET_COLUMN] == 2).mean():.1%}")
col3.metric("Sensors", len(SENSOR_COLUMNS))

st.subheader("📊 Sensor Distributions by Condition")
sensor_pick = st.selectbox("Sensor", SENSOR_COLUMNS)
fig, ax = plt.subplots(figsize=(7, 3))
for label in ["Normal", "Warning", "Critical"]:
    subset = df.loc[df["Condition_Label"] == label, sensor_pick]
    ax.hist(subset, bins=40, alpha=0.6, label=label)
ax.set_xlabel(sensor_pick)
ax.legend()
st.pyplot(fig)

st.subheader("🔍 Condition-Associated Rules (Apriori)")
onehot = to_onehot_basket(df_disc)
rules = mine_rules(onehot, min_support=min_support, min_confidence=min_confidence, max_len=3)

tab1, tab2, tab3 = st.tabs(["Critical", "Warning", "Normal"])
for tab, label in zip([tab1, tab2, tab3], ["Critical", "Warning", "Normal"]):
    with tab:
        class_rules = condition_specific_rules(rules, label=label)
        if class_rules.empty:
            st.write("No rules found at the current support/confidence thresholds.")
        else:
            st.dataframe(
                class_rules.rename(
                    columns={
                        "antecedents_str": "If (sensor state)",
                        "consequents_str": "Then",
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift",
                    }
                ),
                use_container_width=True,
            )

st.subheader("🌲 Random Forest Validation (3-class)")
with st.spinner("Training classifier..."):
    clf, X_test, y_test, metrics = train_and_validate(df)
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
m3.metric("ROC-AUC (OvR)", f"{metrics['roc_auc_ovr_macro']:.3f}")

st.subheader("💡 SHAP Explainability (Critical class)")
sample = X_test.sample(min(200, len(X_test)), random_state=42)
with st.spinner("Computing SHAP values..."):
    shap_values = compute_shap_values(clf, sample, class_index=2)

fig2, ax2 = plt.subplots(figsize=(7, 4))
shap.summary_plot(shap_values, sample, show=False, plot_size=None)
st.pyplot(fig2)

st.caption(
    "Real data: [Kaggle — Engine Fault Detection Data]"
    "(https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data). "
    "Condition label mapping (0=Normal, 1=Warning, 2=Critical) is an assumption "
    "based on class balance, not a confirmed dataset annotation."
)
