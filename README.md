# 🔧 Automotive Engine Fault Pattern Mining
---

## 📌 Project Overview

Traditional ML models predict *whether* a fault will occur — but not *why*.
This project uses **Apriori association rule mining** to extract human-readable co-occurrence patterns from engine sensor data, enabling engineers to act on specific root causes rather than black-box predictions.

**Core question answered:**
> *"Which combinations of sensor anomalies consistently co-occur with each engine condition?"*

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Engine Fault Detection Data](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data) |
| Rows | 10,000 records |
| Features | 11 vibration / acoustic / thermal / pressure sensor readings |
| Target | `Engine_Condition` (3-class: 0 / 1 / 2) |

**Sensors:** Vibration Amplitude, RMS Vibration, Vibration Frequency, Surface Temperature, Exhaust Temperature, Acoustic dB, Acoustic Frequency, Intake Pressure, Exhaust Pressure, Frequency Band Energy, Amplitude Mean.

**⚠️ Note on the target label:** the dataset card does not document what each `Engine_Condition` value means. Based on the class balance (0: 59.6%, 1: 30.4%, 2: 10.0% — a typical severity long-tail), this project treats it as an ordinal severity scale: **0 = Normal, 1 = Warning, 2 = Critical**. This is a stated assumption, not a confirmed annotation.

---

## 🏗️ Methodology

```
Raw sensor data
      │
      ▼
Discretization (Low / Mid / High per sensor, tertile bins)
      │
      ▼
Apriori — association rule extraction (per condition class)
      │
      ├── Support / Confidence / Lift filtering
      │
      ▼
Random Forest — 3-class validation + feature importance
      │
      ▼
SHAP — explainability layer (per-class)
      │
      ▼
Streamlit dashboard — interactive exploration
```

---

## 🔍 Key Results

### 1) Pipeline validation on synthetic data (known ground truth)

Before trusting the pipeline on real data, it was validated on a schema-matched synthetic dataset (`data/generate_synthetic_data.py`) with two deliberately injected fault mechanisms. The pipeline correctly recovered both:

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| `acoustic_dB_High + RMS_vibration_High → Critical` | 0.061 | 0.532 | **7.87** |
| `surface_temp_High + vibration_amplitude_High → Warning` | 0.087 | 0.778 | **7.08** |

Random Forest on the same synthetic set: **Accuracy 0.959, F1 (macro) 0.911, ROC-AUC (OvR) 0.909**. SHAP correctly ranked `RMS_Vibration` and `Acoustic_dB` as the top drivers of the Critical class — exactly the two sensors used to generate that class. This confirms the mining → validation → explainability pipeline works correctly when a real signal exists.

### 2) Real Kaggle data

Running the identical pipeline on the real dataset tells a different, more honest story: sensor-condition associations are weak (lift ≈ 1.05–1.13 for the strongest Warning/Normal rules — close to statistical independence), and **no rule for the Critical class clears the mining thresholds**.

Random Forest baseline: **Accuracy 0.546, F1 (macro) 0.324, ROC-AUC (OvR) 0.506** — barely better than chance for a 3-class problem.

This is a genuine, if unglamorous, finding: on this particular dataset, the 11 sensor readings carry very little predictive signal for `Engine_Condition` as labeled, which suggests either substantial sensor noise or that the features and label were generated largely independently. The value of running the pipeline anyway is exactly this: it distinguishes a dataset with real learnable structure (the synthetic validation set) from one without (the real Kaggle set), rather than reporting an optimistic number that wouldn't replicate.

Full machine-readable results for both runs are in `reports/findings_summary_synthetic_validation.json` and `reports/findings_summary_real_data.json`.

---

## 💡 Why This Matters for Automotive

Unlike accuracy-only ML models, association rules are:

- **Interpretable** — engineers can read and validate them directly
- **Actionable** — each rule points to a specific sensor combination to investigate
- **A diagnostic for the data itself** — weak rules and weak classifier performance are a signal about data quality, not just model quality, which matters as much in a regulated / safety-relevant context as a strong result would

This aligns with **Industry 4.0 predictive maintenance** pipelines used in modern automotive manufacturing, where knowing *when a dataset doesn't support a claim* is as important as extracting the claim when it does.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, NumPy |
| Association mining | mlxtend (`apriori`, `association_rules`) |
| ML validation | scikit-learn (RandomForestClassifier, multi-class) |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |

---

## 🚀 Getting Started

```bash
git clone https://github.com/BUAksakal/engine-fault-pattern-mining
cd engine-fault-pattern-mining
pip install -r requirements.txt
```

**Get the data** — download the real dataset from [Kaggle](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data) and place it at `data/engine_fault_data.csv`, and/or generate the synthetic validation set:
```bash
python data/generate_synthetic_data.py --out data/synthetic_validation_data.csv
```

**Run the full pipeline** (mining + validation + SHAP, saves results to `reports/`):
```bash
python run_pipeline.py
```

**Launch the interactive dashboard** (toggle between real and synthetic data in the sidebar):
```bash
streamlit run app/dashboard.py
```

---

## 📁 Repository Structure

```
engine-fault-pattern-mining/
│
├── data/
│   ├── generate_synthetic_data.py     # schema-matched synthetic validation set generator
│   ├── engine_fault_data.csv          # real Kaggle CSV (not committed -- see Getting Started)
│   └── synthetic_validation_data.csv  # generated locally
│
├── src/
│   ├── data_prep.py     # loading, discretization, one-hot basket construction
│   ├── mining.py        # Apriori + per-class association rule extraction
│   └── validation.py    # Random Forest (3-class) validation + SHAP explainability
│
├── app/
│   └── dashboard.py     # interactive Streamlit dashboard (real / synthetic toggle)
│
├── reports/
│   ├── findings_summary_real_data.json
│   └── findings_summary_synthetic_validation.json
│
├── run_pipeline.py       # end-to-end script (mining -> validation -> SHAP)
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Berke Ugur Aksakal**
M.Sc. Artificial Intelligence for Smart Sensors and Actuators
Technische Hochschule Deggendorf — Campus Cham

🌐 [berkeuguraksakal.com](https://berkeuguraksakal.com) · 💻 [github.com/BUAksakal](https://github.com/BUAksakal)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
