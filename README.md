# 🔧 Automotive Engine Fault Pattern Mining
---

## 📌 Project Overview

Traditional ML models predict *whether* a fault will occur — but not *why*.  
This project uses **Apriori association rule mining** to extract human-readable co-occurrence patterns from engine sensor data, enabling engineers to act on specific root causes rather than black-box predictions.

**Core question answered:**  
> *"Which combinations of sensor anomalies consistently co-occur with engine faults?"*

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Engine Fault Detection Data](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data) |
| Rows | ~20,000 records |
| Features | 6 sensor readings + 1 target |
| Target | Engine Condition (0 = Normal, 1 = Fault) |

**Sensors:**
- Engine RPM
- Lubricating oil pressure (bar)
- Fuel pressure (bar)
- Coolant pressure (bar)
- Lubricating oil temperature (°C)
- Coolant temperature (°C)

---

## 🏗️ Methodology

```
Raw sensor data
      │
      ▼
EDA + Outlier removal
      │
      ▼
Discretization (Low / Mid / High per sensor)
      │
      ▼
Apriori — association rule extraction
      │
      ├── Support / Confidence / Lift filtering
      ├── Fault-specific rule isolation
      │
      ▼
Random Forest — rule validation + feature importance
      │
      ▼
SHAP — explainability layer
      │
      ▼
Streamlit dashboard (interactive demo)
```

---

## 🔍 Key Results

> ⚠️ *Results will be updated after full experimental run.*

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| RPM_low + oil_pressure_low → Fault | TBD | TBD | TBD |
| fuel_pressure_low + oil_temp_high → Fault | TBD | TBD | TBD |
| coolant_pressure_low + RPM_high → Fault | TBD | TBD | TBD |
| oil_pressure_mid + fuel_pressure_mid → Normal | TBD | TBD | TBD |

**Classification baseline (Random Forest):**
- Accuracy: TBD
- F1-score (Fault class): TBD
- ROC-AUC: TBD

---

## 💡 Why This Matters for Automotive

Unlike accuracy-only ML models, association rules are:

- **Interpretable** — engineers can read and validate them directly
- **Actionable** — each rule points to a specific sensor combination to investigate
- **Scalable** — same approach applies to OBD-II logs, CAN bus data, or production line telemetry

This aligns with **Industry 4.0 predictive maintenance** pipelines used in modern automotive manufacturing.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, NumPy |
| Association mining | mlxtend (`apriori`, `association_rules`) |
| ML validation | scikit-learn (RandomForestClassifier) |
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

**Run the analysis:**
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_apriori_mining.ipynb
jupyter notebook notebooks/03_validation.ipynb
```

**Launch dashboard:**
```bash
streamlit run app/dashboard.py
```

---

## 📁 Repository Structure

```
engine-fault-pattern-mining/
│
├── data/
│   └── engine_fault_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_apriori_mining.ipynb
│   └── 03_validation.ipynb
│
├── app/
│   └── dashboard.py
│
├── reports/
│   └── findings_summary.pdf
│
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
