# ChurnWave 📡
### Telecom Customer Churn Prediction

> A production-style ML web app that predicts telecom customer churn using an ensemble of 4 machine learning models — built with Python, Scikit-learn, and Streamlit.

Demo - [ChurnWave](https://churnwave-egohu7ca9tdeptxdpj5faf.streamlit.app/)

---

## 🎯 What It Does

ChurnWave helps telecom businesses identify customers at risk of churning before they leave. A customer service rep can enter a customer's profile and instantly get a churn probability score with actionable retention recommendations.

---

## ✨ Features

- **🧙 4-Step Wizard Form** — Clean, guided single customer prediction (no overwhelming 30-field forms)
- **📂 Bulk Prediction** — Upload a CSV/Excel of customers, get risk scores for all of them instantly
- **📊 Model Insights** — Real feature importances, performance metrics, radar chart comparisons, dataset analytics
- **🔀 Model Selector** — Toggle any of the 4 models on/off from the sidebar
- **🤖 Ensemble Scoring** — Final prediction averages across all active models
- **💡 Retention Recommendations** — Actionable next steps based on risk tier (High / Medium / Low)
- **📥 Download Results** — Export bulk predictions as CSV or Excel

---

## 🧠 ML Models & Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 83.3% | 71.1% | 62.6% | 66.6% | **0.887** ← best |
| XGBoost | 83.0% | 69.8% | 63.6% | 66.6% | 0.886 |
| Logistic Regression | 82.3% | 68.1% | 62.6% | 65.3% | 0.878 |
| Random Forest | 81.6% | 69.5% | 54.8% | 61.3% | 0.873 |

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **ML Models** | Scikit-learn, XGBoost |
| **Data** | Pandas, NumPy |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
ChurnWave/
├── main.py                  # Streamlit app — single entry point
├── zip_lookup.pkl           # Zip code → coordinates lookup
├── requirements.txt
└── models/
    ├── random_forest_model.pkl
    ├── gradient_boosting_model.pkl
    ├── xgboost_model.pkl
    ├── logistic_regression_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── feature_names.pkl
    └── model_performance.csv
```

---

## 🔍 Key Engineering Decisions

- **Wizard UI** — Broke 30 fields into 4 logical steps to reduce cognitive load
- **Auto-derived fields** — Latitude/Longitude resolved from zip code; Total Revenue and Avg Monthly LD Charges calculated automatically — no redundant inputs
- **Ensemble approach** — Averages probabilities across selected models for more robust predictions
- **Real feature importances** — Pulled live from trained Random Forest and Gradient Boosting models, not hardcoded

---

## 🚀 Run Locally

```bash
git clone https://github.com/Sukriti-124/ChurnWave.git
cd ChurnWave

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run main.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset** (extended version) — 7,043 customers, 30 features.  
Source: [IBM Cognos Analytics Sample Data](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

- **Churn rate:** 26.5% (1,869 / 7,043 customers)
- **Target:** Predict whether a customer will churn based on their profile
- **Features include:** Demographics, service subscriptions, billing history, geographic info
---

*Built by Sukriti Srivastava*
