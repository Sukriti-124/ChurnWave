# ChurnWave — Telecom Customer Churn Predictor

A Streamlit app for predicting telecom customer churn using an ensemble of 4 ML models.

## Repo Structure

```
churnwave/
├── main.py                     # Streamlit app (single entry point)
├── zip_lookup.pkl              # Zip code → lat/long lookup (CA zips)
├── requirements.txt
├── .gitignore
├── README.md
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

> **Note:** `telecom_customer_churn.csv` is excluded from the repo (gitignored).  
> The Dataset Overview tab in Model Insights requires it. Place it in the project root if needed.

---

## Setup

```bash
git clone https://github.com/yourname/churnwave.git
cd churnwave

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run main.py
```

---

## Features

### 🎯 Single Prediction (4-step wizard)
| Step | Fields |
|------|--------|
| 1 · Demographics | Gender, Age, Married, Dependents, Referrals, Tenure, Zip Code |
| 2 · Core Services | Phone, Multiple Lines, Internet, Unlimited Data, Avg GB/month |
| 3 · Add-ons | Online Security, Backup, Device Protection, Tech Support, Streaming (TV/Movies/Music), Paperless Billing |
| 4 · Financial | Monthly Charge, Total Charges, Total Refunds, Extra Data Charges, Total LD Charges |

Auto-derived (not asked): Latitude/Longitude (from zip), Avg Monthly LD Charges (Total LD ÷ Tenure), Total Revenue (Charges − Refunds + Extra + LD).

### 📂 Bulk Prediction
Upload a CSV/Excel → get risk scores for every row → download results.

### 📊 Model Insights
- Model performance comparison (accuracy, precision, recall, F1, ROC-AUC)
- Real feature importances from trained RF and GB models
- Dataset overview charts
- Business recommendations by risk tier

---

## Models

| Model | ROC-AUC |
|-------|---------|
| Gradient Boosting | **0.887** ← best |
| Random Forest | ~0.88 |
| XGBoost | ~0.87 |
| Logistic Regression | ~0.84 |

Ensemble = average of all active model probabilities.  
Risk tiers: **High** ≥ 70% · **Medium** 35–70% · **Low** < 35%

---

## Regenerating `zip_lookup.pkl`

If you need to rebuild it from the raw dataset:

```python
import pandas as pd, pickle

df = pd.read_csv('telecom_customer_churn.csv')
lookup = (df.drop_duplicates('Zip Code')
            .set_index('Zip Code')[['Latitude', 'Longitude', 'City']]
            .to_dict(orient='index'))

with open('zip_lookup.pkl', 'wb') as f:
    pickle.dump(lookup, f)
```
