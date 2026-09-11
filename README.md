# ChurnIQ — Customer Churn Analytics & Prediction Platform

An end-to-end data analytics project that identifies which customers are likely to churn, why they churn, and how much revenue is at risk — built to mirror the workflow of a Data Analyst at a subscription/telecom business.

**Live demo:** http://192.168.1.65:8501/

## What this project demonstrates

| Skill | Where it shows up |
|---|---|
| SQL (joins, CTEs, window functions, aggregations) | `src/churn_analysis.sql`, live SQL queries in the Segment Analysis tab |
| Python data engineering | `src/generate_data.py`, `src/load_to_sql.py` |
| Feature engineering & EDA | `src/train_model.py` |
| Machine learning (classification, model comparison) | Logistic Regression vs. Random Forest, evaluated on Accuracy/Precision/Recall/F1/ROC-AUC |
| BI dashboarding | Interactive Streamlit app with 4 views: KPIs, segment drill-down, live predictor, model diagnostics |
| Business storytelling | Revenue-at-risk framing, not just accuracy metrics |

## Project structure

```text
ChurnIQ/
├── app.py                       # Streamlit Cloud entry point; launches src/app.py
├── data/
│   ├── telecom_customers.csv    # generated dataset (7,043 customers)
│   └── churniq.db               # SQLite database for SQL analysis
├── src/
│   ├── generate_data.py         # synthetic data generator (realistic churn drivers)
│   ├── load_to_sql.py           # loads CSV into SQLite with derived fields
│   ├── churn_analysis.sql       # 6 advanced SQL analytics queries
│   ├── train_model.py           # feature engineering + model training/comparison
│   └── app.py                   # Streamlit dashboard (4 tabs)
├── models/
│   ├── churn_model.joblib       # trained model + encoders
│   └── metrics.json             # evaluation results
├── requirements.txt
└── README.md
```

## How to run locally

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/load_to_sql.py
python src/train_model.py
streamlit run src/app.py
```

## Deploy on Streamlit Community Cloud

The repository includes a root-level `app.py` specifically for Streamlit deployment.

1. Open Streamlit Community Cloud: https://share.streamlit.io/
2. Select **Create app**.
3. Choose the GitHub repository `roshan467/churniq`.
4. Select branch `main`.
5. Set **Main file path** to:

```text
app.py
```

6. Click **Deploy**.

The root `app.py` launches the main dashboard from `src/app.py` while keeping the existing project structure unchanged.

### Streamlit deployment troubleshooting

If Streamlit reports that the main file does not exist, make sure the deployment configuration uses:

```text
Repository: roshan467/churniq
Branch: main
Main file path: app.py
```

Do not use `src\app.py` as the deployment entry path.

## Key results

- **7,043** customer records analyzed
- **26.1%** overall churn rate (calibrated to match real-world telecom benchmarks)
- **Random Forest** selected as best model: **70.1% ROC-AUC**, 66.6% recall on churners
- Top churn drivers identified: **contract type** (33.7% importance), **tenure**, **total charges**
- Segment analysis surfaces the highest-risk group: month-to-month + electronic check + no tech support customers — a segment with disproportionately high revenue at risk

## About the dataset

The dataset is synthetically generated (`generate_data.py`) with churn probabilities driven by realistic business logic (contract type, tenure, payment method, support services) rather than randomly labeled. This makes the analysis and model results reproducible without external downloads.

For interviews, describe it honestly: *"I built a synthetic dataset with realistic churn drivers modeled on published telecom churn research, so I could build and ship the full pipeline without dataset licensing issues."*

## Tech stack

Python · Pandas · NumPy · scikit-learn · SQLite · Streamlit · Plotly
