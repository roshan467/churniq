# ChurnIQ — Customer Churn Analytics & Prediction Platform

An end-to-end data analytics project that identifies which customers are likely to churn, why they churn, and how much revenue is at risk — built to mirror the actual workflow of a Data Analyst at a subscription/telecom business.

**Live demo:** *(deploy to Streamlit Community Cloud — see below — then paste the link here and on your resume)*

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

```
ChurnIQ/
├── data/
│   ├── telecom_customers.csv     # generated dataset (7,043 customers)
│   └── churniq.db                # SQLite database for SQL analysis
├── src/
│   ├── generate_data.py          # synthetic data generator (realistic churn drivers)
│   ├── load_to_sql.py            # loads CSV into SQLite with derived fields
│   ├── churn_analysis.sql        # 6 advanced SQL analytics queries
│   ├── train_model.py            # feature engineering + model training/comparison
│   └── app.py                    # Streamlit dashboard (4 tabs)
├── models/
│   ├── churn_model.joblib        # trained model + encoders
│   └── metrics.json              # evaluation results
└── requirements.txt
```

## How to run locally

```bash
pip install -r requirements.txt
python src/generate_data.py     # generate the dataset
python src/load_to_sql.py       # load into SQLite
python src/train_model.py       # train and evaluate models
streamlit run src/app.py        # launch the dashboard
```

Open http://localhost:8501

## Key results

- **7,043** customer records analyzed
- **26.1%** overall churn rate (calibrated to match real-world telecom benchmarks)
- **Random Forest** selected as best model: **70.1% ROC-AUC**, 66.6% recall on churners
- Top churn drivers identified: **contract type** (33.7% importance), **tenure**, **total charges**
- Segment analysis surfaces the highest-risk group: month-to-month + electronic check + no tech support customers — a segment with disproportionately high revenue at risk

## About the dataset

The dataset is synthetically generated (`generate_data.py`) with churn probabilities driven by realistic business logic (contract type, tenure, payment method, support services) rather than randomly labeled — this makes the analysis and model results behave like a real churn dataset while keeping the project fully reproducible without external downloads. This is disclosed here and should be mentioned honestly if asked in an interview: *"I built a synthetic dataset with realistic churn drivers modeled on published telecom churn research, so I could build and ship the full pipeline without dataset licensing issues."*

## Deploying to Streamlit Community Cloud (free, gets you a public link)

1. Push this folder to a public GitHub repo
2. Go to https://share.streamlit.io → "New app" → connect your repo
3. Set the main file path to `src/app.py`
4. Deploy — you'll get a public URL like `https://your-app.streamlit.app`
5. Add that link to your resume's Portfolio/Projects section

## Tech stack

Python · Pandas · NumPy · scikit-learn · SQLite · Streamlit · Plotly
