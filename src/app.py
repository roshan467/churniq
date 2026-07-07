"""
app.py — ChurnIQ: Customer Churn Analytics & Prediction Platform
Run: streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ChurnIQ", page_icon="📊", layout="wide")

DATA_PATH = "data/telecom_customers.csv"
DB_PATH = "data/churniq.db"
MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"]
    )
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

df = load_data()
model_bundle = load_model()
metrics = load_metrics()

st.title("📊 ChurnIQ — Customer Churn Analytics & Prediction")
st.caption("SQL-driven analytics + ML-powered churn prediction for a telecom customer base")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🔍 Segment Analysis", "🤖 Churn Predictor", "🧠 Model Performance"]
)

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    total = len(df)
    churned = (df["churn"] == "Yes").sum()
    churn_rate = churned / total
    revenue_at_risk = df.loc[df["churn"] == "Yes", "monthly_charges"].sum()
    avg_tenure = df["tenure_months"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churn Rate", f"{churn_rate:.1%}")
    c3.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
    c4.metric("Avg. Tenure", f"{avg_tenure:.1f} mo")

    col1, col2 = st.columns(2)
    with col1:
        churn_by_contract = df.groupby("contract")["churn"].apply(
            lambda x: (x == "Yes").mean()
        ).reset_index(name="churn_rate")
        fig = px.bar(churn_by_contract, x="contract", y="churn_rate",
                     title="Churn Rate by Contract Type", text_auto=".1%",
                     color="churn_rate", color_continuous_scale="Reds")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_by_tenure = df.groupby("tenure_bucket", observed=True)["churn"].apply(
            lambda x: (x == "Yes").mean()
        ).reset_index(name="churn_rate")
        fig2 = px.line(churn_by_tenure, x="tenure_bucket", y="churn_rate", markers=True,
                        title="Churn Rate by Tenure Cohort")
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        rev_by_service = df[df["churn"] == "Yes"].groupby("internet_service")["monthly_charges"].sum().reset_index()
        fig3 = px.pie(rev_by_service, names="internet_service", values="monthly_charges",
                      title="Lost Monthly Revenue by Internet Service Type")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        pay_churn = df.groupby("payment_method")["churn"].apply(lambda x: (x == "Yes").mean()).reset_index(name="churn_rate")
        fig4 = px.bar(pay_churn.sort_values("churn_rate"), x="churn_rate", y="payment_method",
                      orientation="h", title="Churn Rate by Payment Method", text_auto=".1%",
                      color="churn_rate", color_continuous_scale="Oranges")
        fig4.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

# ---------------- TAB 2: SEGMENT ANALYSIS (SQL-driven) ----------------
with tab2:
    st.subheader("SQL-Driven High-Risk Segment Finder")
    st.caption("Query the underlying SQLite database directly — filters map to real SQL WHERE clauses")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        contract_f = st.multiselect("Contract", df["contract"].unique(), default=list(df["contract"].unique()))
    with colf2:
        payment_f = st.multiselect("Payment Method", df["payment_method"].unique(), default=list(df["payment_method"].unique()))
    with colf3:
        tech_support_f = st.selectbox("Tech Support", ["All"] + list(df["tech_support"].unique()))

    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT contract, payment_method, tech_support, internet_service,
               COUNT(*) as customers,
               ROUND(100.0*SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END)/COUNT(*),2) as churn_rate_pct,
               ROUND(AVG(monthly_charges),2) as avg_monthly_charges,
               ROUND(SUM(CASE WHEN churn='Yes' THEN monthly_charges ELSE 0 END),2) as revenue_at_risk
        FROM customers
        WHERE contract IN ({','.join(['?']*len(contract_f))})
          AND payment_method IN ({','.join(['?']*len(payment_f))})
          {"AND tech_support = ?" if tech_support_f != "All" else ""}
        GROUP BY contract, payment_method, tech_support, internet_service
        ORDER BY churn_rate_pct DESC
        LIMIT 20
    """
    params = contract_f + payment_f + ([tech_support_f] if tech_support_f != "All" else [])
    result = pd.read_sql_query(query, conn, params=params)
    conn.close()

    st.dataframe(result, use_container_width=True)
    with st.expander("View underlying SQL query"):
        st.code(query, language="sql")

# ---------------- TAB 3: CHURN PREDICTOR ----------------
with tab3:
    st.subheader("Predict Churn Risk for a Customer")
    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    model_name = model_bundle["model_name"]

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", [0, 1])
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    total_charges = monthly_charges * tenure

    if st.button("Predict Churn Risk", type="primary"):
        row = {
            "gender": gender, "partner": partner, "dependents": dependents,
            "contract": contract, "payment_method": payment, "internet_service": internet,
            "online_security": online_sec, "tech_support": tech_sup, "paperless_billing": paperless,
            "senior_citizen": senior, "tenure_months": tenure,
            "monthly_charges": monthly_charges, "total_charges": total_charges,
        }
        row["avg_charge_per_month"] = total_charges / max(tenure, 1)
        row["is_new_customer"] = int(tenure <= 6)
        row["is_high_value"] = int(monthly_charges > df["monthly_charges"].median())

        X_input = pd.DataFrame([row])[feature_cols]
        for col, le in encoders.items():
            X_input[col] = le.transform(X_input[col])

        if scaler is not None:
            X_input_final = scaler.transform(X_input)
        else:
            X_input_final = X_input

        proba = model.predict_proba(X_input_final)[0, 1]

        st.divider()
        colr1, colr2 = st.columns([1, 2])
        with colr1:
            st.metric("Churn Probability", f"{proba:.1%}")
            risk = "🔴 High Risk" if proba > 0.5 else ("🟡 Medium Risk" if proba > 0.3 else "🟢 Low Risk")
            st.markdown(f"### {risk}")
        with colr2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "darkred" if proba > 0.5 else "orange" if proba > 0.3 else "green"},
                       "steps": [{"range": [0, 30], "color": "#d4edda"},
                                 {"range": [30, 50], "color": "#fff3cd"},
                                 {"range": [50, 100], "color": "#f8d7da"}]},
                title={"text": "Churn Risk Score"}
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: MODEL PERFORMANCE ----------------
with tab4:
    st.subheader(f"Model: {metrics['best_model'].replace('_', ' ').title()}")
    results = metrics["results"][metrics["best_model"]]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{results['accuracy']:.1%}")
    c2.metric("Precision", f"{results['precision']:.1%}")
    c3.metric("Recall", f"{results['recall']:.1%}")
    c4.metric("F1 Score", f"{results['f1']:.1%}")
    c5.metric("ROC-AUC", f"{results['roc_auc']:.3f}")

    st.markdown("#### Model Comparison")
    comp_df = pd.DataFrame(metrics["results"]).T
    st.dataframe(comp_df, use_container_width=True)

    if metrics.get("feature_importance"):
        st.markdown("#### Top Churn Drivers (Feature Importance)")
        fi_df = pd.DataFrame(list(metrics["feature_importance"].items()),
                              columns=["feature", "importance"]).head(10)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                     title="What drives churn the most?")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("ChurnIQ — built with Python, SQLite, scikit-learn, and Streamlit | Data: synthetic dataset modeled on real telecom churn patterns")
