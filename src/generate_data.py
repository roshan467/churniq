"""
generate_data.py
Generates a realistic synthetic telecom customer dataset with built-in churn
patterns (mirrors the structure of the well-known Telco Customer Churn dataset,
but produced programmatically so the pipeline is fully reproducible offline).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043  # match the classic Telco dataset size

def generate():
    customer_id = [f"CUST-{10000+i}" for i in range(N)]
    gender = np.random.choice(["Male", "Female"], N)
    senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

    tenure = np.random.exponential(scale=24, size=N).astype(int)
    tenure = np.clip(tenure, 0, 72)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        N, p=[0.34, 0.23, 0.22, 0.21]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]
    )
    online_security = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.5, 0.21])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.5, 0.21])
    paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])

    monthly_charges = np.round(np.random.normal(65, 30, N).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.05, N) + 20, 2)

    # ---- Build churn probability from realistic business drivers ----
    churn_prob = np.full(N, 0.02)
    churn_prob += np.where(contract == "Month-to-month", 0.20, 0)
    churn_prob += np.where(contract == "One year", 0.02, 0)
    churn_prob -= np.where(contract == "Two year", 0.05, 0)
    churn_prob += np.where(tenure < 6, 0.16, 0)
    churn_prob -= np.where(tenure > 48, 0.08, 0)
    churn_prob += np.where(internet_service == "Fiber optic", 0.07, 0)
    churn_prob += np.where(payment_method == "Electronic check", 0.06, 0)
    churn_prob += np.where(online_security == "No", 0.05, 0)
    churn_prob += np.where(tech_support == "No", 0.05, 0)
    churn_prob += np.where(monthly_charges > 80, 0.04, 0)
    churn_prob += np.where(senior_citizen == 1, 0.03, 0)
    churn_prob -= np.where(dependents == "Yes", 0.03, 0)
    churn_prob = np.clip(churn_prob, 0.02, 0.85)

    churn = np.array(["Yes" if np.random.rand() < p else "No" for p in churn_prob])

    df = pd.DataFrame({
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/telecom_customers.csv", index=False)
    print(f"Generated {len(df)} rows. Churn rate: {(df['churn']=='Yes').mean():.2%}")
    print(df.head())
