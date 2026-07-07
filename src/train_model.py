"""
train_model.py
End-to-end ML pipeline for churn prediction:
- Feature engineering
- Train/test split
- Trains Logistic Regression + Random Forest, compares them
- Saves the best model + feature importances for the dashboard
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = "/home/claude/ChurnIQ/data/telecom_customers.csv"
MODEL_PATH = "/home/claude/ChurnIQ/models/churn_model.joblib"
METRICS_PATH = "/home/claude/ChurnIQ/models/metrics.json"

CATEGORICAL = [
    "gender", "partner", "dependents", "contract", "payment_method",
    "internet_service", "online_security", "tech_support", "paperless_billing"
]
NUMERIC = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]

def engineer_features(df):
    df = df.copy()
    df["avg_charge_per_month"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["is_high_value"] = (df["monthly_charges"] > df["monthly_charges"].median()).astype(int)
    return df

def build_pipeline():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = CATEGORICAL + NUMERIC + ["avg_charge_per_month", "is_new_customer", "is_high_value"]
    X = df[feature_cols].copy()
    y = (df["churn"] == "Yes").astype(int)

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    results["logistic_regression"] = evaluate(y_test, lr_pred, lr_proba)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)  # tree model doesn't need scaling
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = evaluate(y_test, rf_pred, rf_proba)

    # pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = rf if best_name == "random_forest" else lr

    feature_importance = None
    if best_name == "random_forest":
        feature_importance = dict(sorted(
            zip(feature_cols, rf.feature_importances_.round(4).tolist()),
            key=lambda x: -x[1]
        ))

    joblib.dump({
        "model": best_model,
        "scaler": scaler if best_name == "logistic_regression" else None,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "model_name": best_name,
    }, MODEL_PATH)

    metrics_out = {
        "results": results,
        "best_model": best_name,
        "feature_importance": feature_importance,
        "churn_rate": float(y.mean()),
        "n_samples": len(df),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(json.dumps(metrics_out, indent=2))
    return metrics_out

def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }

if __name__ == "__main__":
    build_pipeline()
