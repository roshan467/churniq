"""
load_to_sql.py
Loads the cleaned customer dataset into a SQLite database so the project
demonstrates real SQL analysis (not just pandas), the way a Data Analyst
would work against a company database.
"""
import sqlite3
import pandas as pd

DB_PATH = "/home/claude/ChurnIQ/data/churniq.db"
CSV_PATH = "/home/claude/ChurnIQ/data/telecom_customers.csv"

def load():
    df = pd.read_csv(CSV_PATH)
    # tenure bucket for cohort-style SQL analysis
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"]
    )
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into {DB_PATH}")

if __name__ == "__main__":
    load()
