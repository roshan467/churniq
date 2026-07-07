-- ============================================================
-- churn_analysis.sql
-- Advanced SQL analytics queries for ChurnIQ
-- Demonstrates: joins, window functions, CTEs, aggregations,
-- cohort analysis — run against data/churniq.db (SQLite)
-- ============================================================

-- 1. Overall churn rate (headline KPI)
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 2. Churn rate by contract type, ranked (window function)
SELECT
    contract,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    RANK() OVER (ORDER BY 100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*) DESC) AS risk_rank
FROM customers
GROUP BY contract;

-- 3. Cohort analysis: churn rate by tenure bucket
SELECT
    tenure_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY tenure_bucket
ORDER BY
    CASE tenure_bucket
        WHEN '0-6mo' THEN 1 WHEN '7-12mo' THEN 2 WHEN '13-24mo' THEN 3
        WHEN '25-48mo' THEN 4 ELSE 5 END;

-- 4. High-risk segment identification (CTE + multi-condition filter)
WITH risk_flagged AS (
    SELECT *,
        CASE
            WHEN contract = 'Month-to-month'
                 AND payment_method = 'Electronic check'
                 AND tech_support = 'No'
            THEN 1 ELSE 0
        END AS high_risk_flag
    FROM customers
)
SELECT
    high_risk_flag,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue_at_stake
FROM risk_flagged
GROUP BY high_risk_flag;

-- 5. Revenue impact of churn by internet service type
SELECT
    internet_service,
    SUM(CASE WHEN churn='Yes' THEN monthly_charges ELSE 0 END) AS monthly_revenue_lost,
    COUNT(CASE WHEN churn='Yes' THEN 1 END) AS churned_customers
FROM customers
GROUP BY internet_service
ORDER BY monthly_revenue_lost DESC;

-- 6. Payment method churn ranking with running total (window function)
SELECT
    payment_method,
    churned,
    SUM(churned) OVER (ORDER BY churned DESC) AS running_total_churned
FROM (
    SELECT payment_method, SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) AS churned
    FROM customers
    GROUP BY payment_method
) t
ORDER BY churned DESC;
