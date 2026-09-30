# src/utils/query_warehouse.py
"""
Query and explore the data warehouse
"""

import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import DB_URL

def run_sample_queries():
    """Run sample analytical queries"""
    engine = create_engine(DB_URL)
    
    print("="*70)
    print("PMJAY DATA WAREHOUSE - SAMPLE QUERIES")
    print("="*70)
    
    # Query 1: Top 10 hospitals by claim volume
    print("\n1. Top 10 Hospitals by Claim Volume:")
    print("-"*70)
    query1 = """
    SELECT 
        h.hospital_name,
        s.state_name,
        SUM(f.claim_volume) as total_claims,
        SUM(f.claim_value) as total_value,
        AVG(f.rejection_rate) as avg_rejection_rate,
        SUM(CASE WHEN f.is_fraudulent = 1 THEN f.claim_volume ELSE 0 END) as fraud_claims
    FROM fact_hospital_claims_risk f
    JOIN dim_hospital h ON f.hospital_key = h.hospital_key
    JOIN dim_district d ON h.district_key = d.district_key
    JOIN dim_state s ON d.state_key = s.state_key
    GROUP BY h.hospital_name, s.state_name
    ORDER BY total_claims DESC
    LIMIT 10
    """
    df1 = pd.read_sql(query1, engine)
    df1['total_value_cr'] = df1['total_value'] / 10000000
    print(df1.to_string(index=False))
    
    # Query 2: Fraud by state
    print("\n\n2. Fraud Statistics by State:")
    print("-"*70)
    query2 = """
    SELECT 
        s.state_name,
        SUM(f.claim_volume) as total_claims,
        SUM(CASE WHEN f.is_fraudulent = 1 THEN f.claim_volume ELSE 0 END) as fraud_claims,
        ROUND(100.0 * SUM(CASE WHEN f.is_fraudulent = 1 THEN f.claim_volume ELSE 0 END) / 
              SUM(f.claim_volume), 2) as fraud_rate_pct,
        ROUND(SUM(f.claim_value) / 10000000, 2) as total_value_cr
    FROM fact_hospital_claims_risk f
    JOIN dim_hospital h ON f.hospital_key = h.hospital_key
    JOIN dim_district d ON h.district_key = d.district_key
    JOIN dim_state s ON d.state_key = s.state_key
    GROUP BY s.state_name
    ORDER BY fraud_rate_pct DESC
    LIMIT 10
    """
    df2 = pd.read_sql(query2, engine)
    print(df2.to_string(index=False))
    
    # Query 3: Fraud by type
    print("\n\n3. Claims by Fraud Type:")
    print("-"*70)
    query3 = """
    SELECT 
        ff.flag_type,
        ff.severity,
        SUM(f.claim_volume) as total_claims,
        ROUND(SUM(f.claim_value) / 10000000, 2) as total_value_cr,
        ROUND(AVG(f.rejection_rate) * 100, 2) as avg_rejection_rate_pct
    FROM fact_hospital_claims_risk f
    JOIN dim_fraud_flag_type ff ON f.fraud_flag_key = ff.fraud_flag_key
    GROUP BY ff.flag_type, ff.severity
    ORDER BY total_claims DESC
    """
    df3 = pd.read_sql(query3, engine)
    print(df3.to_string(index=False))
    
    # Query 4: Temporal trend
    print("\n\n4. Claims Trend by Financial Year:")
    print("-"*70)
    query4 = """
    SELECT 
        d.financial_year,
        SUM(f.claim_volume) as total_claims,
        SUM(CASE WHEN f.is_fraudulent = 1 THEN f.claim_volume ELSE 0 END) as fraud_claims,
        ROUND(SUM(f.claim_value) / 10000000, 2) as total_value_cr
    FROM fact_hospital_claims_risk f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.financial_year
    ORDER BY d.financial_year
    """
    df4 = pd.read_sql(query4, engine)
    print(df4.to_string(index=False))
    
    # Query 5: Package/Procedure analysis
    print("\n\n5. Top 10 Procedures by Volume:")
    print("-"*70)
    query5 = """
    SELECT 
        p.procedure_name,
        p.specialty,
        p.package_tier,
        SUM(f.claim_volume) as total_claims,
        ROUND(AVG(f.avg_claim_value), 2) as avg_claim_value,
        ROUND(SUM(f.claim_value) / 10000000, 2) as total_value_cr
    FROM fact_hospital_claims_risk f
    JOIN dim_package p ON f.package_key = p.package_key
    GROUP BY p.procedure_name, p.specialty, p.package_tier
    ORDER BY total_claims DESC
    LIMIT 10
    """
    df5 = pd.read_sql(query5, engine)
    print(df5.to_string(index=False))
    
    print("\n" + "="*70)
    print("Query execution complete!")
    print("="*70)

if __name__ == "__main__":
    run_sample_queries()