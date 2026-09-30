# src/utils/export_clean_powerbi.py
"""
Export warehouse data for Power BI (claim-level)
"""

import pandas as pd
import sys
from pathlib import Path
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_URL, PROJECT_ROOT

def export_for_powerbi():
    print("=" * 70)
    print("EXPORTING DATA FOR POWER BI")
    print("=" * 70)
    
    engine = create_engine(DB_URL)
    output_dir = PROJECT_ROOT / 'data' / 'powerbi'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalized view
    query = """
    SELECT 
        f.fact_key,
        f.claim_volume,
        f.claim_value,
        f.total_approved_value,
        f.total_rejected_value,
        f.rejection_rate,
        f.avg_claim_value,
        f.is_fraudulent,
        f.risk_score_classical,
        f.risk_score_quantum,
        
        d.full_date as admission_date,
        d.financial_year,
        d.quarter,
        d.month,
        d.year,
        
        h.hospital_key,
        h.hospital_name,
        h.ownership_type,
        h.total_beds,
        
        dist.district_name,
        s.state_name,
        
        p.procedure_name,
        p.specialty,
        p.package_tier,
        p.base_value as package_base_value,
        
        ff.flag_type as fraud_flag_type,
        ff.severity as fraud_severity
    FROM fact_hospital_claims_risk f
    LEFT JOIN dim_date d ON f.date_key = d.date_key
    LEFT JOIN dim_hospital h ON f.hospital_key = h.hospital_key
    LEFT JOIN dim_district dist ON h.district_key = dist.district_key
    LEFT JOIN dim_state s ON dist.state_key = s.state_key
    LEFT JOIN dim_package p ON f.package_key = p.package_key
    LEFT JOIN dim_fraud_flag_type ff ON f.fraud_flag_key = ff.fraud_flag_key
    """
    
    print("\nExporting denormalized view...")
    df = pd.read_sql(query, engine)
    
    # Add calculated fields
    df['claim_value_cr'] = df['claim_value'] / 10000000
    df['claim_value_lakhs'] = df['claim_value'] / 100000
    
    # Export
    output_file = output_dir / 'pmjay_claims.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Exported {len(df):,} claims")
    print(f"  File: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Verify fraud rate
    fraud_rate = df['is_fraudulent'].mean()
    print(f"\n✓ Fraud rate in export: {fraud_rate*100:.2f}%")
    print(f"  Fraudulent: {df['is_fraudulent'].sum():,}")
    print(f"  Non-fraud: {(df['is_fraudulent']==0).sum():,}")
    
    print("\n" + "=" * 70)
    print("✓ READY FOR POWER BI")
    print("=" * 70)
    print(f"\nImport this file into Power BI:")
    print(f"  {output_file}")

if __name__ == "__main__":
    export_for_powerbi()