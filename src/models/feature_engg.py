# src/models/feature_engineering.py
"""
Feature Engineering for Fraud Detection - FIXED VERSION
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sqlalchemy import create_engine

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import DB_URL, PROJECT_ROOT

def print_flush(message):
    """Print with immediate flush"""
    print(message, flush=True)

class FraudFeatureEngineer:
    def __init__(self, db_url=None):
        self.db_url = db_url if db_url else DB_URL
        self.engine = create_engine(self.db_url)
        
    def extract_base_data(self):
        """Extract denormalized data from warehouse"""
        print_flush("Extracting base data from warehouse...")
        
        query = """
        SELECT 
            f.fact_key,
            d.full_date as admission_date,
            d.financial_year,
            d.quarter,
            d.month,
            h.hospital_key,
            h.hospital_name,
            h.ownership_type,
            h.risk_category,
            h.total_beds,
            s.state_name,
            p.procedure_name,
            p.specialty,
            p.package_tier,
            p.base_value as package_base_value,
            ff.flag_type as fraud_flag_type,
            ff.severity as fraud_severity,
            f.claim_volume,
            f.claim_value,
            f.total_approved_value,
            f.total_rejected_value,
            f.rejection_rate,
            f.avg_claim_value,
            f.is_fraudulent,
            f.data_completeness_score
        FROM fact_hospital_claims_risk f
        LEFT JOIN dim_date d ON f.date_key = d.date_key
        LEFT JOIN dim_hospital h ON f.hospital_key = h.hospital_key
        LEFT JOIN dim_district dist ON h.district_key = dist.district_key
        LEFT JOIN dim_state s ON dist.state_key = s.state_key
        LEFT JOIN dim_package p ON f.package_key = p.package_key
        LEFT JOIN dim_fraud_flag_type ff ON f.fraud_flag_key = ff.fraud_flag_key
        """
        
        df = pd.read_sql(query, self.engine)
        print_flush(f"✓ Extracted {len(df)} base records")
        
        # Debug: Check fraud rate
        fraud_rate = df['is_fraudulent'].mean()
        print_flush(f"  Initial fraud rate: {fraud_rate*100:.2f}%")
        
        if fraud_rate > 0.95:
            print_flush("  ⚠️  WARNING: Very high fraud rate detected!")
            print_flush("  This likely means the fact table aggregation included only fraud records")
            print_flush("  Checking flag types...")
            print_flush(df['fraud_flag_type'].value_counts())
        
        return df
    
    def create_hospital_features(self, df):
        """Create hospital-level aggregated features"""
        print_flush("\nCreating hospital-level features...")
        
        hospital_agg = df.groupby('hospital_key').agg({
            'claim_volume': 'sum',
            'claim_value': ['sum', 'mean', 'std'],
            'rejection_rate': ['mean', 'max'],
            'avg_claim_value': ['mean', 'std'],
            'is_fraudulent': ['sum', 'mean'],
            'fact_key': 'count'
        }).reset_index()
        
        hospital_agg.columns = [
            'hospital_key',
            'hosp_total_claims',
            'hosp_total_value',
            'hosp_avg_claim_value',
            'hosp_std_claim_value',
            'hosp_avg_rejection_rate',
            'hosp_max_rejection_rate',
            'hosp_avg_avg_claim',
            'hosp_std_avg_claim',
            'hosp_fraud_count',
            'hosp_fraud_rate',
            'hosp_num_transactions'
        ]
        
        hospital_agg['hosp_value_per_claim'] = (
            hospital_agg['hosp_total_value'] / (hospital_agg['hosp_total_claims'] + 1)
        )
        
        hospital_agg['hosp_cv_claim_value'] = (
            hospital_agg['hosp_std_claim_value'] / (hospital_agg['hosp_avg_claim_value'] + 1)
        )
        
        # Fill NaN with 0
        hospital_agg = hospital_agg.fillna(0)
        
        print_flush(f"  ✓ Created {len(hospital_agg.columns)-1} hospital features")
        
        return hospital_agg
    
    def create_temporal_features(self, df):
        """Create time-based features"""
        print_flush("\nCreating temporal features...")
        
        df['admission_date'] = pd.to_datetime(df['admission_date'])
        df['days_since_start'] = (df['admission_date'] - df['admission_date'].min()).dt.days
        df['month_num'] = df['admission_date'].dt.month
        df['quarter_num'] = df['admission_date'].dt.quarter
        df['year'] = df['admission_date'].dt.year
        df['day_of_week'] = df['admission_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print_flush(f"  ✓ Created temporal features")
        
        return df
    
    def create_package_features(self, df):
        """Create procedure/package-based features"""
        print_flush("\nCreating package features...")
        
        # Up-coding indicator
        df['claim_to_base_ratio'] = df['avg_claim_value'] / (df['package_base_value'] + 1)
        df['is_upcoded'] = (df['claim_to_base_ratio'] > 1.5).astype(int)
        
        # Package tier encoding
        tier_map = {'Tier-1': 1, 'Tier-2': 2, 'Tier-3': 3}
        df['package_tier_num'] = df['package_tier'].map(tier_map).fillna(0)
        
        # Specialty-level fraud rate
        specialty_fraud = df.groupby('specialty')['is_fraudulent'].mean().to_dict()
        df['specialty_fraud_rate'] = df['specialty'].map(specialty_fraud).fillna(0)
        
        print_flush(f"  ✓ Created package features")
        
        return df
    
    def create_fraud_pattern_features(self, df):
        """Create fraud pattern indicators"""
        print_flush("\nCreating fraud pattern features...")
        
        # Severity encoding
        severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['fraud_severity_num'] = df['fraud_severity'].map(severity_map).fillna(0)
        
        # Rejection-based features
        df['high_rejection'] = (df['rejection_rate'] > 0.3).astype(int)
        
        # One-hot encode fraud flag types
        df['is_disabled_card'] = (df['fraud_flag_type'] == 'Disabled-Card Claim').astype(int)
        df['is_post_rejection'] = (df['fraud_flag_type'] == 'Post-Rejection Claim').astype(int)
        df['is_ineligible'] = (df['fraud_flag_type'] == 'Ineligible-Beneficiary Claim').astype(int)
        df['is_duplicate'] = (df['fraud_flag_type'] == 'Duplicate Claim').astype(int)
        df['is_upcoding_flag'] = (df['fraud_flag_type'] == 'Package Up-coding').astype(int)
        df['is_ghost'] = (df['fraud_flag_type'] == 'Ghost Patient').astype(int)
        
        print_flush(f"  ✓ Created fraud pattern features")
        
        return df
    
    def create_hospital_comparison_features(self, df):
        """Compare hospital metrics to state averages"""
        print_flush("\nCreating comparison features...")
        
        state_avg = df.groupby('state_name').agg({
            'avg_claim_value': 'mean',
            'rejection_rate': 'mean',
            'is_fraudulent': 'mean'
        }).reset_index()
        state_avg.columns = ['state_name', 'state_avg_claim', 'state_avg_rejection', 'state_fraud_rate']
        
        df = df.merge(state_avg, on='state_name', how='left')
        
        df['claim_value_vs_state'] = df['avg_claim_value'] / (df['state_avg_claim'] + 1)
        df['rejection_vs_state'] = df['rejection_rate'] / (df['state_avg_rejection'] + 0.01)
        
        print_flush(f"  ✓ Created comparison features")
        
        return df
    
    def build_feature_matrix(self):
        """Build complete feature matrix for ML"""
        print_flush("="*70)
        print_flush("BUILDING FEATURE MATRIX FOR ML")
        print_flush("="*70)
        
        # Extract base data
        df = self.extract_base_data()
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_package_features(df)
        df = self.create_fraud_pattern_features(df)
        df = self.create_hospital_comparison_features(df)
        
        # Hospital-level features
        hospital_features = self.create_hospital_features(df)
        df = df.merge(hospital_features, on='hospital_key', how='left')
        
        # Encode categorical variables
        print_flush("\nEncoding categorical variables...")
        
        # Ownership type
        for ownership in df['ownership_type'].dropna().unique():
            col_name = f'ownership_{ownership.replace(" ", "_")}'
            df[col_name] = (df['ownership_type'] == ownership).astype(int)
        
        # Risk category
        for risk in df['risk_category'].dropna().unique():
            col_name = f'risk_cat_{risk}'
            df[col_name] = (df['risk_category'] == risk).astype(int)
        
        print_flush(f"  ✓ Encoded categorical variables")
        
        # Select ONLY numeric features
        print_flush("\nSelecting numeric features only...")
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifiers and target
        exclude_cols = ['fact_key', 'is_fraudulent', 'month', 'year', 'quarter_num']
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        print_flush(f"  Selected {len(feature_columns)} numeric features")
        
        # Create feature matrix
        X = df[feature_columns].copy()
        y = df['is_fraudulent'].copy()
        
        # Handle missing values
        print_flush("\nHandling missing values...")
        X = X.fillna(X.median())
        
        # Handle infinity values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print_flush("\n" + "="*70)
        print_flush("FEATURE MATRIX SUMMARY")
        print_flush("="*70)
        print_flush(f"Total samples:        {len(X):,}")
        print_flush(f"Total features:       {len(feature_columns)}")
        print_flush(f"Fraudulent samples:   {y.sum():,} ({y.mean()*100:.2f}%)")
        print_flush(f"Non-fraud samples:    {(~y.astype(bool)).sum():,} ({(~y.astype(bool)).mean()*100:.2f}%)")
        
        # Data type check
        print_flush(f"\nData type verification:")
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print_flush(f"  ⚠️  WARNING: Found non-numeric columns: {list(non_numeric)}")
        else:
            print_flush(f"  ✓ All features are numeric")
        
        # Save
        output_dir = PROJECT_ROOT / 'data' / 'ml_features'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X.to_csv(output_dir / 'X_features.csv', index=False)
        y.to_csv(output_dir / 'y_target.csv', index=False, header=['is_fraudulent'])
        
        with open(output_dir / 'feature_names.txt', 'w') as f:
            for feat in feature_columns:
                f.write(f"{feat}\n")
        
        print_flush(f"\n✓ Feature matrix saved to {output_dir}")
        print_flush("="*70)
        
        return X, y, feature_columns, df

if __name__ == "__main__":
    engineer = FraudFeatureEngineer()
    X, y, features, df_full = engineer.build_feature_matrix()
    
    print_flush("\n✓ Feature engineering complete!")