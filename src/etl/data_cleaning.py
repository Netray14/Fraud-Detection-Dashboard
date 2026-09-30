# src/etl/data_cleaning.py
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from pathlib import Path
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, CLEANED_DATA_DIR

def print_flush(message):
    """Print with immediate flush"""
    print(message, flush=True)

class PMJAYDataCleaner:
    def __init__(self, raw_dir=None, cleaned_dir=None):
        self.raw_dir = raw_dir if raw_dir else RAW_DATA_DIR
        self.cleaned_dir = cleaned_dir if cleaned_dir else CLEANED_DATA_DIR
        
        self.raw_dir = Path(self.raw_dir)
        self.cleaned_dir = Path(self.cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        print_flush(f"Raw data directory:     {self.raw_dir.absolute()}")
        print_flush(f"Cleaned data directory: {self.cleaned_dir.absolute()}")
    
    def normalize_currency(self, df, columns):
        """Convert all currency columns to rupees"""
        print_flush(f"  Normalizing {len(columns)} currency columns...")
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self._convert_to_rupees)
        print_flush(f"  ✓ Currency normalization complete")
        return df
    
    def _convert_to_rupees(self, value):
        """Convert various currency formats to rupees"""
        if pd.isna(value):
            return np.nan
        
        value_str = str(value).lower().replace(',', '').strip()
        
        try:
            numeric = float(''.join(c for c in value_str if c.isdigit() or c == '.'))
            
            if 'crore' in value_str or 'cr' in value_str:
                return numeric * 10000000
            elif 'lakh' in value_str or 'lac' in value_str:
                return numeric * 100000
            else:
                return numeric
        except:
            return np.nan
    
    def fuzzy_match_hospitals(self, claims_df, hospital_dir_df, threshold=85):
        """Match hospital names using fuzzy matching"""
        print_flush("  Performing fuzzy matching for hospital names...")
        print_flush(f"  Processing {len(claims_df)} claims against {len(hospital_dir_df)} hospitals...")
        
        canonical_hospitals = hospital_dir_df[['Hospital_Key', 'Hospital_Name', 'State']].to_dict('records')
        
        def find_best_match(name, state):
            state_hospitals = [h for h in canonical_hospitals if h['State'] == state]
            if not state_hospitals:
                return None
            
            hospital_names = [h['Hospital_Name'] for h in state_hospitals]
            match = process.extractOne(name, hospital_names, scorer=fuzz.ratio)
            
            if match and match[1] >= threshold:
                matched_hospital = next(h for h in state_hospitals if h['Hospital_Name'] == match[0])
                return matched_hospital['Hospital_Key']
            return None
        
        matched_count = 0
        if 'Hospital_Name' in claims_df.columns:
            total = len(claims_df)
            for idx, row in enumerate(claims_df.itertuples(), 1):
                if idx % 10000 == 0:
                    print_flush(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")
                
                if pd.isna(row.Hospital_Key):
                    matched_key = find_best_match(row.Hospital_Name, row.State)
                    if matched_key:
                        claims_df.at[row.Index, 'Hospital_Key'] = matched_key
                        matched_count += 1
        
        print_flush(f"  ✓ Matched {matched_count} hospitals")
        return claims_df
    
    def handle_missing_values(self, df, strategy='flag'):
        """Handle missing values with proper categorization"""
        print_flush("  Analyzing missing values...")
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            print_flush("\n  Missing Values Summary:")
            for col, count in missing_cols.items():
                print_flush(f"    {col:30s}: {count:6d} ({count/len(df)*100:5.1f}%)")
        else:
            print_flush("  ✓ No missing values found")
        
        df['Data_Completeness_Score'] = 1 - (df.isnull().sum(axis=1) / len(df.columns))
        
        if 'State' in df.columns:
            non_participating = ['West Bengal', 'Odisha', 'Delhi']
            df['Is_Structurally_Missing'] = df['State'].isin(non_participating)
        
        print_flush(f"  ✓ Missing value handling complete")
        return df
    
    def encode_fraud_flags(self, df, audit_findings_df):
        """Convert free-text audit findings to structured flags"""
        print_flush("  Encoding fraud flags from audit findings...")
        
        fraud_types = [
            'Flag_DisabledCard',
            'Flag_PostRejection',
            'Flag_IneligibleBeneficiary',
            'Flag_Duplicate',
            'Flag_UpCoding'
        ]
        
        for flag in fraud_types:
            df[flag] = 0
        
        if 'Fraud_Flag' in df.columns:
            df['Flag_DisabledCard'] = (df['Fraud_Flag'] == 'Disabled-Card Claim').astype(int)
            df['Flag_PostRejection'] = (df['Fraud_Flag'] == 'Post-Rejection Claim').astype(int)
            df['Flag_IneligibleBeneficiary'] = (df['Fraud_Flag'] == 'Ineligible-Beneficiary Claim').astype(int)
            df['Flag_Duplicate'] = (df['Fraud_Flag'] == 'Duplicate Claim').astype(int)
            df['Flag_UpCoding'] = (df['Fraud_Flag'] == 'Package Up-coding').astype(int)
            
            fraud_count = df[fraud_types].sum().sum()
            print_flush(f"  ✓ Encoded {fraud_count} fraud flags across {len(fraud_types)} categories")
        
        return df
    
    def run_full_pipeline(self):
        """Execute complete cleaning pipeline"""
        print_flush("="*60)
        print_flush("PMJAY DATA CLEANING PIPELINE")
        print_flush("="*60)
        
        # 1. Load raw data
        print_flush("\n1. Loading raw data...")
        try:
            hospitals = pd.read_csv(self.raw_dir / 'hospital_directory.csv')
            print_flush(f"   ✓ Loaded {len(hospitals)} hospitals")
        except FileNotFoundError as e:
            print_flush(f"   ✗ ERROR: {e}")
            print_flush(f"   Looking for: {self.raw_dir / 'hospital_directory.csv'}")
            return None, None, None
        
        try:
            claims = pd.read_csv(self.raw_dir / 'pmjay_admissions.csv')
            print_flush(f"   ✓ Loaded {len(claims)} claims")
        except FileNotFoundError as e:
            print_flush(f"   ✗ ERROR: {e}")
            print_flush(f"   Looking for: {self.raw_dir / 'pmjay_admissions.csv'}")
            return None, None, None
        
        try:
            audit_findings = pd.read_csv(self.raw_dir / 'cag_audit_findings.csv')
            print_flush(f"   ✓ Loaded {len(audit_findings)} audit findings")
        except FileNotFoundError as e:
            print_flush(f"   ✗ ERROR: {e}")
            print_flush(f"   Looking for: {self.raw_dir / 'cag_audit_findings.csv'}")
            return None, None, None
        
        # 2. Normalize currency
        print_flush("\n2. Normalizing currency values...")
        claims = self.normalize_currency(claims, ['Claim_Value', 'Package_Base_Value'])
        audit_findings = self.normalize_currency(audit_findings, ['Amount_Lakhs'])
        
        # 3. Fuzzy match hospitals
        print_flush("\n3. Resolving hospital entities...")
        # Since our mock data already has Hospital_Key, skip fuzzy matching
        if claims['Hospital_Key'].isnull().any():
            claims = self.fuzzy_match_hospitals(claims, hospitals)
        else:
            print_flush("  ✓ Hospital keys already present, skipping fuzzy matching")
        
        # 4. Handle missing values
        print_flush("\n4. Handling missing values...")
        print_flush("  Processing claims...")
        claims = self.handle_missing_values(claims)
        print_flush("  Processing hospitals...")
        hospitals = self.handle_missing_values(hospitals)
        
        # 5. Encode fraud flags
        print_flush("\n5. Encoding fraud flags...")
        claims = self.encode_fraud_flags(claims, audit_findings)
        
        # 6. Save cleaned data
        print_flush("\n6. Saving cleaned data...")
        claims.to_csv(self.cleaned_dir / 'claims_cleaned.csv', index=False)
        print_flush(f"   ✓ Saved claims to {self.cleaned_dir / 'claims_cleaned.csv'}")
        
        hospitals.to_csv(self.cleaned_dir / 'hospitals_cleaned.csv', index=False)
        print_flush(f"   ✓ Saved hospitals to {self.cleaned_dir / 'hospitals_cleaned.csv'}")
        
        audit_findings.to_csv(self.cleaned_dir / 'audit_cleaned.csv', index=False)
        print_flush(f"   ✓ Saved audit findings to {self.cleaned_dir / 'audit_cleaned.csv'}")
        
        print_flush("\n" + "="*60)
        print_flush("✓ DATA CLEANING COMPLETE!")
        print_flush("="*60)
        print_flush(f"\nCleaned data location: {self.cleaned_dir.absolute()}")
        print_flush(f"  - claims_cleaned.csv:    {len(claims):,} records")
        print_flush(f"  - hospitals_cleaned.csv: {len(hospitals):,} records")
        print_flush(f"  - audit_cleaned.csv:     {len(audit_findings):,} records")
        
        return claims, hospitals, audit_findings

if __name__ == "__main__":
    print_flush("\nStarting data cleaning process...")
    print_flush(f"Python version: {sys.version}")
    print_flush(f"Working directory: {Path.cwd()}\n")
    
    cleaner = PMJAYDataCleaner()
    claims, hospitals, audit = cleaner.run_full_pipeline()
    
    if claims is not None:
        print_flush("\n✓ Pipeline completed successfully!")
    else:
        print_flush("\n✗ Pipeline failed - check errors above")