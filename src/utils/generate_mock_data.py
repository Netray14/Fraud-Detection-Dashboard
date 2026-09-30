# src/utils/generate_clean_mock_data.py
"""
Generate clean mock PMJAY data with correct fraud distribution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT

def print_flush(message):
    print(message, flush=True)

class CleanMockDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.states = [
            'Uttar Pradesh', 'Maharashtra', 'Bihar', 'West Bengal',
            'Madhya Pradesh', 'Tamil Nadu', 'Rajasthan', 'Karnataka',
            'Gujarat', 'Andhra Pradesh', 'Telangana', 'Kerala'
        ]
        
        self.procedures = [
            ('Cardiac Bypass', 'Cardiology', 'Tier-3', 150000),
            ('Hip Replacement', 'Orthopedics', 'Tier-3', 120000),
            ('Cataract Surgery', 'Ophthalmology', 'Tier-1', 15000),
            ('Appendectomy', 'General Surgery', 'Tier-2', 35000),
            ('Dialysis Session', 'Nephrology', 'Tier-2', 3000),
            ('Normal Delivery', 'Obstetrics', 'Tier-1', 8000),
            ('C-Section', 'Obstetrics', 'Tier-2', 15000),
            ('Knee Replacement', 'Orthopedics', 'Tier-3', 110000),
            ('Angioplasty', 'Cardiology', 'Tier-3', 95000),
            ('Hernia Repair', 'General Surgery', 'Tier-2', 25000),
        ]
        
        self.fraud_types = [
            'Disabled-Card Claim',
            'Post-Rejection Claim',
            'Ineligible-Beneficiary Claim',
            'Duplicate Claim',
            'Package Up-coding',
            'Ghost Patient'
        ]
    
    def generate_hospitals(self, n=500):
        """Generate hospital master data"""
        print_flush("Generating hospitals...")
        
        hospitals = []
        prefixes = ["Apollo", "Fortis", "Max", "AIIMS", "District", 
                   "Community", "Medical College", "Civil"]
        
        for i in range(n):
            state = random.choice(self.states)
            hospitals.append({
                'Hospital_Key': f'H{i+1:05d}',
                'Hospital_Name': f'{random.choice(prefixes)} Hospital {state[:3].upper()}-{i+1}',
                'State': state,
                'District': f'{state} District {random.randint(1, 10)}',
                'Ownership_Type': random.choice(['Public', 'Private', 'Private', 'Trust']),
                'Empanelment_Date': (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1800))).strftime('%Y-%m-%d'),
                'Total_Beds': random.randint(50, 500),
                'Risk_Category': random.choices(['Low', 'Medium', 'High'], weights=[0.7, 0.2, 0.1])[0]
            })
        
        df = pd.DataFrame(hospitals)
        print_flush(f"  ✓ Generated {len(df)} hospitals")
        return df
    
    def generate_claims(self, hospitals_df, n=30000):
        """Generate claim-level data with CORRECT fraud distribution"""
        print_flush(f"Generating {n:,} claims...")
        
        claims = []
        start_date = datetime(2018, 4, 1)
        end_date = datetime(2023, 3, 31)
        days_range = (end_date - start_date).days
        
        for i in range(n):
            if (i + 1) % 10000 == 0:
                print_flush(f"  Generated {i+1:,}/{n:,}...")
            
            # Select hospital
            hospital = hospitals_df.sample(1).iloc[0]
            procedure = random.choice(self.procedures)
            
            # Random date
            admission_date = start_date + timedelta(days=random.randint(0, days_range))
            
            # Financial year
            if admission_date.month >= 4:
                fy = f'FY{admission_date.year}-{(admission_date.year+1) % 100:02d}'
            else:
                fy = f'FY{admission_date.year-1}-{admission_date.year % 100:02d}'
            
            # **KEY FIX: Correct fraud probability**
            # Low-risk hospitals: 3% fraud
            # Medium-risk: 8% fraud
            # High-risk: 15% fraud
            if hospital['Risk_Category'] == 'High':
                fraud_prob = 0.15
            elif hospital['Risk_Category'] == 'Medium':
                fraud_prob = 0.08
            else:
                fraud_prob = 0.03
            
            is_fraudulent = random.random() < fraud_prob
            fraud_flag = random.choice(self.fraud_types) if is_fraudulent else None
            
            # Claim value with variations
            base_value = procedure[3]
            claim_value = base_value * random.uniform(0.9, 1.3)
            
            # Up-code fraud adds 50-100% more
            if fraud_flag == 'Package Up-coding':
                claim_value *= random.uniform(1.5, 2.0)
            
            # Claim status
            if is_fraudulent:
                claim_status = random.choices(['Rejected', 'Approved', 'Pending'], 
                                            weights=[0.7, 0.2, 0.1])[0]
            else:
                claim_status = random.choices(['Approved', 'Rejected', 'Pending'], 
                                            weights=[0.90, 0.05, 0.05])[0]
            
            claims.append({
                'Claim_ID': f'C{i+1:08d}',
                'Hospital_Key': hospital['Hospital_Key'],
                'Hospital_Name': hospital['Hospital_Name'],
                'State': hospital['State'],
                'District': hospital['District'],
                'Admission_Date': admission_date.strftime('%Y-%m-%d'),
                'Financial_Year': fy,
                'Procedure_Name': procedure[0],
                'Specialty': procedure[1],
                'Package_Tier': procedure[2],
                'Package_Base_Value': procedure[3],
                'Claim_Value': round(claim_value, 2),
                'Claim_Status': claim_status,
                'Fraud_Flag': fraud_flag,
                'Is_Fraudulent': int(is_fraudulent),
                'Beneficiary_ID': f'BEN{random.randint(1, 30000000):08d}',
                'Age': random.randint(5, 85),
                'Gender': random.choice(['M', 'F']),
                'Length_of_Stay': random.randint(1, 30)
            })
        
        df = pd.DataFrame(claims)
        print_flush(f"  ✓ Generated {len(df):,} claims")
        
        # Verify distribution
        fraud_rate = df['Is_Fraudulent'].mean()
        print_flush(f"  Fraud rate: {fraud_rate*100:.2f}%")
        print_flush(f"  Fraudulent: {df['Is_Fraudulent'].sum():,}")
        print_flush(f"  Non-fraud: {(df['Is_Fraudulent']==0).sum():,}")
        
        return df
    
    def save_datasets(self):
        """Save all datasets"""
        print_flush("\n" + "="*70)
        print_flush("GENERATING CLEAN DATASETS")
        print_flush("="*70)
        
        # Generate
        hospitals = self.generate_hospitals(500)
        claims = self.generate_claims(hospitals, 30000)
        
        # Save to cleaned data directory
        cleaned_dir = PROJECT_ROOT / 'data' / 'cleaned'
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Save claims
        claims_file = cleaned_dir / 'claims_cleaned.csv'
        claims.to_csv(claims_file, index=False)
        print_flush(f"\n✓ Saved claims: {claims_file}")
        print_flush(f"  Records: {len(claims):,}")
        print_flush(f"  Size: {claims_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save hospitals
        hospitals_file = cleaned_dir / 'hospitals_cleaned.csv'
        hospitals.to_csv(hospitals_file, index=False)
        print_flush(f"\n✓ Saved hospitals: {hospitals_file}")
        print_flush(f"  Records: {len(hospitals):,}")
        
        print_flush("\n" + "="*70)
        print_flush("✓ DATASETS READY FOR WAREHOUSE BUILD")
        print_flush("="*70)
        
        return claims, hospitals

if __name__ == "__main__":
    generator = CleanMockDataGenerator()
    claims, hospitals = generator.save_datasets()