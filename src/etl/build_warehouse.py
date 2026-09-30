# src/etl/build_warehouse.py
"""
Build Star/Snowflake Schema Data Warehouse for PMJAY Fraud Detection
FIXED VERSION - Preserves claim-level records (no aggregation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Add src to path and import config
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import CLEANED_DATA_DIR, DB_URL, DB_PATH

Base = declarative_base()

def print_flush(message):
    """Print with immediate flush"""
    print(message, flush=True)

# ============================================================
# DIMENSION TABLES
# ============================================================

class DimDate(Base):
    """Date dimension table"""
    __tablename__ = 'dim_date'
    
    date_key = Column(Integer, primary_key=True)
    full_date = Column(Date, unique=True, nullable=False)
    financial_year = Column(String(10))
    quarter = Column(String(2))
    month = Column(Integer)
    year = Column(Integer)
    day_of_week = Column(String(10))
    
    def __repr__(self):
        return f"<DimDate(date_key={self.date_key}, fy={self.financial_year})>"

class DimState(Base):
    """State dimension table (Snowflake parent)"""
    __tablename__ = 'dim_state'
    
    state_key = Column(Integer, primary_key=True, autoincrement=True)
    state_name = Column(String(100), unique=True, nullable=False)
    region = Column(String(50))
    population_millions = Column(Float)
    
    # Snowflake relationship
    districts = relationship("DimDistrict", back_populates="state")
    
    def __repr__(self):
        return f"<DimState(state_key={self.state_key}, name={self.state_name})>"

class DimDistrict(Base):
    """District dimension table (Snowflake child of State)"""
    __tablename__ = 'dim_district'
    
    district_key = Column(Integer, primary_key=True, autoincrement=True)
    district_name = Column(String(100), nullable=False)
    state_key = Column(Integer, ForeignKey('dim_state.state_key'))
    
    # Snowflake relationships
    state = relationship("DimState", back_populates="districts")
    hospitals = relationship("DimHospital", back_populates="district")
    
    def __repr__(self):
        return f"<DimDistrict(district_key={self.district_key}, name={self.district_name})>"

class DimHospital(Base):
    """Hospital dimension table (Star dimension referencing Snowflake)"""
    __tablename__ = 'dim_hospital'
    
    hospital_key = Column(String(20), primary_key=True)
    hospital_name = Column(String(200), nullable=False)
    ownership_type = Column(String(50))
    total_beds = Column(Integer)
    risk_category = Column(String(20))
    empanelment_date = Column(Date)
    district_key = Column(Integer, ForeignKey('dim_district.district_key'))
    
    # Snowflake relationship
    district = relationship("DimDistrict", back_populates="hospitals")
    
    def __repr__(self):
        return f"<DimHospital(key={self.hospital_key}, name={self.hospital_name})>"

class DimPackage(Base):
    """Medical package/procedure dimension table"""
    __tablename__ = 'dim_package'
    
    package_key = Column(Integer, primary_key=True, autoincrement=True)
    procedure_name = Column(String(200), unique=True, nullable=False)
    specialty = Column(String(100))
    package_tier = Column(String(20))
    base_value = Column(Float)
    
    def __repr__(self):
        return f"<DimPackage(key={self.package_key}, procedure={self.procedure_name})>"

class DimFraudFlagType(Base):
    """Fraud flag type dimension table"""
    __tablename__ = 'dim_fraud_flag_type'
    
    fraud_flag_key = Column(Integer, primary_key=True, autoincrement=True)
    flag_type = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    severity = Column(String(20))
    
    def __repr__(self):
        return f"<DimFraudFlagType(key={self.fraud_flag_key}, type={self.flag_type})>"

# ============================================================
# FACT TABLE
# ============================================================

class FactHospitalClaimsRisk(Base):
    """Main fact table for hospital claims and risk scoring"""
    __tablename__ = 'fact_hospital_claims_risk'
    
    fact_key = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys to dimensions
    date_key = Column(Integer, ForeignKey('dim_date.date_key'))
    hospital_key = Column(String(20), ForeignKey('dim_hospital.hospital_key'))
    package_key = Column(Integer, ForeignKey('dim_package.package_key'))
    fraud_flag_key = Column(Integer, ForeignKey('dim_fraud_flag_type.fraud_flag_key'))
    
    # Measures (additive and semi-additive)
    claim_volume = Column(Integer, default=0)
    claim_value = Column(Float, default=0.0)
    total_approved_value = Column(Float, default=0.0)
    total_rejected_value = Column(Float, default=0.0)
    
    # Calculated measures
    rejection_rate = Column(Float)
    avg_claim_value = Column(Float)
    
    # Risk scoring (to be populated by ML models)
    risk_score_classical = Column(Float)
    risk_score_quantum = Column(Float)
    risk_score_hybrid = Column(Float)
    
    # Fraud indicators
    is_fraudulent = Column(Boolean, default=False)
    fraud_confirmed = Column(Boolean)
    
    # Data quality
    data_completeness_score = Column(Float)
    source_flag = Column(String(50))
    
    def __repr__(self):
        return f"<FactHospitalClaimsRisk(fact_key={self.fact_key}, hospital={self.hospital_key}, value={self.claim_value})>"

# ============================================================
# DATA WAREHOUSE BUILDER CLASS
# ============================================================

class DataWarehouseBuilder:
    def __init__(self, db_url=None):
        self.db_url = db_url if db_url else DB_URL
        self.db_path = DB_PATH
        
        print_flush(f"Database URL: {self.db_url}")
        print_flush(f"Database Path: {self.db_path}")
        
        # Create engine and tables
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        
        self.Session = sessionmaker(bind=self.engine)
        
        print_flush(f"✓ Database initialized at {self.db_path}")
    
    def load_dim_date(self, claims_df):
        """Load date dimension from claims data"""
        print_flush("\n  Loading DimDate...")
        session = self.Session()
        
        try:
            # Get unique dates from claims
            dates = pd.to_datetime(claims_df['Admission_Date']).unique()
            date_records = []
            
            for date in sorted(dates):
                date_obj = pd.to_datetime(date)
                
                # Calculate financial year
                if date_obj.month >= 4:
                    fy = f"FY{date_obj.year}-{(date_obj.year+1) % 100:02d}"
                else:
                    fy = f"FY{date_obj.year-1}-{date_obj.year % 100:02d}"
                
                # Calculate quarter (based on FY)
                fy_month = date_obj.month - 3 if date_obj.month >= 4 else date_obj.month + 9
                quarter = f"Q{(fy_month - 1) // 3 + 1}"
                
                dim_date = DimDate(
                    date_key=int(date_obj.strftime('%Y%m%d')),
                    full_date=date_obj.date(),
                    financial_year=fy,
                    quarter=quarter,
                    month=date_obj.month,
                    year=date_obj.year,
                    day_of_week=date_obj.strftime('%A')
                )
                date_records.append(dim_date)
            
            # Bulk insert
            session.bulk_save_objects(date_records)
            session.commit()
            
            print_flush(f"    ✓ Loaded {len(date_records)} date records")
            return len(date_records)
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading DimDate: {e}")
            raise
        finally:
            session.close()
    
    def load_dim_state_district(self, hospitals_df):
        """Load state and district dimensions (snowflake)"""
        print_flush("\n  Loading DimState and DimDistrict (Snowflake)...")
        session = self.Session()
        
        try:
            # 1. Load States
            states = hospitals_df['State'].unique()
            state_mapping = {}
            
            for idx, state in enumerate(sorted(states), 1):
                dim_state = DimState(
                    state_key=idx,
                    state_name=state,
                    region='India',
                    population_millions=None
                )
                session.add(dim_state)
                state_mapping[state] = idx
            
            session.commit()
            print_flush(f"    ✓ Loaded {len(states)} states")
            
            # 2. Load Districts (snowflaked from State)
            districts = hospitals_df[['District', 'State']].drop_duplicates()
            district_mapping = {}
            
            for idx, row in enumerate(districts.itertuples(), 1):
                dim_district = DimDistrict(
                    district_key=idx,
                    district_name=row.District,
                    state_key=state_mapping[row.State]
                )
                session.add(dim_district)
                district_mapping[row.District] = idx
            
            session.commit()
            print_flush(f"    ✓ Loaded {len(districts)} districts")
            
            return state_mapping, district_mapping
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading State/District: {e}")
            raise
        finally:
            session.close()
    
    def load_dim_hospital(self, hospitals_df, district_mapping):
        """Load hospital dimension"""
        print_flush("\n  Loading DimHospital...")
        session = self.Session()
        
        try:
            hospital_records = []
            
            for _, hosp in hospitals_df.iterrows():
                dim_hospital = DimHospital(
                    hospital_key=hosp['Hospital_Key'],
                    hospital_name=hosp['Hospital_Name'],
                    ownership_type=hosp.get('Ownership_Type'),
                    total_beds=int(hosp['Total_Beds']) if pd.notna(hosp.get('Total_Beds')) else None,
                    risk_category=hosp.get('Risk_Category'),
                    empanelment_date=pd.to_datetime(hosp['Empanelment_Date']).date() 
                        if pd.notna(hosp.get('Empanelment_Date')) else None,
                    district_key=district_mapping.get(hosp['District'])
                )
                hospital_records.append(dim_hospital)
            
            session.bulk_save_objects(hospital_records)
            session.commit()
            
            print_flush(f"    ✓ Loaded {len(hospital_records)} hospitals")
            return len(hospital_records)
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading DimHospital: {e}")
            raise
        finally:
            session.close()
    
    def load_dim_package(self, claims_df):
        """Load package/procedure dimension"""
        print_flush("\n  Loading DimPackage...")
        session = self.Session()
        
        try:
            packages = claims_df[['Procedure_Name', 'Specialty', 'Package_Tier', 'Package_Base_Value']].drop_duplicates()
            package_mapping = {}
            package_records = []
            
            for idx, row in enumerate(packages.itertuples(), 1):
                dim_package = DimPackage(
                    package_key=idx,
                    procedure_name=row.Procedure_Name,
                    specialty=row.Specialty,
                    package_tier=row.Package_Tier,
                    base_value=float(row.Package_Base_Value) if pd.notna(row.Package_Base_Value) else None
                )
                package_records.append(dim_package)
                package_mapping[row.Procedure_Name] = idx
            
            session.bulk_save_objects(package_records)
            session.commit()
            
            print_flush(f"    ✓ Loaded {len(package_records)} packages")
            return package_mapping
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading DimPackage: {e}")
            raise
        finally:
            session.close()
    
    def load_dim_fraud_flag_type(self):
        """Load fraud flag type dimension"""
        print_flush("\n  Loading DimFraudFlagType...")
        session = self.Session()
        
        try:
            fraud_types = [
                (1, 'Disabled-Card Claim', 'Claims initiated on Aadhaar-disabled cards', 'High'),
                (2, 'Post-Rejection Claim', 'Claims initiated after their own rejection date', 'High'),
                (3, 'Ineligible-Beneficiary Claim', 'Claims for ineligible pensioner families', 'High'),
                (4, 'Duplicate Claim', 'Same beneficiary multiple claims for same procedure', 'Medium'),
                (5, 'Package Up-coding', 'Billed at highest-tier package inappropriately', 'Medium'),
                (6, 'Ghost Patient', 'Non-existent patient claims', 'High'),
                (7, 'None', 'No fraud indicators detected', 'Low')
            ]
            
            fraud_records = []
            for key, flag_type, desc, severity in fraud_types:
                dim_fraud = DimFraudFlagType(
                    fraud_flag_key=key,
                    flag_type=flag_type,
                    description=desc,
                    severity=severity
                )
                fraud_records.append(dim_fraud)
            
            session.bulk_save_objects(fraud_records)
            session.commit()
            
            print_flush(f"    ✓ Loaded {len(fraud_records)} fraud flag types")
            return len(fraud_records)
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading DimFraudFlagType: {e}")
            raise
        finally:
            session.close()
    
    def load_fact_table(self, claims_df, package_mapping):
        """Load fact table - FIXED: Keep individual claims, don't aggregate"""
        print_flush("\n  Loading FactHospitalClaimsRisk...")
        session = self.Session()
        
        try:
            # Map fraud flags
            fraud_flag_map = {
                'Disabled-Card Claim': 1,
                'Post-Rejection Claim': 2,
                'Ineligible-Beneficiary Claim': 3,
                'Duplicate Claim': 4,
                'Package Up-coding': 5,
                'Ghost Patient': 6,
            }
            
            print_flush("    Preparing claims for warehouse...")
            
            # Create fraud flag key column
            claims_df['Fraud_Flag_Key'] = claims_df['Fraud_Flag'].map(fraud_flag_map).fillna(7).astype(int)
            
            # Check distribution BEFORE loading
            fraud_count = claims_df['Is_Fraudulent'].sum()
            total_count = len(claims_df)
            print_flush(f"    Total claims: {total_count:,}")
            print_flush(f"    Fraudulent: {fraud_count:,} ({fraud_count/total_count*100:.2f}%)")
            print_flush(f"    Non-fraudulent: {total_count - fraud_count:,} ({(total_count-fraud_count)/total_count*100:.2f}%)")
            
            # **KEY FIX: Don't aggregate - keep individual claim records**
            print_flush("    Creating fact records (one per claim - NO AGGREGATION)...")
            
            fact_records = []
            for idx, row in enumerate(claims_df.itertuples(), 1):
                if idx % 5000 == 0:
                    print_flush(f"      Processing {idx:,}/{total_count:,}...")
                
                date_key = int(pd.to_datetime(row.Admission_Date).strftime('%Y%m%d'))
                
                # Determine approved/rejected values
                if row.Claim_Status == 'Approved':
                    approved_value = float(row.Claim_Value)
                    rejected_value = 0.0
                    rejection_rate = 0.0
                elif row.Claim_Status == 'Rejected':
                    approved_value = 0.0
                    rejected_value = float(row.Claim_Value)
                    rejection_rate = 1.0
                else:  # Pending
                    approved_value = 0.0
                    rejected_value = 0.0
                    rejection_rate = 0.0
                
                fact = FactHospitalClaimsRisk(
                    date_key=date_key,
                    hospital_key=row.Hospital_Key,
                    package_key=package_mapping.get(row.Procedure_Name),
                    fraud_flag_key=int(row.Fraud_Flag_Key),
                    claim_volume=1,  # One claim per record (NOT aggregated)
                    claim_value=float(row.Claim_Value),
                    total_approved_value=approved_value,
                    total_rejected_value=rejected_value,
                    rejection_rate=rejection_rate,
                    avg_claim_value=float(row.Claim_Value),
                    is_fraudulent=bool(row.Is_Fraudulent),
                    fraud_confirmed=None,
                    risk_score_classical=None,
                    risk_score_quantum=None,
                    risk_score_hybrid=None,
                    data_completeness_score=1.0,
                    source_flag='Primary'
                )
                fact_records.append(fact)
            
            # Bulk insert in batches
            print_flush(f"    Inserting {len(fact_records):,} records...")
            batch_size = 2000
            for i in range(0, len(fact_records), batch_size):
                batch = fact_records[i:i+batch_size]
                session.bulk_save_objects(batch)
                session.commit()
                if (i + batch_size) % 10000 == 0:
                    print_flush(f"      Inserted {min(i+batch_size, len(fact_records)):,}...")
            
            # Final verification
            final_fraud_count = sum(1 for f in fact_records if f.is_fraudulent)
            final_fraud_rate = final_fraud_count / len(fact_records) if fact_records else 0
            
            print_flush(f"    ✓ Loaded {len(fact_records):,} fact records")
            print_flush(f"    Final fraud distribution:")
            print_flush(f"      Fraudulent: {final_fraud_count:,} ({final_fraud_rate*100:.2f}%)")
            print_flush(f"      Non-fraud: {len(fact_records) - final_fraud_count:,} ({(1-final_fraud_rate)*100:.2f}%)")
            
            return len(fact_records)
            
        except Exception as e:
            session.rollback()
            print_flush(f"    ✗ Error loading FactHospitalClaimsRisk: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            session.close()
    
    def build_warehouse(self, 
                       claims_path=None, 
                       hospitals_path=None):
        """Build complete data warehouse"""
        
        claims_path = claims_path if claims_path else CLEANED_DATA_DIR / 'claims_cleaned.csv'
        hospitals_path = hospitals_path if hospitals_path else CLEANED_DATA_DIR / 'hospitals_cleaned.csv'
        
        print_flush("="*70)
        print_flush("BUILDING PMJAY DATA WAREHOUSE")
        print_flush("="*70)
        print_flush(f"\nSource files:")
        print_flush(f"  Claims:    {claims_path}")
        print_flush(f"  Hospitals: {hospitals_path}")
        print_flush(f"\nTarget database: {self.db_path}")
        
        # Load source data
        print_flush("\n" + "-"*70)
        print_flush("Loading source data...")
        print_flush("-"*70)
        
        claims_df = pd.read_csv(claims_path)
        hospitals_df = pd.read_csv(hospitals_path)
        
        print_flush(f"✓ Loaded {len(claims_df):,} claims")
        print_flush(f"✓ Loaded {len(hospitals_df):,} hospitals")
        
        # Build dimensions
        print_flush("\n" + "-"*70)
        print_flush("Building Dimension Tables...")
        print_flush("-"*70)
        
        self.load_dim_date(claims_df)
        state_mapping, district_mapping = self.load_dim_state_district(hospitals_df)
        self.load_dim_hospital(hospitals_df, district_mapping)
        package_mapping = self.load_dim_package(claims_df)
        self.load_dim_fraud_flag_type()
        
        # Build fact table
        print_flush("\n" + "-"*70)
        print_flush("Building Fact Table (Claim-Level)...")
        print_flush("-"*70)
        
        self.load_fact_table(claims_df, package_mapping)
        
        # Summary
        print_flush("\n" + "="*70)
        print_flush("✓ DATA WAREHOUSE BUILD COMPLETE!")
        print_flush("="*70)
        
        self.print_warehouse_summary()
    
    def print_warehouse_summary(self):
        """Print warehouse statistics"""
        session = self.Session()
        
        try:
            from sqlalchemy import func
            
            print_flush("\n📊 Warehouse Summary:")
            print_flush("-"*70)
            
            # Count records in each table
            tables = [
                ('DimDate', DimDate),
                ('DimState', DimState),
                ('DimDistrict', DimDistrict),
                ('DimHospital', DimHospital),
                ('DimPackage', DimPackage),
                ('DimFraudFlagType', DimFraudFlagType),
                ('FactHospitalClaimsRisk', FactHospitalClaimsRisk)
            ]
            
            for table_name, table_class in tables:
                count = session.query(table_class).count()
                print_flush(f"  {table_name:30s} {count:10,} records")
            
            # Fact table statistics
            total_volume = session.query(
                func.sum(FactHospitalClaimsRisk.claim_volume)
            ).scalar() or 0
            
            total_value = session.query(
                func.sum(FactHospitalClaimsRisk.claim_value)
            ).scalar() or 0
            
            fraud_records = session.query(FactHospitalClaimsRisk).filter(
                FactHospitalClaimsRisk.is_fraudulent == True
            ).count()
            
            total_records = session.query(FactHospitalClaimsRisk).count()
            
            fraud_volume = session.query(
                func.sum(FactHospitalClaimsRisk.claim_volume)
            ).filter(
                FactHospitalClaimsRisk.is_fraudulent == True
            ).scalar() or 0
            
            print_flush("\n📈 Fact Table Metrics:")
            print_flush("-"*70)
            print_flush(f"  Total Records:         {total_records:,}")
            print_flush(f"  Fraud Records:         {fraud_records:,} ({fraud_records/total_records*100:.2f}%)")
            print_flush(f"  Non-Fraud Records:     {total_records - fraud_records:,} ({(total_records-fraud_records)/total_records*100:.2f}%)")
            print_flush(f"  Total Claims Volume:   {total_volume:,}")
            print_flush(f"  Total Value:           ₹{total_value/10000000:,.2f} Cr")
            print_flush(f"  Fraudulent Claims:     {fraud_volume:,} ({fraud_volume/total_volume*100:.2f}%)")
            
            print_flush("\n" + "="*70)
            print_flush(f"Database location: {self.db_path.absolute()}")
            print_flush(f"Database size: {self.db_path.stat().st_size / 1024 / 1024:.2f} MB")
            print_flush("="*70)
            
        except Exception as e:
            print_flush(f"Error generating summary: {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print_flush("\nStarting warehouse build process...")
    print_flush(f"Python version: {sys.version}")
    print_flush(f"Working directory: {Path.cwd()}\n")
    
    builder = DataWarehouseBuilder()
    builder.build_warehouse()
    
    print_flush("\n✓ Warehouse build completed successfully!")
    print_flush(f"\nNext steps:")
    print_flush("  1. Verify database: python check_database.py")
    print_flush("  2. Export for Power BI: python src/utils/export_clean_powerbi.py")
    print_flush("  3. Build Power BI dashboard")