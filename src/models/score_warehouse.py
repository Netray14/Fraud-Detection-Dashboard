# src/models/score_warehouse.py
"""
Score the Data Warehouse with Twin-Track Risk Scores
Updates fact_hospital_claims_risk with:
- risk_score_classical (XGBoost)
- risk_score_quantum (Quantum-Inspired)
- risk_score_hybrid (average of both)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sqlalchemy import create_engine, text

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import DB_URL, PROJECT_ROOT

def print_flush(message):
    print(message, flush=True)

class WarehouseScorer:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        self.classical_dir = PROJECT_ROOT / 'models' / 'classical'
        self.quantum_dir = PROJECT_ROOT / 'models' / 'quantum'
        self.features_dir = PROJECT_ROOT / 'data' / 'ml_features'
    
    def load_features_with_keys(self):
        """Load features along with fact_keys for joining back"""
        print_flush("Loading features and fact keys...")
        
        X = pd.read_csv(self.features_dir / 'X_features.csv')
        
        # Get fact keys in the same order as features were generated
        query = """
        SELECT f.fact_key
        FROM fact_hospital_claims_risk f
        LEFT JOIN dim_date d ON f.date_key = d.date_key
        LEFT JOIN dim_hospital h ON f.hospital_key = h.hospital_key
        LEFT JOIN dim_district dist ON h.district_key = dist.district_key
        LEFT JOIN dim_state s ON dist.state_key = s.state_key
        LEFT JOIN dim_package p ON f.package_key = p.package_key
        LEFT JOIN dim_fraud_flag_type ff ON f.fraud_flag_key = ff.fraud_flag_key
        """
        fact_keys = pd.read_sql(query, self.engine)['fact_key'].values
        
        if len(fact_keys) != len(X):
            print_flush(f"⚠️  Warning: {len(fact_keys)} fact keys vs {len(X)} feature rows")
            n = min(len(fact_keys), len(X))
            fact_keys = fact_keys[:n]
            X = X.iloc[:n]
        
        print_flush(f"✓ Loaded {len(X):,} records to score")
        return X, fact_keys
    
    def score_classical(self, X):
        """Generate classical XGBoost risk scores"""
        print_flush("\n📊 Generating Classical (XGBoost) risk scores...")
        
        model = joblib.load(self.classical_dir / 'xgboost_model.pkl')
        scaler = joblib.load(self.classical_dir / 'scaler.pkl')
        
        X_scaled = scaler.transform(X)
        scores = model.predict_proba(X_scaled)[:, 1]
        
        print_flush(f"✓ Classical scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        return scores
    
    def score_quantum(self, X):
        """Generate quantum-inspired risk scores"""
        print_flush("\n⚛️  Generating Quantum-Inspired risk scores...")
        
        pca = joblib.load(self.quantum_dir / 'pca.pkl')
        scaler = joblib.load(self.quantum_dir / 'scaler.pkl')
        quantum_weights = joblib.load(self.quantum_dir / 'quantum_weights.pkl')
        
        X_scaled = scaler.transform(X)
        X_reduced = pca.transform(X_scaled)
        
        # Apply quantum-inspired transform (same as training)
        X_quantum = np.zeros_like(X_reduced)
        for i in range(X_reduced.shape[1]):
            for j in range(i, X_reduced.shape[1]):
                X_quantum[:, i] += np.tanh(X_reduced[:, i] * X_reduced[:, j])
        X_quantum = (X_quantum - X_quantum.mean(axis=0)) / (X_quantum.std(axis=0) + 1e-8)
        
        scores = np.dot(X_quantum, quantum_weights)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        print_flush(f"✓ Quantum scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        return scores
    
    def update_warehouse(self, fact_keys, classical_scores, quantum_scores):
        """Write scores back to fact table"""
        print_flush("\n💾 Writing risk scores to warehouse...")
        
        hybrid_scores = (classical_scores + quantum_scores) / 2
        
        # Build update dataframe
        updates = pd.DataFrame({
            'fact_key': fact_keys,
            'risk_score_classical': classical_scores,
            'risk_score_quantum': quantum_scores,
            'risk_score_hybrid': hybrid_scores
        })
        
        # Write to temp table then update via SQL (fast)
        updates.to_sql('temp_scores', self.engine, if_exists='replace', index=False)
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE fact_hospital_claims_risk
                SET risk_score_classical = (
                        SELECT risk_score_classical FROM temp_scores 
                        WHERE temp_scores.fact_key = fact_hospital_claims_risk.fact_key),
                    risk_score_quantum = (
                        SELECT risk_score_quantum FROM temp_scores 
                        WHERE temp_scores.fact_key = fact_hospital_claims_risk.fact_key),
                    risk_score_hybrid = (
                        SELECT risk_score_hybrid FROM temp_scores 
                        WHERE temp_scores.fact_key = fact_hospital_claims_risk.fact_key)
                WHERE fact_key IN (SELECT fact_key FROM temp_scores)
            """))
            conn.execute(text("DROP TABLE temp_scores"))
        
        print_flush(f"✓ Updated {len(updates):,} fact records with twin-track risk scores")
    
    def generate_hospital_vulnerability_index(self):
        """Generate the Hospital Vulnerability Index (your report's key KPI!)"""
        print_flush("\n🏥 Generating Hospital Vulnerability Index...")
        
        query = """
        SELECT 
            h.hospital_key,
            h.hospital_name,
            h.ownership_type,
            s.state_name,
            SUM(f.claim_volume) as total_claims,
            ROUND(SUM(f.claim_value)/10000000, 2) as total_value_cr,
            ROUND(AVG(f.rejection_rate), 4) as avg_rejection_rate,
            ROUND(AVG(f.risk_score_classical), 4) as avg_classical_score,
            ROUND(AVG(f.risk_score_quantum), 4) as avg_quantum_score,
            ROUND(AVG(f.risk_score_hybrid), 4) as avg_hybrid_score,
            SUM(CASE WHEN f.is_fraudulent = 1 THEN f.claim_volume ELSE 0 END) as fraud_claims,
            -- Agreement/disagreement flag (KPI from your report!)
            ROUND(AVG(ABS(f.risk_score_classical - f.risk_score_quantum)), 4) as avg_score_disagreement
        FROM fact_hospital_claims_risk f
        JOIN dim_hospital h ON f.hospital_key = h.hospital_key
        JOIN dim_district d ON h.district_key = d.district_key
        JOIN dim_state s ON d.state_key = s.state_key
        GROUP BY h.hospital_key, h.hospital_name, h.ownership_type, s.state_name
        """
        
        hvi = pd.read_sql(query, self.engine)
        
        # Composite Hospital Vulnerability Index (0-100 scale)
        hvi['vulnerability_index'] = (
            hvi['avg_hybrid_score'].fillna(0) * 40 +          # ML risk score (40%)
            hvi['avg_rejection_rate'].fillna(0) * 30 +        # Rejection rate (30%)
            (hvi['fraud_claims'] / hvi['total_claims']).fillna(0) * 30  # Historic fraud (30%)
        ) * 100
        
        hvi['vulnerability_index'] = hvi['vulnerability_index'].round(2)
        hvi = hvi.sort_values('vulnerability_index', ascending=False)
        
        # Assign priority tiers for SIU investigation
        hvi['investigation_priority'] = pd.cut(
            hvi['vulnerability_index'],
            bins=[-1, 25, 50, 75, 200],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Save
        output_dir = PROJECT_ROOT / 'data' / 'powerbi_exports'
        output_dir.mkdir(parents=True, exist_ok=True)
        hvi.to_csv(output_dir / 'hospital_vulnerability_index.csv', index=False)
        
        print_flush(f"✓ Generated HVI for {len(hvi)} hospitals")
        print_flush("\n🚨 TOP 10 HOSPITALS FOR SIU INVESTIGATION:")
        print_flush("-"*70)
        top10 = hvi.head(10)[['hospital_name', 'state_name', 'vulnerability_index', 
                               'investigation_priority', 'avg_score_disagreement']]
        print_flush(top10.to_string(index=False))
        
        return hvi
    
    def run(self):
        """Execute complete scoring pipeline"""
        print_flush("="*70)
        print_flush("TWIN-TRACK WAREHOUSE SCORING")
        print_flush("(Classical XGBoost + Quantum-Inspired)")
        print_flush("="*70)
        
        X, fact_keys = self.load_features_with_keys()
        
        classical_scores = self.score_classical(X)
        quantum_scores = self.score_quantum(X)
        
        self.update_warehouse(fact_keys, classical_scores, quantum_scores)
        
        hvi = self.generate_hospital_vulnerability_index()
        
        print_flush("\n" + "="*70)
        print_flush("✓ WAREHOUSE SCORING COMPLETE!")
        print_flush("="*70)

if __name__ == "__main__":
    scorer = WarehouseScorer()
    scorer.run()