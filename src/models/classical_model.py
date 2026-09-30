# src/models/classical_model.py
"""
Classical Machine Learning Model - XGBoost Classifier
For Fraud Risk Scoring
UPDATED VERSION - Compatible with all XGBoost versions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT

def print_flush(message):
    """Print with immediate flush"""
    print(message, flush=True)

class ClassicalFraudDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_dir = PROJECT_ROOT / 'models' / 'classical'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load feature matrix"""
        print_flush("Loading feature matrix...")
        
        features_dir = PROJECT_ROOT / 'data' / 'ml_features'
        
        X = pd.read_csv(features_dir / 'X_features.csv')
        y = pd.read_csv(features_dir / 'y_target.csv').values.ravel()
        
        # Load feature names
        with open(features_dir / 'feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print_flush(f"✓ Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
        print_flush(f"  Fraud rate: {y.mean()*100:.2f}%")
        print_flush(f"  Non-fraud rate: {(1-y.mean())*100:.2f}%")
        
        return X, y
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using SMOTE or undersampling"""
        print_flush(f"\nHandling class imbalance using {method.upper()}...")
        
        print_flush(f"  Before resampling:")
        print_flush(f"    Class 0 (Non-fraud): {(y_train == 0).sum():,}")
        print_flush(f"    Class 1 (Fraud):     {(y_train == 1).sum():,}")
        
        try:
            if method == 'smote':
                # SMOTE: Oversample minority class
                smote = SMOTE(random_state=self.random_state, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
            elif method == 'hybrid':
                # Hybrid: SMOTE + Random undersampling
                over = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
                under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
                
                pipeline = ImbPipeline([
                    ('over', over),
                    ('under', under)
                ])
                X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
            
            else:
                # No resampling
                X_resampled, y_resampled = X_train, y_train
            
            print_flush(f"  After resampling:")
            print_flush(f"    Class 0 (Non-fraud): {(y_resampled == 0).sum():,}")
            print_flush(f"    Class 1 (Fraud):     {(y_resampled == 1).sum():,}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print_flush(f"  ⚠️  Resampling failed: {e}")
            print_flush(f"  Continuing without resampling...")
            return X_train, y_train
    
    def train_model(self, X, y, use_resampling=True):
        """Train XGBoost classifier"""
        print_flush("\n" + "="*70)
        print_flush("TRAINING XGBOOST CLASSIFIER")
        print_flush("="*70)
        
        # Check class distribution
        unique_classes = np.unique(y)
        print_flush(f"\nUnique classes in target: {unique_classes}")
        
        if len(unique_classes) < 2:
            print_flush("\n✗ ERROR: Only one class present in data!")
            print_flush("  Cannot train a binary classifier with single class")
            print_flush("  Please check data pipeline")
            return None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        
        print_flush(f"\nData Split:")
        print_flush(f"  Train set: {X_train.shape[0]:,} samples")
        print_flush(f"  Test set:  {X_test.shape[0]:,} samples")
        print_flush(f"  Train fraud rate: {y_train.mean()*100:.2f}%")
        print_flush(f"  Test fraud rate:  {y_test.mean()*100:.2f}%")
        
        # Handle class imbalance
        if use_resampling and len(np.unique(y_train)) >= 2:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train, y_train, method='hybrid'
            )
        else:
            print_flush("\n⚠️  Skipping resampling")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features
        print_flush("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        print_flush("  ✓ Features scaled using StandardScaler")
        
        # XGBoost parameters - Compatible with all versions
        print_flush("\nConfiguring XGBoost model...")
        
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # Check XGBoost version and add version-specific params
        try:
            xgb_version = xgb.__version__
            print_flush(f"  XGBoost version: {xgb_version}")
            
            # For newer versions
            if hasattr(xgb.XGBClassifier(), 'use_label_encoder'):
                params['use_label_encoder'] = False
        except:
            pass
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train model
        print_flush("\nTraining XGBoost model...")
        print_flush("  (This may take a few minutes...)")
        
        self.model.fit(
            X_train_scaled, 
            y_train_balanced,
            verbose=False
        )
        
        print_flush(f"✓ Model trained successfully")
        print_flush(f"  Estimators used: {self.model.n_estimators}")
        
        # Cross-validation
        print_flush("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_balanced,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print_flush(f"  CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print_flush(f"  CV scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        # Evaluate on test set
        print_flush("\n" + "="*70)
        print_flush("MODEL EVALUATION ON TEST SET")
        print_flush("="*70)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Classification Report
        print_flush("\nClassification Report:")
        print_flush("-"*70)
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'], digits=4)
        print_flush(report)
        
        # Confusion Matrix
        print_flush("\nConfusion Matrix:")
        print_flush("-"*70)
        cm = confusion_matrix(y_test, y_pred)
        
        print_flush(f"")
        print_flush(f"                    Predicted")
        print_flush(f"                Non-Fraud    Fraud")
        print_flush(f"  Actual")
        print_flush(f"  Non-Fraud     {cm[0,0]:6,}    {cm[0,1]:6,}")
        print_flush(f"  Fraud         {cm[1,0]:6,}    {cm[1,1]:6,}")
        print_flush(f"")
        
        # Key Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print_flush("\nKey Performance Metrics:")
        print_flush("-"*70)
        print_flush(f"  AUC-ROC:   {auc:.4f}")
        print_flush(f"  Precision: {precision:.4f}  (of predicted frauds, how many are actual frauds)")
        print_flush(f"  Recall:    {recall:.4f}  (of actual frauds, how many we catch)")
        print_flush(f"  F1-Score:  {f1:.4f}  (harmonic mean of precision & recall)")
        
        # Business Impact Metrics
        print_flush("\nBusiness Impact Metrics:")
        print_flush("-"*70)
        
        # True/False Positives/Negatives
        tn, fp, fn, tp = cm.ravel()
        
        # False Positive Rate (Type I Error)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print_flush(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        print_flush(f"    → {fp:,} non-frauds incorrectly flagged")
        
        # False Negative Rate (Type II Error)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print_flush(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
        print_flush(f"    → {fn:,} frauds missed by the model")
        
        # True Positive Rate (Sensitivity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        print_flush(f"  True Positive Rate:  {tpr:.4f} ({tpr*100:.2f}%)")
        print_flush(f"    → {tp:,} frauds correctly detected")
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print_flush(f"  Specificity:         {specificity:.4f} ({specificity*100:.2f}%)")
        print_flush(f"    → {tn:,} non-frauds correctly identified")
        
        # Save model
        self.save_model()
        
        # Plot results
        self.plot_results(y_test, y_pred_proba, X_test_scaled)
        
        # Print summary
        print_flush("\n" + "="*70)
        print_flush("✓ TRAINING COMPLETE!")
        print_flush("="*70)
        print_flush(f"\nModel Performance Summary:")
        print_flush(f"  Overall Accuracy:  {(tp + tn) / (tp + tn + fp + fn):.4f}")
        print_flush(f"  AUC-ROC:          {auc:.4f}")
        print_flush(f"  Precision:        {precision:.4f}")
        print_flush(f"  Recall:           {recall:.4f}")
        print_flush(f"  F1-Score:         {f1:.4f}")
        
        return {
            'model': self.model,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': {
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'fnr': fnr,
                'tpr': tpr,
                'specificity': specificity
            }
        }
    
    def plot_results(self, y_test, y_pred_proba, X_test):
        """Plot ROC curve, PR curve, and feature importance"""
        print_flush("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.4f})', linewidth=2, color='#2E86AB')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
        axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
        axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
        axes[0, 0].set_title('ROC Curve - Classical XGBoost', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, linewidth=2, color='#A23B72')
        axes[0, 1].set_xlabel('Recall', fontsize=11)
        axes[0, 1].set_ylabel('Precision', fontsize=11)
        axes[0, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=y_test.mean(), color='red', linestyle='--', 
                          label=f'Baseline ({y_test.mean():.3f})', alpha=0.5)
        axes[0, 1].legend(fontsize=10)
        
        # 3. Feature Importance (Top 20)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
            axes[1, 0].barh(range(len(feature_importance)), 
                           feature_importance['importance'],
                           color=colors)
            axes[1, 0].set_yticks(range(len(feature_importance)))
            axes[1, 0].set_yticklabels(feature_importance['feature'], fontsize=9)
            axes[1, 0].invert_yaxis()
            axes[1, 0].set_xlabel('Importance Score', fontsize=11)
            axes[1, 0].set_title('Top 20 Most Important Features', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Risk Score Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, 
                       label='Non-Fraud', density=True, color='#06A77D')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, 
                       label='Fraud', density=True, color='#D62246')
        axes[1, 1].axvline(0.5, color='black', linestyle='--', linewidth=1.5, 
                          label='Decision Threshold', alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Fraud Probability', fontsize=11)
        axes[1, 1].set_ylabel('Density', fontsize=11)
        axes[1, 1].set_title('Risk Score Distribution by True Label', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.model_dir / 'classical_model_evaluation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print_flush(f"✓ Saved visualization to {output_path}")
        
        # Save feature importance separately
        if hasattr(self.model, 'feature_importances_'):
            feature_importance_full = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_full.to_csv(
                self.model_dir / 'feature_importance.csv', 
                index=False
            )
            print_flush(f"✓ Saved feature importance to feature_importance.csv")
        
        plt.close()
    
    def save_model(self):
        """Save trained model and scaler"""
        print_flush(f"\nSaving model to {self.model_dir}...")
        
        joblib.dump(self.model, self.model_dir / 'xgboost_model.pkl')
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        joblib.dump(self.feature_names, self.model_dir / 'feature_names.pkl')
        
        # Save model metadata
        metadata = {
            'n_features': len(self.feature_names),
            'model_type': 'XGBoost',
            'random_state': self.random_state,
            'xgboost_version': xgb.__version__
        }
        joblib.dump(metadata, self.model_dir / 'metadata.pkl')
        
        print_flush(f"✓ Model saved successfully")
        print_flush(f"  - xgboost_model.pkl")
        print_flush(f"  - scaler.pkl")
        print_flush(f"  - feature_names.pkl")
        print_flush(f"  - metadata.pkl")
    
    def load_model(self):
        """Load saved model"""
        print_flush("Loading model from disk...")
        
        self.model = joblib.load(self.model_dir / 'xgboost_model.pkl')
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
        
        print_flush("✓ Model loaded successfully")
    
    def predict_risk_scores(self, X):
        """Predict fraud risk scores for new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        risk_scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return risk_scores

if __name__ == "__main__":
    print_flush("="*70)
    print_flush("CLASSICAL FRAUD DETECTION MODEL - XGBOOST")
    print_flush("="*70)
    print_flush(f"\nInitializing...")
    
    detector = ClassicalFraudDetector()
    
    # Load data
    X, y = detector.load_data()
    
    # Train model
    results = detector.train_model(X, y, use_resampling=True)
    
    if results:
        print_flush("\n" + "="*70)
        print_flush("✓ CLASSICAL MODEL TRAINING COMPLETE!")
        print_flush("="*70)
        print_flush(f"\nModel artifacts saved to: {detector.model_dir}")
        print_flush(f"\nFiles generated:")
        print_flush(f"  1. xgboost_model.pkl           - Trained model")
        print_flush(f"  2. scaler.pkl                  - Feature scaler")
        print_flush(f"  3. feature_names.pkl           - Feature list")
        print_flush(f"  4. feature_importance.csv      - Feature rankings")
        print_flush(f"  5. classical_model_evaluation.png - Performance plots")
        
        print_flush("\n" + "="*70)
        print_flush("Next steps:")
        print_flush("  1. Review the evaluation plots")
        print_flush("  2. Check feature_importance.csv for key fraud indicators")
        print_flush("  3. Run model comparison: python src/models/compare_models.py")
        print_flush("  4. Score the warehouse with risk scores")
        print_flush("="*70)
    else:
        print_flush("\n✗ Model training failed - check errors above")