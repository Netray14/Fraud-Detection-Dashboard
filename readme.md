# PMJAY Fraud Detection BI

## Project Overview

**Hybrid Quantum Fraud Detection BI: A Business Intelligence Framework for Prioritizing Suspect Health Insurance Claims using Classical (XGBoost) and Quantum (VQC) Risk Scoring over Raw Government Claims Data**

This project develops a comprehensive Business Intelligence (BI) solution for detecting and prioritizing fraudulent claims in the Ayushman Bharat – Pradhan Mantri Jan Arogya Yojana (AB-PMJAY) health insurance scheme.

## Key Features

- 🏥 **Data Integration**: Consolidates PMJAY claims data from multiple government sources (data.gov.in, Parliamentary replies, CAG audits)
- 🔍 **Fraud Detection**: Twin-track risk scoring using:
  - **Classical ML**: XGBoost classifier for baseline fraud detection
  - **Quantum ML**: Variational Quantum Classifier (VQC) for quantum-inspired scoring
- 📊 **Business Intelligence**: Star/Snowflake data warehouse with Power BI dashboards
- 🎯 **Hospital Vulnerability Index**: Composite fraud-risk scoring for hospital prioritization

## Business Value

- **Target Users**: NHA's National Anti-Fraud Unit (NAFU), State Anti-Fraud Units (SAFUs), State Health Agencies
- **Impact**: Enables proactive, data-driven fraud investigation prioritization across 30,000+ empanelled hospitals
- **Scale**: Processes 50,000+ synthetic claim records representing real PMJAY data patterns

## Project Structure
pmjay-fraud-detection-bi/
├── data/
│ ├── raw/ # Raw data extracts
│ ├── cleaned/ # Cleaned/prepared data
│ ├── warehouse/ # SQLite data warehouse
│ └── powerbi/ # Power BI exports
├── src/
│ ├── config.py # Central configuration
│ ├── etl/
│ │ ├── data_cleaning.py # Data cleaning pipeline
│ │ └── build_warehouse.py # Warehouse schema & loading
│ ├── models/
│ │ ├── feature_engineering.py # Feature creation
│ │ ├── classical_model.py # XGBoost classifier
│ │ ├── quantum_model.py # Quantum VQC model
│ │ └── score_warehouse.py # Risk scoring
│ └── utils/
│ ├── generate_clean_mock_data.py
│ ├── export_powerbi_with_scores.py
│ └── query_warehouse.py
├── dashboards/
│ └── PMJAY_Dashboard.pbix # Power BI dashboard
├── docs/
│ └── README.md
├── notebooks/
│ └── (Jupyter exploratory analysis)
├── requirements.txt
├── check_database.py
├── setup_and_run.py
└── README.md


## Quick Start

### Prerequisites
- Python 3.8+
- Git
- Power BI Desktop
- SQLite3

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/pmjay-fraud-detection-bi.git
cd pmjay-fraud-detection-bi