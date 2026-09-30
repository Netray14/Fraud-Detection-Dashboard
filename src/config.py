# src/config.py
"""
Central configuration for all paths
"""
from pathlib import Path

# Get the project root directory (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
CLEANED_DATA_DIR = PROJECT_ROOT / 'data' / 'cleaned'
WAREHOUSE_DIR = PROJECT_ROOT / 'data' / 'warehouse'

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = WAREHOUSE_DIR / 'pmjay_warehouse.db'
DB_URL = f'sqlite:///{DB_PATH}'

# Print paths for debugging
if __name__ == "__main__":
    print("Project Configuration:")
    print(f"  PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"  RAW_DATA_DIR:     {RAW_DATA_DIR}")
    print(f"  CLEANED_DATA_DIR: {CLEANED_DATA_DIR}")
    print(f"  WAREHOUSE_DIR:    {WAREHOUSE_DIR}")
    print(f"  DB_PATH:          {DB_PATH}")