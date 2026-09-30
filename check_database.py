# check_database.py
"""
Check actual database tables and structure
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from config import DB_URL, DB_PATH

def check_database():
    print("=" * 70)
    print("DATABASE STRUCTURE CHECK")
    print("=" * 70)
    print(f"Database: {DB_PATH}")

    if not Path(DB_PATH).exists():
        print("\n✗ Database file not found. Run build_warehouse.py first.")
        return

    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\nFound {len(tables)} tables:")
    print("-" * 70)

    with engine.connect() as conn:
        for table in tables:
            print(f"\nTable: {table}")
            print("  Columns:")
            for col in inspector.get_columns(table):
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"    - {col['name']:30s} {str(col['type']):15s} {nullable}")

            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  Records: {count:,}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_database()