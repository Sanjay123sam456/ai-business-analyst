import pandas as pd
import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "business.db")

def _read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        "Could not decode CSV. Tried utf-8, utf-8-sig, cp1252, latin1. "
        "Please re-save your file as UTF-8 CSV and upload again."
    ) from last_error


def _normalize_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        c.strip().replace("\xa0", " ").lower().replace(" ", "_").replace("-", "_")
        for c in cleaned.columns
    ]

    # Convert likely date columns to ISO format so SQLite date functions work.
    for col in cleaned.columns:
        if any(k in col for k in ["date", "time", "month", "year", "day"]):
            parsed = pd.to_datetime(cleaned[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:
                cleaned[col] = parsed.dt.strftime("%Y-%m-%d")

    return cleaned

def load_csv_to_db(csv_path: str, table_name: str = "sales"):
    df = _normalize_for_sqlite(_read_csv_with_fallback(csv_path))
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return df.shape, list(df.columns)

def load_sample_data():
    """Generate sample superstore-like sales data"""
    import numpy as np
    from datetime import datetime, timedelta
    import random

    random.seed(42)
    np.random.seed(42)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Food", "Furniture"]
    products = {
        "Electronics": ["Laptop", "Phone", "Tablet", "TV"],
        "Clothing": ["Shirt", "Pants", "Shoes", "Jacket"],
        "Food": ["Groceries", "Snacks", "Beverages", "Dairy"],
        "Furniture": ["Chair", "Table", "Sofa", "Bed"]
    }

    rows = []
    start_date = datetime(2023, 1, 1)

    for i in range(2000):
        date = start_date + timedelta(days=random.randint(0, 730))
        region = random.choice(regions)
        category = random.choice(categories)
        product = random.choice(products[category])

        # Simulate March 2024 revenue drop
        base_revenue = random.uniform(100, 2000)
        if date.year == 2024 and date.month == 3:
            base_revenue *= 0.4  # 60% drop in March 2024

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "month": date.month,
            "year": date.year,
            "region": region,
            "category": category,
            "product": product,
            "revenue": round(base_revenue, 2),
            "quantity": random.randint(1, 50),
            "customers": random.randint(1, 100),
            "conversion_rate": round(random.uniform(0.1, 0.9), 2)
        })

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    sample_csv_path = os.path.join(DATA_DIR, "sample_sales.csv")
    df.to_csv(sample_csv_path, index=False)
    load_csv_to_db(sample_csv_path, "sales")
    return df.shape, list(df.columns)
