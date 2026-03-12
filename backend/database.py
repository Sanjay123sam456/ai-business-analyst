import sqlite3
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "business.db")

def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        raise ValueError(f"SQL Error: {e}")
    finally:
        conn.close()

def get_schema() -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema = []
    for (table,) in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_names = [f"{c[1]} ({c[2]})" for c in cols]
        schema.append(f"Table: {table}\nColumns: {', '.join(col_names)}")
    conn.close()
    return "\n\n".join(schema)
