import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame) -> dict:
    """Detect anomalies in query results"""
    anomalies = []
    summary = {}

    if df.empty:
        return {"anomalies": [], "summary": "No data found."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 2:
            continue

        mean = values.mean()
        std = values.std()
        min_val = values.min()
        max_val = values.max()

        # Detect significant drops or spikes
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            outliers = values[z_scores > 2]
            if len(outliers) > 0:
                anomalies.append(f"Column '{col}' has {len(outliers)} outlier(s). Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean:.2f}")

        # Detect percentage change if there's a time-ordered column
        if len(values) > 1:
            pct_changes = values.pct_change().dropna()
            big_drops = pct_changes[pct_changes < -0.3]
            big_spikes = pct_changes[pct_changes > 0.3]

            if len(big_drops) > 0:
                anomalies.append(f"Column '{col}' shows drops > 30% at {len(big_drops)} point(s)")
            if len(big_spikes) > 0:
                anomalies.append(f"Column '{col}' shows spikes > 30% at {len(big_spikes)} point(s)")

        summary[col] = {
            "mean": round(mean, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "std": round(std, 2)
        }

    return {
        "anomalies": anomalies,
        "summary": summary,
        "row_count": len(df),
        "columns": list(df.columns)
    }
