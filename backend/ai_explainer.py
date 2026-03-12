from backend.gemini_client import generate_text

def explain_data(question: str, df_preview: str, anomaly_report: dict) -> str:
    anomaly_text = "\n".join(anomaly_report.get("anomalies", [])) or "No major anomalies detected."
    stats_text = str(anomaly_report.get("summary", {}))

    prompt = f"""You are an expert business analyst AI assistant. Analyze the data and answer the user's business question clearly.

User Question: {question}

Data Preview:
{df_preview}

Statistical Summary:
{stats_text}

Detected Anomalies:
{anomaly_text}

Instructions:
- Answer the question directly and clearly
- Explain what the data shows in plain business language
- Identify the root cause if there is a problem
- Give 2-3 specific actionable recommendations
- Be concise but insightful
- Format your response with clear sections

Your Analysis:"""

    return generate_text(prompt, max_output_tokens=600, temperature=0.2)
