import re

from backend.gemini_client import generate_text


def _extract_sql(raw_text: str) -> str:
    text = (raw_text or "").replace("```sql", "").replace("```", "").strip()

    # Remove common labels/prefixes models add before the query.
    for prefix in ("sql query:", "query:", "sql:", "sqlite:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # Keep only the SQL statement portion if extra text appears before it.
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():].strip()

    # If model returns multiple statements, keep the first one.
    if ";" in text:
        first, _sep, _rest = text.partition(";")
        text = first.strip() + ";"

    return text

def nl_to_sql(question: str, schema: str) -> str:
    prompt = f"""You are an expert SQL assistant. Convert the user's business question into a valid SQLite SQL query.

Database Schema:
{schema}

Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- Always use lowercase column names
- For date filtering use strftime('%Y-%m', date) for year-month
- Keep queries simple and readable

User Question: {question}

SQL Query:"""

    raw = generate_text(prompt, max_output_tokens=300, temperature=0.1)
    sql = _extract_sql(raw)
    if not sql.lower().startswith(("select", "with")):
        raise ValueError(f"Generated SQL is invalid: {raw}")
    return sql
