import re

from backend.gemini_client import generate_text


def _extract_column_specs(schema: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for line in (schema or "").splitlines():
        if not line.lower().startswith("columns:"):
            continue
        parts = line.split(":", 1)[1]
        for item in parts.split(","):
            item = item.strip()
            match = re.match(r"([a-zA-Z_][\w]*)\s*\(([^)]*)\)", item)
            if not match:
                continue
            specs.append((match.group(1).strip(), match.group(2).strip().lower()))
    return specs


def _classify_columns(schema: str) -> dict[str, list[str]]:
    specs = _extract_column_specs(schema)
    cols = [name for name, _ in specs]
    numeric: list[str] = []
    date_like: list[str] = []
    text_like: list[str] = []

    for name, col_type in specs:
        lowered = name.lower()
        if any(k in lowered for k in ["date", "time", "month", "year", "day"]):
            date_like.append(name)
        if any(k in col_type for k in ["int", "real", "float", "double", "numeric", "decimal"]):
            numeric.append(name)
        elif col_type in ("text", ""):
            text_like.append(name)

    return {
        "all": cols,
        "numeric": numeric,
        "date_like": date_like,
        "text_like": text_like,
    }


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


def _extract_unanswerable(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.lower().startswith("unanswerable:"):
        return text.split(":", 1)[1].strip()
    return ""


def nl_to_sql(question: str, schema: str) -> str:
    profile = _classify_columns(schema)
    all_cols = profile["all"]
    numeric_cols = profile["numeric"]
    date_cols = profile["date_like"]

    if not all_cols:
        raise ValueError("No table schema found. Please upload or load a dataset first.")

    prompt = f"""You are an expert SQL assistant. Convert the user's business question into a valid SQLite SQL query.

Database Schema:
{schema}

Available Columns:
{", ".join(all_cols)}

Detected Numeric Columns:
{", ".join(numeric_cols) if numeric_cols else "None"}

Detected Date-like Columns:
{", ".join(date_cols) if date_cols else "None"}

Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- Always use lowercase column names
- Use ONLY columns listed in Available Columns
- Never invent or assume missing columns
- For date grouping/filtering use strftime('%Y-%m', <date_column>) only when a date-like column exists
- For ranking/comparison questions, include at least one numeric metric column when possible
- Avoid returning only id columns (e.g., customer_id only) for "top/best/highest" questions
- Prefer human-readable fields when present (e.g., customer_name over customer_id)
- If the question cannot be answered with current columns, return:
  UNANSWERABLE: <one short reason and which columns are missing>

User Question: {question}

SQL Query:"""

    raw = generate_text(prompt, max_output_tokens=300, temperature=0.1)
    cannot_answer_reason = _extract_unanswerable(raw)
    if cannot_answer_reason:
        raise ValueError(
            f"Cannot answer this question with current dataset schema: {cannot_answer_reason}"
        )

    sql = _extract_sql(raw)
    if not sql.lower().startswith(("select", "with")):
        preview_cols = ", ".join(all_cols[:10])
        if len(all_cols) > 10:
            preview_cols += ", ..."
        raise ValueError(
            "Could not generate valid SQL for this schema. "
            f"Try rephrasing with available columns: {preview_cols}"
        )
    return sql
