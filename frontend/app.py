import io
import html
import os
import re
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_explainer import explain_data
from backend.anomaly_detector import detect_anomalies
from backend.database import DB_PATH, get_schema, run_query
from backend.sql_engine import nl_to_sql
from utils.data_loader import load_sample_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADED_CSV_PATH = os.path.join(DATA_DIR, "uploaded.csv")

# Load local environment variables for development (Render uses dashboard env vars).
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_EXAMPLES = [
    "Show top 10 rows from this dataset.",
    "How many total records are in this dataset?",
    "What are the main columns and data quality issues?",
    "Which fields have the highest missing values?",
    "Give a quick summary of this dataset.",
]


def _humanize(col_name: str) -> str:
    return col_name.replace("_", " ").strip()


def _pick_first(columns: list[str], keywords: list[str]) -> str:
    for col in columns:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            return col
    return ""


def _is_id_col(col_name: str) -> bool:
    name = col_name.lower()
    return name == "id" or name.endswith("_id") or name.startswith("id_")


def build_example_questions(columns: list[str]) -> list[str]:
    if not columns:
        return DEFAULT_EXAMPLES.copy()

    cols = [c.strip().lower() for c in columns if c and c.strip()]

    date_col = _pick_first(cols, ["date", "time", "month", "year", "day"])
    geo_col = _pick_first(cols, ["region", "state", "city", "country", "market", "zone"])
    category_col = _pick_first(cols, ["category", "sub_category", "segment", "type", "class", "department"])
    customer_col = _pick_first(cols, ["customer", "client", "user"])
    product_col = _pick_first(cols, ["product", "item", "sku", "brand"])

    metric_priority = [
        "sales",
        "revenue",
        "profit",
        "amount",
        "price",
        "cost",
        "quantity",
        "qty",
        "discount",
        "score",
        "value",
    ]
    metric_col = _pick_first(cols, metric_priority)

    if not metric_col:
        for col in cols:
            if not _is_id_col(col) and col != date_col:
                metric_col = col
                break

    dimension_col = category_col or geo_col or customer_col or product_col
    if not dimension_col:
        for col in cols:
            if col not in {metric_col, date_col} and not _is_id_col(col):
                dimension_col = col
                break

    questions: list[str] = []

    def add(question: str):
        if question and question not in questions:
            questions.append(question)

    if metric_col and dimension_col:
        add(f"Which {_humanize(dimension_col)} has the highest total {_humanize(metric_col)}?")
        add(f"Compare total {_humanize(metric_col)} by {_humanize(dimension_col)}.")
    if date_col and metric_col:
        add(f"Show monthly trend of {_humanize(metric_col)} using {_humanize(date_col)}.")
    if customer_col and metric_col:
        add(f"Who are the top 10 {_humanize(customer_col)} by {_humanize(metric_col)}?")
    if product_col and metric_col:
        add(f"Which {_humanize(product_col)} contributes the most {_humanize(metric_col)}?")
    if metric_col:
        add(f"What is the average {_humanize(metric_col)}?")

    add("How many total records are in this dataset?")

    if len(questions) < 5:
        for q in DEFAULT_EXAMPLES:
            add(q)

    return questions[:5]


def load_uploaded_csv_to_db(uploaded_file, table_name: str = "sales"):
    raw = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError as err:
            last_error = err

    if df is None:
        raise ValueError(
            "Could not decode CSV. Tried utf-8, utf-8-sig, cp1252, latin1."
        ) from last_error

    df.columns = [
        c.strip().replace("\xa0", " ").lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    for col in df.columns:
        if any(k in col for k in ["date", "time", "month", "year", "day"]):
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed.dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return df.shape, list(df.columns)


def _humanize_label(col_name: str) -> str:
    return col_name.replace("_", " ").title()


def _is_date_like(col_name: str) -> bool:
    name = col_name.lower()
    return any(k in name for k in ["date", "time", "month", "year", "day"])


def _style_dark_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#EAF2FF", size=12),
        title_font=dict(size=15, color="#EAF2FF"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(
        gridcolor="rgba(180,200,220,0.12)",
        zerolinecolor="rgba(180,200,220,0.18)",
        tickfont=dict(color="#DDE7F5"),
    )
    fig.update_yaxes(
        gridcolor="rgba(180,200,220,0.12)",
        zerolinecolor="rgba(180,200,220,0.18)",
        tickfont=dict(color="#DDE7F5"),
    )


def build_visualization(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
        if len(non_numeric_cols) == 1 and len(df) > 1:
            col = non_numeric_cols[0]
            counts = (
                df[col]
                .astype(str)
                .value_counts(dropna=False)
                .head(20)
                .rename_axis(col)
                .reset_index(name="count")
            )
            fig = px.bar(
                counts,
                x=col,
                y="count",
                color_discrete_sequence=["#37A2FF"],
                title=f"Top Values in {_humanize_label(col)}",
            )
            _style_dark_plot(fig)
            return fig, None
        return None, "Result has no numeric metric to plot."

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    date_cols = [c for c in df.columns if _is_date_like(c)]
    color_seq = ["#37A2FF"]

    # Summary query: single row with multiple numeric metrics.
    if len(df) == 1 and len(numeric_cols) >= 2:
        metric_frame = pd.DataFrame(
            {
                "metric": [_humanize_label(c) for c in numeric_cols],
                "value": [float(df.iloc[0][c]) for c in numeric_cols],
            }
        )
        fig = px.bar(
            metric_frame,
            x="metric",
            y="value",
            color_discrete_sequence=color_seq,
            title="Summary Metrics",
        )
        _style_dark_plot(fig)
        return fig, None

    # Single scalar doesn't need a chart.
    if len(df) == 1 and len(numeric_cols) == 1:
        return None, "Query returned one numeric value, so chart is skipped."

    # Time-series style chart when date-like column exists.
    if date_cols and numeric_cols:
        x_col = date_cols[0]
        y_col = numeric_cols[0]
        temp = df.copy()
        parsed = pd.to_datetime(temp[x_col], errors="coerce")
        if parsed.notna().sum() > 0:
            temp = temp.assign(_parsed_date=parsed).sort_values("_parsed_date")
            fig = px.line(
                temp,
                x=x_col,
                y=y_col,
                color_discrete_sequence=color_seq,
                title=f"{_humanize_label(y_col)} Trend",
            )
            _style_dark_plot(fig)
            return fig, None

    # Category vs metric chart.
    if non_numeric_cols and numeric_cols:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]
        temp = (
            df.groupby(x_col, as_index=False)[y_col]
            .sum()
            .sort_values(y_col, ascending=False)
            .head(20)
        )
        fig = px.bar(
            temp,
            x=x_col,
            y=y_col,
            color_discrete_sequence=color_seq,
            title=f"{_humanize_label(y_col)} by {_humanize_label(x_col)}",
        )
        _style_dark_plot(fig)
        return fig, None

    # Fallback: first numeric over row index.
    fallback = df.reset_index().rename(columns={"index": "row_index"})
    y_col = numeric_cols[0]
    fig = px.line(
        fallback,
        x="row_index",
        y=y_col,
        color_discrete_sequence=color_seq,
        title=f"{_humanize_label(y_col)} by Row",
    )
    _style_dark_plot(fig)
    return fig, None


def _query_flag(flag_name: str) -> bool:
    params = st.query_params
    if flag_name not in params:
        return False

    raw_value = params.get(flag_name, "1")
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else "1"

    return str(raw_value).strip().lower() in {"", "1", "true", "yes", "y", "on"}


def _sqlite_ready() -> tuple[bool, str]:
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _handle_health_checks():
    if _query_flag("healthz"):
        st.json({
            "status": "ok",
            "mode": "liveness",
            "app": "alive",
        })
        st.stop()

    if _query_flag("readyz"):
        has_api_key = bool(os.getenv("OPENROUTER_API_KEY"))
        db_ok, db_error = _sqlite_ready()

        checks = {
            "openrouter_api_key": "ok" if has_api_key else "missing",
            "sqlite": "ok" if db_ok else f"error: {db_error}",
        }
        overall_ok = has_api_key and db_ok

        st.json({
            "status": "ok" if overall_ok else "degraded",
            "mode": "readiness",
            "checks": checks,
        })
        st.stop()


st.set_page_config(
    page_title="AI Business Analyst Assistant",
    page_icon="AI",
    layout="wide",
)

_handle_health_checks()

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2196F3);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 4px 0 0; opacity: 0.9; font-size: 1rem; }

    .sql-box {
        background: #1e1e1e;
        color: #00ff88;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 8px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #132217, #1b2d21);
        border-left: 4px solid #4CAF50;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        color: #e8f5e9;
        line-height: 1.65;
        white-space: pre-wrap;
        max-height: 340px;
        overflow-y: auto;
        font-size: 1rem;
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
        word-break: break-word;
    }
    .anomaly-box {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #3e2723;
    }
    .viz-box {
        background: linear-gradient(135deg, #101a2a, #0c1624);
        border: 1px solid #2f5b87;
        border-left: 4px solid #37A2FF;
        border-radius: 10px;
        padding: 14px 14px 6px 14px;
        margin: 12px 0 14px;
    }
    .viz-title {
        color: #dbeaff;
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2196F3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>AI Business Analyst Assistant</h1>
    <p>Ask business questions in plain English and get SQL, data, charts, and AI insights.</p>
</div>
""",
    unsafe_allow_html=True,
)

if "example_questions" not in st.session_state:
    st.session_state["example_questions"] = DEFAULT_EXAMPLES.copy()
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "pending_question" in st.session_state:
    st.session_state["question_input"] = st.session_state.pop("pending_question")

with st.sidebar:
    st.header("Configuration")
    if os.getenv("OPENROUTER_API_KEY"):
        st.success("OpenRouter key is loaded from server environment.")
    else:
        st.error("Server is missing OPENROUTER_API_KEY.")
        st.caption("Set OPENROUTER_API_KEY in Render Environment Variables.")

    st.divider()
    st.header("Data Source")
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])

    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(UPLOADED_CSV_PATH, "wb") as file_obj:
                    file_obj.write(uploaded_file.getvalue())

                shape, cols = load_uploaded_csv_to_db(uploaded_file, "sales")
                st.session_state["active_columns"] = cols
                st.session_state["example_questions"] = build_example_questions(cols)

                st.success(f"Loaded {shape[0]:,} rows x {shape[1]} columns")
                preview_cols = ", ".join(cols[:8])
                if len(cols) > 8:
                    preview_cols += f", +{len(cols) - 8} more"
                st.caption(f"Columns ({len(cols)}): {preview_cols}")
                with st.expander("View all columns"):
                    st.write(", ".join(cols))
            except Exception as err:
                st.error(f"Could not load CSV: {err}")
                st.info("Tip: Save your file as UTF-8 CSV, then upload again.")
    else:
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample sales data..."):
                shape, cols = load_sample_data()
                st.session_state["active_columns"] = cols
                st.session_state["example_questions"] = build_example_questions(cols)
                st.success(f"Loaded {shape[0]:,} rows x {shape[1]} columns")
                preview_cols = ", ".join(cols[:8])
                if len(cols) > 8:
                    preview_cols += f", +{len(cols) - 8} more"
                st.caption(f"Columns ({len(cols)}): {preview_cols}")
                with st.expander("View all columns"):
                    st.write(", ".join(cols))

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Ask a business question",
        key="question_input",
        placeholder="e.g. Which category has the highest total sales?",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze", use_container_width=True)

st.markdown("**Suggested Questions**")
if st.session_state.get("active_columns"):
    st.caption("Auto-generated from your uploaded dataset.")
else:
    st.caption("Load data to get column-specific suggestions.")

examples = st.session_state.get("example_questions", DEFAULT_EXAMPLES)
example_cols = st.columns(2)
for idx, ex in enumerate(examples):
    with example_cols[idx % 2]:
        if st.button(ex, key=f"main_example_{idx}", use_container_width=True):
            st.session_state["pending_question"] = ex
            st.rerun()

if analyze and question:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Server is missing OPENROUTER_API_KEY. Please set it in Render environment variables.")
        st.stop()

    try:
        schema = get_schema()
        if not schema:
            st.warning("No data loaded. Please load sample data or upload a CSV first.")
            st.stop()

        with st.spinner("Converting question to SQL..."):
            sql = nl_to_sql(question, schema)

        st.markdown("**Generated SQL Query:**")
        st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)

        with st.spinner("Running query on database..."):
            df = run_query(sql)

        if df.empty:
            st.warning("Query returned no results. Try rephrasing your question.")
            st.stop()

        st.markdown(f"**Query Results** - {len(df):,} rows returned")
        st.dataframe(df, use_container_width=True)

        fig, viz_note = build_visualization(df)
        st.markdown('<div class="viz-box"><div class="viz-title">Data Visualization</div>', unsafe_allow_html=True)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Visualization skipped: {viz_note or 'No plottable output.'}")
        st.markdown("</div>", unsafe_allow_html=True)

        anomaly_report = detect_anomalies(df)
        if anomaly_report.get("anomalies"):
            st.markdown("**Anomalies Detected:**")
            for item in anomaly_report["anomalies"]:
                st.markdown(f'<div class="anomaly-box">{item}</div>', unsafe_allow_html=True)

        with st.spinner("Generating AI insights..."):
            df_preview = df.head(20).to_string()
            explanation = explain_data(question, df_preview, anomaly_report)
            explanation = re.sub(r"\n{3,}", "\n\n", explanation).strip()
            explanation = re.sub(r"([,.;:])(?=\S)", r"\1 ", explanation)
            explanation = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", explanation)

        st.markdown("**AI Business Analysis:**")
        st.markdown(
            f'<div class="insight-box">{html.escape(explanation)}</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        csv = df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, "results.csv", "text/csv")

    except Exception as err:
        message = str(err)
        if "Cannot answer this question with current dataset schema" in message:
            st.warning(message)
            active_cols = st.session_state.get("active_columns", [])
            if active_cols:
                preview_cols = ", ".join(active_cols[:12])
                if len(active_cols) > 12:
                    preview_cols += ", ..."
                st.caption(f"Available columns: {preview_cols}")
            st.caption("Try asking using available columns or load a different dataset.")
        else:
            st.error(f"Error: {message}")
            st.caption("Try rephrasing your question or check your API key.")

elif not question and analyze:
    st.warning("Please enter a business question first.")

st.divider()
st.markdown(
    """
<div style='text-align:center; color:#888; font-size:0.8rem;'>
    Built by Sanjay Kumar | Python - SQL - LLM - Streamlit - Plotly
</div>
""",
    unsafe_allow_html=True,
)

