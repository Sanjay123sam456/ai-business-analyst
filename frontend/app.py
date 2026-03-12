import io
import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_explainer import explain_data
from backend.anomaly_detector import detect_anomalies
from backend.database import DB_PATH, get_schema, run_query
from backend.sql_engine import nl_to_sql
from utils.data_loader import load_sample_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADED_CSV_PATH = os.path.join(DATA_DIR, "uploaded.csv")

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

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return df.shape, list(df.columns)


st.set_page_config(
    page_title="AI Business Analyst Assistant",
    page_icon="AI",
    layout="wide",
)

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
    }
    .anomaly-box {
        background: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #3e2723;
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
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

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
        st.error("Please enter your OpenRouter API Key in the sidebar.")
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

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(df.columns) >= 2 and numeric_cols:
            st.markdown("**Data Visualization:**")
            x_col = df.columns[0]
            y_col = numeric_cols[0]
            if len(df) <= 50:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    color_discrete_sequence=["#2196F3"],
                    title=f"{y_col} by {x_col}",
                )
            else:
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    color_discrete_sequence=["#2196F3"],
                    title=f"{y_col} trend",
                )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Arial"),
                title_font_size=14,
            )
            st.plotly_chart(fig, use_container_width=True)

        anomaly_report = detect_anomalies(df)
        if anomaly_report.get("anomalies"):
            st.markdown("**Anomalies Detected:**")
            for item in anomaly_report["anomalies"]:
                st.markdown(f'<div class="anomaly-box">{item}</div>', unsafe_allow_html=True)

        with st.spinner("Generating AI insights..."):
            df_preview = df.head(20).to_string()
            explanation = explain_data(question, df_preview, anomaly_report)

        st.markdown("**AI Business Analysis:**")
        st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)

        st.divider()
        csv = df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, "results.csv", "text/csv")

    except Exception as err:
        st.error(f"Error: {err}")
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
