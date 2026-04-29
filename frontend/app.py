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
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed.dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

    return df.shape, list(df.columns)


def remember_loaded_dataset(source: str, shape: tuple[int, int], cols: list[str]):
    st.session_state["active_data_source"] = source
    st.session_state["active_shape"] = shape
    st.session_state["active_columns"] = cols
    st.session_state["example_questions"] = build_example_questions(cols)


def show_loaded_dataset(shape: tuple[int, int], cols: list[str]):
    st.success(f"Loaded {shape[0]:,} rows x {shape[1]} columns")
    preview_cols = ", ".join(cols[:8])
    if len(cols) > 8:
        preview_cols += f", +{len(cols) - 8} more"
    st.caption(f"Columns ({len(cols)}): {preview_cols}")
    with st.expander("View all columns"):
        st.write(", ".join(cols))


def _humanize_label(col_name: str) -> str:
    return col_name.replace("_", " ").title()


def _is_date_like(col_name: str) -> bool:
    name = col_name.lower()
    return any(k in name for k in ["date", "time", "month", "year", "day"])


def _style_dark_plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#d7e2ee", size=12),
        title_font=dict(size=15, color="#f6d365"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(246,211,101,0.2)",
        tickfont=dict(color="#b7c4d3"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(246,211,101,0.2)",
        tickfont=dict(color="#b7c4d3"),
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
                color_discrete_sequence=["#f6d365"],
                title=f"Top Values in {_humanize_label(col)}",
            )
            _style_dark_plot(fig)
            return fig, None
        return None, "Result has no numeric metric to plot."

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    date_cols = [c for c in df.columns if _is_date_like(c)]
    color_seq = ["#f6d365"]

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
    :root {
        --canvas: #070b12;
        --panel: #0f1724;
        --panel-2: #151f2e;
        --ink: #f4f7fb;
        --muted: #93a4b8;
        --line: #273244;
        --teal: #2dd4bf;
        --teal-dark: #14b8a6;
        --coral: #fb7185;
        --amber: #f6d365;
    }

    .stApp {
        background:
            linear-gradient(90deg, rgba(45,212,191,0.055) 1px, transparent 1px),
            linear-gradient(180deg, rgba(246,211,101,0.04) 1px, transparent 1px),
            var(--canvas);
        background-size: 44px 44px;
        color: var(--ink);
    }
    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    section[data-testid="stSidebar"] {
        background: #0b1018;
        border-right: 1px solid var(--line);
        box-shadow: 14px 0 36px rgba(0,0,0,0.28);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink);
    }
    section[data-testid="stSidebar"] .stAlert {
        border-radius: 4px;
        border: 1px solid rgba(45,212,191,0.26);
        background: rgba(20,184,166,0.12);
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--line);
    }
    .main-header {
        background:
            linear-gradient(135deg, rgba(15,23,36,0.95), rgba(10,15,24,0.96));
        border: 1px solid rgba(45,212,191,0.24);
        border-top: 4px solid var(--amber);
        padding: 0;
        border-radius: 10px;
        color: var(--ink);
        margin-bottom: 18px;
        overflow: hidden;
        box-shadow: 0 24px 70px rgba(0,0,0,0.38);
    }
    .console-bar {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 18px;
        background: rgba(255,255,255,0.035);
        border-bottom: 1px solid rgba(148,163,184,0.14);
        font-family: "Cascadia Mono", Consolas, monospace;
        font-size: 0.78rem;
        color: var(--muted);
    }
    .console-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 7px;
        background: var(--teal);
        box-shadow: 0 0 14px rgba(45,212,191,0.8);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.7fr);
        gap: 28px;
        padding: 28px;
        align-items: stretch;
    }
    .main-header .eyebrow {
        color: var(--amber);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.05;
        color: var(--ink);
        letter-spacing: 0;
        max-width: 740px;
    }
    .main-header p {
        margin: 12px 0 0;
        color: var(--muted);
        font-size: 1rem;
        max-width: 780px;
    }
    .hud-panel {
        border: 1px solid rgba(246,211,101,0.26);
        background: rgba(246,211,101,0.07);
        border-radius: 8px;
        padding: 16px;
        min-height: 100%;
    }
    .hud-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        padding: 11px 0;
        border-bottom: 1px solid rgba(246,211,101,0.12);
    }
    .hud-row:last-child {
        border-bottom: none;
    }
    .hud-label {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .hud-value {
        color: var(--amber);
        font-family: "Cascadia Mono", Consolas, monospace;
        font-weight: 800;
        white-space: nowrap;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.045);
        color: var(--ink);
        border: 1px solid rgba(148,163,184,0.35);
        border-radius: 4px;
        min-height: 50px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        font-family: "Cascadia Mono", Consolas, monospace;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 3px rgba(45,212,191,0.16);
    }
    label, .stMarkdown, .stCaption, p, span, div {
        color: inherit;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }

    .sql-box {
        background: #050a10;
        color: #9af5de;
        padding: 13px 16px;
        border-radius: 4px;
        border: 1px solid rgba(45,212,191,0.2);
        border-left: 4px solid var(--teal);
        font-family: "Cascadia Mono", Consolas, monospace;
        font-size: 0.9rem;
        margin: 8px 0 14px;
    }
    .insight-box {
        background: rgba(15,23,36,0.92);
        border: 1px solid rgba(251,113,133,0.28);
        border-left: 4px solid var(--coral);
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        color: var(--ink);
        line-height: 1.65;
        white-space: pre-wrap;
        max-height: 340px;
        overflow-y: auto;
        font-size: 1rem;
        font-family: Inter, "Segoe UI", "Helvetica Neue", sans-serif;
        word-break: break-word;
        box-shadow: 0 18px 34px rgba(0,0,0,0.28);
    }
    .anomaly-box {
        background: rgba(246,211,101,0.09);
        border-left: 4px solid var(--amber);
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #fff4bf;
    }
    .viz-box {
        background: rgba(15,23,36,0.9);
        border: 1px solid rgba(45,212,191,0.22);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        padding: 14px 14px 6px 14px;
        margin: 12px 0 14px;
        box-shadow: 0 18px 34px rgba(0,0,0,0.26);
    }
    .viz-title {
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .section-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: 6px 0 10px;
        color: var(--amber);
        font-family: "Cascadia Mono", Consolas, monospace;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-label::before {
        content: "";
        width: 28px;
        height: 1px;
        background: var(--amber);
        display: inline-block;
    }
    .stButton > button {
        background: rgba(45,212,191,0.09);
        color: #dffcf7;
        border: 1px solid rgba(45,212,191,0.38);
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        min-height: 44px;
        box-shadow: none;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: rgba(246,211,101,0.16);
        border-color: rgba(246,211,101,0.7);
        color: white;
        transform: translateY(-1px);
    }
    .stButton > button:focus {
        color: white;
        box-shadow: 0 0 0 3px rgba(45,212,191,0.18);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 18px 34px rgba(0,0,0,0.24);
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
        gap: 0.75rem;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.25rem;
    }
    .stDownloadButton > button {
        background: rgba(251,113,133,0.11);
        border-color: rgba(251,113,133,0.45);
    }
    @media (max-width: 780px) {
        .hero-grid {
            grid-template-columns: 1fr;
            padding: 20px;
        }
        .main-header h1 {
            font-size: 1.7rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <div class="console-bar">
        <span><span class="console-dot"></span>analyst-console / sample-dataset online</span>
        <span>SQL + CHARTS + INSIGHTS</span>
    </div>
    <div class="hero-grid">
        <div>
            <div class="eyebrow">Decision Desk</div>
            <h1>Revenue Intelligence Console</h1>
            <p>Ask a business question, inspect the generated SQL, and review chart-ready findings without forcing recruiters to upload a file.</p>
        </div>
        <div class="hud-panel">
            <div class="hud-row"><span class="hud-label">Dataset</span><span class="hud-value">sample_sales</span></div>
            <div class="hud-row"><span class="hud-label">Rows</span><span class="hud-value">2,000</span></div>
            <div class="hud-row"><span class="hud-label">Mode</span><span class="hud-value">auto-load</span></div>
        </div>
    </div>
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
    st.header("Control Room")
    if os.getenv("OPENROUTER_API_KEY"):
        st.success("OpenRouter key is loaded from server environment.")
    else:
        st.error("Server is missing OPENROUTER_API_KEY.")
        st.caption("Set OPENROUTER_API_KEY in Render Environment Variables.")

    st.divider()
    st.header("Data Feed")
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])

    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(UPLOADED_CSV_PATH, "wb") as file_obj:
                    file_obj.write(uploaded_file.getvalue())

                shape, cols = load_uploaded_csv_to_db(uploaded_file, "sales")
                remember_loaded_dataset("upload", shape, cols)
                show_loaded_dataset(shape, cols)
            except Exception as err:
                st.error(f"Could not load CSV: {err}")
                st.info("Tip: Save your file as UTF-8 CSV, then upload again.")
    else:
        sample_is_active = st.session_state.get("active_data_source") == "sample"
        should_load_sample = not sample_is_active

        if should_load_sample:
            with st.spinner("Loading sample sales data..."):
                shape, cols = load_sample_data()
                remember_loaded_dataset("sample", shape, cols)

        if st.button("Reload Sample Data"):
            with st.spinner("Reloading sample sales data..."):
                shape, cols = load_sample_data()
                remember_loaded_dataset("sample", shape, cols)

        if st.session_state.get("active_data_source") == "sample":
            show_loaded_dataset(
                st.session_state["active_shape"],
                st.session_state["active_columns"],
            )

st.markdown('<div class="section-label">Command Input</div>', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Question",
        key="question_input",
        placeholder="Type a question, e.g. Which category has the highest total sales?",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze", use_container_width=True)

st.markdown('<div class="section-label">Quick Prompts</div>', unsafe_allow_html=True)
if st.session_state.get("active_columns"):
    st.caption("Generated from the active dataset schema.")
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

        st.markdown('<div class="section-label">SQL Trace</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)

        with st.spinner("Running query on database..."):
            df = run_query(sql)

        if df.empty:
            st.warning("Query returned no results. Try rephrasing your question.")
            st.stop()

        st.markdown(f'<div class="section-label">Data Return - {len(df):,} rows</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-label">Analyst Notes</div>', unsafe_allow_html=True)
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

