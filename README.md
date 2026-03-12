# 🤖 AI Business Analyst Assistant

Ask any business question in plain English — get SQL, data, charts and AI-powered insights instantly.

## What it does

1. **Natural Language → SQL** — Type a business question, AI converts it to SQL automatically
2. **Anomaly Detection** — Automatically detects drops, spikes and outliers in your data
3. **AI Root Cause Analysis** — LLM explains WHY metrics changed and suggests actions
4. **Interactive Charts** — Auto-generates charts from query results
5. **CSV Export** — Download results for further analysis

## Example Questions

- "Why did revenue drop in March 2024?"
- "Which region has the highest revenue?"
- "Show monthly revenue trend for 2024"
- "Which product category performs best?"

## Tech Stack

- **Python** — Core language
- **SQLite** — Database
- **Pandas** — Data manipulation
- **Google Gemini API (via OpenRouter)** — LLM for NL→SQL and AI explanations
- **Streamlit** — Frontend UI
- **Plotly** — Interactive charts
- **scikit-learn** — Anomaly detection

## Setup

```bash
# Clone repo
git clone https://github.com/Sanjay123sam456/ai-business-analyst
cd ai-business-analyst

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY=your_key_here

# Run app
streamlit run frontend/app.py
```

## Project Structure

```
ai-business-analyst/
├── data/                    # Database and CSV files
├── backend/
│   ├── database.py          # SQLite connection and query runner
│   ├── sql_engine.py        # Natural language to SQL conversion
│   ├── anomaly_detector.py  # Statistical anomaly detection
│   └── ai_explainer.py      # LLM root cause analysis
├── frontend/
│   └── app.py               # Streamlit UI
├── utils/
│   └── data_loader.py       # CSV to SQLite loader
└── requirements.txt
```

## Live Demo

🔗 [Live App](#) — deployed on Render

## Built By

**Sanjay Kumar** — MCA Graduate, BIT Mesra  
[LinkedIn](https://linkedin.com/in/sanjay-kumar-ai) | [GitHub](https://github.com/Sanjay123sam456)
