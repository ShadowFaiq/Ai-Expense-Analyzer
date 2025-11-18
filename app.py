import os
import json
import sqlite3
from datetime import datetime, date
from typing import Optional, Dict, Any, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import streamlit as st

# --------------------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="AI Expense Analyzer", page_icon="💸", layout="wide")
st.title("💸 AI Expense Analyzer — Enhanced Version")

# --------------------------------------------------------------
# DATABASE LAYER
# --------------------------------------------------------------
DB_FILE = "expenses.db"

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            category TEXT,
            description TEXT,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_exp_date ON expenses(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_exp_cat ON expenses(category)")
    conn.commit()
    conn.close()


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM expenses", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])  # date-only field
    return df


def add_expense(amount: float, category: str, description: str, dt: date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses (amount, category, description, date, created_at) VALUES (?,?,?,?,?)",
        (amount, category, description, dt.isoformat(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    load_data.clear()  # refresh cached data

# --------------------------------------------------------------
# AGENT ROUTER CLIENT (IMPROVED)
# --------------------------------------------------------------
class AgentRouterClient:
    def __init__(self):
        self.endpoint = os.getenv("AGENT_ROUTER_ENDPOINT")
        self.api_key = os.getenv("AGENT_ROUTER_API_KEY")
        self.session = requests.Session()

        # retries on 429 / 500+ errors
        retry_cfg = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def ready(self):
        return bool(self.endpoint and self.api_key)

    def call(self, prompt: str) -> str:
        if not self.ready():
            return "LLM Not Configured"  # graceful fallback

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"prompt": prompt, "context": {}}

        try:
            r = self.session.post(self.endpoint, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            return f"Error contacting Agent Router: {str(e)}"

        try:
            data = r.json()
        except:
            return r.text

        # flexible key extraction
        for key in ["text", "output", "response", "result"]:
            if key in data:
                return str(data[key])

        return str(data)


router = AgentRouterClient()

# --------------------------------------------------------------
# PROMPT HELPERS
# --------------------------------------------------------------
def prompt_category(desc: str) -> str:
    return f"""
You classify an expense into one category.
Categories: food, utilities, shopping, transport, education, health, entertainment, misc.
Output only one category.

Expense: {desc}
    """.strip()


def prompt_pattern_analysis(df: pd.DataFrame) -> str:
    sample = df.tail(10).to_dict(orient="records")
    return f"""
Analyze these expenses and identify spending patterns, risky behaviors, and budgeting recommendations.
Return detailed bullet points.
    
Expenses Data JSON:
{json.dumps(sample, indent=2)}
    """.strip()

# --------------------------------------------------------------
# PARSERS
# --------------------------------------------------------------
def parse_bullets(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = []
    import re
    for l in lines:
        clean = re.sub(r"^[\-*•·\s]*\d*[.)\-]?\s*", "", l)
        if len(clean) > 2:
            bullets.append(clean)
    return bullets

# --------------------------------------------------------------
# VISUALIZATIONS (ENHANCED)
# --------------------------------------------------------------
def plot_monthly(df):
    monthly = df.resample("M", on="date")["amount"].sum()
    fig = px.line(monthly, title="Monthly Spending Trend", markers=True)
    fig.update_layout(height=350)
    return fig


def plot_categories(df):
    fig = px.pie(df, names="category", values="amount", title="Spending by Category")
    fig.update_layout(height=350)
    return fig


def plot_heatmap(df):
    df["weekday"] = df["date"].dt.weekday  # 0=Mon
    df["week"] = df["date"].dt.isocalendar().week
    heat = df.pivot_table(index="week", columns="weekday", values="amount", aggfunc="sum", fill_value=0)
    fig = px.imshow(heat, aspect="auto", title="Spending Heatmap (Week vs Weekday)")
    fig.update_layout(height=350)
    return fig


def plot_daily(df):
    daily = df.resample("D", on="date")["amount"].sum()
    fig = px.area(daily, title="Daily Spending", markers=True)
    fig.update_layout(height=300)
    return fig

# --------------------------------------------------------------
# UI — SIDEBAR INPUT
# --------------------------------------------------------------
st.sidebar.header("Add New Expense")
with st.sidebar.form("add_expense_form"):
    amount = st.number_input("Amount", min_value=0.0, step=0.5)
    desc = st.text_input("Description")
    auto_cat = st.checkbox("Auto-categorize with AI", value=True)
    date_input = st.date_input("Date", value=date.today())
    submitted = st.form_submit_button("Add Expense")

if submitted:
    if amount <= 0:
        st.sidebar.error("Amount must be greater than zero.")
    else:
        if auto_cat and router.ready():
            category = router.call(prompt_category(desc))
        else:
            category = st.sidebar.selectbox("Category", ["misc", "food", "transport", "utilities", "shopping"],
                                           index=0, key="manual_cat_select")

        add_expense(amount, category, desc, date_input)
        st.sidebar.success(f"Added: {amount} ({category})")

# --------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------
df = load_data()

if df.empty:
    st.info("No expenses recorded yet.")
    st.stop()

# --------------------------------------------------------------
# LAYOUT
# --------------------------------------------------------------
t1, t2 = st.tabs(["📊 Dashboard", "🤖 AI Insights"])

with t1:
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(plot_monthly(df), use_container_width=True)
    with colB:
        st.plotly_chart(plot_categories(df), use_container_width=True)

    st.plotly_chart(plot_daily(df), use_container_width=True)
    st.plotly_chart(plot_heatmap(df), use_container_width=True)


with t2:
    st.subheader("AI Pattern Analysis")
    if router.ready():
        prompt = prompt_pattern_analysis(df)
        raw = router.call(prompt)
        bullets = parse_bullets(raw)
        st.write("### Insights")
        for b in bullets:
            st.markdown(f"- {b}")
        st.write("### Raw Model Output")
        st.code(raw)
    else:
        st.warning("Agent Router not configured. Set environment variables to enable AI.")
