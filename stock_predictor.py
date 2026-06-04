
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.title("🔮 Stock Predictor")
st.caption("Historical chart + AI-powered outlook using Prophet forecasting")

# ── LLM helper ───────────────────────────────────────────────
def get_llm_outlook(ticker, latest, pct_change, high_52, low_52, future_price, trend, lower, upper):
    prompt = f"""
You are a financial analyst assistant. Given the following stock data for {ticker}:
- Current price: ${latest:.2f}
- 1-year price change: {pct_change:+.1f}%
- 52-week high: ${high_52:.2f}
- 52-week low: ${low_52:.2f}
- Prophet 30-day projected price: ${future_price:.2f} ({trend})
- Forecast confidence interval: ${lower:.2f} to ${upper:.2f}

In 3-4 sentences, give a brief, balanced outlook on this stock's potential near-term performance.
Mention the trend, the confidence interval range, and remind the user this is not financial advice.
"""
    # Try Ollama first
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception:
        pass

    # Fallback: Groq
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Rule-based fallback
    direction = "upward" if future_price > latest else "downward"
    return (
        f"**{ticker} Outlook:** Based on the past year, {ticker} has moved "
        f"{pct_change:+.1f}% and is currently trading at \${latest:.2f}. "
        f"The model projects a {direction} trajectory over the next 30 days toward "
        f"\${future_price:.2f}, with a confidence range of \${lower:.2f}–\${upper:.2f}. "
    )

# ── Stock analysis helper ─────────────────────────────────────
def analyze_ticker(ticker, container):
    with container:
        with st.spinner(f"Fetching data for {ticker}..."):
            end = datetime.today()
            start = end - timedelta(days=365)
            df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            st.error(f"No data found for {ticker}.")
            return

        

        # ── Key stats ─────────────────────────────────────────
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()

        # ── Chart ─────────────────────────────────────────────
        st.subheader(f"{ticker} – Past 12 Months")
        chart_df = pd.DataFrame({
            "Price":   close,
            "50-Day MA":  close.rolling(window=50).mean(),
            "200-Day MA": close.rolling(window=200).mean(),
        })
        st.line_chart(chart_df)

        latest     = float(close.iloc[-1])
        year_ago   = float(close.iloc[0])
        pct_change = ((latest - year_ago) / year_ago) * 100
        high_52    = float(high.max())
        low_52     = float(low.min())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${latest:.2f}")
        col2.metric("1-Year Change", f"{pct_change:+.1f}%")
        col3.metric("52-Week High",  f"${high_52:.2f}")
        col4.metric("52-Week Low",   f"${low_52:.2f}")

        # ── Prophet forecast ──────────────────────────────────
        
        with st.spinner("Running Prophet model..."):
            # Prophet needs columns named exactly 'ds' (date) and 'y' (value)
            prophet_df = pd.DataFrame({
                "ds": close.index,
                "y":  close.values
            })
            # Remove timezone info if present — Prophet doesn't like it
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05  # controls how flexible the trend is
            )
            model.fit(prophet_df)

            # Create 30 future trading days
            future    = model.make_future_dataframe(periods=30)
            forecast  = model.predict(future)

            # Pull out the key numbers
            future_price = float(forecast["yhat"].iloc[-1])
            lower        = float(forecast["yhat_lower"].iloc[-1])
            upper        = float(forecast["yhat_upper"].iloc[-1])
            trend        = "📈 Upward" if future_price > latest else "📉 Downward"

        # ── Forecast chart ────────────────────────────────────
        st.subheader(f"30-Day Forecast   {trend}")
        # Combine historical close + forecast into one clean chart
        hist_chart = pd.DataFrame({
            "Historical": close.values
        }, index=close.index)

        forecast_only = forecast[forecast["ds"] > close.index[-1]][["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
        forecast_chart = pd.DataFrame({
            "Forecast":   forecast_only["yhat"],
            "Upper Band": forecast_only["yhat_upper"],
            "Lower Band": forecast_only["yhat_lower"],
        })

        st.line_chart(pd.concat([
            hist_chart.rename(columns={"Historical": "Historical Close"}),
            forecast_chart
        ]))

        # Forecast summary metrics
        fc1, fc2, fc3 = st.columns(3)
        delta_value = future_price - latest
        fc1.metric(
            "30-Day Projection",
            f"${future_price:.2f}",
            delta=f"{'+' if delta_value >= 0 else ''}{delta_value:.2f}",
            delta_color="normal"
        )
        fc2.metric("Forecast Low",      f"${lower:.2f}")
        fc3.metric("Forecast High",     f"${upper:.2f}")

        # ── AI Outlook ────────────────────────────────────────
        st.subheader("AI Outlook")
        outlook = get_llm_outlook(
            ticker, latest, pct_change, high_52, low_52,
            future_price, trend, lower, upper
        )
        st.info(outlook)
        

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Single Stock", "Compare Two Stocks"])

def stock_search(label, key):
    query = st.sidebar.text_input(label, value="", key=f"input_{key}")
    ticker = None

    if len(query) >= 2:  # start searching after 2 characters
        try:
            results = yf.Search(query, max_results=8)
            quotes  = results.quotes
            if quotes:
                options = ["-- Select a company --"] + [
                    f"{q.get('longname') or q.get('shortname', 'Unknown')} ({q.get('symbol', '')})"
                    for q in quotes if q.get('symbol')
                ]
                selected = st.sidebar.selectbox(
                    "Results",
                    options,
                    key=f"select_{key}"
                )
                if selected != "-- Select a company --":
                    ticker = selected.split("(")[-1].replace(")", "").strip()
            else:
                st.sidebar.caption("No matches found.")
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")
    elif len(query) == 1:
        st.sidebar.caption("Keep typing to search...")

    return ticker

if mode == "Single Stock":
    ticker1 = stock_search("Search Company", key="one")
    analyze = st.sidebar.button("Analyze Stock", disabled=ticker1 is None)
    if analyze and ticker1:
        analyze_ticker(ticker1, st.container())

else:
    ticker1 = stock_search("Search Company 1", key="one")
    ticker2 = stock_search("Search Company 2", key="two")
    both_ready = ticker1 is not None and ticker2 is not None
    analyze = st.sidebar.button("Compare Stocks", disabled=not both_ready)
    if analyze and both_ready:
        col_a, col_b = st.columns(2)
        analyze_ticker(ticker1, col_a)
        analyze_ticker(ticker2, col_b)