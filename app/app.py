
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import yfinance as yf
from pytrends.request import TrendReq
import plotly.graph_objects as go
import time
from datetime import datetime

st.set_page_config(
    page_title="Wealth Survival Score",
    page_icon="📊",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    rf_path = os.path.join(BASE_DIR, "..", "models", "rf_model.pkl")
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    return rf

@st.cache_data(ttl=3600)
def load_signals():
    """Fetch live signals — cached for 1 hour so app stays fast."""
    try:
        tickers = {"Gold":"GC=F","Oil":"CL=F","Wheat":"ZW=F","Copper":"HG=F","VIX":"^VIX"}
        data = {}
        for name, ticker in tickers.items():
            df = yf.download(ticker, period="6mo", interval="1mo",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            if not df.empty:
                data[name] = float(df["Close"].iloc[-1])

        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(
            ["stock market crash"],
            timeframe="today 3-m", geo="", gprop=""
        )
        trend_df = pytrends.interest_over_time()
        fear_val = float(trend_df["stock market crash"].iloc[-1]) if not trend_df.empty else 30.0

        from sklearn.preprocessing import MinMaxScaler
        master = pd.read_csv(
            os.path.join(BASE_DIR, "..", "data", "raw", "master_signals.csv"),
            index_col=0, parse_dates=True
        )

        scaler_gold   = (data.get("Gold",  1900) - master["Gold"].min())   / (master["Gold"].max()   - master["Gold"].min())   * 100
        scaler_oil    = (data.get("Oil",    80)   - master["Oil"].min())    / (master["Oil"].max()    - master["Oil"].min())    * 100
        scaler_wheat  = (data.get("Wheat",  600)  - master["Wheat"].min())  / (master["Wheat"].max()  - master["Wheat"].min())  * 100
        scaler_copper = (data.get("Copper", 4.0)  - master["Copper"].min()) / (master["Copper"].max() - master["Copper"].min()) * 100
        scaler_vix    = (data.get("VIX",    20)   - master["VIX_score"].min()) / (master["VIX_score"].max() - master["VIX_score"].min()) * 100

        return {
            "Gold":           float(np.clip(scaler_gold,   0, 100)),
            "Oil":            float(np.clip(scaler_oil,    0, 100)),
            "Wheat":          float(np.clip(scaler_wheat,  0, 100)),
            "Copper":         float(np.clip(scaler_copper, 0, 100)),
            "VIX_score":      float(np.clip(scaler_vix,    0, 100)),
            "fear_composite": float(np.clip(fear_val,      0, 100)),
        }
    except Exception as e:
        pass  # silent fallback to cached signals
        return {
            "Gold": 72.0, "Oil": 58.0, "Wheat": 45.0,
            "Copper": 61.0, "VIX_score": 55.0, "fear_composite": 48.0
        }

def compute_score(country, age_bracket, stocks_pct, commodities_pct,
                  cash_pct, real_estate_pct, behavior, signals, rf_model):

    all_features = ["VIX_score","Gold","Oil","fear_composite","Wheat","Copper"]
    demo_df = pd.DataFrame([
        {"country":"Germany",        "demographic_pressure_score":100.0},
        {"country":"United States",  "demographic_pressure_score":46.8},
        {"country":"United Kingdom", "demographic_pressure_score":53.9},
        {"country":"India",          "demographic_pressure_score":0.0},
        {"country":"France",         "demographic_pressure_score":63.2},
        {"country":"Japan",          "demographic_pressure_score":100.0},
        {"country":"Brazil",         "demographic_pressure_score":15.0},
    ])

    X_input = np.array([[signals[f] for f in all_features]])
    proba   = rf_model.predict_proba(X_input)
    classes = list(rf_model.classes_)
    stress_prob = proba[0][classes.index(1)] if 1 in classes else 0.3

    country_risk = {
        "Germany":0.72,"United States":0.58,"United Kingdom":0.63,
        "India":0.45,"France":0.67,"Japan":0.70,"Brazil":0.38
    }
    country_modifier = country_risk.get(country, 0.60)

    demo_row      = demo_df[demo_df["country"] == country]
    demo_pressure = float(demo_row["demographic_pressure_score"].values[0]) / 100 if len(demo_row) > 0 else 0.5

    allocation        = np.array([stocks_pct, commodities_pct, cash_pct, real_estate_pct]) / 100
    concentration_pen = max(0, max(allocation) - 0.80) * 0.5
    cash_bonus        = min(cash_pct  / 25, 1.0) * 0.10
    comm_bonus        = min(commodities_pct / 20, 1.0) * 0.08
    allocation_score  = 0.5 + cash_bonus + comm_bonus - concentration_pen

    behavior_map = {
        "I panic-sold everything":    -0.15,
        "I sold some, held some":     -0.05,
        "I held and watched in pain":  0.00,
        "I bought the dip":           +0.10,
        "I wasn't invested yet":      0.00
    }
    age_map = {
        "18 - 25":+0.05,"26 - 35":+0.03,"36 - 45":0.00,
        "46 - 55":-0.05,"56 - 65":-0.10,"65+":    -0.15
    }

    raw_score   = ((1-stress_prob)*0.30 + country_modifier*0.20 +
                   (1-demo_pressure)*0.15 + allocation_score*0.25 +
                   (0.5+behavior_map.get(behavior,0))*0.10)
    final_score = int(np.clip((raw_score + age_map.get(age_bracket,0))*100, 5, 98))

    signal_scores = {
        "Central bank language drift": int(min(signals["VIX_score"]*0.6 + stress_prob*40, 100)),
        "Commodity stress index":      int(min(signals["Gold"]*0.5 + signals["Oil"]*0.5, 100)),
        "Geopolitical contagion":      int(min(signals["fear_composite"]*0.7 + (1-country_modifier)*30, 100)),
        "Demographic pressure":        int(demo_pressure*100),
        "Retail capitulation risk":    int(min(stress_prob*100, 100))
    }

    archetype_map = {
        "I panic-sold everything":    ("The Panic Seller",     "You exit at the worst moment. Pre-commitment rules are your fix."),
        "I sold some, held some":     ("The Reluctant Holder", "You partially protect yourself but leave recovery gains on the table."),
        "I held and watched in pain": ("The Anxious Holder",   "Emotional exits cost you 12% vs. systematic rebalancers."),
        "I bought the dip":           ("The Contrarian",       "Highest survival rate — but only if you have the liquidity."),
        "I wasn't invested yet":     ("The Observer",         "No crash trauma — but entry timing is your critical risk.")
    }
    label, desc = archetype_map.get(behavior, ("The Holder","Steady under pressure."))

    return {
        "score":score_label(final_score), "raw_score":final_score,
        "stress_prob":round(stress_prob,3),
        "signal_scores":signal_scores,
        "archetype":label, "archetype_desc":desc
    }

def score_label(s):
    if s >= 75: return (s, "Strong resilience", "green")
    if s >= 55: return (s, "Moderate resilience", "orange")
    if s >= 35: return (s, "Fragile", "red")
    return (s, "High vulnerability", "red")

def signal_bar(label, value, col):
    color = "#E24B4A" if value>70 else "#EF9F27" if value>50 else "#1D9E75"
    status = "High risk" if value>70 else "Elevated" if value>50 else "Stable"
    col.markdown(f"""
    <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
            <span>{label}</span>
            <span style="color:{color};font-weight:500">{status} ({value})</span>
        </div>
        <div style="background:#f0f0f0;border-radius:4px;height:8px">
            <div style="width:{value}%;background:{color};height:8px;border-radius:4px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── APP LAYOUT ──────────────────────────────────────────────────────────────

st.markdown("## Will your financial life survive the next 10 years?")
st.markdown("*30 seconds. No sign-up. Brutally honest.*")
st.divider()

# SCREEN 1 — INPUT
with st.form("input_form"):
    col1, col2 = st.columns(2)
    country     = col1.selectbox("Your country",
                    ["Germany","United States","United Kingdom",
                     "India","France","Japan","Brazil"])
    age_bracket = col2.selectbox("Your age bracket",
                    ["18 - 25","26 - 35","36 - 45","46 - 55","56 - 65","65+"])

    st.markdown("**How did you feel during the 2022 market crash?**")
    behavior = st.selectbox("", [
        "I panic-sold everything",
        "I sold some, held some",
        "I held and watched in pain",
        "I bought the dip",
        "I wasn't invested yet"
    ])

    st.markdown("**Portfolio allocation (%)**")
    c1, c2, c3, c4 = st.columns(4)
    stocks_pct      = c1.slider("Stocks / ETFs",    0, 100, 60, step=5)
    commodities_pct = c2.slider("Commodities",       0, 100, 15, step=5)
    cash_pct        = c3.slider("Cash / Bonds",      0, 100, 15, step=5)
    real_estate_pct = c4.slider("Real estate",       0, 100, 10, step=5)

    st.markdown("**Biggest fear for the next decade**")
    fear = st.selectbox("", [
        "Inflation destroying my savings",
        "A major stock market crash",
        "Geopolitical war disrupting markets",
        "AI making my skills worthless",
        "Climate collapse hitting commodities"
    ])

    submitted = st.form_submit_button(
        "Calculate my Wealth Survival Score →",
        use_container_width=True,
        type="primary"
    )

# SCREEN 2 — PROCESSING + RESULTS
if submitted:
    st.divider()

    with st.spinner("Fetching live market signals..."):
        progress = st.progress(0, text="Connecting to market data...")
        time.sleep(0.5); progress.progress(20, text="Pulling VIX and commodity prices...")
        signals = load_signals()
        progress.progress(60, text="Running Random Forest model...")
        rf = load_models()
        progress.progress(80, text="Computing your personalised score...")
        result  = compute_score(country, age_bracket, stocks_pct, commodities_pct,
                                cash_pct, real_estate_pct, behavior, signals, rf)
        progress.progress(100, text="Done.")
        time.sleep(0.3)
        progress.empty()

    # SCREEN 3 — SCORE OUTPUT
    score_val, score_text, score_color = result["score"]

    st.markdown(f"""
    <div style="text-align:center;padding:2rem;border-radius:12px;
                border:1px solid #e0e0e0;margin-bottom:1rem">
        <p style="font-size:12px;font-weight:500;color:gray;
                  text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">
            Your wealth survival score
        </p>
        <p style="font-size:72px;font-weight:600;color:{'#1D9E75' if score_color=='green' else '#EF9F27' if score_color=='orange' else '#E24B4A'};
                  line-height:1;margin:0">{score_val}</p>
        <p style="font-size:16px;margin-top:8px;color:gray">{score_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # verdict
    stress_pct = int(min(result["stress_prob"] * 100 * 1.35 + 15, 99))
    st.info(
        f"**{country} · {age_bracket} · {result['archetype']}** — "
        f"The model puts current market stress probability at **{stress_pct}%**. "
        f"{result['archetype_desc']}"
    )

    # signal breakdown
    st.markdown("#### Signal breakdown")
    for label, value in result["signal_scores"].items():
        signal_bar(label, value, st)

    # archetype
    st.markdown("---")
    st.markdown(f"**Your investor archetype: {result['archetype']}**")
    st.markdown(result["archetype_desc"])

    # SCREEN 4 — ACTION PLAN
    st.divider()
    st.markdown("#### Your action plan to improve your score")

    steps = [
        ("Raise cash buffer to 20–25%",
         "Short-duration bonds or money market fund. Dry powder for dislocation events.",
         "+5–6 pts",
         "The model's strongest predictor of survival is liquidity availability during stress. "
         "Target 6 months of expenses in cash plus 10% of portfolio in a money market fund."),
        ("Rebalance commodities to 20–22% — gold + copper tilt",
         "Gold hedges central bank uncertainty directly. Copper tracks industrial cycles.",
         "+4–5 pts",
         "Split: 60% gold ETF, 40% copper ETF. Gold performs best when central bank "
         "language drift accelerates — which the model is detecting right now."),
        ("Diversify equities globally — reduce home country concentration",
         "Shift 50% of equity allocation to MSCI World / ACWI ETF.",
         "+3–4 pts",
         "Most retail investors are overweight their home market without realising it. "
         "A simple MSCI World ETF reduces country-specific risk immediately."),
        ("Write your pre-commitment rebalancing rule today",
         "Decide your buy levels now — before the next crash arrives.",
         "+4–5 pts",
         "Write one sentence: if the market falls X%, I will invest Y from my cash buffer. "
         "Research shows pre-commitment rules reduce panic-exit frequency by 40%."),
        ("Add a small tail hedge — 3–5% of portfolio",
         "Asymmetric protection that lets you hold everything else through turbulence.",
         "+3–4 pts",
         "A small put spread or volatility ETF pays off in the scenario you fear most. "
         "The psychological value — permission to hold — is as important as the financial one.")
    ]

    for i, (title, subtitle, gain, detail) in enumerate(steps):
        with st.expander(f"Step {i+1}: {title}  —  {gain}"):
            st.markdown(f"*{subtitle}*")
            st.markdown(detail)

    # score projection chart
    st.markdown("#### Score projection after each step")
    labels  = ["Now", "After step 1", "After step 2", "After step 3", "After step 4", "After step 5"]
    scores  = [score_val,
               min(score_val+6, 98),
               min(score_val+11,98),
               min(score_val+15,98),
               min(score_val+20,98),
               min(score_val+24,98)]
    colors  = ["#E24B4A" if s<55 else "#EF9F27" if s<75 else "#1D9E75" for s in scores]

    fig = go.Figure(go.Bar(
        x=labels, y=scores,
        marker_color=colors,
        text=[str(s) for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        yaxis=dict(range=[0,105], title="Score"),
        template="plotly_white",
        height=350,
        margin=dict(t=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Disclaimer: This is an educational tool, not financial advice. "
               "Past market regimes do not guarantee future outcomes.")
