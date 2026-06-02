# Wealth Survival Score

> *I built an app that tells you whether your financial life will survive the next 10 years.*

## What it does

You input your country, age, portfolio allocation, and how you behaved during the last crash. The app runs 5 real-time signals through a machine learning pipeline and returns a personalised **Wealth Survival Score** from 0-100 with a plain-English verdict.

**Live app:** [Launch on Streamlit](YOUR-STREAMLIT-URL-HERE)

## The 5 signals

| Signal | Source | Why it matters |
|--------|--------|----------------|
| Central bank language drift | VIX + Fed/ECB speeches | Policy uncertainty before it hits prices |
| Commodity stress index | Gold, Oil, Wheat, Copper | Real-economy stress visible before equities |
| Geopolitical contagion | Google Trends fear terms | Retail fear before institutional reaction |
| Demographic pressure | World Bank data | Forced pension selling pressure by country |
| Retail capitulation risk | Random Forest on all signals | Probability you are about to panic-sell |

## Key finding

Traditional financial features (VIX, Gold, Oil) explain a limited portion of market stress variance. Adding Google Trends fear signals and commodity breadth improves prediction. The most powerful signal is **behavioural**: knowing which investor archetype you are predicts survival rate better than any single market metric.

## Sample result

Germany, Age 36-45, Anxious Holder, 60% stocks / 15% commodities

    Wealth Survival Score:      48 / 100
    Market stress probability:  69.5%
    Archetype:                  The Anxious Holder

## ML toolkit used

- **Transformers** — FinBERT embeddings on central bank speech text
- **K-Means + Elbow Method** — market regime clustering (optimal k=3)
- **Random Forest** — stress event prediction + SHAP feature importance
- **Linear Regression + R2** — baseline vs alternative data comparison

## Regime distribution (K-Means k=3)

- Calm market:      75 months (50%)
- Elevated stress:  59 months (39%)
- Crisis / panic:   16 months (11%)

## Project structure

    wealth-survival-score/
    notebooks/    01_data_collection.ipynb
                  02_processing_and_viz.ipynb
                  03_models.ipynb
    data/raw/     all CSV signal files
    models/       trained model pickle files
    app/          Streamlit app (app.py)
    utils/        score_engine.py
    requirements.txt

## How to run locally

    git clone https://github.com/YOUR-USERNAME/wealth-survival-score
    cd wealth-survival-score
    pip install -r requirements.txt
    streamlit run app/app.py

## Results summary

| Model | CV R2 |
|-------|-------|
| Traditional features only (VIX, Gold, Oil) | baseline |
| Traditional + Alternative signals | improved |

---
Built with Python, scikit-learn, SHAP, yfinance, pytrends, Streamlit.
