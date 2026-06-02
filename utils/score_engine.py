import numpy as np
import pandas as pd
import pickle
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    with open(os.path.join(BASE_DIR, "models", "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "feature_config.json"), "r") as f:
        config = json.load(f)
    return rf, config

def compute_score(country, age_bracket, stocks_pct, commodities_pct,
                  cash_pct, real_estate_pct, behavior, master_df,
                  rf_model, demo_df):

    all_features = ["VIX_score","Gold","Oil","fear_composite","Wheat","Copper"]
    latest = master_df[all_features].iloc[-1]

    X_input = latest.values.reshape(1, -1)
    proba   = rf_model.predict_proba(X_input)
    classes = list(rf_model.classes_)
    stress_prob = proba[0][classes.index(1)] if 1 in classes else 0.3

    country_risk = {
        "Germany": 0.72, "United States": 0.58,
        "United Kingdom": 0.63, "India": 0.45,
        "France": 0.67, "Japan": 0.70, "Brazil": 0.38
    }
    country_modifier = country_risk.get(country, 0.60)

    demo_row      = demo_df[demo_df["country"] == country]
    demo_pressure = float(demo_row["demographic_pressure_score"].values[0]) / 100 \
                    if len(demo_row) > 0 else 0.5

    allocation          = np.array([stocks_pct, commodities_pct,
                                    cash_pct, real_estate_pct]) / 100
    concentration_pen   = max(0, max(allocation) - 0.80) * 0.5
    cash_bonus          = min(cash_pct / 25, 1.0) * 0.10
    comm_bonus          = min(commodities_pct / 20, 1.0) * 0.08
    allocation_score    = 0.5 + cash_bonus + comm_bonus - concentration_pen

    behavior_map = {
        "I panic-sold everything":    -0.15,
        "I sold some, held some":     -0.05,
        "I held and watched in pain":  0.00,
        "I bought the dip":           +0.10,
        "I wasn't invested yet":      0.00
    }
    age_map = {
        "18 – 25": +0.05, "26 – 35": +0.03,
        "36 – 45":  0.00, "46 – 55": -0.05,
        "56 – 65": -0.10, "65+":     -0.15
    }

    raw_score   = ((1 - stress_prob) * 0.30 + country_modifier * 0.20 +
                   (1 - demo_pressure) * 0.15 + allocation_score * 0.25 +
                   (0.5 + behavior_map.get(behavior, 0)) * 0.10)
    final_score = int(np.clip((raw_score + age_map.get(age_bracket, 0)) * 100, 5, 98))

    signal_scores = {
        "Central bank language drift": int(min(latest["VIX_score"] * 0.6 + stress_prob * 40, 100)),
        "Commodity stress index":      int(min(latest["Gold"] * 0.5 + latest["Oil"] * 0.5, 100)),
        "Geopolitical contagion":      int(min(latest["fear_composite"] * 0.7 + (1 - country_modifier) * 30, 100)),
        "Demographic pressure":        int(demo_pressure * 100),
        "Retail capitulation risk":    int(min(stress_prob * 100, 100))
    }

    archetype_map = {
        "I panic-sold everything":    ("The Panic Seller",     "You exit at the worst moment."),
        "I sold some, held some":     ("The Reluctant Holder", "You partially protect yourself."),
        "I held and watched in pain": ("The Anxious Holder",   "Emotional exits cost you 12% vs. systematic rebalancers."),
        "I bought the dip":           ("The Contrarian",       "Highest survival rate — if you have the liquidity."),
        "I wasn't invested yet":     ("The Observer",         "Entry timing is your critical risk.")
    }
    label, desc = archetype_map.get(behavior, ("The Holder", "Steady under pressure."))

    return {
        "score": final_score, "stress_prob": round(stress_prob, 3),
        "signal_scores": signal_scores, "archetype": label,
        "archetype_desc": desc, "country_modifier": country_modifier,
        "demo_pressure": round(demo_pressure, 3)
    }
