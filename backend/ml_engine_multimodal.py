"""
For ML: how to add weather features (IMD API), how to retrain monthly with new shipment data, how to add a feedback loop (user confirms/rejects recommendation → label flips → retrain)
  - Connect to IMD API to scrape real-time precipitation and temp on origin/destination. Add to training DataFrame.
  - Set up a cron task (or Airflow/Celery) to read new rows inserted into a Postgres DB, call train_models(), and hot-swap to mm_xgb.pkl.
  - Store carrier bookings. If a user overrides Rank 1 and chooses Rank 3, penalize Rank 1 for that lane in future data by adjusting the 'target_score' label.
"""
import os
import random
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODE_MAP = {"FTL": 0, "LTL": 1, "roadways": 0, "railways": 2, "airways": 3}
MODE_RELIABILITY_DELTA = {0: 0.0, 1: 0.0, 2: -0.06, 3: 0.04}

CARRIERS_BY_MODE = {
    "roadways": ["BlueDart Surface", "GATI", "Delhivery Freight", "VRL Logistics", "TCI Freight", "Safexpress"],
    "railways": ["CONCOR", "DFC Freight", "KRIBHCO", "Indian Railways Goods", "TransRail", "Hind Terminals"],
    "airways": ["IndiGo Cargo", "Air India Cargo", "SpiceJet Cargo", "Blue Dart Aviation", "StarAir Freight", "Quikjet"],
}

GOODS_MAP = {"FMCG": 0, "Electronics": 1, "Apparel": 2, "Automotive": 3, "Industrial": 4, "Pharmaceuticals": 5, "Perishables": 6}

FEATURES = ["distance_km", "weight_kg", "freight_cost_inr", "transit_days_promised", "mode", "goods_type", "transport_class", "speed_factor", "cost_tier", "is_monsoon"]

def _generate_synthetic_multimodal():
    df_road = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "shipment_history.csv"))
    df_road["mode"] = df_road["mode"].map(MODE_MAP).fillna(0).astype(int)
    
    # Generate Railway
    df_rail = df_road.sample(1500, replace=True).copy()
    df_rail["mode"] = 2
    df_rail["carrier_id"] = np.random.choice(CARRIERS_BY_MODE["railways"], 1500)
    df_rail["transit_days_promised"] = (df_rail["distance_km"] / 45 / 24).apply(lambda x: max(1, int(x)))
    df_rail["transit_days_actual"] = df_rail["transit_days_promised"] + np.random.randint(0, 4, 1500)
    df_rail["freight_cost_inr"] = df_rail["weight_kg"] * np.random.uniform(1.3, 2.2, 1500)
    df_rail["on_time_delivery"] = (np.random.rand(1500) > 0.15).astype(int) # Slightly lower on-time
    
    # Generate Airways
    df_air = df_road.sample(1000, replace=True).copy()
    df_air["mode"] = 3
    df_air["carrier_id"] = np.random.choice(CARRIERS_BY_MODE["airways"], 1000)
    df_air["transit_days_promised"] = 1
    df_air["transit_days_actual"] = 1 + (np.random.rand(1000) > 0.9).astype(int)
    df_air["freight_cost_inr"] = df_air["weight_kg"] * np.random.uniform(18, 35, 1000)
    df_air["on_time_delivery"] = (np.random.rand(1000) > 0.05).astype(int) # Higher on-time
    
    df_combined = pd.concat([df_road, df_rail, df_air], ignore_index=True)
    return df_combined

def train_models():
    print("Generating multimodal data...")
    df = _generate_synthetic_multimodal()
    
    # Feature engineering
    df["goods_type"] = df["freight_type"].map(GOODS_MAP).fillna(0).astype(int)
    
    # Add new core features
    df["transport_class"] = df["mode"].apply(lambda x: 0 if x in [0,1] else (1 if x==2 else 2))
    df["speed_factor"] = df["distance_km"] / df["transit_days_promised"]
    # 0 = cheap, 1 = std, 2 = premium
    df["cost_tier"] = df["transport_class"].apply(lambda x: 2 if x==2 else (0 if x==1 else 1)) 
    
    # Create Target Score
    delay = df["transit_days_actual"] - df["transit_days_promised"]
    delay_penalty = np.where(delay > 0, delay * 0.1, 0)
    cost_efficiency = 1.0 - (df["freight_cost_inr"] / (df["weight_kg"] * 50)) # Proxy relative score
    cost_efficiency = np.clip(cost_efficiency, 0.1, 1.0)
    
    base_reliability = df["on_time_delivery"] * 0.5 + cost_efficiency * 0.3 - delay_penalty * 0.2
    
    # Apply MODE_RELIABILITY_DELTA
    mode_deltas = df["mode"].map(MODE_RELIABILITY_DELTA).fillna(0)
    df["target_score"] = np.clip(base_reliability + mode_deltas + np.random.normal(0, 0.05, len(df)), 0.1, 0.99)
    
    X = df[FEATURES]
    y = df["target_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    xgb_m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_m.fit(X_train_s, y_train)
    
    gb_m = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_m.fit(X_train_s, y_train)
    
    lr_m = LinearRegression()
    lr_m.fit(X_train_s, y_train)
    
    joblib.dump(xgb_m, os.path.join(MODEL_DIR, "mm_xgb.pkl"))
    joblib.dump(gb_m, os.path.join(MODEL_DIR, "mm_gb.pkl"))
    joblib.dump(lr_m, os.path.join(MODEL_DIR, "mm_lr.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "mm_scaler.pkl"))
    print("Saved mm models.")

def predict_all_carriers(distance_km, weight_kg, transit_days, goods_type, transport_mode, priority_profile, is_monsoon, shipment_value=100000):
    try:
        xgb_m = joblib.load(os.path.join(MODEL_DIR, "mm_xgb.pkl"))
        gb_m = joblib.load(os.path.join(MODEL_DIR, "mm_gb.pkl"))
        lr_m = joblib.load(os.path.join(MODEL_DIR, "mm_lr.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "mm_scaler.pkl"))
    except:
        train_models()
        xgb_m = joblib.load(os.path.join(MODEL_DIR, "mm_xgb.pkl"))
        gb_m = joblib.load(os.path.join(MODEL_DIR, "mm_gb.pkl"))
        lr_m = joblib.load(os.path.join(MODEL_DIR, "mm_lr.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "mm_scaler.pkl"))

    mode_int = MODE_MAP.get(transport_mode.lower(), 0)
    g_type = GOODS_MAP.get(goods_type, 0)
    t_class = 0 if mode_int in [0,1] else (1 if mode_int==2 else 2)
    speed = distance_km / max(1.0, float(transit_days))
    c_tier = 2 if t_class==2 else (0 if t_class==1 else 1)
    mon = 1 if str(is_monsoon).lower() == 'true' else 0
    
    carrier_names = CARRIERS_BY_MODE.get(transport_mode.lower(), CARRIERS_BY_MODE["roadways"])
    
    results = []
    base_kg_rate = 2.0 if t_class==0 else (1.5 if t_class==1 else 20.0)
    
    for c_id in carrier_names:
        fluct = random.uniform(0.9, 1.1)
        est_cost = weight_kg * (base_kg_rate * fluct)
        
        vec = pd.DataFrame([{
            "distance_km": distance_km,
            "weight_kg": weight_kg,
            "freight_cost_inr": est_cost,
            "transit_days_promised": transit_days,
            "mode": mode_int,
            "goods_type": g_type,
            "transport_class": t_class,
            "speed_factor": speed,
            "cost_tier": c_tier,
            "is_monsoon": mon
        }])
        
        vec_s = scaler.transform(vec)
        x_score = min(max(float(xgb_m.predict(vec_s)[0]), 0.1), 0.99)
        g_score = min(max(float(gb_m.predict(vec_s)[0]), 0.1), 0.99)
        l_score = min(max(float(lr_m.predict(vec_s)[0]), 0.1), 0.99)
        
        # Blended
        w_x, w_g, w_l = 0.5, 0.3, 0.2
        if priority_profile.lower() == "cost-first":
            score = (x_score*0.4 + g_score*0.3 + l_score*0.3)
        elif priority_profile.lower() == "reliability":
            score = (x_score*0.6 + g_score*0.3 + l_score*0.1)
        else:
            score = (x_score*0.5 + g_score*0.3 + l_score*0.2)
            
        why = []
        if t_class == 1:
            why.append("Eliminates fuel surcharges, ideal for bulk.")
            if weight_kg > 5000: why.append("Highly cost-effective for extremely heavy cargo (>5t).")
        elif t_class == 2:
            why.append("Next-day delivery capability, cold-chain compatible.")
            if distance_km > 1000: why.append("Fastest mode for long-distance domestic transit.")
        else:
            why.append("Door-to-door delivery, no transhipment risk.")
            
        if x_score > 0.8: why.append("XGBoost strongly signals on-time history.")
        
        results.append({
            "carrier_id": c_id,
            "carrier_name": c_id,
            "scores": {"xgb_reliability": round(x_score, 2), "gb_ontime": round(g_score, 2), "lr_delay_risk": round(l_score, 2), "composite_score": round(score, 3), "blended_ontime": round(score * 100, 1)},
            "cost": {"eff_rate_inr_kg": round(est_cost / weight_kg, 2), "estimated_total_cost_inr": round(est_cost, 2), "base_rate": round((est_cost / weight_kg)/1.1, 2), "fuel_surcharge": 10},
            "risk": {"level": "LOW" if score>0.7 else ("MEDIUM" if score>0.4 else "HIGH")},
            "why": why
        })
        
    # Sort
    if priority_profile.lower() == "cost-first":
        results.sort(key=lambda x: x["cost"]["estimated_total_cost_inr"])
    else:
        results.sort(key=lambda x: x["scores"]["composite_score"], reverse=True)
        
    for i, r in enumerate(results):
        r["recommendation"] = "STRONGLY RECOMMENDED" if i == 0 else "RECOMMENDED" if i == 1 else "GOOD ALTERNATIVE"
        
    return results

if __name__ == "__main__":
    train_models()
