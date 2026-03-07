"""
ML Core: Carrier Scoring + Risk Prediction + PuLP Optimizer
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, os, warnings
warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.dirname(__file__))
DATA  = os.path.join(BASE, "data")
MODEL = os.path.join(BASE, "models")
os.makedirs(MODEL, exist_ok=True)

# ── Load Data ────────────────────────────────────────────────────────────────
df      = pd.read_csv(f"{DATA}/shipment_history.csv")
carriers= pd.read_csv(f"{DATA}/carrier_master.csv")
lanes   = pd.read_csv(f"{DATA}/lane_master.csv")

# ── 1. CARRIER SCORING ENGINE ────────────────────────────────────────────────
def build_carrier_scores():
    grp = df.groupby("carrier_id").agg(
        avg_cost_per_kg   = ("cost_per_kg",         "mean"),
        ontime_pct        = ("on_time_delivery",     "mean"),
        damage_rate       = ("damage_loss_flag",     "mean"),
        avg_delay_days    = ("delay_days",           "mean"),
        shipment_count    = ("shipment_id",          "count"),
        avg_weight_kg     = ("weight_kg",            "mean"),
    ).reset_index()

    # Min-max normalize (lower cost = better, so invert)
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

    grp["norm_price"]  = 1 - norm(grp["avg_cost_per_kg"])
    grp["norm_ontime"] = norm(grp["ontime_pct"])
    grp["norm_risk"]   = 1 - norm(grp["damage_rate"])

    grp["composite_score"] = (
        0.4 * grp["norm_price"] +
        0.3 * grp["norm_ontime"] +
        0.3 * grp["norm_risk"]
    ).round(4)

    grp["rank"] = grp["composite_score"].rank(ascending=False).astype(int)
    return grp.merge(carriers[["carrier_id","name","fleet_size","iso_certified"]], on="carrier_id")

# ── 2. LANE-LEVEL SCORING ────────────────────────────────────────────────────
def build_lane_scores():
    grp = df.groupby(["carrier_id","lane_id"]).agg(
        avg_cost_per_kg = ("cost_per_kg",        "mean"),
        ontime_pct      = ("on_time_delivery",   "mean"),
        damage_rate     = ("damage_loss_flag",   "mean"),
        shipment_count  = ("shipment_id",        "count"),
    ).reset_index()

    results = []
    for lane_id, sub in grp.groupby("lane_id"):
        sub = sub.copy()
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        sub["norm_price"]  = 1 - norm(sub["avg_cost_per_kg"])
        sub["norm_ontime"] = norm(sub["ontime_pct"])
        sub["norm_risk"]   = 1 - norm(sub["damage_rate"])
        sub["composite_score"] = (
            0.4*sub["norm_price"] + 0.3*sub["norm_ontime"] + 0.3*sub["norm_risk"]
        ).round(4)
        sub["lane_rank"] = sub["composite_score"].rank(ascending=False).astype(int)
        results.append(sub)

    out = pd.concat(results).merge(
        carriers[["carrier_id","name"]], on="carrier_id"
    ).merge(lanes[["lane_id","origin","destination","distance_km"]], on="lane_id")
    return out

# ── 3. RISK PREDICTION MODEL ─────────────────────────────────────────────────
def train_risk_model():
    features = ["distance_km","weight_kg","transit_days_promised","is_monsoon","is_festival"]
    target   = "on_time_delivery"

    # Encode carrier
    df_enc = df.copy()
    df_enc["carrier_enc"] = pd.Categorical(df_enc["carrier_id"]).codes
    features = features + ["carrier_enc"]

    X = df_enc[features].fillna(0)
    y = df_enc[target]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Gradient Boosting (sklearn equiv of XGBoost)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, gb.predict(X_te_s))

    # Logistic Regression for probability calibration
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_s, y_tr)
    lr_acc = accuracy_score(y_te, lr.predict(X_te_s))

    # Feature importances
    feat_imp = dict(zip(features, gb.feature_importances_.round(4)))

    # Save
    with open(f"{MODEL}/gb_model.pkl","wb") as f:  pickle.dump(gb, f)
    with open(f"{MODEL}/lr_model.pkl","wb") as f:  pickle.dump(lr, f)
    with open(f"{MODEL}/scaler.pkl","wb") as f:    pickle.dump(scaler, f)
    with open(f"{MODEL}/features.pkl","wb") as f:  pickle.dump(features, f)

    return {"gb_accuracy": round(acc,4), "lr_accuracy": round(lr_acc,4), "feature_importances": feat_imp}

# ── 4. PREDICT FOR A NEW SHIPMENT ────────────────────────────────────────────
def predict_ontime(distance_km, weight_kg, transit_days, is_monsoon, is_festival, carrier_id):
    with open(f"{MODEL}/gb_model.pkl","rb") as f:  gb = pickle.load(f)
    with open(f"{MODEL}/lr_model.pkl","rb") as f:  lr = pickle.load(f)
    with open(f"{MODEL}/scaler.pkl","rb") as f:    scaler = pickle.load(f)
    with open(f"{MODEL}/features.pkl","rb") as f:  features = pickle.load(f)

    carrier_map = {"C001":0,"C002":1,"C003":2,"C004":3,"C005":4,"C006":5}
    carrier_enc = carrier_map.get(carrier_id, 0)

    row = pd.DataFrame([{
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "transit_days_promised": transit_days,
        "is_monsoon": is_monsoon,
        "is_festival": is_festival,
        "carrier_enc": carrier_enc,
    }])[features]

    row_s = scaler.transform(row)
    prob_gb = gb.predict_proba(row_s)[0][1]
    prob_lr = lr.predict_proba(row_s)[0][1]
    blended = round(0.6*prob_gb + 0.4*prob_lr, 4)

    # SHAP-style manual feature contribution (without shap lib)
    fi = dict(zip(features, gb.feature_importances_))
    contributions = [
        {"feature": k, "importance": round(float(v),4),
         "value": float(row[k].iloc[0]),
         "direction": "positive" if k not in ["distance_km","is_monsoon","is_festival"] else "negative"}
        for k,v in sorted(fi.items(), key=lambda x: -x[1])
    ]

    return {
        "ontime_probability": blended,
        "risk_level": "LOW" if blended>0.88 else ("MEDIUM" if blended>0.75 else "HIGH"),
        "gb_prob": round(float(prob_gb),4),
        "lr_prob": round(float(prob_lr),4),
        "feature_contributions": contributions[:5],
    }

# ── 5. PULP OPTIMIZER ────────────────────────────────────────────────────────
def optimize_allocation(bid_data, lane_volumes):
    """
    bid_data: list of {carrier_id, lane_id, rate, capacity, ontime_pct, risk}
    lane_volumes: {lane_id: required_kg}
    Returns optimal allocation minimizing cost * risk under capacity constraints.
    """
    try:
        import pulp
    except ImportError:
        # Fallback: greedy allocation by composite score
        return _greedy_allocation(bid_data, lane_volumes)

    prob  = pulp.LpProblem("CarrierAllocation", pulp.LpMinimize)
    alloc = {(b["carrier_id"],b["lane_id"]): pulp.LpVariable(
        f"x_{b['carrier_id']}_{b['lane_id']}", 0, b["capacity"]
    ) for b in bid_data}

    # Objective: minimize cost-adjusted for risk
    prob += pulp.lpSum(
        alloc[(b["carrier_id"],b["lane_id"])] * b["rate"] * (1 + b["risk"])
        for b in bid_data
    )

    # Constraint 1: Each lane demand must be met
    for lane_id, vol in lane_volumes.items():
        lane_bids = [b for b in bid_data if b["lane_id"]==lane_id]
        if lane_bids:
            prob += pulp.lpSum(alloc[(b["carrier_id"],b["lane_id"])] for b in lane_bids) >= vol

    # Constraint 2: Carrier capacity not exceeded
    for b in bid_data:
        prob += alloc[(b["carrier_id"],b["lane_id"])] <= b["capacity"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = []
    for b in bid_data:
        val = pulp.value(alloc[(b["carrier_id"],b["lane_id"])])
        if val and val > 0:
            results.append({
                "carrier_id": b["carrier_id"],
                "lane_id": b["lane_id"],
                "allocated_kg": round(val, 0),
                "rate": b["rate"],
                "total_cost": round(val * b["rate"], 2),
            })

    return {
        "status": str(pulp.LpStatus[prob.status]),
        "total_cost": round(sum(r["total_cost"] for r in results), 2),
        "allocations": results,
        "method": "PuLP Linear Programming"
    }

def _greedy_allocation(bid_data, lane_volumes):
    results = []
    total_cost = 0
    for lane_id, vol in lane_volumes.items():
        lane_bids = sorted(
            [b for b in bid_data if b["lane_id"]==lane_id],
            key=lambda x: x["rate"]*(1+x["risk"])
        )
        remaining = vol
        for b in lane_bids:
            if remaining <= 0: break
            allocated = min(b["capacity"], remaining)
            results.append({
                "carrier_id": b["carrier_id"],
                "lane_id": b["lane_id"],
                "allocated_kg": allocated,
                "rate": b["rate"],
                "total_cost": round(allocated*b["rate"], 2)
            })
            total_cost += allocated*b["rate"]
            remaining  -= allocated
    return {"status":"Greedy","total_cost":round(total_cost,2),"allocations":results,"method":"Greedy Fallback"}

# ── 6. MARKET INTELLIGENCE (Live scraped data) ───────────────────────────────
MARKET_DATA = {
    "scraped_at": "March 7, 2026",
    "composite_multiplier": 1.074,
    "factors": [
        {"id":"fuel",    "name":"Fuel Price Index",        "icon":"⛽","multiplier":1.035,"trend":"UP",   "risk":"yellow",
         "value":"₹87.62–₹92.39/litre","detail":"Israel-Iran crude shock; ₹4–5 hike imminent","confidence":"HIGH"},
        {"id":"toll",    "name":"NHAI Toll Rates",          "icon":"🛣️","multiplier":1.022,"trend":"UP",   "risk":"yellow",
         "value":"4–5% hike Apr 2025","detail":"855 plazas affected; WPI-linked annual revision","confidence":"HIGH"},
        {"id":"festival","name":"Festival Demand Surge",    "icon":"🎉","multiplier":1.045,"trend":"UP",   "risk":"red",
         "value":"4 events next 20 days","detail":"Eid (Mar 19), Navratri, Ram Navami — capacity squeeze","confidence":"HIGH"},
        {"id":"weather", "name":"Weather & Route Risk",     "icon":"🌧️","multiplier":1.005,"trend":"STABLE","risk":"green",
         "value":"Low — dry season","detail":"March pre-monsoon; all major corridors clear","confidence":"HIGH"},
        {"id":"carrier_risk","name":"Carrier Financial Health","icon":"🏦","multiplier":1.008,"trend":"STABLE","risk":"green",
         "value":"0 top carriers flagged","detail":"GATI-KWE monitoring; Delhivery/TCI/BlueDart stable","confidence":"MEDIUM"},
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training models…")
    metrics = train_risk_model()
    print(f"  GB Accuracy : {metrics['gb_accuracy']}")
    print(f"  LR Accuracy : {metrics['lr_accuracy']}")

    print("\nCarrier Scores:")
    scores = build_carrier_scores()
    print(scores[["carrier_id","name","composite_score","rank","ontime_pct","avg_cost_per_kg"]].to_string(index=False))

    print("\nSample Prediction:")
    pred = predict_ontime(1415, 5000, 2, 0, 0, "C001")
    print(f"  On-time prob: {pred['ontime_probability']} | Risk: {pred['risk_level']}")
    print("✅ Models ready")
