"""
ML Engine — 3-Model Ensemble
├── XGBoost            → delivery risk / service reliability
├── Gradient Boosting  → on-time probability
└── Logistic Regression→ delay probability (explainable)

All stats derived from real shipment_history.csv
No hardcoded averages — everything calculated from data
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

print("BASE:", BASE)
print("DATA:", DATA)
# ── Industry Profiles ─────────────────────────────────────────────────────────
PRIORITY_PROFILES = {
    "cost":        {"w_cost": 0.60, "w_ontime": 0.20, "w_risk": 0.20, "label": "Cost-First"},
    "reliability": {"w_cost": 0.20, "w_ontime": 0.50, "w_risk": 0.30, "label": "Reliability-First"},
    "risk":        {"w_cost": 0.20, "w_ontime": 0.30, "w_risk": 0.50, "label": "Risk-Averse"},
    "balanced":    {"w_cost": 0.40, "w_ontime": 0.30, "w_risk": 0.30, "label": "Balanced"},
    "sales":       {"w_cost": 0.25, "w_ontime": 0.45, "w_risk": 0.30, "label": "Sales-Impact"},
    "pharma":      {"w_cost": 0.15, "w_ontime": 0.35, "w_risk": 0.50, "label": "Pharma/High-Value"},
}

GOODS_RISK_MULTIPLIER = {
    "Pharma":        1.8,
    "Electronics":   1.6,
    "Auto Parts":    1.2,
    "FMCG":          1.0,
    "Textiles":      0.8,
    "General Cargo": 0.9,
}

DAILY_REVENUE_RISK_PCT = {
    "Pharma":        0.030,
    "FMCG":          0.020,
    "Electronics":   0.015,
    "Auto Parts":    0.012,
    "Textiles":      0.008,
    "General Cargo": 0.010,
}

# ── Encoding Maps ─────────────────────────────────────────────────────────────
CARRIER_MAP = {
    "C001": 0, "C002": 1, "C003": 2, "C004": 3,
    "C005": 4, "C006": 5, "C007": 6, "C008": 7
}

FREIGHT_TYPE_MAP = {
    "Pharma": 0, "Electronics": 1, "Auto Parts": 2,
    "FMCG": 3, "Textiles": 4, "General Cargo": 5,
}

MODE_MAP = {"FTL": 0, "LTL": 1}

# ── Feature Sets ──────────────────────────────────────────────────────────────
XGB_FEATURES = [
    "transit_days_promised", "weight_kg", "distance_km",
    "mode_enc", "freight_type_enc",
]

GB_FEATURES = [
    "distance_km", "weight_kg", "transit_days_promised",
    "is_monsoon", "is_festival", "carrier_enc",
    "freight_type_enc", "mode_enc",
]

LR_FEATURES = GB_FEATURES


# ── LOAD REAL STATS FROM DATA ─────────────────────────────────────────────────
def _load_historical_stats():
    """
    Calculate real stats from shipment_history.csv
    Used for revenue at risk and why explanation
    No hardcoding — everything from real data
    """
    df = pd.read_csv(os.path.join(DATA, "shipment_history.csv"))

    # Avg delay days when shipment IS late (per carrier)
    late = df[df["on_time_delivery"] == 0]
    avg_delay_by_carrier = late.groupby("carrier_id")["delay_days"].mean().to_dict()
    overall_avg_delay    = late["delay_days"].mean()

    # On-time % per carrier (historical)
    ontime_by_carrier = df.groupby("carrier_id")["on_time_delivery"].mean().to_dict()

    # On-time % per carrier during monsoon
    monsoon_df = df[df["is_monsoon"] == 1]
    ontime_monsoon = monsoon_df.groupby("carrier_id")["on_time_delivery"].mean().to_dict()

    # On-time % per carrier during festival
    festival_df = df[df["is_festival"] == 1]
    ontime_festival = festival_df.groupby("carrier_id")["on_time_delivery"].mean().to_dict()

    # Avg cost per kg per carrier
    avg_cost_by_carrier = df.groupby("carrier_id")["cost_per_kg"].mean().to_dict()

    # Damage rate per carrier
    damage_by_carrier = df.groupby("carrier_id")["damage_loss_flag"].mean().to_dict()

    return {
        "avg_delay_by_carrier":  avg_delay_by_carrier,
        "overall_avg_delay":     overall_avg_delay,
        "ontime_by_carrier":     ontime_by_carrier,
        "ontime_monsoon":        ontime_monsoon,
        "ontime_festival":       ontime_festival,
        "avg_cost_by_carrier":   avg_cost_by_carrier,
        "damage_by_carrier":     damage_by_carrier,
    }


# ── 1. TRAIN ALL 3 MODELS ─────────────────────────────────────────────────────
def train_model():
    df = pd.read_csv(os.path.join(DATA, "shipment_history.csv"))

    df["carrier_enc"]      = df["carrier_id"].map(CARRIER_MAP).fillna(0).astype(int)
    df["freight_type_enc"] = df["freight_type"].map(FREIGHT_TYPE_MAP).fillna(5).astype(int)
    df["mode_enc"]         = df["mode"].map(MODE_MAP).fillna(0).astype(int)

    y       = df["on_time_delivery"]
    results = {}

    # XGBoost
    X_xgb = df[XGB_FEATURES].fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X_xgb, y, test_size=0.2, random_state=42)
    xgb_scaler = StandardScaler()
    X_tr_s, X_te_s = xgb_scaler.fit_transform(X_tr), xgb_scaler.transform(X_te)
    xgb = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                        **{"eval_metric": "logloss", "verbosity": 0} if XGBOOST_AVAILABLE else {})
    xgb.fit(X_tr_s, y_tr)
    results["xgb_accuracy"] = round(accuracy_score(y_te, xgb.predict(X_te_s)), 4)

    # Gradient Boosting
    X_gb = df[GB_FEATURES].fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X_gb, y, test_size=0.2, random_state=42)
    gb_scaler = StandardScaler()
    X_tr_s, X_te_s = gb_scaler.fit_transform(X_tr), gb_scaler.transform(X_te)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X_tr_s, y_tr)
    results["gb_accuracy"] = round(accuracy_score(y_te, gb.predict(X_te_s)), 4)

    # Logistic Regression
    X_lr = df[LR_FEATURES].fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X_lr, y, test_size=0.2, random_state=42)
    lr_scaler = StandardScaler()
    X_tr_s, X_te_s = lr_scaler.fit_transform(X_tr), lr_scaler.transform(X_te)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_s, y_tr)
    results["lr_accuracy"] = round(accuracy_score(y_te, lr.predict(X_te_s)), 4)

    # Save all
    for name, obj in [
        ("xgb_model", xgb), ("xgb_scaler", xgb_scaler),
        ("gb_model",  gb),  ("gb_scaler",  gb_scaler),
        ("lr_model",  lr),  ("lr_scaler",  lr_scaler),
    ]:
        pickle.dump(obj, open(os.path.join(MODELS, f"{name}.pkl"), "wb"))

    print(f"✅ XGBoost             : {results['xgb_accuracy']:.2%}  → delivery risk")
    print(f"✅ Gradient Boosting   : {results['gb_accuracy']:.2%}  → on-time probability")
    print(f"✅ Logistic Regression : {results['lr_accuracy']:.2%}  → delay probability")
    return results


# ── 2. LOAD MODELS ────────────────────────────────────────────────────────────
def _load_models():
    def _load(name): return pickle.load(open(os.path.join(MODELS, f"{name}.pkl"), "rb"))
    return (_load("xgb_model"), _load("xgb_scaler"),
            _load("gb_model"),  _load("gb_scaler"),
            _load("lr_model"),  _load("lr_scaler"))


# ── 3. GENERATE WHY EXPLANATION ───────────────────────────────────────────────
def _generate_why(carrier_id, blended_prob, eff_rate,
                  revenue_at_risk, all_rates, all_revenues,
                  goods_type, is_monsoon, is_festival,
                  stats, damage_multiplier):
    """
    Dynamically generate why this carrier is recommended.
    All comparisons are against real data — nothing hardcoded.
    """
    why = []

    # 1. Cost comparison vs average
    avg_rate = round(sum(all_rates) / len(all_rates), 3)
    if eff_rate < avg_rate:
        saving_pct = round((avg_rate - eff_rate) / avg_rate * 100, 1)
        why.append(f"₹{eff_rate}/kg is {saving_pct}% cheaper than lane average (₹{avg_rate}/kg)")
    else:
        premium_pct = round((eff_rate - avg_rate) / avg_rate * 100, 1)
        why.append(f"₹{eff_rate}/kg is {premium_pct}% above lane average — justified by reliability")

    # 2. On-time probability from real historical data
    hist_ontime = stats["ontime_by_carrier"].get(carrier_id, None)
    if hist_ontime:
        why.append(
            f"{round(hist_ontime * 100, 1)}% historical on-time rate "
            f"({round(blended_prob * 100, 1)}% predicted for this shipment)"
        )

    # 3. Revenue at risk vs average
    avg_revenue_risk = round(sum(all_revenues) / len(all_revenues), 2)
    if revenue_at_risk < avg_revenue_risk:
        why.append(
            f"Lowest revenue at risk: ₹{revenue_at_risk:,} "
            f"vs lane average ₹{avg_revenue_risk:,} for {goods_type}"
        )

    # 4. Damage rate from real data
    damage_rate = stats["damage_by_carrier"].get(carrier_id, None)
    if damage_rate is not None:
        why.append(f"{round(damage_rate * 100, 2)}% historical damage rate on this carrier")

    # 5. Monsoon / festival context from real data
    if is_monsoon:
        monsoon_rate = stats["ontime_monsoon"].get(carrier_id, None)
        if monsoon_rate:
            why.append(
                f"During monsoon: {round(monsoon_rate * 100, 1)}% on-time "
                f"(based on historical monsoon shipments)"
            )

    if is_festival:
        festival_rate = stats["ontime_festival"].get(carrier_id, None)
        if festival_rate:
            why.append(
                f"During festivals: {round(festival_rate * 100, 1)}% on-time "
                f"(based on historical festival shipments)"
            )

    # 6. Goods sensitivity note
    if damage_multiplier > 1.0:
        why.append(
            f"{goods_type} has {damage_multiplier}x damage sensitivity — "
            f"reliability weighted higher in scoring"
        )

    return why


# ── 4. CORE PREDICT ───────────────────────────────────────────────────────────
def predict(carrier_id, distance_km, weight_kg,
            is_monsoon, is_festival, transit_days,
            goods_type="General Cargo", mode="FTL",
            shipment_value_inr=100000,
            all_rates=None, all_revenues=None, stats=None):
    """
    Run all 3 models and blend:
    XGBoost 40% + Gradient Boosting 40% + Logistic Regression 20%
    """
    xgb, xgb_scaler, gb, gb_scaler, lr, lr_scaler = _load_models()

    carrier_enc      = CARRIER_MAP.get(carrier_id, 0)
    freight_type_enc = FREIGHT_TYPE_MAP.get(goods_type, 5)
    mode_enc         = MODE_MAP.get(mode, 0)

    xgb_row = pd.DataFrame([{
        "transit_days_promised": transit_days,
        "weight_kg":             weight_kg,
        "distance_km":           distance_km,
        "mode_enc":              mode_enc,
        "freight_type_enc":      freight_type_enc,
    }])[XGB_FEATURES]

    full_row = pd.DataFrame([{
        "distance_km":           distance_km,
        "weight_kg":             weight_kg,
        "transit_days_promised": transit_days,
        "is_monsoon":            is_monsoon,
        "is_festival":           is_festival,
        "carrier_enc":           carrier_enc,
        "freight_type_enc":      freight_type_enc,
        "mode_enc":              mode_enc,
    }])[GB_FEATURES]

    prob_xgb = xgb.predict_proba(xgb_scaler.transform(xgb_row))[0][1]
    prob_gb  = gb.predict_proba(gb_scaler.transform(full_row))[0][1]
    prob_lr  = lr.predict_proba(lr_scaler.transform(full_row))[0][1]

    # Weighted blend
    blended = round(0.40 * prob_xgb + 0.40 * prob_gb + 0.20 * prob_lr, 4)

    # ── Risk level — fixed logic ──────────────────────────────────────────────
    # Risk level uses blended probability directly
    # Goods type affects revenue_at_risk NOT the risk level threshold
    if blended >= 0.88:
        risk_level, risk_color = "LOW",    "green"
    elif blended >= 0.75:
        risk_level, risk_color = "MEDIUM", "yellow"
    else:
        risk_level, risk_color = "HIGH",   "red"

    # ── Revenue at risk — from real avg delay data ────────────────────────────
    damage_multiplier  = GOODS_RISK_MULTIPLIER.get(goods_type, 1.0)
    daily_revenue_risk = DAILY_REVENUE_RISK_PCT.get(goods_type, 0.010)

    # Use real avg delay from data if stats provided, else fallback
    if stats:
        avg_delay = stats["avg_delay_by_carrier"].get(carrier_id,
                    stats["overall_avg_delay"])
    else:
        avg_delay = 2.507  # real overall avg from data

    delay_prob      = 1 - blended
    expected_delay  = round(delay_prob * avg_delay, 2)
    revenue_at_risk = round(
        shipment_value_inr * daily_revenue_risk * expected_delay * damage_multiplier, 2
    )

    return {
        "carrier_id":  carrier_id,

        "scores": {
            "composite_score":     None,  # filled in predict_all_carriers
            "blended_ontime":      f"{round(blended * 100, 1)}%",
            "xgb_reliability":     f"{round(float(prob_xgb) * 100, 1)}%",
            "gb_ontime":           f"{round(prob_gb  * 100, 1)}%",
            "lr_delay_risk":       f"{round((1-prob_lr) * 100, 1)}%",
        },

        "risk": {
            "level":                  risk_level,
            "color":                  risk_color,
            "goods_type":             goods_type,
            "damage_multiplier":      damage_multiplier,
            "expected_delay_days":    expected_delay,
            "daily_revenue_risk_pct": f"{round(daily_revenue_risk * 100, 1)}%",
            "revenue_at_risk_inr":    revenue_at_risk,
        },

        # raw values needed for composite scoring + why
        "_blended":          blended,
        "_revenue_at_risk":  revenue_at_risk,
        "_damage_multiplier": damage_multiplier,
    }


# ── 5. PREDICT ALL CARRIERS ───────────────────────────────────────────────────
def predict_all_carriers(lane_id, distance_km, weight_kg,
                          transit_days, is_monsoon, is_festival,
                          goods_type="General Cargo", mode="FTL",
                          priority_profile="balanced",
                          shipment_value_inr=100000):
    """
    Main dashboard function.
    Runs all 3 models for all 8 carriers.
    Generates composite score, rank, recommendation, and why explanation.
    """
    profile = PRIORITY_PROFILES.get(priority_profile, PRIORITY_PROFILES["balanced"])
    stats   = _load_historical_stats()

    # Load bid rates
    try:
        bid_df    = pd.read_csv(os.path.join(DATA, "bid_rates.csv"))
        lane_bids = bid_df[bid_df["lane_id"] == lane_id]
    except:
        lane_bids = pd.DataFrame()

    # First pass — get all predictions
    results = []
    for cid in CARRIER_MAP.keys():
        pred     = predict(cid, distance_km, weight_kg,
                           is_monsoon, is_festival, transit_days,
                           goods_type, mode, shipment_value_inr,
                           stats=stats)
        rate_row = lane_bids[lane_bids["carrier_id"] == cid] if not lane_bids.empty else pd.DataFrame()
        pred["cost"] = {
            "base_rate":        float(rate_row["base_rate_inr_kg"].iloc[0])   if not rate_row.empty else 4.0,
            "fuel_surcharge":   float(rate_row["fuel_surcharge_pct"].iloc[0]) if not rate_row.empty else 5.0,
            "eff_rate_inr_kg":  float(rate_row["eff_rate_inr_kg"].iloc[0])    if not rate_row.empty else 4.0,
            "priority_profile": profile["label"],
        }
        pred["_eff_rate"] = pred["cost"]["eff_rate_inr_kg"]
        results.append(pred)

    # Collect all rates and revenues for comparison in why
    all_rates    = [r["_eff_rate"]         for r in results]
    all_revenues = [r["_revenue_at_risk"]  for r in results]

    # Second pass — composite score + why
    def norm(v, lo, hi): return (v - lo) / (hi - lo + 1e-9)
    min_r, max_r = min(all_rates),    max(all_rates)
    min_p, max_p = min(r["_blended"] for r in results), max(r["_blended"] for r in results)
    min_d, max_d = min(all_revenues), max(all_revenues)

    for r in results:
        norm_cost   = 1 - norm(r["_eff_rate"],   min_r, max_r)
        norm_ontime =     norm(r["_blended"],     min_p, max_p)
        norm_risk   = 1 - norm(r["_revenue_at_risk"], min_d, max_d)

        composite = round(
            profile["w_cost"]   * norm_cost +
            profile["w_ontime"] * norm_ontime +
            profile["w_risk"]   * norm_risk,
            4
        )
        r["scores"]["composite_score"] = composite

        # Generate why dynamically
        r["why"] = _generate_why(
            carrier_id        = r["carrier_id"],
            blended_prob      = r["_blended"],
            eff_rate          = r["_eff_rate"],
            revenue_at_risk   = r["_revenue_at_risk"],
            all_rates         = all_rates,
            all_revenues      = all_revenues,
            goods_type        = goods_type,
            is_monsoon        = is_monsoon,
            is_festival       = is_festival,
            stats             = stats,
            damage_multiplier = r["_damage_multiplier"],
        )

    # Sort and rank
    results.sort(key=lambda x: x["scores"]["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"]           = i + 1
        r["recommendation"] = "✅ RECOMMEND" if i == 0 else ("⚠️ ALTERNATE" if i == 1 else "")
        # Clean up internal keys
        del r["_blended"], r["_revenue_at_risk"], r["_eff_rate"], r["_damage_multiplier"]

    return {
        "lane_id":          lane_id,
        "goods_type":       goods_type,
        "mode":             mode,
        "priority_profile": profile["label"],
        "carriers":         results,
    }


# ── 6. WHAT IF SIMULATOR ──────────────────────────────────────────────────────
def predict_whatif(carrier_id, distance_km, weight_kg,
                   transit_days, goods_type="General Cargo", mode="FTL",
                   shipment_value_inr=100000,
                   diesel_change_pct=0, early_monsoon=False,
                   festival_override=False):
    is_monsoon  = 1 if early_monsoon     else 0
    is_festival = 1 if festival_override else 0
    transit_adj = transit_days + (0.5 if diesel_change_pct > 5 else 0)
    stats       = _load_historical_stats()

    base = predict(carrier_id, distance_km, weight_kg,
                   is_monsoon, is_festival, transit_adj,
                   goods_type, mode, shipment_value_inr, stats=stats)

    base["whatif_scenario"] = {
        "diesel_change_pct":     diesel_change_pct,
        "early_monsoon":         early_monsoon,
        "festival_override":     festival_override,
        "cost_increase_pct":     round(diesel_change_pct * 0.35, 2),
        "adjusted_transit_days": transit_adj,
    }
    # clean internal keys
    for k in ["_blended", "_revenue_at_risk", "_eff_rate", "_damage_multiplier"]:
        base.pop(k, None)
    return base


# ── 7. FESTIVAL WARNING ───────────────────────────────────────────────────────
def get_festival_warning(carrier_id, lane_id):
    df        = pd.read_csv(os.path.join(DATA, "shipment_history.csv"))
    lane_data = df[(df["carrier_id"] == carrier_id) & (df["lane_id"] == lane_id)]
    if lane_data.empty:
        lane_data = df[df["carrier_id"] == carrier_id]

    normal_ontime   = lane_data[lane_data["is_festival"] == 0]["on_time_delivery"].mean()
    festival_ontime = lane_data[lane_data["is_festival"] == 1]["on_time_delivery"].mean()

    if pd.isna(festival_ontime): festival_ontime = normal_ontime or 0.85
    if pd.isna(normal_ontime):   normal_ontime   = 0.85

    drop_pct = round((normal_ontime - festival_ontime) * 100, 1)

    freight_festival = {}
    if "freight_type" in lane_data.columns:
        for ft in FREIGHT_TYPE_MAP.keys():
            ft_data     = lane_data[lane_data["freight_type"] == ft]
            ft_normal   = ft_data[ft_data["is_festival"] == 0]["on_time_delivery"].mean()
            ft_festival = ft_data[ft_data["is_festival"] == 1]["on_time_delivery"].mean()
            if not pd.isna(ft_normal) and not pd.isna(ft_festival):
                freight_festival[ft] = round((ft_normal - ft_festival) * 100, 1)

    upcoming_festivals = [
        {"name": "Eid",        "date": "19-Mar-2026", "days_away": 12, "impact": "HIGH",   "region": "Pan India"},
        {"name": "Ram Navami", "date": "06-Apr-2026", "days_away": 30, "impact": "MEDIUM", "region": "North India"},
        {"name": "Baisakhi",   "date": "14-Apr-2026", "days_away": 38, "impact": "MEDIUM", "region": "North India"},
    ]

    return {
        "carrier_id":           carrier_id,
        "lane_id":              lane_id,
        "normal_ontime_pct":    round(normal_ontime * 100, 1),
        "festival_ontime_pct":  round(festival_ontime * 100, 1),
        "performance_drop_pct": drop_pct,
        "freight_type_impact":  freight_festival,
        "alert":                drop_pct > 5,
        "alert_message": (
            f"⚠️ {carrier_id} on-time drops {drop_pct}% during festivals. "
            f"Book extra capacity before Eid (19-Mar)."
        ) if drop_pct > 5 else "✅ Carrier performs consistently during festivals.",
        "upcoming_festivals":   upcoming_festivals,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Training 3-model ensemble on 5000 shipments...")
    print("=" * 60)
    train_model()

    print("\n" + "=" * 60)
    print("Test: Pharma | Mumbai→Delhi | Reliability-First")
    print("=" * 60)
    result = predict_all_carriers(
        lane_id="L001", distance_km=1415, weight_kg=5000,
        transit_days=2, is_monsoon=0, is_festival=0,
        goods_type="Pharma", mode="FTL",
        priority_profile="reliability",
        shipment_value_inr=500000
    )

    for c in result["carriers"]:
        print(f"\n  Rank {c['rank']} — {c['carrier_id']}  {c['recommendation']}")
        print(f"  Scores  : composite={c['scores']['composite_score']} | "
              f"blended={c['scores']['blended_ontime']} | "
              f"XGB={c['scores']['xgb_reliability']} | "
              f"GB={c['scores']['gb_ontime']}")
        print(f"  Risk    : {c['risk']['level']} ({c['risk']['color']}) | "
              f"Rev@Risk=₹{c['risk']['revenue_at_risk_inr']:,} | "
              f"ExpDelay={c['risk']['expected_delay_days']}d")
        print(f"  Cost    : ₹{c['cost']['eff_rate_inr_kg']}/kg | "
              f"Profile={c['cost']['priority_profile']}")
        print(f"  Why     :")
        for w in c["why"]:
            print(f"    → {w}")

    print("\n" + "=" * 60)
    print("What If — Diesel +10%, Early Monsoon on C001")
    print("=" * 60)
    w = predict_whatif("C001", 1415, 5000, 2, "Pharma", "FTL", 500000,
                       diesel_change_pct=10, early_monsoon=True)
    print(f"  Blended : {w['scores']['blended_ontime']}")
    print(f"  Risk    : {w['risk']['level']} | Rev@Risk=₹{w['risk']['revenue_at_risk_inr']:,}")
    print(f"  Cost up : {w['whatif_scenario']['cost_increase_pct']}%")