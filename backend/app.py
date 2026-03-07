


"""
FreightIQ — Flask Backend
Wires: scoring.py → optimize.py → market_intel_agent.py

Endpoints
─────────
GET  /api/lanes                  → lane catalog (origin, destination, distance, transit)
POST /api/shipment/analyze       → full pipeline: score + optimize + market intel
GET  /api/market/intelligence    → market factors only (Gemini or demo)
GET  /api/health                 → health check

Run:
    pip install flask flask-cors pulp google-genai pandas scikit-learn xgboost
    DEMO_MODE=true python app.py
    # or for live Gemini:
    GEMINI_API_KEY=your_key DEMO_MODE=false python app.py
"""

import os
import sys
import traceback
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from scoring import get_lane_catalog, run_scoring
from optimize import optimize_allocation
from market_intel_agent import get_market_intelligence_snapshot
from ml_engine import train_model   # trains 3-model ensemble on startup

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Train models on startup ────────────────────────────────────────────────────
print("=" * 56)
print("🚀  FreightIQ backend starting …")
print("=" * 56)
try:
    metrics = train_model()
    print(f"   ✅  XGBoost            : {metrics['xgb_accuracy']:.2%}")
    print(f"   ✅  Gradient Boosting  : {metrics['gb_accuracy']:.2%}")
    print(f"   ✅  Logistic Regression: {metrics['lr_accuracy']:.2%}")
except Exception as exc:
    print(f"   ⚠   Model training failed: {exc}")
    print("       Continuing — models will train on first request.")
print("=" * 56)


# ── Helpers ────────────────────────────────────────────────────────────────────
def success(data):
    return jsonify({"success": True, "data": data})


def failure(msg, code=400):
    return jsonify({"success": False, "error": str(msg)}), code


def safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "demo_mode": os.getenv("DEMO_MODE", "true").lower() == "true",
    })


# ── Lanes ──────────────────────────────────────────────────────────────────────
@app.get("/api/lanes")
def lanes():
    """Return lane catalog used to populate origin/destination dropdowns."""
    try:
        catalog = get_lane_catalog()
        return jsonify({"success": True, "lanes": catalog})
    except Exception as exc:
        return failure(traceback.format_exc())


# ── Market intelligence only ───────────────────────────────────────────────────
@app.get("/api/market/intelligence")
def market_intelligence():
    try:
        bust = request.args.get("bust", "0") == "1"
        data = get_market_intelligence_snapshot(bust_cache=bust)
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        return failure(str(exc))


# ── Full shipment analysis (scoring + optimization + market intel) ─────────────
@app.post("/api/shipment/analyze")
def analyze():
    """
    POST body (JSON):
        lane_id             str   required  e.g. "L001"
        goods_type          str   optional  default "General Cargo"
        mode                str   optional  "FTL" | "LTL"
        priority_profile    str   optional  "balanced"|"cost"|"reliability"|"risk"|"pharma"|"sales"
        weight_kg           float required
        shipment_value_inr  float optional  default 100000
        min_reliability     float optional  default 0.9
        is_monsoon          int   optional  0|1  (auto-detected if omitted)
        is_festival         int   optional  0|1  (auto-detected if omitted)
    """
    try:
        body = request.get_json(force=True) or {}

        lane_id    = body.get("lane_id", "").strip()
        weight_kg  = safe_float(body.get("weight_kg"), 0)

        if not lane_id:
            return failure("lane_id is required", 422)
        if weight_kg <= 0:
            return failure("weight_kg must be positive", 422)

        goods_type          = str(body.get("goods_type",  "General Cargo"))
        mode                = str(body.get("mode",        "FTL"))
        priority_profile    = str(body.get("priority_profile", "balanced"))
        shipment_value_inr  = safe_float(body.get("shipment_value_inr"), 100000)
        min_reliability     = safe_float(body.get("min_reliability"),    0.9)

        month = datetime.datetime.now().month
        is_monsoon  = safe_int(body.get("is_monsoon"),  1 if month in (6,7,8,9) else 0)
        is_festival = safe_int(body.get("is_festival"), 0)

        # 1. Market intelligence
        market = get_market_intelligence_snapshot()

        # 2. Carrier scoring
        scoring_result = run_scoring(
            lane_id            = lane_id,
            weight_kg          = weight_kg,
            goods_type         = goods_type,
            mode               = mode,
            priority_profile   = priority_profile,
            shipment_value_inr = shipment_value_inr,
            is_monsoon         = is_monsoon,
            is_festival        = is_festival,
        )

        # Inject market multiplier into each carrier cost
        composite_mult = float(market.get("composite_multiplier", 1.0))
        for carrier in scoring_result.get("carriers", []):
            cost = carrier.get("cost", {})
            base_rate = float(cost.get("eff_rate_inr_kg", cost.get("base_rate", 4.0)))
            adjusted  = round(base_rate * composite_mult, 4)
            cost["adjusted_eff_rate_inr_kg"]  = adjusted
            cost["market_multiplier"]         = composite_mult
            cost["estimated_total_cost_inr"]  = round(adjusted * weight_kg, 2)
            cost["weight_kg"]                 = weight_kg
            carrier["cost"] = cost

        # 3. LP optimisation
        try:
            optimization = optimize_allocation(
                scoring_output  = scoring_result,
                total_weight_kg = weight_kg,
                min_reliability = min_reliability,
            )
        except Exception as opt_exc:
            optimization = {"status": "error", "error": str(opt_exc)}

        carriers = scoring_result.get("carriers", [])
        best     = carriers[0] if carriers else None

        return jsonify({
            "success": True,
            "shipment": {
                "lane_id":            lane_id,
                "origin":             scoring_result.get("origin"),
                "destination":        scoring_result.get("destination"),
                "distance_km":        scoring_result.get("distance_km"),
                "transit_days":       scoring_result.get("transit_days"),
                "goods_type":         goods_type,
                "mode":               mode,
                "priority_profile":   priority_profile,
                "weight_kg":          weight_kg,
                "shipment_value_inr": shipment_value_inr,
                "is_monsoon":         is_monsoon,
                "is_festival":        is_festival,
            },
            "market_intelligence":  market,
            "carrier_scoring":      scoring_result,
            "best_recommendation":  best,
            "optimization":         optimization,
        })

    except ValueError as ve:
        return failure(str(ve), 422)
    except Exception:
        return failure(traceback.format_exc(), 500)


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🌐  http://0.0.0.0:{port}")
    print(f"    DEMO_MODE  = {os.getenv('DEMO_MODE','true')}")
    print(f"    GEMINI_KEY = {'set' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)