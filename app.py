


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
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.scoring import get_lane_catalog, run_scoring
from backend.ml_engine import predict_all_carriers, train_model  # trains 3-model ensemble on startup
from backend.optimize import optimize_allocation
from backend.market_intel_agent import get_market_intelligence_snapshot
from backend.openai_analysis_agent import run_analysis as run_openai_analysis
from backend.maps_service import get_directions, get_directions_latlng

# ── App setup ─────────────────────────────────────────────────────────────────
from flask import Flask, render_template

app = Flask(__name__,
            template_folder="frontend/templates",
            static_folder="frontend/static")
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route("/")
def home():
    return render_template("index.html")

# ── Train models on startup ────────────────────────────────────────────────────
print("=" * 56)
print(">>  FreightIQ backend starting ...")
print("=" * 56)
try:
    metrics = train_model()
    print(f"   [OK]  XGBoost            : {metrics['xgb_accuracy']:.2%}")
    print(f"   [OK]  Gradient Boosting  : {metrics['gb_accuracy']:.2%}")
    print(f"   [OK]  Logistic Regression: {metrics['lr_accuracy']:.2%}")
except Exception as exc:
    print(f"   [!!]  Model training failed: {exc}")
    print("         Continuing -- models will train on first request.")
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
        "google_maps_key": os.getenv("GOOGLE_MAPS_API_KEY", ""),
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


# ── Google Maps Directions ─────────────────────────────────────────────────────
@app.post("/api/maps/directions")
def maps_directions():
    """Get real driving distance and transit time between two points."""
    try:
        body = request.get_json(force=True) or {}
        # Support both city names and exact lat/lng
        if body.get("origin_lat") and body.get("dest_lat"):
            result = get_directions_latlng(
                origin_lat=float(body["origin_lat"]),
                origin_lng=float(body["origin_lng"]),
                dest_lat=float(body["dest_lat"]),
                dest_lng=float(body["dest_lng"]),
                origin_label=body.get("origin", ""),
                dest_label=body.get("destination", ""),
            )
        else:
            origin = body.get("origin", "").strip()
            destination = body.get("destination", "").strip()
            if not origin or not destination:
                return failure("origin and destination are required", 422)
            result = get_directions(origin, destination)
        return jsonify({"success": True, "directions": result})
    except Exception as exc:
        return failure(str(exc))


# ── Market intelligence only ───────────────────────────────────────────────────
@app.get("/api/market/intelligence")
def market_intelligence():
    try:
        bust = request.args.get("bust", "0") == "1"
        origin = request.args.get("origin", "Delhi")
        destination = request.args.get("destination", "Mumbai")
        data = get_market_intelligence_snapshot(bust_cache=bust, origin=origin, destination=destination)
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

        # Extract origin/destination for route-aware market intel
        origin_city = str(body.get("origin", "Delhi"))
        dest_city   = str(body.get("destination", "Mumbai"))

        # 1. Market intelligence (route-aware: state-specific festivals & weather)
        market = get_market_intelligence_snapshot(origin=origin_city, destination=dest_city)

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

        # 4. OpenAI Analysis — summarise & recommend
        shipment_info = {
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
        }
        try:
            ai_analysis = run_openai_analysis(
                shipment=shipment_info,
                market=market,
                carriers=carriers,
                best=best,
                optimization=optimization,
            )
        except Exception as ai_exc:
            ai_analysis = {"error": str(ai_exc), "agent": "error"}

        return jsonify({
            "success": True,
            "shipment":             shipment_info,
            "market_intelligence":  market,
            "carrier_scoring":      scoring_result,
            "best_recommendation":  best,
            "optimization":         optimization,
            "ai_analysis":          ai_analysis,
        })

    except ValueError as ve:
        return failure(str(ve), 422)
    except Exception:
        return failure(traceback.format_exc(), 500)


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n[*]  http://0.0.0.0:{port}")
    print(f"     DEMO_MODE   = {os.getenv('DEMO_MODE','true')}")
    print(f"     GEMINI_KEY  = {'set' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
    print(f"     OPENAI_KEY  = {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
    