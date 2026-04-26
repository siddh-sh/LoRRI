import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

import backend.transport_network as tn
import backend.ml_engine as ml
import backend.market_intel_agent as mkt
import backend.optimize as opt
import backend.openai_analysis_agent as ai
import backend.news as news_api
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'), 
            static_folder=os.path.join(BASE_DIR, 'frontend', 'static'))
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true",
        "google_maps_key": os.getenv("GOOGLE_MAPS_API_KEY", "")
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    cities = []
    for city, coords in tn.CITY_COORDS.items():
        cities.append({
            "name": city,
            "has_railway": city in tn.CITY_RAIL_CODE,
            "has_airport": city in tn.CITY_IATA,
            "coords": {"lat": coords[0], "lng": coords[1]}
        })
    return jsonify({"success": True, "cities": cities})

@app.route('/api/route', methods=['POST'])
def fetch_route():
    data = request.json
    origin = data.get("origin")
    dest = data.get("destination")
    mode = data.get("mode", "roadways")
    
    route = tn.get_route(origin, dest, mode)
    if not route:
        return jsonify({"success": False, "error": "Route could not be calculated"}), 400
        
    return jsonify({"success": True, "route": route})

@app.route('/api/shipment/analyze', methods=['POST'])
def analyze_shipment():
    data = request.json
    origin = data.get("origin")
    dest = data.get("destination")
    weight_kg = float(data.get("weight_kg", 0))
    transport_mode = data.get("transport_mode", "roadways")
    route_mode = transport_mode
    
    if transport_mode == "roadways":
        road_freight_mode = data.get("road_freight_mode", "FTL").upper()
        if weight_kg > 3000:
            road_freight_mode = "FTL"
        ml_transport_mode = road_freight_mode
    else:
        ml_transport_mode = transport_mode
        
    priority_profile = data.get("priority_profile", "balanced")
    goods_type = data.get("goods_type", "FMCG")
    shipment_value = float(data.get("shipment_value_inr", 0))
    min_reliability = float(data.get("min_reliability", 0.8))
    
    # 1. Route Calculation
    route = tn.get_route(origin, dest, route_mode)
    if not route:
        return jsonify({"success": False, "error": f"Invalid route pair {origin} -> {dest}"}), 400
        
    distance_km = route["distance_km"]
    transit_days = route["transit_days"]
    
    # 2. Market Intelligence
    market_snapshot = mkt.get_market_intelligence_snapshot()
    # In live scenario, extract route-specific from global snapshot
    
    is_monsoon = False  # Derived ideally from market_snapshot
    mult = market_snapshot.get("composite_multiplier", 1.0)
    
    # 3. Carrier Scoring
    carriers = ml.predict_all_carriers(
        distance_km=distance_km,
        weight_kg=weight_kg,
        transit_days=transit_days,
        goods_type=goods_type,
        transport_mode=ml_transport_mode,
        priority_profile=priority_profile,
        is_monsoon=is_monsoon,
        shipment_value=shipment_value
    )
    
    # 4. Inject Market Multiplier
    # Costs are usually adjusted upward when market is constrained
    for c in carriers:
        base_est = c["cost"]["estimated_total_cost_inr"]
        c["cost"]["market_adjusted_cost"] = round(base_est * mult, 2)
        c["cost"]["adjusted_eff_rate"] = round((base_est * mult) / weight_kg, 2)
    
    best_carrier = carriers[0] if carriers else None
    
    # 5. LP Optimization
    scoring_output = {
        "lane_id": f"{origin}|{dest}",
        "carriers": carriers
    }
    optimization = opt.optimize_allocation(scoring_output, weight_kg, min_reliability)

    
    # 6. AI Agent Analysis
    ai_analysis = ai.run_analysis(
        shipment={"lane_id": f"{origin}|{dest}", "distance_km": distance_km, "transit_days": transit_days, "weight_kg": weight_kg, "goods_type": goods_type, "priority_profile": priority_profile},
        market=market_snapshot,
        carriers=carriers,
        best=best_carrier,
        optimization=optimization
    )
    
    return jsonify({
        "success": True,
        "shipment": {
            "origin": origin,
            "destination": dest,
            "transport_mode": transport_mode,
            "weight_kg": weight_kg,
            "distance_km": distance_km,
            "transit_days_est": transit_days,
            "path_nodes": route["path_nodes"],
            "via_hubs": route["via_hubs"]
        },
        "market_intelligence": market_snapshot,
        "carrier_scoring": {"carriers": carriers},
        "best_recommendation": best_carrier,
        "optimization": optimization,
        "ai_analysis": ai_analysis
    })

@app.route('/api/market/intelligence', methods=['GET'])
def get_market_intelligence():
    intel = mkt.get_market_intelligence_snapshot()
    return jsonify({"success": True, "data": intel})

@app.route('/api/news', methods=['GET'])
def get_logistics_news():
    news_items = news_api.fetch_logistics_news(limit=5)
    return jsonify({"success": True, "news": news_items})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
