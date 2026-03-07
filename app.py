"""
Procurement Co-Pilot — Flask Backend API
"""
import os, sys, json, io, traceback
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify

import pandas as pd
import numpy as np

from backend.ml_engine import (
    build_carrier_scores, build_lane_scores,
    train_risk_model, predict_ontime,
    optimize_allocation, MARKET_DATA,
    DATA, MODEL
)

app = Flask(__name__)
app.after_request(lambda r: (r.headers.update({"Access-Control-Allow-Origin":"*","Access-Control-Allow-Headers":"Content-Type","Access-Control-Allow-Methods":"GET,POST,OPTIONS"}), r)[1])

print("🚀 Training models on startup…")
try:
    _metrics = train_risk_model()
    print(f"   ✅ GB={_metrics['gb_accuracy']}  LR={_metrics['lr_accuracy']}")
except Exception as e:
    print(f"   ⚠ Training error: {e}")

def ok(data):  return jsonify({"status":"ok",  "data": data})
def err(msg):  return jsonify({"status":"error","message": str(msg)}), 400

def safe(obj):
    if isinstance(obj, dict):   return {k: safe(v) for k,v in obj.items()}
    if isinstance(obj, list):   return [safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    try:
        if pd.isna(obj): return None
    except: pass
    return obj

@app.route("/api/health")
def health():
    return ok({"message": "Procurement Co-Pilot API running", "version": "1.0"})

@app.route("/api/carriers/scores")
def carrier_scores():
    try:
        df = build_carrier_scores()
        return ok(safe(df.to_dict("records")))
    except Exception as e:
        return err(traceback.format_exc())

@app.route("/api/lanes/scores")
def lane_scores():
    try:
        df = build_lane_scores()
        result = {}
        for lane_id, grp in df.groupby("lane_id"):
            top = grp.sort_values("composite_score", ascending=False).head(3)
            result[lane_id] = safe(top.to_dict("records"))
        return ok(result)
    except Exception as e:
        return err(traceback.format_exc())

@app.route("/api/predict/ontime", methods=["POST"])
def predict():
    try:
        body = request.json or {}
        result = predict_ontime(
            float(body.get("distance_km",1000)), float(body.get("weight_kg",5000)),
            int(body.get("transit_days",2)),     int(body.get("is_monsoon",0)),
            int(body.get("is_festival",0)),      str(body.get("carrier_id","C001")),
        )
        return ok(safe(result))
    except Exception as e:
        return err(str(e))

@app.route("/api/bids/score", methods=["POST"])
def score_bids():
    try:
        bids = request.json.get("bids", [])
        if not bids: return err("No bids provided")
        rates = [b["rate"] for b in bids]
        min_r, max_r = min(rates), max(rates)
        scored = []
        for b in bids:
            np_ = 1-(b["rate"]-min_r)/(max_r-min_r+1e-9)
            ot  = b.get("ontime_pct",0.85)
            rsk = b.get("risk_score",0.10)
            comp= round(0.4*np_+0.3*ot+0.3*(1-rsk),4)
            try:
                pred = predict_ontime(b.get("distance_km",800),b.get("weight_kg",5000),
                                      b.get("transit_days",2),b.get("is_monsoon",0),
                                      b.get("is_festival",0),b.get("carrier_id","C001"))
                ml_p = pred["ontime_probability"]
            except: ml_p = ot
            scored.append({**b,"norm_price":round(float(np_),4),"composite_score":comp,
                           "ml_ontime_prob":ml_p,
                           "market_adj_rate":round(b["rate"]*MARKET_DATA["composite_multiplier"],3),
                           "market_multiplier":MARKET_DATA["composite_multiplier"]})
        scored.sort(key=lambda x: -x["composite_score"])
        for i,s in enumerate(scored):
            s["rank"]=i+1
            s["recommendation"]="RECOMMEND" if i==0 else ("ALTERNATE" if i==1 else "PASS")
        return ok(safe(scored))
    except Exception as e:
        return err(traceback.format_exc())

@app.route("/api/optimize/allocation", methods=["POST"])
def optimize():
    try:
        body = request.json or {}
        result = optimize_allocation(body.get("bid_data",[]), body.get("lane_volumes",{}))
        return ok(safe(result))
    except Exception as e:
        return err(traceback.format_exc())

@app.route("/api/upload/csv", methods=["POST"])
def upload_csv():
    try:
        file = request.files.get("file")
        if not file: return err("No file uploaded")
        content = file.read().decode("utf-8","ignore")
        df = pd.read_csv(io.StringIO(content))
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        col_map = {}
        for col in df.columns:
            if "carrier" in col and "id" in col:      col_map["carrier_id"]=col
            elif "carrier" in col and "name" in col:  col_map["carrier_name"]=col
            elif any(x in col for x in ["rate","price","cost"]): col_map["rate"]=col
            elif "lane" in col:                        col_map["lane_id"]=col
            elif "ontime" in col or "on_time" in col: col_map["ontime_pct"]=col
            elif "capacity" in col:                   col_map["capacity"]=col
            elif "transit" in col:                    col_map["transit_days"]=col
            elif "distance" in col:                   col_map["distance_km"]=col

        scored_preview = []
        if "rate" in col_map:
            bids=[]
            for _,row in df.iterrows():
                try:
                    bids.append({"carrier_id":str(row.get(col_map.get("carrier_id",""),"C001")),
                                 "carrier_name":str(row.get(col_map.get("carrier_name",""),"Unknown")),
                                 "lane_id":str(row.get(col_map.get("lane_id",""),"L001")),
                                 "rate":float(row.get(col_map.get("rate",""),3.5) or 3.5),
                                 "ontime_pct":float(row.get(col_map.get("ontime_pct",""),0.85) or 0.85),
                                 "capacity":float(row.get(col_map.get("capacity",""),10000) or 10000),
                                 "transit_days":int(row.get(col_map.get("transit_days",""),2) or 2),
                                 "distance_km":float(row.get(col_map.get("distance_km",""),800) or 800),
                                 "risk_score":0.10,"is_monsoon":0,"is_festival":0})
                except: pass
            if bids:
                rs=[b["rate"] for b in bids]; mn,mx=min(rs),max(rs)
                for b in bids:
                    np__=1-(b["rate"]-mn)/(mx-mn+1e-9)
                    b["composite_score"]=round(0.4*np__+0.3*b["ontime_pct"]+0.3*0.9,4)
                    b["market_adj_rate"]=round(b["rate"]*MARKET_DATA["composite_multiplier"],3)
                scored_preview=sorted(bids,key=lambda x:-x["composite_score"])[:10]

        return ok(safe({"stats":{"rows":len(df),"columns":list(df.columns),"detected_mapping":col_map},
                        "preview":safe(df.head(5).to_dict("records")),
                        "scored_preview":safe(scored_preview),
                        "message":f"Parsed {len(df)} rows. {len(col_map)} columns auto-mapped."}))
    except Exception as e:
        return err(traceback.format_exc())

@app.route("/api/market/intelligence")
def market_intel():
    return ok(MARKET_DATA)

@app.route("/api/dashboard/summary")
def dashboard_summary():
    try:
        df_hist = pd.read_csv(f"{DATA}/shipment_history.csv")
        cs = build_carrier_scores()
        top3 = cs.nlargest(3,"composite_score")[["carrier_id","name","composite_score","ontime_pct","rank"]].to_dict("records")
        return ok(safe({"total_shipments":int(len(df_hist)),
                        "overall_ontime_pct":round(float(df_hist["on_time_delivery"].mean()*100),1),
                        "avg_cost_per_kg":round(float(df_hist["cost_per_kg"].mean()),3),
                        "avg_delay_days":round(float(df_hist["delay_days"].mean()),2),
                        "damage_rate_pct":round(float(df_hist["damage_loss_flag"].mean()*100),2),
                        "top_carriers":safe(top3),
                        "active_lanes":int(df_hist["lane_id"].nunique()),
                        "total_carriers":int(df_hist["carrier_id"].nunique()),
                        "market_multiplier":MARKET_DATA["composite_multiplier"],
                        "market_risk_level":"MODERATE"}))
    except Exception as e:
        return err(traceback.format_exc())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
