import os
from typing import Dict, List

import pandas as pd

from backend.ml_engine import predict_all_carriers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Lane data is loaded from lane_master.csv at runtime.
# Previously this file contained ~1,330 lines of hardcoded STATIC_LANES as a
# fallback, but lane_master.csv always exists so it was dead code. Removed to
# keep the module clean and data-driven.
_LANE_CACHE = None   # module-level cache so CSV is read only once


def _lane_master_path() -> str:
    return os.path.join(DATA_DIR, "lane_master.csv")


def _load_lane_df() -> pd.DataFrame:
    global _LANE_CACHE
    if _LANE_CACHE is not None:
        return _LANE_CACHE
    path = _lane_master_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"lane_master.csv not found at {path}. "
            "Please ensure backend/data/lane_master.csv exists."
        )
    _LANE_CACHE = pd.read_csv(path)
    return _LANE_CACHE

def get_lane_catalog() -> List[Dict]:
    df = _load_lane_df().copy()
    cols = ["lane_id", "origin", "destination", "distance_km", "std_transit_days"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"lane_master.csv missing required column: {col}")
    return df[cols].sort_values(["origin", "destination"]).to_dict(orient="records")


def load_lane(lane_id: str) -> Dict:
    lane_df = _load_lane_df()
    lane = lane_df[lane_df["lane_id"] == lane_id]
    if lane.empty:
        raise ValueError(f"Lane {lane_id} not found")
    row = lane.iloc[0]
    return {
        "lane_id": row["lane_id"],
        "origin": row["origin"],
        "destination": row["destination"],
        "distance_km": float(row["distance_km"]),
        "transit_days": float(row["std_transit_days"]),
    }


def run_scoring(
    lane_id,
    weight_kg,
    goods_type="General Cargo",
    mode="FTL",
    priority_profile="balanced",
    shipment_value_inr=100000,
    is_monsoon=0,
    is_festival=0
):
    lane = load_lane(lane_id)

    results = predict_all_carriers(
        lane_id=lane["lane_id"],
        distance_km=lane["distance_km"],
        weight_kg=weight_kg,
        transit_days=lane["transit_days"],
        is_monsoon=is_monsoon,
        is_festival=is_festival,
        goods_type=goods_type,
        mode=mode,
        priority_profile=priority_profile,
        shipment_value_inr=shipment_value_inr
    )

    results["origin"] = lane["origin"]
    results["destination"] = lane["destination"]
    results["distance_km"] = lane["distance_km"]
    results["transit_days"] = lane["transit_days"]
    return results
