import os
from typing import Dict, List

import pandas as pd

from ml_engine import predict_all_carriers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

STATIC_LANES: List[Dict] = [
    {"lane_id": "L001", "origin": "Mumbai", "destination": "Delhi", "distance_km": 1415, "std_transit_days": 2},
    {"lane_id": "L002", "origin": "Mumbai", "destination": "Bangalore", "distance_km": 981, "std_transit_days": 2},
    {"lane_id": "L003", "origin": "Mumbai", "destination": "Hyderabad", "distance_km": 710, "std_transit_days": 1},
    {"lane_id": "L004", "origin": "Chennai", "destination": "Hyderabad", "distance_km": 627, "std_transit_days": 1},
    {"lane_id": "L005", "origin": "Pune", "destination": "Ahmedabad", "distance_km": 660, "std_transit_days": 1},
]


def _lane_master_path() -> str:
    return os.path.join(DATA_DIR, "lane_master.csv")


def _load_lane_df() -> pd.DataFrame:
    path = _lane_master_path()
    if os.path.exists(path):
        return pd.read_csv(path)

    df = pd.DataFrame(STATIC_LANES)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


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
