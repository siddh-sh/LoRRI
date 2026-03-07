import pandas as pd
import os
from backend.ml_engine import predict_all_carriers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_lane(lane_id):

    lane_df = pd.read_csv(os.path.join(DATA_DIR, "lane_master.csv"))
    lane = lane_df[lane_df["lane_id"] == lane_id]

    if lane.empty:
        raise ValueError(f"Lane {lane_id} not found")

    lane = lane.iloc[0]

    return {
        "lane_id": lane["lane_id"],
        "distance_km": lane["distance_km"],
        "transit_days": lane["std_transit_days"]
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
    """
    Main scoring function used by API / frontend
    """

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

    return results


if __name__ == "__main__":
    """
    Quick local test
    """

    output = run_scoring(
        lane_id="L001",
        weight_kg=5000,
        goods_type="Pharma",
        mode="FTL",
        priority_profile="reliability",
        shipment_value_inr=500000
    )

    for carrier in output["carriers"]:
        print(
            f"{carrier['rank']} | {carrier['carrier_id']} | "
            f"{carrier['scores']['composite_score']} | "
            f"{carrier['recommendation']}"
        )