import pulp
import pandas as pd
from typing import Dict, Any, List


def _scoring_output_to_df(scoring_output: Dict[str, Any],
                          total_weight_kg: float) -> pd.DataFrame:
    """
    Convert run_scoring(...) output into a tabular form
    suitable for optimization.

    scoring_output: dict returned by scoring.run_scoring(...)
    total_weight_kg: total shipment weight we want to allocate (kg)
    """
    lane_id = scoring_output["lane_id"]
    rows: List[Dict[str, Any]] = []

    for carrier in scoring_output["carriers"]:
        cid = carrier["carrier_id"]
        scores = carrier.get("scores", {})
        cost = carrier.get("cost", {})

        eff_rate = float(cost.get("eff_rate_inr_kg", 0.0))
        ontime_prob = float(scores.get("composite_score", 0.0))  # or scores["ontime_prob"]

        rows.append({
            "lane_id": lane_id,
            "carrier_id": cid,
            "eff_rate_inr_kg": eff_rate,
            "reliability": ontime_prob,
            "total_weight_kg": float(total_weight_kg),
        })

    return pd.DataFrame(rows)


def optimize_allocation(scoring_output: Dict[str, Any],
                        total_weight_kg: float = 5000.0,
                        min_reliability: float = 0.9) -> Dict[str, Any]:
    """
    Optimize allocation of a given shipment across carriers.

    Objective:
        Minimize total cost = sum(x_i * eff_rate_i * total_weight_kg)

    Constraints:
        1) Sum_i x_i = 1          (100% of weight allocated)
        2) Sum_i x_i * reliability_i >= min_reliability

    Returns:
        dict with optimization summary + carrier-level allocation.
    """
    df = _scoring_output_to_df(scoring_output, total_weight_kg)

    if df.empty:
        raise ValueError("No carriers found in scoring_output")

    carriers = df["carrier_id"].tolist()

    # LP problem
    prob = pulp.LpProblem("Carrier_Award_Optimization", pulp.LpMinimize)

    # Decision variables: share of weight to allocate to each carrier (0..1)
    x = pulp.LpVariable.dicts(
        "alloc",
        carriers,
        lowBound=0,
        upBound=1,
        cat="Continuous"
    )

    # Objective: minimize total cost
    cost_terms = []
    for _, row in df.iterrows():
        cid = row["carrier_id"]
        rate = float(row["eff_rate_inr_kg"])
        cost_terms.append(x[cid] * rate * total_weight_kg)

    prob += pulp.lpSum(cost_terms), "Total_Shipping_Cost"

    # Constraint 1: fully allocate the shipment
    prob += pulp.lpSum(x[cid] for cid in carriers) == 1.0, "Full_Allocation"

    # Constraint 2: minimum average reliability
    prob += pulp.lpSum(
        x[row["carrier_id"]] * float(row["reliability"])
        for _, row in df.iterrows()
    ) >= min_reliability, "Min_Reliability"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Collect results
    allocations = []
    total_cost = 0.0
    avg_reliability = 0.0

    for _, row in df.iterrows():
        cid = row["carrier_id"]
        share = x[cid].value() or 0.0
        weight_alloc = share * total_weight_kg
        rate = float(row["eff_rate_inr_kg"])
        reliability = float(row["reliability"])
        cost = weight_alloc * rate

        total_cost += cost
        avg_reliability += share * reliability

        allocations.append({
            "carrier_id": cid,
            "allocation_share": round(share, 4),
            "allocated_weight_kg": round(weight_alloc, 2),
            "eff_rate_inr_kg": round(rate, 2),
            "expected_cost_inr": round(cost, 2),
            "reliability": round(reliability, 4),
        })

    result = {
        "lane_id": scoring_output.get("lane_id"),
        "total_weight_kg": total_weight_kg,
        "min_reliability_constraint": min_reliability,
        "optimized_total_cost_inr": round(total_cost, 2),
        "optimized_avg_reliability": round(avg_reliability, 4),
        "allocations": allocations,
        "status": pulp.LpStatus[prob.status],
    }

    return result