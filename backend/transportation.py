"""
Transportation Mode Module — Airways / Railways / Roadways
Generates realistic ranked transport options for any Indian route.

Data: All fallback-generated from parameterized models. No hardcoding.
Extensible: Swap FallbackDataSource for an API/scraper source.

Sources modeled after:
  - Indian Railways goods tariff schedules
  - Domestic air cargo rate benchmarks (AAI / IATA India)
  - Real Indian city infrastructure connectivity
"""

import math
import random
import hashlib
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# ── Valid Modes ────────────────────────────────────────────────────────────────
VALID_MODES = {"airways", "railways", "roadways"}


def validate_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in VALID_MODES:
        raise ValueError(
            f"Invalid transport mode '{mode}'. Must be one of: {', '.join(sorted(VALID_MODES))}"
        )
    return m


# ── City Infrastructure Registry ──────────────────────────────────────────────
# Tier determines connectivity quality and provider availability.
# tier 1 = major hub, tier 2 = mid-size, tier 3 = small
# has_airport / has_rail_freight derived from tier + geography

CITY_INFRA: Dict[str, Dict[str, Any]] = {}

_CITY_TIERS = {
    1: [
        "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
        "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Chandigarh", "Kochi",
        "Guwahati", "Noida", "Coimbatore",
    ],
    2: [
        "Nagpur", "Indore", "Bhopal", "Patna", "Bhubaneswar", "Ranchi",
        "Varanasi", "Amritsar", "Dehradun", "Surat", "Vadodara", "Rajkot",
        "Visakhapatnam", "Madurai", "Mangalore", "Thiruvananthapuram",
        "Nashik", "Kanpur", "Raipur", "Vijayawada",
    ],
    3: [
        "Agra", "Jabalpur",
    ],
}

# Cities with NO commercial airport for cargo
_NO_AIRPORT = {
    "Agra", "Nashik", "Kanpur", "Raipur", "Jabalpur",
    "Rajkot", "Vadodara", "Ranchi",
}

# Cities with NO direct rail freight terminal
_NO_RAIL_FREIGHT = {
    "Mangalore", "Thiruvananthapuram", "Madurai",
}

for _tier, _cities in _CITY_TIERS.items():
    for _city in _cities:
        CITY_INFRA[_city] = {
            "tier": _tier,
            "has_airport": _city not in _NO_AIRPORT,
            "has_rail_freight": _city not in _NO_RAIL_FREIGHT,
        }


def _city_tier(city: str) -> int:
    return CITY_INFRA.get(city, {}).get("tier", 3)


def _has_airport(city: str) -> bool:
    return CITY_INFRA.get(city, {}).get("has_airport", False)


def _has_rail_freight(city: str) -> bool:
    return CITY_INFRA.get(city, {}).get("has_rail_freight", True)


# ── Provider Registry ─────────────────────────────────────────────────────────
# Each provider has base parameters that feed into the pricing engine.
# No prices are hardcoded — they are computed from these coefficients.

PROVIDERS: Dict[str, List[Dict[str, Any]]] = {
    "airways": [
        {
            "name": "IndiGo Cargo",
            "cost_coeff": 1.00,     # multiplier on base air rate
            "speed_coeff": 0.90,    # lower = faster
            "reliability": 0.93,
            "max_weight_kg": 15000,
            "min_tier": 1,          # only serves tier-1+ airports
            "cargo_restrictions": [],
        },
        {
            "name": "Air India Cargo",
            "cost_coeff": 1.08,
            "speed_coeff": 0.85,
            "reliability": 0.90,
            "max_weight_kg": 25000,
            "min_tier": 1,
            "cargo_restrictions": [],
        },
        {
            "name": "Blue Dart Aviation",
            "cost_coeff": 1.22,
            "speed_coeff": 0.78,
            "reliability": 0.96,
            "max_weight_kg": 10000,
            "min_tier": 1,
            "cargo_restrictions": [],
        },
        {
            "name": "SpiceJet Cargo",
            "cost_coeff": 0.92,
            "speed_coeff": 0.95,
            "reliability": 0.88,
            "max_weight_kg": 12000,
            "min_tier": 2,
            "cargo_restrictions": ["Pharma"],
        },
        {
            "name": "Vistara Cargo",
            "cost_coeff": 1.15,
            "speed_coeff": 0.88,
            "reliability": 0.91,
            "max_weight_kg": 8000,
            "min_tier": 1,
            "cargo_restrictions": [],
        },
    ],
    "railways": [
        {
            "name": "Indian Railways Parcel",
            "cost_coeff": 1.00,
            "speed_coeff": 1.00,
            "reliability": 0.82,
            "max_weight_kg": 50000,
            "min_tier": 3,
            "cargo_restrictions": [],
        },
        {
            "name": "CONCOR (Container Corp)",
            "cost_coeff": 1.12,
            "speed_coeff": 0.88,
            "reliability": 0.89,
            "max_weight_kg": 80000,
            "min_tier": 2,
            "cargo_restrictions": [],
        },
        {
            "name": "GATI-KWE Rail",
            "cost_coeff": 1.18,
            "speed_coeff": 0.82,
            "reliability": 0.91,
            "max_weight_kg": 30000,
            "min_tier": 2,
            "cargo_restrictions": [],
        },
        {
            "name": "DFC Express Freight",
            "cost_coeff": 0.95,
            "speed_coeff": 0.75,
            "reliability": 0.94,
            "max_weight_kg": 100000,
            "min_tier": 1,
            "cargo_restrictions": [],
        },
        {
            "name": "KRIBHCO Rail Logistics",
            "cost_coeff": 0.88,
            "speed_coeff": 1.10,
            "reliability": 0.80,
            "max_weight_kg": 60000,
            "min_tier": 3,
            "cargo_restrictions": ["Pharma", "Electronics"],
        },
    ],
}


# ── Goods-Type Sensitivity Factors ────────────────────────────────────────────
# These affect pricing and compatibility, not hardcoded per-provider.

GOODS_FACTORS: Dict[str, Dict[str, float]] = {
    "Pharma":        {"handling_mult": 1.35, "urgency": 1.4, "temp_sensitive": 1.2},
    "Electronics":   {"handling_mult": 1.25, "urgency": 1.2, "temp_sensitive": 1.0},
    "Auto Parts":    {"handling_mult": 1.10, "urgency": 0.9, "temp_sensitive": 1.0},
    "FMCG":          {"handling_mult": 1.00, "urgency": 1.1, "temp_sensitive": 1.05},
    "Textiles":      {"handling_mult": 0.90, "urgency": 0.8, "temp_sensitive": 1.0},
    "General Cargo": {"handling_mult": 1.00, "urgency": 1.0, "temp_sensitive": 1.0},
}


# ── Deterministic Seeded Randomness ───────────────────────────────────────────
# Ensures same route + params always gives same "realistic" variation,
# while still producing diverse results across different routes.

def _route_seed(origin: str, destination: str, provider: str) -> int:
    key = f"{origin}|{destination}|{provider}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _seeded_variation(seed: int, base: float, pct_range: float = 0.08) -> float:
    """Return base ± pct_range%, deterministic per seed."""
    rng = random.Random(seed)
    factor = 1.0 + rng.uniform(-pct_range, pct_range)
    return round(base * factor, 2)


# ── Pricing Engine ────────────────────────────────────────────────────────────
# All rates computed from distance, weight, goods type, and season.

def _base_air_rate_per_kg(distance_km: float) -> float:
    """
    Domestic air cargo rate model (INR/kg).
    Based on distance bands — modeled after IATA India domestic rates.
    Continuous function, not hardcoded tiers.
    """
    # Base formula: starts ~25/kg, scales logarithmically with distance
    rate = 18.0 + 12.0 * math.log10(max(distance_km, 100) / 100.0)
    # Weight surcharge diminishes at scale (handled in caller)
    return round(rate, 2)


def _base_rail_rate_per_kg(distance_km: float) -> float:
    """
    Indian Railways goods tariff model (INR/kg).
    Based on per-tonne-km rates scaled to per-kg.
    """
    # ~1.8 INR/tonne-km → 0.0018/kg-km, with fixed handling
    per_km = 0.0018 + 0.0004 * math.exp(-distance_km / 1500.0)
    rate = 1.5 + per_km * distance_km
    return round(rate, 2)


def _compute_cost(
    mode: str,
    distance_km: float,
    weight_kg: float,
    provider: Dict[str, Any],
    goods_type: str,
    is_monsoon: int,
    is_festival: int,
    seed: int,
) -> float:
    """Compute total estimated cost for a shipment."""
    if mode == "airways":
        base_rate = _base_air_rate_per_kg(distance_km)
    else:
        base_rate = _base_rail_rate_per_kg(distance_km)

    # Provider coefficient
    rate = base_rate * provider["cost_coeff"]

    # Goods handling multiplier
    gf = GOODS_FACTORS.get(goods_type, GOODS_FACTORS["General Cargo"])
    rate *= gf["handling_mult"]

    # Volume discount for heavy shipments (continuous, not stepped)
    if weight_kg > 1000:
        discount = 1.0 - 0.04 * math.log10(weight_kg / 1000.0)
        rate *= max(discount, 0.82)

    # Seasonal adjustments
    if is_monsoon:
        rate *= 1.06 if mode == "airways" else 1.03
    if is_festival:
        rate *= 1.08 if mode == "airways" else 1.05

    # Deterministic per-route variation
    rate = _seeded_variation(seed, rate, pct_range=0.06)

    return round(rate * weight_kg, 2)


# ── Transit Time Engine ──────────────────────────────────────────────────────

def _compute_transit_hours(
    mode: str,
    distance_km: float,
    provider: Dict[str, Any],
    origin_tier: int,
    dest_tier: int,
    seed: int,
) -> float:
    """Compute estimated transit time in hours."""
    if mode == "airways":
        # Flight time + ground handling
        flight_hours = distance_km / 750.0  # avg domestic cruise ~750 km/h
        # Ground handling depends on city tier
        ground_hours = (5 - origin_tier) * 1.5 + (5 - dest_tier) * 1.5
        base = (flight_hours + ground_hours) * provider["speed_coeff"]
    else:
        # Rail: avg speed ~40-60 km/h including stops
        rail_hours = distance_km / 45.0
        # Terminal handling
        terminal_hours = (5 - origin_tier) * 2.0 + (5 - dest_tier) * 2.0
        base = (rail_hours + terminal_hours) * provider["speed_coeff"]

    return round(_seeded_variation(seed + 1, max(base, 2.0), pct_range=0.05), 1)


def _format_transit_time(hours: float) -> str:
    """Format hours into human-readable string."""
    if hours < 1:
        return f"{int(hours * 60)}m"
    days = int(hours // 24)
    rem_hours = int(hours % 24)
    if days > 0:
        return f"{days}d {rem_hours}h" if rem_hours > 0 else f"{days}d"
    return f"{rem_hours}h {int((hours % 1) * 60)}m"


# ── Cargo Compatibility ──────────────────────────────────────────────────────

def _cargo_compatible(provider: Dict[str, Any], goods_type: str) -> bool:
    return goods_type not in provider.get("cargo_restrictions", [])


def _cargo_score(provider: Dict[str, Any], goods_type: str, mode: str) -> float:
    """Score cargo compatibility 0-1."""
    if not _cargo_compatible(provider, goods_type):
        return 0.0

    gf = GOODS_FACTORS.get(goods_type, GOODS_FACTORS["General Cargo"])
    base = 0.85

    # Air is better for urgent/temp-sensitive goods
    if mode == "airways":
        base += 0.05 * (gf["urgency"] - 1.0)
        base += 0.05 * (gf["temp_sensitive"] - 1.0)
    # Rail is better for heavy, non-urgent goods
    elif mode == "railways":
        base += 0.05 * (1.0 - gf["urgency"])
        base -= 0.03 * (gf["temp_sensitive"] - 1.0)

    return round(max(0.0, min(1.0, base)), 4)


# ── Availability Check ───────────────────────────────────────────────────────

def check_availability(
    origin: str, destination: str, mode: str, weight_kg: float
) -> Tuple[bool, str]:
    """Check if a transport mode is available for a given route."""
    mode = validate_mode(mode)

    if mode == "roadways":
        return True, "Roadways available for all routes"

    if mode == "airways":
        if not _has_airport(origin):
            return False, f"No commercial cargo airport at {origin}"
        if not _has_airport(destination):
            return False, f"No commercial cargo airport at {destination}"
        if weight_kg > 25000:
            return False, f"Weight {weight_kg}kg exceeds max air cargo capacity (25,000 kg)"
        return True, "Airways available"

    if mode == "railways":
        if not _has_rail_freight(origin):
            return False, f"No rail freight terminal at {origin}"
        if not _has_rail_freight(destination):
            return False, f"No rail freight terminal at {destination}"
        return True, "Railways available"

    return False, "Unknown mode"


# ── Ranking Engine ────────────────────────────────────────────────────────────

# Ranking weights — configurable, not hardcoded per-option
RANKING_WEIGHTS = {
    "cost": 0.30,
    "transit_time": 0.25,
    "reliability": 0.20,
    "cargo_compat": 0.15,
    "feasibility": 0.10,
}


def _normalize(values: List[float], invert: bool = False) -> List[float]:
    """Min-max normalize a list. invert=True means lower is better."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    rng = hi - lo if hi != lo else 1e-9
    if invert:
        return [round((hi - v) / rng, 4) for v in values]
    return [round((v - lo) / rng, 4) for v in values]


def rank_options(options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank transport options using multi-factor weighted scoring."""
    if not options:
        return []

    costs = [o["estimated_cost"] for o in options]
    times = [o["_transit_hours"] for o in options]
    reliabilities = [o["_reliability"] for o in options]
    cargo_scores = [o["_cargo_score"] for o in options]
    feasibilities = [o["_feasibility"] for o in options]

    n_cost = _normalize(costs, invert=True)       # lower cost = better
    n_time = _normalize(times, invert=True)        # lower time = better
    n_rel = _normalize(reliabilities, invert=False) # higher = better
    n_cargo = _normalize(cargo_scores, invert=False)
    n_feas = _normalize(feasibilities, invert=False)

    w = RANKING_WEIGHTS
    for i, opt in enumerate(options):
        score = (
            w["cost"] * n_cost[i]
            + w["transit_time"] * n_time[i]
            + w["reliability"] * n_rel[i]
            + w["cargo_compat"] * n_cargo[i]
            + w["feasibility"] * n_feas[i]
        )
        opt["score"] = round(score, 4)

    # Sort best-first
    options.sort(key=lambda x: x["score"], reverse=True)

    # Generate recommendation reasons dynamically (before stripping internal keys)
    for i, opt in enumerate(options):
        opt["recommendation_reason"] = _generate_reason(opt, i, options)

    # Remove internal keys
    for opt in options:
        for k in ["_transit_hours", "_reliability", "_cargo_score", "_feasibility"]:
            opt.pop(k, None)

    return options


def _generate_reason(opt: Dict, rank: int, all_opts: List[Dict]) -> str:
    """Dynamically generate a recommendation reason based on relative position."""
    reasons = []
    costs = [o["estimated_cost"] for o in all_opts]
    times = [o["_transit_hours"] for o in all_opts]

    avg_cost = sum(costs) / len(costs) if costs else 0
    avg_time = sum(times) / len(times) if times else 0

    if opt["estimated_cost"] < avg_cost:
        saving = round((avg_cost - opt["estimated_cost"]) / avg_cost * 100, 1)
        reasons.append(f"{saving}% below average cost")
    elif opt["estimated_cost"] == min(costs):
        reasons.append("Lowest cost option")

    if opt["_transit_hours"] < avg_time:
        faster = round((avg_time - opt["_transit_hours"]) / avg_time * 100, 1)
        reasons.append(f"{faster}% faster than average")
    elif opt["_transit_hours"] == min(times):
        reasons.append("Fastest delivery")

    if opt["_reliability"] >= 0.93:
        reasons.append(f"{round(opt['_reliability'] * 100, 1)}% reliability rating")

    if opt["_cargo_score"] >= 0.9:
        reasons.append("Excellent cargo compatibility")

    if rank == 0:
        reasons.insert(0, "Best overall score")
    elif rank == 1:
        reasons.insert(0, "Strong alternative")

    return " · ".join(reasons[:3]) if reasons else "Available option for this route"


# ── Feasibility Score ─────────────────────────────────────────────────────────

def _compute_feasibility(
    provider: Dict[str, Any],
    weight_kg: float,
    origin_tier: int,
    dest_tier: int,
) -> float:
    """Feasibility based on weight limits and city connectivity."""
    score = 1.0

    # Weight vs capacity
    capacity = provider["max_weight_kg"]
    utilization = weight_kg / capacity
    if utilization > 1.0:
        return 0.0
    if utilization > 0.8:
        score *= 0.85
    elif utilization > 0.5:
        score *= 0.95

    # City tier vs provider min-tier
    min_tier = provider.get("min_tier", 3)
    if origin_tier > min_tier or dest_tier > min_tier:
        score *= 0.7

    return round(score, 4)


# ── Data Source Architecture ──────────────────────────────────────────────────

class TransportDataSource(ABC):
    """Base class for transport data sources. Extend for live APIs."""

    @abstractmethod
    def fetch_options(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        weight_kg: float,
        goods_type: str,
        transport_mode: str,
        is_monsoon: int,
        is_festival: int,
    ) -> List[Dict[str, Any]]:
        ...


class FallbackDataSource(TransportDataSource):
    """
    Generates realistic transport options from parameterized models.
    No external API calls. All data computed from route parameters.

    To integrate live data in the future:
      1. Create a new class extending TransportDataSource
      2. Implement fetch_options() with API calls
      3. Pass the new source to generate_transport_options()

    Potential live sources:
      - Airways: IndiGo Cargo API, Air India Cargo portal
      - Railways: Indian Railways FOIS API, CONCOR tracking
      - Aggregators: FreightBro, Freightwalla, Vahak
    """

    def fetch_options(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        weight_kg: float,
        goods_type: str,
        transport_mode: str,
        is_monsoon: int = 0,
        is_festival: int = 0,
    ) -> List[Dict[str, Any]]:

        providers = PROVIDERS.get(transport_mode, [])
        origin_tier = _city_tier(origin)
        dest_tier = _city_tier(destination)
        options = []

        for prov in providers:
            # Skip if cargo incompatible
            if not _cargo_compatible(prov, goods_type):
                continue

            # Skip if weight exceeds capacity
            if weight_kg > prov["max_weight_kg"]:
                continue

            seed = _route_seed(origin, destination, prov["name"])

            cost = _compute_cost(
                transport_mode, distance_km, weight_kg,
                prov, goods_type, is_monsoon, is_festival, seed,
            )

            transit_hours = _compute_transit_hours(
                transport_mode, distance_km, prov,
                origin_tier, dest_tier, seed,
            )

            feasibility = _compute_feasibility(
                prov, weight_kg, origin_tier, dest_tier,
            )

            # Skip truly infeasible options
            if feasibility <= 0.0:
                continue

            # Apply monsoon/festival reliability penalty
            reliability = prov["reliability"]
            if is_monsoon:
                reliability *= 0.96 if transport_mode == "airways" else 0.93
            if is_festival:
                reliability *= 0.97

            options.append({
                "provider_name": prov["name"],
                "transport_mode": transport_mode,
                "origin": origin,
                "destination": destination,
                "estimated_cost": cost,
                "estimated_transit_time": _format_transit_time(transit_hours),
                "availability": True,
                # Internal — used by ranking, stripped after
                "_transit_hours": transit_hours,
                "_reliability": round(reliability, 4),
                "_cargo_score": _cargo_score(prov, goods_type, transport_mode),
                "_feasibility": feasibility,
            })

        return options


# ── Default data source instance ──────────────────────────────────────────────
_default_source = FallbackDataSource()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def generate_transport_options(
    origin: str,
    destination: str,
    distance_km: float,
    weight_kg: float,
    goods_type: str = "General Cargo",
    transport_mode: str = "roadways",
    is_monsoon: int = 0,
    is_festival: int = 0,
    source: Optional[TransportDataSource] = None,
) -> Dict[str, Any]:
    """
    Main entry point for transportation options.

    Returns:
        {
            "transport_mode": str,
            "available": bool,
            "message": str,
            "options": List[...] | [],    # ranked best→worst
            "total_options": int,
            "origin": str,
            "destination": str,
        }
    """
    mode = validate_mode(transport_mode)

    if mode == "roadways":
        return {
            "transport_mode": "roadways",
            "available": True,
            "message": "Roadways handled by existing carrier pipeline",
            "options": [],
            "total_options": 0,
            "origin": origin,
            "destination": destination,
        }

    # Check route availability
    available, msg = check_availability(origin, destination, mode, weight_kg)
    if not available:
        return {
            "transport_mode": mode,
            "available": False,
            "message": msg,
            "options": [],
            "total_options": 0,
            "origin": origin,
            "destination": destination,
        }

    # Fetch and rank options
    ds = source or _default_source
    raw_options = ds.fetch_options(
        origin=origin,
        destination=destination,
        distance_km=distance_km,
        weight_kg=weight_kg,
        goods_type=goods_type,
        transport_mode=mode,
        is_monsoon=is_monsoon,
        is_festival=is_festival,
    )

    if not raw_options:
        return {
            "transport_mode": mode,
            "available": False,
            "message": f"No {mode} options available for {origin} → {destination} with the given parameters",
            "options": [],
            "total_options": 0,
            "origin": origin,
            "destination": destination,
        }

    ranked = rank_options(raw_options)

    return {
        "transport_mode": mode,
        "available": True,
        "message": f"{len(ranked)} {mode} option(s) found",
        "options": ranked,
        "total_options": len(ranked),
        "origin": origin,
        "destination": destination,
    }
