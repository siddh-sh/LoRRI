"""
Google Maps Service — Directions & Distance Calculation
────────────────────────────────────────────────────────
Uses the Google Maps Directions API to compute:
  - Real driving distance (km)
  - Estimated transit time (hours / days)
  - Polyline for route visualization
"""

import json
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

# Cache directions results
_directions_cache: dict = {}
DIRECTIONS_CACHE_TTL = 3600  # 1 hour


def _cache_key(origin: str, destination: str) -> str:
    return f"{origin.strip().lower()}|{destination.strip().lower()}"


def get_directions(origin: str, destination: str) -> dict:
    """
    Get driving directions between two Indian locations.
    Returns distance_km, duration_hours, duration_text, and route polyline.
    """
    key = _cache_key(origin, destination)
    cached = _directions_cache.get(key)
    if cached and (time.time() - cached["ts"] < DIRECTIONS_CACHE_TTL):
        result = dict(cached["data"])
        result["from_cache"] = True
        return result

    if not GOOGLE_MAPS_API_KEY:
        log.warning("GOOGLE_MAPS_API_KEY not set -- returning fallback")
        return _fallback(origin, destination)

    try:
        params = {
            "origin": f"{origin}, India",
            "destination": f"{destination}, India",
            "key": GOOGLE_MAPS_API_KEY,
            "mode": "driving",
            "units": "metric",
            "region": "in",
            "language": "en",
        }
        resp = requests.get(DIRECTIONS_URL, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "OK":
            log.error("Directions API error: %s", data.get("status"))
            return _fallback(origin, destination)

        route = data["routes"][0]
        leg = route["legs"][0]

        distance_m = leg["distance"]["value"]
        duration_s = leg["duration"]["value"]

        result = {
            "origin": leg["start_address"],
            "destination": leg["end_address"],
            "origin_lat": leg["start_location"]["lat"],
            "origin_lng": leg["start_location"]["lng"],
            "dest_lat": leg["end_location"]["lat"],
            "dest_lng": leg["end_location"]["lng"],
            "distance_km": round(distance_m / 1000, 1),
            "duration_hours": round(duration_s / 3600, 1),
            "duration_text": leg["duration"]["text"],
            "transit_days": _hours_to_transit_days(duration_s / 3600),
            "polyline": route.get("overview_polyline", {}).get("points", ""),
            "steps_count": len(leg.get("steps", [])),
            "via": route.get("summary", ""),
            "from_cache": False,
            "source": "google_maps",
        }

        _directions_cache[key] = {"ts": time.time(), "data": result}
        log.info("Directions: %s -> %s = %.1f km, %s",
                 origin, destination, result["distance_km"], result["duration_text"])
        return result

    except Exception as exc:
        log.error("Directions API call failed: %s", exc)
        return _fallback(origin, destination)


def get_directions_latlng(
    origin_lat: float, origin_lng: float,
    dest_lat: float, dest_lng: float,
    origin_label: str = "", dest_label: str = "",
) -> dict:
    """
    Get driving directions between exact lat/lng coordinates.
    Used when user pins exact pickup/drop points on the map.
    """
    if not GOOGLE_MAPS_API_KEY:
        return {"error": "GOOGLE_MAPS_API_KEY not set", "source": "fallback"}

    try:
        params = {
            "origin": f"{origin_lat},{origin_lng}",
            "destination": f"{dest_lat},{dest_lng}",
            "key": GOOGLE_MAPS_API_KEY,
            "mode": "driving",
            "units": "metric",
            "region": "in",
            "language": "en",
        }
        resp = requests.get(DIRECTIONS_URL, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "OK":
            return {"error": f"API status: {data.get('status')}", "source": "google_maps"}

        route = data["routes"][0]
        leg = route["legs"][0]
        distance_m = leg["distance"]["value"]
        duration_s = leg["duration"]["value"]

        return {
            "origin": leg["start_address"],
            "destination": leg["end_address"],
            "origin_lat": origin_lat,
            "origin_lng": origin_lng,
            "dest_lat": dest_lat,
            "dest_lng": dest_lng,
            "distance_km": round(distance_m / 1000, 1),
            "duration_hours": round(duration_s / 3600, 1),
            "duration_text": leg["duration"]["text"],
            "transit_days": _hours_to_transit_days(duration_s / 3600),
            "polyline": route.get("overview_polyline", {}).get("points", ""),
            "via": route.get("summary", ""),
            "from_cache": False,
            "source": "google_maps",
        }

    except Exception as exc:
        return {"error": str(exc), "source": "google_maps"}


def _hours_to_transit_days(hours: float) -> int:
    """
    Convert driving hours to freight transit days.
    Trucks drive ~10-12 hrs/day with mandatory rest.
    Add 0.5 days for loading/unloading at each end.
    """
    driving_days = hours / 11  # avg 11 hrs driving per day
    total = driving_days + 0.5  # loading/unloading buffer
    return max(1, round(total))


def _fallback(origin: str, destination: str) -> dict:
    """Return empty fallback when API is unavailable."""
    return {
        "origin": origin,
        "destination": destination,
        "distance_km": None,
        "duration_hours": None,
        "duration_text": None,
        "transit_days": None,
        "polyline": "",
        "from_cache": False,
        "source": "fallback",
    }
