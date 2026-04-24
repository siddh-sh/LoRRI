import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from flask import jsonify, request
from google import genai
from google.genai import types
import requests

try:
    from backend.ml_engine import _get_dynamic_festivals
except ImportError:
    def _get_dynamic_festivals(): return ["Upcoming Events"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))
MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_cache: dict = {}


def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < CACHE_TTL_SECONDS):
        cached = dict(entry["data"])
        cached["from_cache"] = True
        return cached
    return None


def cache_set(key: str, data: dict):
    _cache[key] = {"ts": time.time(), "data": data}


def cache_bust(key: str):
    _cache.pop(key, None)


def ist_now() -> str:
    return datetime.now().strftime("%B %d, %Y · %I:%M %p IST")


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-API-Key", "")
        admin_key = os.getenv("ADMIN_KEY", "")
        if admin_key and token != admin_key:
            return jsonify({"error": "Forbidden"}), 403
        return f(*args, **kwargs)
    return decorated


def safe_float(value, default=1.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_multiplier(value: float) -> str:
    return f"×{value:.3f}"


def fmt_above_base(composite: float) -> str:
    pct = (composite - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def sanitize_factor(factor: dict) -> dict:
    return {
        "key": str(factor.get("key", "")),
        "icon": str(factor.get("icon", "•")),
        "label": str(factor.get("label", "Unnamed Factor")),
        "sub": str(factor.get("sub", "")),
        "multiplier_value": safe_float(factor.get("multiplier_value"), 1.0),
        "multiplier": str(factor.get("multiplier", "×1.000")),
        "detail": str(factor.get("detail", "")),
        "direction": str(factor.get("direction", "STABLE")).upper(),
        "confidence": str(factor.get("confidence", "MEDIUM")).upper(),
        "style": str(factor.get("style", "yellow")),
        "source_url": str(factor.get("source_url", "")),
    }


def recompute_composite(data: dict) -> dict:
    product = 1.0
    factors = [sanitize_factor(f) for f in data.get("factors", [])]
    for factor in factors:
        mv = safe_float(factor.get("multiplier_value"), 1.0)
        factor["multiplier_value"] = mv
        factor["multiplier"] = fmt_multiplier(mv)
        product *= mv

    product = round(product, 3)
    data["factors"] = factors
    data["composite_multiplier"] = product
    data["above_base_pct"] = fmt_above_base(product)
    data["cache_ttl_seconds"] = CACHE_TTL_SECONDS
    return data


INTELLIGENCE_PROMPT = """You are a live freight market intelligence AI for India.
Today's date is {date}.
The shipment route is: {origin} to {destination}.

Use Google Search to find REAL and CURRENT data for:
1. Current petrol and diesel retail prices in India (check PPAC or IOCL).
2. Latest NHAI toll revision information.
3. Upcoming Indian national holidays AND state-specific holidays/festivals in {origin_state} and {destination_state} within the next 30 days from {date}. Include gazetted holidays, restricted holidays, and major festivals. Do NOT include past events.
4. Weather disruptions on the {origin}–{destination} corridor and related logistics routes.
5. Financial or operational health signals for major carriers such as Blue Dart, Delhivery, TCI, GATI, VRL.

Return ONLY a valid JSON object, with this exact schema:
{{
  "date": "{date}",
  "scraped_at": "{date}",
  "factors": [
    {{
      "key": "fuel",
      "icon": "fuel",
      "label": "Fuel Price Index",
      "sub": "string with actual price range",
      "multiplier_value": 1.035,
      "detail": "string",
      "direction": "UP",
      "confidence": "HIGH",
      "style": "yellow",
      "source_url": "https://example.com"
    }},
    {{
      "key": "toll",
      "icon": "toll",
      "label": "NHAI Toll Rates",
      "sub": "string",
      "multiplier_value": 1.022,
      "detail": "string",
      "direction": "UP",
      "confidence": "HIGH",
      "style": "yellow",
      "source_url": "https://example.com"
    }},
    {{
      "key": "festival",
      "icon": "festival",
      "label": "Festival & Holiday Impact",
      "sub": "list nearest upcoming festival/holiday name and date",
      "multiplier_value": 1.045,
      "detail": "List ALL upcoming national and state holidays for {origin_state} and {destination_state} in next 30 days with names and dates. Mention impact on freight capacity.",
      "direction": "UP",
      "confidence": "HIGH",
      "style": "pink",
      "source_url": "https://example.com"
    }},
    {{
      "key": "weather",
      "icon": "weather",
      "label": "Weather & Route Risk",
      "sub": "string",
      "multiplier_value": 1.005,
      "detail": "string about weather on {origin}-{destination} route",
      "direction": "STABLE",
      "confidence": "HIGH",
      "style": "teal",
      "source_url": "https://example.com"
    }},
    {{
      "key": "carrier",
      "icon": "carrier",
      "label": "Carrier Financial Health",
      "sub": "string",
      "multiplier_value": 1.008,
      "detail": "string",
      "direction": "STABLE",
      "confidence": "HIGH",
      "style": "teal",
      "source_url": "https://example.com"
    }}
  ]
}}

Important:
- Do not include markdown.
- Do not include commentary outside JSON.
- Keep values realistic and current.
- source_url should be a real source.
- For festivals: ONLY list events from TODAY ({date}) onwards. Do NOT mention past events.
"""


def _call_gemini_agent(bust_cache: bool = False, origin: str = "Delhi", destination: str = "Mumbai") -> dict:
    origin_state = CITY_STATE_MAP.get(origin, "India")
    dest_state = CITY_STATE_MAP.get(destination, "India")
    cache_key = f"market_intelligence:{origin}:{destination}"

    if not bust_cache:
        cached = cache_get(cache_key)
        if cached:
            return cached

    if not client:
        raise RuntimeError("GEMINI_API_KEY not set and DEMO_MODE is false")

    date_str = ist_now()
    prompt = INTELLIGENCE_PROMPT.format(
        date=date_str,
        origin=origin,
        destination=destination,
        origin_state=origin_state,
        destination_state=dest_state,
    )
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
            ),
        )

        raw = json.loads(response.text)
        data = {
            "date": str(raw.get("date", date_str)),
            "scraped_at": str(raw.get("scraped_at", date_str)),
            "factors": raw.get("factors", []),
            "origin": origin,
            "destination": destination,
            "origin_state": origin_state,
            "destination_state": dest_state,
            "from_cache": False,
            "demo_mode": False,
            "model_used": MODEL,
        }
        data = recompute_composite(data)
        cache_set(cache_key, data)
        return data
    except Exception:
        return _demo_data(origin, destination)


def _demo_data(origin: str = "Delhi", destination: str = "Mumbai") -> dict:
    """Live scraper fallback when Gemini API is unavailable (avoids hardcoded data)."""
    date_str = ist_now()
    
    # 1. Fetch live crude oil prices (Yahoo Finance)
    oil_price, oil_trend = "85.00", "STABLE"
    try:
        r = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/BZ=F', headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            price = r.json()['chart']['result'][0]['meta']['regularMarketPrice']
            oil_price = f"${price:.2f}/bbl"
            oil_trend = "UP" if price > 80 else "DOWN"
    except Exception as e:
        log.warning("Live oil price fetch failed: %s", e)
        
    # 2. Fetch live weather warnings (wttr.in)
    weather_desc, weather_trend = "Clear", "STABLE"
    try:
        r = requests.get(f'https://wttr.in/{destination.split(",")[0].strip()}?format=j1', headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            weather_desc = r.json()['current_condition'][0]['weatherDesc'][0]['value']
            temp = int(r.json()['current_condition'][0]['temp_C'])
            if "Rain" in weather_desc or "Storm" in weather_desc:
                weather_trend = "UP"
            weather_desc = f"{temp}°C, {weather_desc}"
    except Exception as e:
        log.warning("Live weather fetch failed: %s", e)
        
    # 3. Dynamic festival check
    festivals = _get_dynamic_festivals()
    festival_text = f"Upcoming: {', '.join(festivals[:2])}" if festivals else "No major events in next 10 days"

    data = {
        "date": date_str,
        "scraped_at": date_str,
        "origin": origin,
        "destination": destination,
        "from_cache": False,
        "demo_mode": True,
        "model_used": "live-api-fallback",
        "factors": [
            {
                "key": "fuel",
                "icon": "⛽",
                "label": "Global Crude Index",
                "sub": "Live from Brent Crude",
                "multiplier_value": 1.025 if oil_trend == "UP" else 0.99,
                "detail": f"Current market price: {oil_price}. Indian domestic fuel adjustments may follow.",
                "direction": oil_trend,
                "confidence": "HIGH",
                "style": "pink",
            },
            {
                "key": "tolls",
                "icon": "🛣️",
                "label": "NHAI Toll Rates",
                "sub": "4-5% annual revision",
                "multiplier_value": 1.015,
                "detail": "Standard annual highway toll indexing integrated into base carrier costs.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "yellow",
            },
            {
                "key": "festivals",
                "icon": "🎉",
                "label": "Festival Demand",
                "sub": "Dynamic local events",
                "multiplier_value": 1.04 if festivals else 1.00,
                "detail": festival_text,
                "direction": "UP" if festivals else "STABLE",
                "confidence": "HIGH",
                "style": "yellow",
            },
            {
                "key": "weather",
                "icon": "⛅",
                "label": "Weather Risk",
                "sub": f"Live at {destination.split(',')[0]}",
                "multiplier_value": 1.03 if weather_trend == "UP" else 1.00,
                "detail": f"Current condition: {weather_desc}.",
                "direction": weather_trend,
                "confidence": "HIGH",
                "style": "teal",
            },
        ],
    }
    return recompute_composite(data)


def get_market_intelligence_snapshot(
    bust_cache: bool = False,
    origin: str = "Delhi",
    destination: str = "Mumbai",
) -> dict:
    if DEMO_MODE:
        return _demo_data(origin, destination)
    return _call_gemini_agent(bust_cache=bust_cache, origin=origin, destination=destination)


# ── City-to-state mapping for Indian logistics hubs ───────────
CITY_STATE_MAP = {
    "Agra": "Uttar Pradesh", "Ahmedabad": "Gujarat", "Amritsar": "Punjab",
    "Bangalore": "Karnataka", "Bhopal": "Madhya Pradesh", "Bhubaneswar": "Odisha",
    "Chandigarh": "Chandigarh/Punjab", "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Dehradun": "Uttarakhand", "Delhi": "Delhi", "Guwahati": "Assam",
    "Hyderabad": "Telangana", "Indore": "Madhya Pradesh", "Jabalpur": "Madhya Pradesh",
    "Jaipur": "Rajasthan", "Kanpur": "Uttar Pradesh", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Madurai": "Tamil Nadu",
    "Mangalore": "Karnataka", "Mumbai": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Noida": "Uttar Pradesh", "Patna": "Bihar",
    "Pune": "Maharashtra", "Raipur": "Chhattisgarh", "Rajkot": "Gujarat",
    "Ranchi": "Jharkhand", "Surat": "Gujarat", "Thiruvananthapuram": "Kerala",
    "Vadodara": "Gujarat", "Varanasi": "Uttar Pradesh",
    "Vijayawada": "Andhra Pradesh", "Visakhapatnam": "Andhra Pradesh",
}
