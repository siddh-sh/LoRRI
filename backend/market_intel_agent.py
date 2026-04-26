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

# No hardcoded imports needed — all data is scraped live

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# DEMO_MODE removed — always use live data
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
        log.info("No Gemini key — falling back to live scraper")
        return _scrape_live_intelligence(origin, destination)

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
    except Exception as exc:
        log.warning("Gemini call failed (%s), falling back to live scraper", exc)
        return _scrape_live_intelligence(origin, destination)


def fetch_live_news(query: str, default_text: str) -> str:
    """Fetch live news snippet using Google News RSS."""
    try:
        import urllib.parse
        import xml.etree.ElementTree as ET
        safe_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            items = root.findall(".//item")
            for item in items:
                title = item.findtext("title", "")
                if title:
                    return title.rsplit(" - ", 1)[0]
    except Exception as e:
        log.warning("Live news fetch failed for %s: %s", query, e)
    return default_text

def fetch_live_city_event(city: str) -> str:
    return fetch_live_news(f"{city} festival OR event", f"Standard local conditions in {city}")


def _scrape_live_intelligence(origin: str = "Delhi", destination: str = "Mumbai") -> dict:
    """Scrape REAL live market intelligence from the internet for the given route."""
    date_str = ist_now()
    origin_state = CITY_STATE_MAP.get(origin, "India")
    dest_state = CITY_STATE_MAP.get(destination, "India")

    # 1. Live fuel / crude oil news
    fuel_news = fetch_live_news(
        f"India diesel petrol price today {origin}",
        "Fuel prices steady across major Indian metros.")
    fuel_headline = fetch_live_news(
        "Brent crude oil price today",
        "Global crude markets stable.")
    fuel_detail = f"{fuel_news}. Global: {fuel_headline}"
    fuel_dir = "UP" if any(w in fuel_detail.lower() for w in ["hike", "rise", "surge", "increase"]) else "STABLE"
    fuel_mult = 1.03 if fuel_dir == "UP" else 1.00

    # 2. Live toll / highway news
    toll_news = fetch_live_news(
        f"NHAI toll revision highway {origin} {destination}",
        "No major toll revisions reported on this corridor.")
    toll_dir = "UP" if any(w in toll_news.lower() for w in ["hike", "revision", "increase"]) else "STABLE"
    toll_mult = 1.02 if toll_dir == "UP" else 1.00

    # 3. Live weather for BOTH origin and destination
    o_weather = _fetch_weather(origin)
    d_weather = _fetch_weather(destination)
    weather_detail = f"{origin}: {o_weather['desc']} | {destination}: {d_weather['desc']}"
    weather_dir = "UP" if o_weather["risky"] or d_weather["risky"] else "STABLE"
    weather_mult = 1.04 if weather_dir == "UP" else 1.00

    # 4. Live events / festivals at origin AND destination
    origin_event = fetch_live_news(
        f"{origin} {origin_state} festival OR holiday OR event OR strike transport",
        f"No major disruptions reported in {origin}.")
    dest_event = fetch_live_news(
        f"{destination} {dest_state} festival OR holiday OR event OR strike transport",
        f"No major disruptions reported in {destination}.")
    event_detail = f"Origin ({origin}): {origin_event} | Dest ({destination}): {dest_event}"
    event_dir = "UP" if any(w in event_detail.lower() for w in ["strike", "bandh", "festival", "holiday", "blockade"]) else "STABLE"
    event_mult = 1.05 if event_dir == "UP" else 1.00

    # 5. Live carrier / logistics corridor news
    carrier_news = fetch_live_news(
        f"India freight logistics carrier {origin} {destination} road rail air cargo",
        "Carrier operations normal across major corridors.")
    carrier_dir = "UP" if any(w in carrier_news.lower() for w in ["delay", "disruption", "strike", "shortage"]) else "STABLE"
    carrier_mult = 1.02 if carrier_dir == "UP" else 1.00

    data = {
        "date": date_str,
        "scraped_at": date_str,
        "origin": origin,
        "destination": destination,
        "origin_state": origin_state,
        "destination_state": dest_state,
        "from_cache": False,
        "demo_mode": False,
        "model_used": "live-scraper",
        "factors": [
            {"key": "fuel", "icon": "⛽", "label": "Fuel Price Index",
             "sub": fuel_news[:60], "multiplier_value": fuel_mult,
             "detail": fuel_detail, "direction": fuel_dir,
             "confidence": "HIGH", "style": "pink"},
            {"key": "tolls", "icon": "🛣️", "label": "NHAI Toll Rates",
             "sub": toll_news[:60], "multiplier_value": toll_mult,
             "detail": toll_news, "direction": toll_dir,
             "confidence": "HIGH", "style": "yellow"},
            {"key": "festivals", "icon": "🎉", "label": "Events & Disruptions",
             "sub": f"Live from {origin} & {destination}",
             "multiplier_value": event_mult, "detail": event_detail,
             "direction": event_dir, "confidence": "HIGH", "style": "yellow"},
            {"key": "weather", "icon": "⛅", "label": "Weather & Route Risk",
             "sub": f"Live: {origin} & {destination}",
             "multiplier_value": weather_mult, "detail": weather_detail,
             "direction": weather_dir, "confidence": "HIGH", "style": "teal"},
            {"key": "carrier", "icon": "🚚", "label": "Carrier & Corridor Intel",
             "sub": carrier_news[:60], "multiplier_value": carrier_mult,
             "detail": carrier_news, "direction": carrier_dir,
             "confidence": "MEDIUM", "style": "teal"},
        ],
    }
    return recompute_composite(data)


def _fetch_weather(city: str) -> dict:
    """Fetch live weather for a city from wttr.in."""
    try:
        r = requests.get(f'https://wttr.in/{city}?format=j1',
                         headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            cond = r.json()['current_condition'][0]
            desc = f"{cond['temp_C']}°C, {cond['weatherDesc'][0]['value']}"
            risky = any(w in desc.lower() for w in ["rain", "storm", "thunder", "flood", "cyclone"])
            return {"desc": desc, "risky": risky}
    except Exception as e:
        log.warning("Weather fetch failed for %s: %s", city, e)
    return {"desc": "Data unavailable", "risky": False}


def get_market_intelligence_snapshot(
    bust_cache: bool = False,
    origin: str = "Delhi",
    destination: str = "Mumbai",
) -> dict:
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
