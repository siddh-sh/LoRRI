import json
import logging
import os
import time
from datetime import datetime
from functools import wraps

from flask import jsonify, request
from google import genai
from google.genai import types

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

Use Google Search to find REAL and CURRENT data for:
1. Current petrol and diesel retail prices in India.
2. Latest NHAI toll revision information.
3. Upcoming Indian festivals and holidays in the next 30 days affecting freight demand.
4. Weather disruptions on major logistics corridors in India.
5. Financial or operational health signals for major carriers such as Blue Dart, Delhivery, TCI, GATI, VRL.

Return ONLY a valid JSON object, with this exact schema:
{{
  "date": "{date}",
  "scraped_at": "{date}",
  "factors": [
    {{
      "key": "fuel",
      "icon": "⛽",
      "label": "Fuel Price Index",
      "sub": "string",
      "multiplier_value": 1.035,
      "detail": "string",
      "direction": "UP",
      "confidence": "HIGH",
      "style": "yellow",
      "source_url": "https://example.com"
    }},
    {{
      "key": "toll",
      "icon": "🛣️",
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
      "icon": "🎉",
      "label": "Festival Demand Surge",
      "sub": "string",
      "multiplier_value": 1.045,
      "detail": "string",
      "direction": "UP",
      "confidence": "HIGH",
      "style": "pink",
      "source_url": "https://example.com"
    }},
    {{
      "key": "weather",
      "icon": "🌦️",
      "label": "Weather & Route Risk",
      "sub": "string",
      "multiplier_value": 1.005,
      "detail": "string",
      "direction": "STABLE",
      "confidence": "HIGH",
      "style": "teal",
      "source_url": "https://example.com"
    }},
    {{
      "key": "carrier",
      "icon": "🏦",
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
"""


def _call_gemini_agent(bust_cache: bool = False) -> dict:
    cache_key = "market_intelligence"

    if not bust_cache:
        cached = cache_get(cache_key)
        if cached:
            return cached

    if not client:
        raise RuntimeError("GEMINI_API_KEY not set and DEMO_MODE is false")

    date_str = ist_now()
    prompt = INTELLIGENCE_PROMPT.format(date=date_str)
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
        "from_cache": False,
        "demo_mode": False,
        "model_used": MODEL,
    }
    data = recompute_composite(data)
    cache_set(cache_key, data)
    return data


def _demo_data() -> dict:
    data = {
        "date": ist_now(),
        "scraped_at": ist_now(),
        "from_cache": False,
        "demo_mode": True,
        "model_used": MODEL,
        "factors": [
            {
                "key": "fuel",
                "icon": "⛽",
                "label": "Fuel Price Index",
                "sub": "₹87.62–92.39/L",
                "multiplier_value": 1.035,
                "detail": "Fuel prices remain elevated with geopolitical pressure and refining cost concerns.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "yellow",
                "source_url": "https://ppac.gov.in",
            },
            {
                "key": "toll",
                "icon": "🛣️",
                "label": "NHAI Toll Rates",
                "sub": "4–5% annual revision",
                "multiplier_value": 1.022,
                "detail": "WPI-linked annual toll adjustments continue to increase route cost on national corridors.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "yellow",
                "source_url": "https://nhai.gov.in",
            },
            {
                "key": "festival",
                "icon": "🎉",
                "label": "Festival Demand Surge",
                "sub": "Multiple events in next 20 days",
                "multiplier_value": 1.045,
                "detail": "Festival-linked freight demand is driving short-term capacity tightening across key lanes.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "pink",
                "source_url": "https://www.india.gov.in/calendar",
            },
            {
                "key": "weather",
                "icon": "🌦️",
                "label": "Weather & Route Risk",
                "sub": "Low disruption risk",
                "multiplier_value": 1.005,
                "detail": "No major weather-driven corridor disruption currently indicated on major routes.",
                "direction": "STABLE",
                "confidence": "HIGH",
                "style": "teal",
                "source_url": "https://mausam.imd.gov.in",
            },
            {
                "key": "carrier",
                "icon": "🏦",
                "label": "Carrier Financial Health",
                "sub": "No major distress flagged",
                "multiplier_value": 1.008,
                "detail": "Major listed carriers appear broadly stable, with no acute short-term risk materially affecting pricing.",
                "direction": "STABLE",
                "confidence": "HIGH",
                "style": "teal",
                "source_url": "https://www.moneycontrol.com",
            },
        ],
    }
    return recompute_composite(data)


def get_market_intelligence_snapshot(bust_cache: bool = False) -> dict:
    try:
        if DEMO_MODE:
            return _demo_data()
        return _call_gemini_agent(bust_cache=bust_cache)
    except Exception as e:
        log.warning("Falling back to demo market intelligence: %s", e)
        return _demo_data()
