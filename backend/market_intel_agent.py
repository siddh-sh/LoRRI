import os
import json
import time
import logging
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "AIzaSyCZAVJCE4_QZ1SnqntQ-t8Zl46KGGcTsJg")
DEMO_MODE          = os.getenv("DEMO_MODE", "true").lower() == "false"
CACHE_TTL_SECONDS  = int(os.getenv("CACHE_TTL", "300"))   # 5-min cache
MODEL              = "gemini-2.0-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Simple in-memory cache ─────────────────────────────────────────────────────
_cache: dict = {}

def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < CACHE_TTL_SECONDS):
        log.info(f"Cache HIT: {key}")
        return entry["data"]
    return None

def cache_set(key: str, data: dict):
    _cache[key] = {"ts": time.time(), "data": data}
    log.info(f"Cache SET: {key}")

def cache_bust(key: str):
    _cache.pop(key, None)

# ── Helpers ────────────────────────────────────────────────────────────────────
def ist_now() -> str:
    """Return human-readable timestamp."""
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

# ── Gemini Agentic Scraper Logic ───────────────────────────────────────────────
INTELLIGENCE_PROMPT = """You are a live freight market intelligence AI for India.
Today's date is {date}.

Your task: Use Google Search to find REAL, CURRENT data for:
1. Current petrol & diesel retail prices (IOCL, BPCL, HPCL).
2. Latest NHAI toll rate revisions or WPI-linked notifications.
3. Upcoming Indian festivals/holidays in the next 30 days affecting logistics.
4. Weather disruptions on major corridors (NH-44, NH-48, NH-8).
5. Financial health news for carriers (BlueDart, Delhivery, TCI, GATI, VRL).

Return ONLY a valid JSON object. Compute 'composite_multiplier' as the product of all 'multiplier_value' fields.
JSON Schema:
{{
  "date": "{date}",
  "scraped_at": "{date}",
  "composite_multiplier": float,
  "above_base_pct": "string (e.g. +7.4%)",
  "factors": [
    {{
      "key": "fuel",
      "icon": "⛽",
      "label": "Fuel Price Index",
      "sub": "string",
      "multiplier": "string (e.g. ×1.035)",
      "multiplier_value": float,
      "detail": "string",
      "direction": "UP/STABLE/DOWN",
      "confidence": "HIGH",
      "style": "yellow",
      "source_url": "url"
    }},
    ... (toll, festival, weather, carrier)
  ]
}}
"""

def _call_gemini_agent(bust_cache: bool = False) -> dict:
    cache_key = "market_intelligence"
    if not bust_cache:
        cached = cache_get(cache_key)
        if cached: return cached

    log.info("Calling Gemini agent with Google Search grounding...")
    date_str = ist_now()
    prompt = INTELLIGENCE_PROMPT.format(date=date_str)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json"
        )
    )

    data = json.loads(response.text)
    data["from_cache"] = False
    data["model_used"] = MODEL
    
    cache_set(cache_key, data)
    return data

def _demo_data() -> dict:
    """Realistic demo data for testing."""
    return {
        "date": ist_now(),
        "scraped_at": ist_now(),
        "composite_multiplier": 1.074,
        "above_base_pct": "+7.4%",
        "from_cache": False,
        "demo_mode": True,
        "model_used": MODEL,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "factors": [
            {
                "key": "fuel",
                "icon": "⛽",
                "label": "Fuel Price Index",
                "sub": "₹87.62–92.39/L",
                "multiplier": "×1.035",
                "multiplier_value": 1.035,
                "detail": "Israel-Iran geopolitical tension; PPAC data shows ₹4–5/L hike imminent across major cities.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "yellow",
                "source_url": "https://ppac.gov.in",
            },
            {
                "key": "toll",
                "icon": "🛣️",
                "label": "NHAI Toll Rates",
                "sub": "4–5% hike Apr 2025",
                "multiplier": "×1.022",
                "multiplier_value": 1.022,
                "detail": "855 plazas affected by WPI-linked annual revision effective April 1, 2025.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "yellow",
                "source_url": "https://nhai.gov.in",
            },
            {
                "key": "festival",
                "icon": "🎉",
                "label": "Festival Demand Surge",
                "sub": "4 events next 20d",
                "multiplier": "×1.045",
                "multiplier_value": 1.045,
                "detail": "Eid-ul-Fitr Mar 31, Navratri Mar 30 – Apr 7, Ram Navami Apr 6, Baisakhi Apr 13.",
                "direction": "UP",
                "confidence": "HIGH",
                "style": "pink",
                "source_url": "https://www.india.gov.in/calendar",
            },
            {
                "key": "weather",
                "icon": "🌦️",
                "label": "Weather & Route Risk",
                "sub": "Low — dry season",
                "multiplier": "×1.005",
                "multiplier_value": 1.005,
                "detail": "March pre-monsoon; all major corridors (NH-44, NH-48, NH-8) are clear. IMD: no disruptions.",
                "direction": "STABLE",
                "confidence": "HIGH",
                "style": "teal",
                "source_url": "https://mausam.imd.gov.in",
            },
            {
                "key": "carrier",
                "icon": "🏦",
                "label": "Carrier Financial Health",
                "sub": "0 carriers flagged",
                "multiplier": "×1.008",
                "multiplier_value": 1.008,
                "detail": "GATI-KWE under monitoring post-merger; BlueDart, TCI, Delhivery all financially stable.",
                "direction": "STABLE",
                "confidence": "HIGH",
                "style": "teal",
                "source_url": "https://www.moneycontrol.com",
            },
        ],
    }

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "model": MODEL,
        "timestamp": ist_now(),
    })

@app.get("/api/market/intelligence")
def get_market_intelligence():
    bust = request.args.get("bust", "0") == "1"
    try:
        if DEMO_MODE:
            log.info("DEMO_MODE=true — returning demo data")
            data = _demo_data()
        else:
            if not GEMINI_API_KEY or "YOUR_GEMINI" in GEMINI_API_KEY:
                return jsonify({"error": "GEMINI_API_KEY not set"}), 500
            data = _call_gemini_agent(bust_cache=bust)

        return jsonify({"success": True, "data": data})

    except Exception as e:
        log.exception("Unexpected error in /api/market/intelligence")
        return jsonify({"success": False, "error": str(e)}), 500

@app.post("/api/market/intelligence/refresh")
@require_api_key
def force_refresh():
    cache_bust("market_intelligence")
    return jsonify({"success": True, "message": "Cache cleared."})

@app.get("/api/market/intelligence/cache-status")
def cache_status():
    entry = _cache.get("market_intelligence")
    if entry:
        age = int(time.time() - entry["ts"])
        return jsonify({
            "cached": True,
            "age_seconds": age,
            "expires_in_seconds": max(0, CACHE_TTL_SECONDS - age)
        })
    return jsonify({"cached": False})

# ── Dev Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Starting FreightIQ backend on port {port} | DEMO_MODE={DEMO_MODE}")
    app.run(host="0.0.0.0", port=port, debug=True)