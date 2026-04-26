"""
OpenAI Analysis Agent — Carrier Recommendation & Market Summary
────────────────────────────────────────────────────────────────
Takes the Gemini-scraped live market intelligence + carrier scoring
results and produces a clear, actionable analysis for the user:
  • Natural-language summary of current market conditions
  • Why the #1 carrier was chosen
  • Comparison of top carriers with trade-offs
  • Actionable recommendation with risk considerations
"""

import json
import logging
import os
import time

from openai import OpenAI

log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── Cache for analysis results (avoid repeated API calls) ─────
_analysis_cache: dict = {}
ANALYSIS_CACHE_TTL = int(os.getenv("ANALYSIS_CACHE_TTL", "600"))


def _cache_key(shipment: dict, market: dict) -> str:
    """Generate a unique cache key from shipment + market data."""
    parts = [
        str(shipment.get("lane_id", "")),
        str(shipment.get("weight_kg", "")),
        str(shipment.get("goods_type", "")),
        str(shipment.get("priority_profile", "")),
        str(market.get("scraped_at", "")),
    ]
    return "|".join(parts)


def _get_cached(key: str):
    entry = _analysis_cache.get(key)
    if entry and (time.time() - entry["ts"] < ANALYSIS_CACHE_TTL):
        result = dict(entry["data"])
        result["from_cache"] = True
        return result
    return None


def _set_cache(key: str, data: dict):
    _analysis_cache[key] = {"ts": time.time(), "data": data}


ANALYSIS_PROMPT = """You are an expert freight logistics analyst.
You have been given LIVE market intelligence data and carrier scoring results from our ML pipeline.

Your job is to produce a clear, professional analysis that helps a logistics manager
understand the current market and make the best carrier decision. You must explicitly answer:
1. Which option is best?
2. Why is it best for this shipment?
3. What tradeoff is being made (e.g. cost vs reliability vs urgency)?
4. Are there any viable alternatives (e.g. faster or cheaper)?
5. How does this align with the user's selected priority profile?

## SHIPMENT DETAILS
{shipment_json}

## LIVE MARKET INTELLIGENCE
{market_json}

## CARRIER RANKINGS
{carriers_json}

## BEST RECOMMENDATION
{best_json}

## OPTIMISATION RESULTS
{optimization_json}

---

Produce a JSON response with this exact structure:
{{
  "final_summary": "1-2 clear sentences stating which option is best and a brief explanation of the market conditions affecting it.",
  "best_option_explanation": "2-3 sentences explaining exactly why this carrier/mode was chosen based on the user's priority profile and specific shipment parameters.",
  "tradeoff_analysis": "1-2 sentences explaining what trade-offs are being made (e.g., 'By choosing X for reliability, you are paying a 5% premium over Y').",
  "alternatives": "1-2 sentences highlighting a faster or cheaper alternative if one exists."
}}

IMPORTANT:
- Use REAL data from the inputs — do not fabricate numbers.
- Be specific: mention carrier names, actual costs, and percentages.
- Return ONLY valid JSON, no markdown, no commentary outside JSON.
"""


def run_analysis(
    shipment: dict,
    market: dict,
    carriers: list,
    best: dict,
    optimization: dict,
) -> dict:
    """
    Call OpenAI to produce a natural-language analysis of the
    carrier scoring + market intelligence results.

    Returns a dict with the analysis fields, or a fallback if
    OpenAI is unavailable.
    """
    cache_k = _cache_key(shipment, market)

    # 1. Check cache
    cached = _get_cached(cache_k)
    if cached:
        log.info("OpenAI analysis served from cache")
        return cached

    # 2. If no client, return demo analysis
    if not _client:
        log.warning("OPENAI_API_KEY not set — returning demo analysis")
        return _demo_analysis(shipment, market, carriers, best)

    # 3. Build prompt
    prompt = ANALYSIS_PROMPT.format(
        shipment_json=json.dumps(shipment, indent=2, default=str),
        market_json=json.dumps(market, indent=2, default=str),
        carriers_json=json.dumps(
            [_slim_carrier(c) for c in (carriers or [])[:5]],
            indent=2,
            default=str,
        ),
        best_json=json.dumps(_slim_carrier(best) if best else {}, indent=2, default=str),
        optimization_json=json.dumps(optimization or {}, indent=2, default=str),
    )

    # 4. Call OpenAI
    try:
        log.info("Calling OpenAI %s for carrier analysis...", OPENAI_MODEL)
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a freight logistics analyst. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)
        result = {
            "market_summary": str(raw.get("market_summary", "")),
            "recommendation_headline": str(raw.get("recommendation_headline", "")),
            "carrier_analysis": str(raw.get("carrier_analysis", "")),
            "risk_advisory": str(raw.get("risk_advisory", "")),
            "cost_insight": str(raw.get("cost_insight", "")),
            "action_items": raw.get("action_items", []),
            "confidence_level": str(raw.get("confidence_level", "MEDIUM")).upper(),
            "model_used": OPENAI_MODEL,
            "from_cache": False,
            "agent": "openai",
        }
        _set_cache(cache_k, result)
        log.info("OpenAI analysis complete ✓")
        return result

    except Exception as exc:
        log.error("OpenAI analysis failed: %s", exc)
        return _demo_analysis(shipment, market, carriers, best)


def _slim_carrier(c: dict) -> dict:
    """Reduce carrier dict to essentials for the prompt (saves tokens)."""
    if not c:
        return {}
    return {
        "carrier_id": c.get("carrier_id"),
        "scores": c.get("scores"),
        "cost": c.get("cost"),
        "risk": c.get("risk"),
        "recommendation": c.get("recommendation"),
        "why": (c.get("why") or [])[:3],
    }


def _demo_analysis(shipment: dict, market: dict, carriers: list, best: dict) -> dict:
    """Fallback analysis when OpenAI is unavailable."""
    best_name = best.get("carrier_id", "Top Carrier") if best else "Top Carrier"
    composite = market.get("composite_multiplier", 1.0)
    above_pct = market.get("above_base_pct", "+0%")

    top3 = []
    for c in (carriers or [])[:3]:
        top3.append(c.get("carrier_id", "Unknown"))

    return {
        "final_summary": f"Based on current market conditions, {best_name} is the best option for this {shipment.get('priority_profile', 'balanced')} shipment.",
        "best_option_explanation": f"It perfectly balances your priority profile with an expected cost of ₹{best.get('cost', {}).get('estimated_total_cost_inr', 0)} and strong reliability.",
        "tradeoff_analysis": "You are trading a slightly higher cost for guaranteed capacity in a tight market.",
        "alternatives": "A lower cost alternative exists, but it carries a higher risk of delay due to current weather patterns.",
        "from_fallback": True
    }
