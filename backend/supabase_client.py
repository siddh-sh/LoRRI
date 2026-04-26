"""
Supabase Client — Shared database layer for FreightIQ
─────────────────────────────────────────────────────
Uses PostgREST client directly (avoids heavy storage3/pyiceberg deps).
Provides helper functions to persist shipment analyses and
logistics news to Supabase. Gracefully degrades if credentials
are missing (logs a warning, never crashes the app).
"""

import logging
import os
import threading

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

log = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_client = None


def get_client():
    """Lazy-initialise and return a PostgREST SyncPostgrestClient."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL or SUPABASE_KEY not set — database persistence disabled")
        return None
    try:
        from postgrest import SyncPostgrestClient

        rest_url = SUPABASE_URL.rstrip("/") + "/rest/v1"
        _client = SyncPostgrestClient(
            base_url=rest_url,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        log.info("Supabase PostgREST client initialised ✓")
        return _client
    except Exception as exc:
        log.error("Failed to create Supabase client: %s", exc)
        return None


# ── Shipment Analysis ─────────────────────────────────────────

def save_shipment_analysis(
    input_data: dict,
    carrier_scoring: list,
    best_carrier: dict | None,
    optimization: dict,
    market_intelligence: dict,
    ai_analysis: dict,
    route_info: dict,
):
    """
    Persist a shipment analysis to Supabase (fire-and-forget).
    Called from app.py after building the response.
    """
    def _insert():
        client = get_client()
        if not client:
            return

        best = best_carrier or {}
        row = {
            # Input params
            "origin": str(input_data.get("origin", "")),
            "destination": str(input_data.get("destination", "")),
            "transport_mode": str(input_data.get("transport_mode", "roadways")),
            "weight_kg": float(input_data.get("weight_kg", 0)),
            "goods_type": str(input_data.get("goods_type", "FMCG")),
            "priority_profile": str(input_data.get("priority_profile", "balanced")),
            "shipment_value_inr": float(input_data.get("shipment_value_inr", 0)),
            "min_reliability": float(input_data.get("min_reliability", 0.8)),

            # Route
            "distance_km": float(route_info.get("distance_km", 0)),
            "transit_days_est": float(route_info.get("transit_days", 0)),

            # Best carrier summary
            "best_carrier_id": str(best.get("carrier_id", "")),
            "best_carrier_score": float(best.get("scores", {}).get("composite_score", 0)),
            "best_carrier_cost": float(best.get("cost", {}).get("estimated_total_cost_inr", 0)),

            # Optimization
            "optimized_total_cost": float(optimization.get("optimized_total_cost_inr", 0)),

            # Market
            "composite_multiplier": float(market_intelligence.get("composite_multiplier", 1.0)),

            # Full JSON payloads
            "carrier_scoring": carrier_scoring,
            "optimization": optimization,
            "market_intelligence": market_intelligence,
            "ai_analysis": ai_analysis,
        }

        try:
            client.from_("shipment_analyses").insert(row).execute()
            log.info("Shipment analysis saved to Supabase ✓")
        except Exception as exc:
            log.error("Failed to save shipment analysis: %s", exc)

    # Fire-and-forget in a background thread
    threading.Thread(target=_insert, daemon=True).start()


# ── Logistics News ────────────────────────────────────────────

def save_news_items(news_items: list):
    """
    Persist fetched news items to Supabase.
    Uses upsert on URL to prevent duplicates.
    """
    def _upsert():
        client = get_client()
        if not client or not news_items:
            return

        rows = []
        for item in news_items:
            url = item.get("url", "")
            if not url or url == "#":
                continue  # Skip placeholder items
            rows.append({
                "title": str(item.get("title", "")),
                "source": str(item.get("source", "Unknown")),
                "published_at": str(item.get("published_at", "")),
                "url": url,
                "relevance_score": int(item.get("relevance_score", 0)),
                "category": str(item.get("category", "Logistics")),
            })

        if not rows:
            return

        try:
            client.from_("logistics_news").upsert(
                rows, on_conflict="url"
            ).execute()
            log.info("Saved %d news items to Supabase ✓", len(rows))
        except Exception as exc:
            log.error("Failed to save news items: %s", exc)

    # Fire-and-forget
    threading.Thread(target=_upsert, daemon=True).start()
