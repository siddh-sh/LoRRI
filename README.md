# LoRRI
# FreightIQ - Advanced Procurement Optimization & Agentic Logistics

FreightIQ is an intelligent procurement assistant that optimizes freight carrier selection using predictive ML, linear programming, and an agentic AI layer for explainable, end‑to‑end decision support.

---

## 1. System Overview

FreightIQ replaces manual spreadsheet‑based freight procurement with an orchestrated pipeline: from shipment input and carrier bids to optimized, explainable booking recommendations.

> High‑level architecture:

**Key capabilities:**

- Data‑driven scoring of carriers on cost, reliability, and risk.
- PuLP‑based shipment allocation under capacity and reliability constraints.
- Agentic AI layer that explains recommendations and supports what‑if simulations.
- Web dashboard for interactive visualization of bids, scores, and risk.

---

## 2. User Journey

The user interacts with FreightIQ via a browser‑based dashboard.

1. **Dashboard**
   - Open the procurement dashboard and select “New Shipment”.

2. **Enter Shipment Details**
   - Origin, destination, weight/volume, freight type, and delivery deadline.

3. **Fetch Carrier Bids**
   - System loads available carrier bids (file, DB, or live APIs).

4. **Preprocessing & Risk Prediction**
   - Data is cleaned and normalized; ML model predicts delay probability and reliability.

5. **Scoring & Optimization**
   - Composite scores computed, then allocated across carriers via linear programming.

6. **Agent Recommendation**
   - Agentic layer produces a natural‑language recommendation and reasoning.

7. **Review & Confirm**
   - Dashboard shows ranked carriers, cost and risk breakdowns, and explanation.
   - User accepts a recommendation to confirm shipment booking or runs a what‑if scenario.

---

## 3. Architecture and Components


### 3.1 Frontend UI

- HTML, CSS, JavaScript (or your chosen framework).
- Screens:
  - Shipment form.
  - Carrier results dashboard.
  - Risk & reliability visualization.

### 3.2 Flask Backend (API Layer)

- Exposes routes such as:

```text
/predict-risk        # Predict delay probability & reliability
/carrier-ranking     # Composite scores & ranking
/market-intelligence # External signals (fuel, traffic, weather, etc.)
/recommendation      # End‑to‑end recommendation object
