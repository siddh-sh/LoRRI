# 🚛 FreightIQ (Project LoRRI)

<div align="center">
  <p><strong>Agentic AI for Predictive, Optimized Freight Procurement</strong></p>
  <p><i>Winner / Finalist - LogisticsNow Hackathon 2025 • Track: Ops Automation</i></p>
</div>

---

## 📖 Overview

**FreightIQ** replaces manual, spreadsheet-based freight procurement with an orchestrated, intelligent pipeline. It analyzes live market data and carrier bids to recommend optimal load allocations based on cost, reliability, and risk. 

By combining **Predictive Machine Learning**, **Linear Programming**, and an **Explainable AI Agent**, FreightIQ eliminates the "black box" of logistics algorithms and empowers supply chain teams with fast, data-backed, and fully explained decisions.

---

## ✨ Key Features

- 🧠 **Predictive Risk Modeling:** Evaluates carriers across Road (FTL/LTL), Rail, and Air using an ensemble of XGBoost, Gradient Boosting, and Linear Regression to predict delay probabilities.
- ⚖️ **Mathematical Optimization (PuLP):** Dynamically allocates freight volume across multiple carriers to strictly minimize costs or maximize reliability based on your selected priority profile.
- 📰 **Live Market Intelligence:** Scrapes real-time unstructured data (Google News RSS) to detect and penalize routes affected by weather warnings, fuel spikes, or strikes.
- 🤖 **Agentic Explainability:** Integrates with OpenAI to translate complex ML probabilities and LP math into simple, natural-language business justifications.
- ⚡ **Lightning-Fast UI:** Built with raw HTML/JS/CSS and Leaflet.js to ensure an extremely fast, zero-bloat dashboard with a built-in dark/light mode engine.

---

## 🏗️ Architecture & Data Flow

FreightIQ operates on a synchronous 5-stage pipeline:

1. **Input (UI):** User enters shipment constraints via the dashboard.
2. **Context (APIs):** System fetches geographic constraints (Google Maps) and live market disruptions (News RSS).
3. **Scoring (ML):** Features are scaled and fed into XGBoost/GBR models to generate risk-adjusted composite scores.
4. **Allocation (LP):** The PuLP engine takes the ranked scores and calculates the exact kilogram allocation per carrier to meet capacity and budget constraints.
5. **Output (GenAI & DB):** The LLM translates the mathematical decision into plain English. The entire payload is then asynchronously logged to **Supabase (PostgreSQL)** for persistence.

---

## 💻 Tech Stack

### Frontend
* **Core:** HTML5, CSS3, Vanilla JavaScript
* **Templating:** Jinja2
* **Mapping:** Leaflet.js, Google Maps API

### Backend
* **Framework:** Python 3, Flask, Gunicorn
* **Intelligence:** Google News RSS, NetworkX

### AI, ML & Optimization
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib
* **Optimization:** PuLP (Linear Programming)
* **Agentic AI:** OpenAI API (`gpt-4` / `gpt-3.5-turbo`)

### Database & Storage
* **Persistence:** Supabase (PostgreSQL)

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Valid API Keys for Google Maps, OpenAI, and Supabase.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siddh-sh/LoRRI.git
   cd LoRRI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your keys:
   ```env
   GOOGLE_MAPS_API_KEY=your_google_maps_key
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```
   The application will be live at `http://127.0.0.1:5000`

---

## 🤝 Team
*Built with ❤️ for the LogisticsNow 2025 Hackathon.*
