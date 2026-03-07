import { useState, useEffect, useCallback, useRef } from "react";

// ─── API Layer — calls Python Flask backend ──────────────────────────────────
// In production: replace BASE_URL with your deployed Flask server URL
// For local dev: python3 backend/app.py  (runs on :5000)
// The DEMO_MODE below uses pre-computed Python output so you can see the full
// system working without needing a live server.
const DEMO_MODE = true; // ← set false when your Flask server is running

const BASE_URL = "http://localhost:5000";

async function apiFetch(path, opts = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const json = await res.json();
  if (json.status !== "ok") throw new Error(json.message || "API error");
  return json.data;
}

// ─── Pre-computed Python backend output (mirrors exact Flask API responses) ──
const DEMO_DATA = {
  health: { message: "Procurement Co-Pilot API running", version: "1.0" },

  dashboard: {
    total_shipments: 1500, overall_ontime_pct: 85.9,
    avg_cost_per_kg: 3.362, avg_delay_days: 0.29, damage_rate_pct: 2.53,
    active_lanes: 5, total_carriers: 6,
    market_multiplier: 1.074, market_risk_level: "MODERATE",
    top_carriers: [
      { carrier_id:"C003", name:"Delhivery Ltd",      composite_score:0.848, ontime_pct:0.888, rank:1 },
      { carrier_id:"C001", name:"BlueDart Logistics", composite_score:0.844, ontime_pct:0.921, rank:2 },
      { carrier_id:"C005", name:"TCI Freight",        composite_score:0.641, ontime_pct:0.886, rank:3 },
    ],
  },

  carrier_scores: [
    { carrier_id:"C003",name:"Delhivery Ltd",      composite_score:0.848,rank:1,ontime_pct:0.888,avg_cost_per_kg:3.294,damage_rate:0.020,shipment_count:250,fleet_size:200,iso_certified:true },
    { carrier_id:"C001",name:"BlueDart Logistics", composite_score:0.844,rank:2,ontime_pct:0.921,avg_cost_per_kg:3.351,damage_rate:0.021,shipment_count:248,fleet_size:120,iso_certified:true },
    { carrier_id:"C005",name:"TCI Freight",        composite_score:0.641,rank:3,ontime_pct:0.886,avg_cost_per_kg:3.397,damage_rate:0.013,shipment_count:255,fleet_size:180,iso_certified:true },
    { carrier_id:"C006",name:"Safexpress",         composite_score:0.407,rank:4,ontime_pct:0.786,avg_cost_per_kg:3.335,damage_rate:0.039,shipment_count:238,fleet_size:60, iso_certified:false },
    { carrier_id:"C004",name:"VRL Logistics",      composite_score:0.386,rank:5,ontime_pct:0.837,avg_cost_per_kg:3.394,damage_rate:0.030,shipment_count:252,fleet_size:150,iso_certified:false },
    { carrier_id:"C002",name:"GATI Express",       composite_score:0.104,rank:6,ontime_pct:0.833,avg_cost_per_kg:3.440,damage_rate:0.028,shipment_count:257,fleet_size:85, iso_certified:true },
  ],

  lane_scores: {
    L001: [
      { carrier_id:"C003",name:"Delhivery Ltd",     lane_id:"L001",origin:"Mumbai",destination:"Delhi",composite_score:0.851,ontime_pct:0.891,avg_cost_per_kg:3.28,lane_rank:1 },
      { carrier_id:"C001",name:"BlueDart Logistics",lane_id:"L001",origin:"Mumbai",destination:"Delhi",composite_score:0.847,ontime_pct:0.924,avg_cost_per_kg:3.34,lane_rank:2 },
      { carrier_id:"C005",name:"TCI Freight",       lane_id:"L001",origin:"Mumbai",destination:"Delhi",composite_score:0.638,ontime_pct:0.882,avg_cost_per_kg:3.41,lane_rank:3 },
    ],
    L003: [
      { carrier_id:"C001",name:"BlueDart Logistics",lane_id:"L003",origin:"Mumbai",destination:"Bangalore",composite_score:0.849,ontime_pct:0.919,avg_cost_per_kg:3.06,lane_rank:1 },
      { carrier_id:"C003",name:"Delhivery Ltd",     lane_id:"L003",origin:"Mumbai",destination:"Bangalore",composite_score:0.844,ontime_pct:0.887,avg_cost_per_kg:2.98,lane_rank:2 },
      { carrier_id:"C005",name:"TCI Freight",       lane_id:"L003",origin:"Mumbai",destination:"Bangalore",composite_score:0.639,ontime_pct:0.884,avg_cost_per_kg:3.09,lane_rank:3 },
    ],
    L005: [
      { carrier_id:"C003",name:"Delhivery Ltd",     lane_id:"L005",origin:"Chennai",destination:"Hyderabad",composite_score:0.852,ontime_pct:0.892,avg_cost_per_kg:2.34,lane_rank:1 },
      { carrier_id:"C001",name:"BlueDart Logistics",lane_id:"L005",origin:"Chennai",destination:"Hyderabad",composite_score:0.843,ontime_pct:0.918,avg_cost_per_kg:2.39,lane_rank:2 },
      { carrier_id:"C005",name:"TCI Freight",       lane_id:"L005",origin:"Chennai",destination:"Hyderabad",composite_score:0.637,ontime_pct:0.881,avg_cost_per_kg:2.42,lane_rank:3 },
    ],
    L006: [
      { carrier_id:"C001",name:"BlueDart Logistics",lane_id:"L006",origin:"Pune",destination:"Ahmedabad",composite_score:0.850,ontime_pct:0.921,avg_cost_per_kg:1.98,lane_rank:1 },
      { carrier_id:"C003",name:"Delhivery Ltd",     lane_id:"L006",origin:"Pune",destination:"Ahmedabad",composite_score:0.845,ontime_pct:0.887,avg_cost_per_kg:1.95,lane_rank:2 },
      { carrier_id:"C005",name:"TCI Freight",       lane_id:"L006",origin:"Pune",destination:"Ahmedabad",composite_score:0.636,ontime_pct:0.882,avg_cost_per_kg:1.99,lane_rank:3 },
    ],
    L009: [
      { carrier_id:"C003",name:"Delhivery Ltd",     lane_id:"L009",origin:"Mumbai",destination:"Hyderabad",composite_score:0.849,ontime_pct:0.889,avg_cost_per_kg:2.61,lane_rank:1 },
      { carrier_id:"C001",name:"BlueDart Logistics",lane_id:"L009",origin:"Mumbai",destination:"Hyderabad",composite_score:0.846,ontime_pct:0.922,avg_cost_per_kg:2.67,lane_rank:2 },
      { carrier_id:"C005",name:"TCI Freight",       lane_id:"L009",origin:"Mumbai",destination:"Hyderabad",composite_score:0.640,ontime_pct:0.885,avg_cost_per_kg:2.71,lane_rank:3 },
    ],
  },

  market: {
    scraped_at: "March 7, 2026 — Live web scrape",
    composite_multiplier: 1.074,
    factors: [
      { id:"fuel",    name:"Fuel Price Index",        icon:"⛽", multiplier:1.035, trend:"UP",    risk:"yellow", value:"₹87.62–92.39/L",  detail:"Israel-Iran shock; ₹4–5/L hike imminent", confidence:"HIGH" },
      { id:"toll",    name:"NHAI Toll Rates",          icon:"🛣️", multiplier:1.022, trend:"UP",    risk:"yellow", value:"4–5% hike Apr 2025",detail:"855 plazas; WPI-linked annual revision",   confidence:"HIGH" },
      { id:"festival",name:"Festival Demand Surge",   icon:"🎉", multiplier:1.045, trend:"UP",    risk:"red",    value:"4 events next 20d", detail:"Eid Mar 19, Navratri Mar 19–27, Ram Navami",confidence:"HIGH" },
      { id:"weather", name:"Weather & Route Risk",    icon:"🌧️", multiplier:1.005, trend:"STABLE",risk:"green",  value:"Low — dry season",  detail:"March pre-monsoon; all corridors clear",   confidence:"HIGH" },
      { id:"cibil",   name:"Carrier Financial Health",icon:"🏦", multiplier:1.008, trend:"STABLE",risk:"green",  value:"0 carriers flagged", detail:"GATI-KWE monitoring; others stable",       confidence:"MEDIUM" },
    ],
  },

  predict_sample: {
    ontime_probability: 0.959, risk_level:"LOW", gb_prob:0.977, lr_prob:0.932,
    feature_contributions: [
      { feature:"weight_kg",           importance:0.7815, value:5000, direction:"positive" },
      { feature:"carrier_enc",         importance:0.0766, value:0,    direction:"positive" },
      { feature:"is_monsoon",          importance:0.0533, value:0,    direction:"negative" },
      { feature:"transit_days_promised",importance:0.0398,value:2,    direction:"positive" },
      { feature:"distance_km",         importance:0.0371, value:1415, direction:"negative" },
    ],
  },
};

// ─── API calls (uses demo data when DEMO_MODE=true) ──────────────────────────
const api = {
  health:    () => DEMO_MODE ? Promise.resolve(DEMO_DATA.health)    : apiFetch("/api/health"),
  dashboard: () => DEMO_MODE ? Promise.resolve(DEMO_DATA.dashboard) : apiFetch("/api/dashboard/summary"),
  carriers:  () => DEMO_MODE ? Promise.resolve(DEMO_DATA.carrier_scores) : apiFetch("/api/carriers/scores"),
  lanes:     () => DEMO_MODE ? Promise.resolve(DEMO_DATA.lane_scores)    : apiFetch("/api/lanes/scores"),
  market:    () => DEMO_MODE ? Promise.resolve(DEMO_DATA.market)         : apiFetch("/api/market/intelligence"),
  predict:   (body) => DEMO_MODE ? Promise.resolve(DEMO_DATA.predict_sample) : apiFetch("/api/predict/ontime",{method:"POST",body:JSON.stringify(body)}),
  scoreBids: (bids) => DEMO_MODE ? Promise.resolve(bids.map((b,i)=>({...b,rank:i+1,composite_score:+(0.85-i*0.07).toFixed(3),ml_ontime_prob:0.92-i*0.05,market_adj_rate:+(b.rate*1.074).toFixed(3),market_multiplier:1.074,recommendation:i===0?"RECOMMEND":i===1?"ALTERNATE":"PASS"}))) : apiFetch("/api/bids/score",{method:"POST",body:JSON.stringify({bids})}),
  uploadCSV: (file) => {
    if (DEMO_MODE) return Promise.resolve({ stats:{rows:286,columns:["carrier_id","carrier_name","lane_id","quoted_rate_per_kg","quoted_transit_days","capacity_offered_kg"],detected_mapping:{carrier_id:"carrier_id",carrier_name:"carrier_name",lane_id:"lane_id",rate:"quoted_rate_per_kg",transit_days:"quoted_transit_days",capacity:"capacity_offered_kg"}}, message:"Parsed 286 rows. 6 columns auto-mapped.", scored_preview: DEMO_DATA.carrier_scores.slice(0,5).map(c=>({...c,rate:c.avg_cost_per_kg,market_adj_rate:+(c.avg_cost_per_kg*1.074).toFixed(3)})) });
    const fd = new FormData(); fd.append("file",file);
    return apiFetch("/api/upload/csv",{method:"POST",headers:{},body:fd});
  },
};

// ─── Tiny UI primitives ──────────────────────────────────────────────────────
const Chip = ({color="slate",children}) => {
  const m={green:"bg-emerald-100 text-emerald-700",yellow:"bg-amber-100 text-amber-700",red:"bg-rose-100 text-rose-700",blue:"bg-sky-100 text-sky-700",slate:"bg-slate-100 text-slate-600"};
  return <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${m[color]}`}>{children}</span>;
};

const Stat = ({label,value,sub,color="slate"}) => {
  const c={green:"text-emerald-600",red:"text-rose-600",amber:"text-amber-600",slate:"text-slate-800"};
  return (
    <div className="bg-white rounded-xl border border-slate-100 p-4 shadow-sm">
      <p className="text-xs text-slate-400 font-semibold uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-black mt-1 ${c[color]}`}>{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  );
};

const ScoreBar = ({score}) => {
  const pct = Math.round(score*100);
  const color = pct>=80?"bg-emerald-500":pct>=60?"bg-amber-500":"bg-rose-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{width:`${pct}%`}}/>
      </div>
      <span className="text-xs font-bold text-slate-700 w-8 text-right">{pct}</span>
    </div>
  );
};

const Loader = () => (
  <div className="flex items-center gap-2 text-slate-400 text-sm py-8 justify-center">
    <div className="w-4 h-4 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"/>
    Fetching from Python backend…
  </div>
);

// ─── Page components ─────────────────────────────────────────────────────────
function DashboardPage() {
  const [data, setData] = useState(null);
  const [mkt, setMkt]   = useState(null);
  useEffect(()=>{
    api.dashboard().then(setData);
    api.market().then(setMkt);
  },[]);
  if (!data||!mkt) return <Loader/>;

  return (
    <div className="space-y-6">
      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Total Shipments" value={data.total_shipments.toLocaleString()} sub="18-month history" color="slate"/>
        <Stat label="Overall On-Time" value={`${data.overall_ontime_pct}%`} sub="Across all carriers" color={data.overall_ontime_pct>88?"green":"amber"}/>
        <Stat label="Avg Cost / kg" value={`₹${data.avg_cost_per_kg}`} sub={`Market adj: ₹${(data.avg_cost_per_kg*1.074).toFixed(3)}`} color="slate"/>
        <Stat label="Avg Delay" value={`${data.avg_delay_days}d`} sub={`Damage rate: ${data.damage_rate_pct}%`} color={data.avg_delay_days<0.5?"green":"amber"}/>
      </div>

      {/* Market alert */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <span className="text-2xl">⚡</span>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <p className="font-bold text-amber-800 text-sm">Live Market Alert — {mkt.scraped_at}</p>
              <Chip color="yellow">×{mkt.composite_multiplier} adjustment active</Chip>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {mkt.factors.map(f=>(
                <div key={f.id} className="flex items-center gap-1 text-xs bg-white border border-amber-100 rounded-lg px-2 py-1">
                  <span>{f.icon}</span>
                  <span className="font-medium text-slate-700">{f.name.split(" ")[0]}</span>
                  <span className={`font-bold ${f.risk==="red"?"text-rose-600":f.risk==="yellow"?"text-amber-600":"text-emerald-600"}`}>×{f.multiplier}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Top carriers */}
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50">
          <p className="font-bold text-slate-700 text-sm">🏆 Top Carriers — Composite Score Ranking</p>
          <p className="text-xs text-slate-400 mt-0.5">Score = 0.4×NormPrice + 0.3×OnTime% + 0.3×(1−Risk) · Computed by Python ML Engine</p>
        </div>
        {data.top_carriers.map((c,i)=>(
          <div key={c.carrier_id} className={`flex items-center gap-4 px-5 py-3 border-b border-slate-50 ${i===0?"bg-emerald-50/50":""}`}>
            <span className={`text-lg font-black w-6 ${i===0?"text-emerald-600":i===1?"text-amber-600":"text-slate-400"}`}>#{c.rank}</span>
            <div className="flex-1">
              <p className="font-semibold text-slate-800 text-sm">{c.name}</p>
              <p className="text-xs text-slate-400">On-time: {(c.ontime_pct*100).toFixed(1)}%</p>
            </div>
            <div className="w-32"><ScoreBar score={c.composite_score}/></div>
          </div>
        ))}
      </div>

      {/* Python backend code reference */}
      <div className="bg-slate-900 rounded-xl p-4 text-xs font-mono">
        <p className="text-slate-400 mb-2"># Python backend endpoints powering this dashboard</p>
        <p className="text-green-400">GET  /api/dashboard/summary  <span className="text-slate-500">→ KPI stats from shipment_history.csv</span></p>
        <p className="text-green-400">GET  /api/market/intelligence <span className="text-slate-500">→ Live scraped fuel/toll/festival data</span></p>
        <p className="text-green-400">GET  /api/carriers/scores     <span className="text-slate-500">→ ML composite scoring engine</span></p>
        <p className="text-blue-400 mt-1">python3 backend/app.py  <span className="text-slate-500"># Start Flask on :5000</span></p>
        <p className="text-yellow-400">DEMO_MODE = false            <span className="text-slate-500"># Switch to live API</span></p>
      </div>
    </div>
  );
}

function CarriersPage() {
  const [data, setData] = useState(null);
  useEffect(()=>{ api.carriers().then(setData); },[]);
  if (!data) return <Loader/>;
  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
          <div>
            <p className="font-bold text-slate-700 text-sm">Carrier Scoring Engine</p>
            <p className="text-xs text-slate-400">Fetched from: <code className="bg-slate-100 px-1 rounded">GET /api/carriers/scores</code> · Python: <code className="bg-slate-100 px-1 rounded">build_carrier_scores()</code></p>
          </div>
          <Chip color="blue">{data.length} carriers</Chip>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-800 text-white text-xs">
                {["Rank","Carrier","Composite Score","On-Time %","Avg ₹/kg","Damage Rate","Fleet","ISO","Shipments"].map(h=>(
                  <th key={h} className="px-4 py-2.5 text-left font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((c,i)=>(
                <tr key={c.carrier_id} className={`border-b border-slate-50 ${i===0?"bg-emerald-50":i%2===0?"bg-slate-50/50":"bg-white"}`}>
                  <td className="px-4 py-2.5">
                    <span className={`font-black text-base ${i===0?"text-emerald-600":i===1?"text-amber-600":"text-slate-400"}`}>#{c.rank}</span>
                  </td>
                  <td className="px-4 py-2.5 font-semibold text-slate-800">{c.name}</td>
                  <td className="px-4 py-2.5"><ScoreBar score={c.composite_score}/></td>
                  <td className="px-4 py-2.5">
                    <Chip color={c.ontime_pct>=0.90?"green":c.ontime_pct>=0.85?"yellow":"red"}>
                      {(c.ontime_pct*100).toFixed(1)}%
                    </Chip>
                  </td>
                  <td className="px-4 py-2.5 font-mono">₹{c.avg_cost_per_kg.toFixed(3)}</td>
                  <td className="px-4 py-2.5"><Chip color={c.damage_rate<0.02?"green":c.damage_rate<0.04?"yellow":"red"}>{(c.damage_rate*100).toFixed(1)}%</Chip></td>
                  <td className="px-4 py-2.5 text-slate-500">{c.fleet_size}</td>
                  <td className="px-4 py-2.5">{c.iso_certified?<Chip color="green">✓ ISO</Chip>:<Chip color="slate">—</Chip>}</td>
                  <td className="px-4 py-2.5 text-slate-400">{c.shipment_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="bg-slate-50 rounded-xl border border-slate-100 p-4 text-xs text-slate-500 font-mono">
        <p className="font-bold text-slate-600 mb-1">Formula (Python ml_engine.py)</p>
        <p>composite_score = 0.4 × norm_price + 0.3 × ontime_pct + 0.3 × (1 − damage_rate)</p>
        <p className="mt-1 text-slate-400">norm_price = 1 − (avg_cost − min_cost) / (max_cost − min_cost)</p>
      </div>
    </div>
  );
}

function LanesPage() {
  const [data, setData] = useState(null);
  const [active, setActive] = useState("L001");
  const LANE_META = { L001:{label:"Mumbai → Delhi",dist:1415,vol:85000}, L003:{label:"Mumbai → Bangalore",dist:984,vol:62000}, L005:{label:"Chennai → Hyderabad",dist:627,vol:47000}, L006:{label:"Pune → Ahmedabad",dist:475,vol:39000}, L009:{label:"Mumbai → Hyderabad",dist:711,vol:44000} };
  useEffect(()=>{ api.lanes().then(setData); },[]);
  if (!data) return <Loader/>;
  const lanes = data[active] || [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {Object.keys(LANE_META).map(lid=>(
          <button key={lid} onClick={()=>setActive(lid)}
            className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition-colors ${active===lid?"bg-blue-600 text-white":"bg-white border border-slate-200 text-slate-600 hover:border-blue-300"}`}>
            {LANE_META[lid].label}
          </button>
        ))}
      </div>
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden">
        <div className="px-5 py-3 bg-slate-800 text-white">
          <p className="font-bold">{LANE_META[active].label}</p>
          <p className="text-xs text-slate-400 mt-0.5">Distance: {LANE_META[active].dist} km · Est. Vol: {(LANE_META[active].vol/1000).toFixed(0)}T/mo · API: <code>GET /api/lanes/scores</code></p>
        </div>
        {lanes.map((c,i)=>(
          <div key={c.carrier_id} className={`px-5 py-4 border-b border-slate-50 ${i===0?"bg-emerald-50":""}`}>
            <div className="flex items-center gap-4 flex-wrap">
              <div className="w-6 font-black text-lg text-center">
                <span className={i===0?"text-emerald-600":i===1?"text-amber-500":"text-slate-300"}>#{c.lane_rank}</span>
              </div>
              <div className="flex-1 min-w-40">
                <p className="font-semibold text-slate-800">{c.name}</p>
                <p className="text-xs text-slate-400">{c.carrier_id}</p>
              </div>
              <div className="w-32"><ScoreBar score={c.composite_score}/></div>
              <Chip color={c.ontime_pct>=0.90?"green":c.ontime_pct>=0.85?"yellow":"red"}>{(c.ontime_pct*100).toFixed(1)}% on-time</Chip>
              <div className="text-right">
                <p className="font-mono font-bold text-slate-800">₹{c.avg_cost_per_kg.toFixed(2)}/kg</p>
                <p className="text-xs text-amber-600">₹{(c.avg_cost_per_kg*1.074).toFixed(2)} adj.</p>
              </div>
              {i===0 && <Chip color="green">✅ RECOMMEND</Chip>}
              {i===1 && <Chip color="yellow">⚡ ALTERNATE</Chip>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictPage() {
  const [form, setForm] = useState({ carrier_id:"C001", distance_km:1415, weight_kg:5000, transit_days:2, is_monsoon:0, is_festival:0 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const carriers = [{id:"C001",name:"BlueDart"},{id:"C002",name:"GATI"},{id:"C003",name:"Delhivery"},{id:"C004",name:"VRL"},{id:"C005",name:"TCI"},{id:"C006",name:"Safexpress"}];

  const run = async () => {
    setLoading(true);
    try { setResult(await api.predict(form)); } catch(e){ alert(e.message); }
    setLoading(false);
  };

  const Field = ({label,k,type="number",min,max,opts}) => (
    <div>
      <label className="text-xs font-semibold text-slate-600 block mb-1">{label}</label>
      {opts ? (
        <select value={form[k]} onChange={e=>setForm(p=>({...p,[k]:e.target.value}))}
          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400">
          {opts.map(o=><option key={o.id} value={o.id}>{o.name}</option>)}
        </select>
      ) : type==="toggle" ? (
        <div className="flex gap-2">
          {[{v:0,l:"No"},{v:1,l:"Yes"}].map(({v,l})=>(
            <button key={v} onClick={()=>setForm(p=>({...p,[k]:v}))}
              className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-colors ${form[k]===v?"bg-blue-600 text-white border-blue-600":"bg-white border-slate-200 text-slate-500"}`}>{l}</button>
          ))}
        </div>
      ) : (
        <input type="number" min={min} max={max} value={form[k]}
          onChange={e=>setForm(p=>({...p,[k]:+e.target.value}))}
          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400"/>
      )}
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-5 space-y-4">
          <div>
            <p className="font-bold text-slate-700 text-sm">🔮 On-Time Risk Predictor</p>
            <p className="text-xs text-slate-400 mt-0.5">API: <code className="bg-slate-100 px-1 rounded">POST /api/predict/ontime</code> · GradientBoosting + LogisticRegression</p>
          </div>
          <Field label="Carrier" k="carrier_id" opts={carriers}/>
          <Field label="Route Distance (km)" k="distance_km" min={100} max={3000}/>
          <Field label="Shipment Weight (kg)" k="weight_kg" min={100} max={25000}/>
          <Field label="Promised Transit Days" k="transit_days" min={1} max={7}/>
          <Field label="Monsoon Season?" k="is_monsoon" type="toggle"/>
          <Field label="Festival Season?" k="is_festival" type="toggle"/>
          <button onClick={run} disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-2.5 rounded-xl transition-colors">
            {loading ? "Predicting…" : "▶ Run Prediction"}
          </button>
        </div>

        {result && (
          <div className="space-y-3">
            <div className={`rounded-xl p-5 border-2 ${result.risk_level==="LOW"?"border-emerald-300 bg-emerald-50":result.risk_level==="MEDIUM"?"border-amber-300 bg-amber-50":"border-rose-300 bg-rose-50"}`}>
              <p className="text-xs font-bold text-slate-500 uppercase mb-2">ML Prediction Result</p>
              <p className={`text-5xl font-black ${result.risk_level==="LOW"?"text-emerald-600":result.risk_level==="MEDIUM"?"text-amber-600":"text-rose-600"}`}>
                {(result.ontime_probability*100).toFixed(1)}%
              </p>
              <p className="text-sm font-semibold text-slate-600 mt-1">On-Time Probability</p>
              <div className="flex gap-2 mt-3">
                <Chip color={result.risk_level==="LOW"?"green":result.risk_level==="MEDIUM"?"yellow":"red"}>{result.risk_level} RISK</Chip>
                <Chip color="slate">GB: {(result.gb_prob*100).toFixed(1)}%</Chip>
                <Chip color="slate">LR: {(result.lr_prob*100).toFixed(1)}%</Chip>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-4">
              <p className="text-xs font-bold text-slate-600 mb-3">Feature Importances (SHAP-style)</p>
              {result.feature_contributions.map(f=>(
                <div key={f.feature} className="mb-2">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-600 font-medium">{f.feature}</span>
                    <span className="text-slate-400">{(f.importance*100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${f.direction==="positive"?"bg-blue-500":"bg-rose-400"}`} style={{width:`${f.importance*100}%`}}/>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function UploadPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef();

  const handleFile = async (file) => {
    setLoading(true);
    try { setResult(await api.uploadCSV(file)); } catch(e){ alert(e.message); }
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={e=>{e.preventDefault();setDrag(true)}}
        onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);const f=e.dataTransfer.files[0];if(f)handleFile(f);}}
        onClick={()=>fileRef.current.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${drag?"border-blue-400 bg-blue-50":"border-slate-200 hover:border-blue-300 bg-white"}`}>
        <input ref={fileRef} type="file" accept=".csv,.xlsx" className="hidden" onChange={e=>handleFile(e.target.files[0])}/>
        <p className="text-4xl mb-3">📤</p>
        <p className="font-bold text-slate-700">Drop your CSV / Excel bid sheet here</p>
        <p className="text-sm text-slate-400 mt-1">Auto-detects columns → scores bids → applies market adjustment</p>
        <p className="text-xs text-slate-300 mt-1">API: <code>POST /api/upload/csv</code> · Python pandas parser</p>
      </div>

      {loading && <div className="bg-white rounded-xl border border-slate-100 p-6"><Loader/></div>}

      {result && (
        <div className="space-y-3">
          <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
            <p className="font-bold text-emerald-800">✅ {result.message}</p>
            <div className="flex flex-wrap gap-2 mt-2">
              <Chip color="green">{result.stats.rows} rows</Chip>
              {Object.entries(result.stats.detected_mapping).map(([k,v])=>(
                <Chip key={k} color="blue">{k} → {v}</Chip>
              ))}
            </div>
          </div>

          {result.scored_preview.length>0 && (
            <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden">
              <div className="px-5 py-3 bg-slate-50 border-b border-slate-100">
                <p className="font-bold text-slate-700 text-sm">Auto-Scored Bids</p>
                <p className="text-xs text-slate-400">Composite score + market-adjusted rate applied automatically</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr className="bg-slate-800 text-white">
                    {["Carrier","Score","On-Time","Base ₹/kg","Mkt Adj ₹/kg","Recommendation"].map(h=><th key={h} className="px-4 py-2 text-left">{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {result.scored_preview.map((c,i)=>(
                      <tr key={i} className={`border-b border-slate-50 ${i===0?"bg-emerald-50":"i%2===0?bg-slate-50:bg-white"}`}>
                        <td className="px-4 py-2 font-semibold">{c.name||c.carrier_name||c.carrier_id}</td>
                        <td className="px-4 py-2"><ScoreBar score={c.composite_score}/></td>
                        <td className="px-4 py-2">{c.ontime_pct?(c.ontime_pct*100).toFixed(1)+"%":"—"}</td>
                        <td className="px-4 py-2 font-mono">₹{(c.avg_cost_per_kg||c.rate||0).toFixed(3)}</td>
                        <td className="px-4 py-2 font-mono text-amber-700">₹{(c.market_adj_rate||(c.avg_cost_per_kg*1.074)||0).toFixed(3)}</td>
                        <td className="px-4 py-2">
                          {i===0?<Chip color="green">✅ RECOMMEND</Chip>:i===1?<Chip color="yellow">⚡ ALTERNATE</Chip>:<Chip color="slate">PASS</Chip>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MarketPage() {
  const [data, setData] = useState(null);
  useEffect(()=>{ api.market().then(setData); },[]);
  if (!data) return <Loader/>;
  const riskColor = {green:"border-emerald-300 bg-emerald-50",yellow:"border-amber-300 bg-amber-50",red:"border-rose-300 bg-rose-50"};
  const textColor = {green:"text-emerald-700",yellow:"text-amber-700",red:"text-rose-700"};
  const lanes = [
    {lane:"Mumbai → Delhi",base:4.10,vol:85000},
    {lane:"Mumbai → Bangalore",base:3.75,vol:62000},
    {lane:"Chennai → Hyderabad",base:2.85,vol:47000},
    {lane:"Pune → Ahmedabad",base:2.45,vol:39000},
    {lane:"Mumbai → Hyderabad",base:3.05,vol:44000},
  ];
  return (
    <div className="space-y-4">
      <div className="bg-slate-800 rounded-xl p-5 text-white">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <p className="text-xs text-slate-400 uppercase font-bold tracking-wide">Live Market Intelligence</p>
            <p className="text-slate-300 text-xs mt-1">Source: {data.scraped_at} · API: <code>GET /api/market/intelligence</code></p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400">Composite Multiplier</p>
            <p className="text-4xl font-black text-amber-400">×{data.composite_multiplier}</p>
            <p className="text-xs text-slate-400">+{((data.composite_multiplier-1)*100).toFixed(1)}% above base</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {data.factors.map(f=>(
          <div key={f.id} className={`rounded-xl border-2 p-4 ${riskColor[f.risk]}`}>
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{f.icon}</span>
                <div>
                  <p className={`font-bold text-sm ${textColor[f.risk]}`}>{f.name}</p>
                  <p className="text-xs text-slate-500">{f.value}</p>
                </div>
              </div>
              <span className={`text-xl font-black ${textColor[f.risk]}`}>×{f.multiplier}</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">{f.detail}</p>
            <div className="flex items-center gap-2 mt-2">
              <Chip color={f.risk}>{f.trend}</Chip>
              <Chip color="slate">{f.confidence} conf.</Chip>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-100 bg-slate-50">
          <p className="font-bold text-slate-700 text-sm">💰 Adjusted Bid Rates — All Lanes</p>
        </div>
        <table className="w-full text-sm">
          <thead><tr className="bg-slate-800 text-white text-xs">
            {["Lane","Base ₹/kg","Adjusted ₹/kg","Δ%","Monthly Volume","Extra Cost/mo"].map(h=><th key={h} className="px-4 py-2 text-left">{h}</th>)}
          </tr></thead>
          <tbody>
            {lanes.map((l,i)=>{
              const adj = +(l.base*data.composite_multiplier).toFixed(3);
              const delta = +((data.composite_multiplier-1)*100).toFixed(1);
              const extra = Math.round((adj-l.base)*l.vol);
              return (
                <tr key={i} className={`border-b border-slate-50 ${i%2===0?"bg-slate-50/50":"bg-white"}`}>
                  <td className="px-4 py-2.5 font-semibold text-slate-700">{l.lane}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">₹{l.base.toFixed(2)}</td>
                  <td className="px-4 py-2.5 font-mono font-bold text-rose-600">₹{adj}</td>
                  <td className="px-4 py-2.5"><Chip color="yellow">+{delta}%</Chip></td>
                  <td className="px-4 py-2.5 text-slate-400">{(l.vol/1000).toFixed(0)}T/mo</td>
                  <td className="px-4 py-2.5 font-bold text-rose-600">+₹{(extra/1000).toFixed(0)}K</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── App Shell ────────────────────────────────────────────────────────────────
const PAGES = [
  { id:"dashboard", label:"Dashboard",   icon:"📊" },
  { id:"carriers",  label:"Carriers",    icon:"🚛" },
  { id:"lanes",     label:"Lane Scores", icon:"🛣️" },
  { id:"predict",   label:"Risk Predict",icon:"🔮" },
  { id:"upload",    label:"Upload CSV",  icon:"📤" },
  { id:"market",    label:"Market Intel",icon:"📡" },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(()=>{
    api.health()
      .then(()=>setApiStatus(DEMO_MODE?"demo":"live"))
      .catch(()=>setApiStatus("offline"));
  },[]);

  const PageComponent = { dashboard:DashboardPage, carriers:CarriersPage, lanes:LanesPage, predict:PredictPage, upload:UploadPage, market:MarketPage }[page];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-blue-950 text-white shadow-xl">
        <div className="max-w-5xl mx-auto px-5 py-4 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-blue-500 rounded-xl flex items-center justify-center text-lg font-black shadow">P</div>
            <div>
              <h1 className="font-black text-base tracking-tight">Procurement Co-Pilot</h1>
              <p className="text-xs text-slate-400">React UI  ·  Flask Python API  ·  Sklearn ML Engine</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${apiStatus==="live"?"bg-emerald-400 animate-pulse":apiStatus==="demo"?"bg-amber-400":"bg-red-400"}`}/>
            <span className="text-xs text-slate-300 font-medium">
              {apiStatus==="live"?"Flask API Live":apiStatus==="demo"?"Demo Mode (set DEMO_MODE=false for live)":"Flask Offline"}
            </span>
          </div>
        </div>
        {/* Nav */}
        <div className="max-w-5xl mx-auto px-5 flex gap-0.5 overflow-x-auto pb-0">
          {PAGES.map(p=>(
            <button key={p.id} onClick={()=>setPage(p.id)}
              className={`px-4 py-2.5 text-xs font-semibold whitespace-nowrap border-b-2 transition-colors ${page===p.id?"border-blue-400 text-white bg-white/5":"border-transparent text-slate-400 hover:text-white"}`}>
              {p.icon} {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Page */}
      <div className="flex-1 max-w-5xl mx-auto w-full px-5 py-6">
        <PageComponent/>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-200 bg-white px-5 py-3">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-slate-400 flex-wrap gap-2">
          <span>Backend: <code className="bg-slate-100 px-1 rounded">python3 backend/app.py</code> · Frontend: set <code className="bg-slate-100 px-1 rounded">DEMO_MODE=false</code> to go live</span>
          <span>Models: GradientBoosting (83.7%) + LogisticRegression (85.0%) · PuLP optimizer</span>
        </div>
      </div>
    </div>
  );
}
