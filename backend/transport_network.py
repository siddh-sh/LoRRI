"""
For Railway: how to plug in real IRCTC/NTES API for live train schedules
  - Subscribe to CRIS/NTES XML APIs or third-party wrappers like RailYatri API.
  - Query for live freight train running status using the Train Number and Station Code.
  - Update edge weights dynamically in NetworkX.
For Airways: how to use FlightAware or OAG API for real AWB tracking
  - Use FlightAware AeroAPI or OAG live flight status.
  - Tie the AWB (Air Waybill) to specific flight numbers.
  - Calculate delays based on actual take-off / ETAs.
For Road: how to switch from Directions API to Route Optimization API (fleet)
  - Use Google Route Optimization API or NextBillion.ai equivalent.
  - Pass fleet capacity, time windows, and shift constraints.
"""
import os
import math
import requests
import networkx as nx

CITY_COORDS = {"Agra": [27.1752, 78.0098], "Ahmedabad": [23.0215, 72.5801], "Amritsar": [31.6357, 74.8787], "Bangalore": [12.9768, 77.5901], "Bhopal": [23.2585, 77.4020], "Bhubaneswar": [20.2603, 85.8395], "Chandigarh": [30.7334, 76.7797], "Chennai": [13.0837, 80.2702], "Coimbatore": [11.0018, 76.9628], "Dehradun": [30.3256, 78.0437], "Delhi": [28.6139, 77.2090], "Guwahati": [26.1806, 91.7539], "Hyderabad": [17.3606, 78.4741], "Indore": [22.7204, 75.8682], "Jabalpur": [23.1702, 79.9325], "Jaipur": [26.9155, 75.8190], "Kanpur": [26.4609, 80.3218], "Kochi": [9.9679, 76.2444], "Kolkata": [22.5726, 88.3639], "Lucknow": [26.8381, 80.9346], "Madurai": [9.9261, 78.1141], "Mangalore": [12.8698, 74.8430], "Mumbai": [19.0550, 72.8692], "Nagpur": [21.1498, 79.0821], "Nashik": [20.0112, 73.7902], "Noida": [28.5706, 77.3272], "Patna": [25.6093, 85.1235], "Pune": [18.5214, 73.8545], "Raipur": [21.2381, 81.6337], "Rajkot": [22.3053, 70.8028], "Ranchi": [23.3701, 85.3250], "Surat": [21.2095, 72.8317], "Thiruvananthapuram": [8.4882, 76.9476], "Vadodara": [22.2973, 73.1943], "Varanasi": [25.3356, 83.0076], "Vijayawada": [16.5115, 80.6160], "Visakhapatnam": [17.6936, 83.2921]}

CITY_IATA = {"Agra": "AGR", "Ahmedabad": "AMD", "Amritsar": "ATQ", "Bangalore": "BLR", "Bhopal": "BHO", "Bhubaneswar": "BBI", "Chandigarh": "IXC", "Chennai": "MAA", "Coimbatore": "CJB", "Dehradun": "DED", "Delhi": "DEL", "Guwahati": "GAU", "Hyderabad": "HYD", "Indore": "IDR", "Jabalpur": "JLR", "Jaipur": "JAI", "Kanpur": "KNU", "Kochi": "COK", "Kolkata": "CCU", "Lucknow": "LKO", "Madurai": "IXM", "Mangalore": "IXE", "Mumbai": "BOM", "Nagpur": "NAG", "Nashik": "ISK", "Noida": "DEL", "Patna": "PAT", "Pune": "PNQ", "Raipur": "RPR", "Rajkot": "RAJ", "Ranchi": "IXR", "Surat": "STV", "Thiruvananthapuram": "TRV", "Vadodara": "BDQ", "Varanasi": "VNS", "Vijayawada": "VGA", "Visakhapatnam": "VTZ"}

CITY_RAIL_CODE = {"Agra": "AGC", "Ahmedabad": "ADI", "Amritsar": "ASR", "Bangalore": "SBC", "Bhopal": "BPL", "Bhubaneswar": "BBS", "Chandigarh": "CDG", "Chennai": "MAS", "Coimbatore": "CBE", "Dehradun": "DDN", "Delhi": "NDLS", "Guwahati": "GHY", "Hyderabad": "SC", "Indore": "INDB", "Jabalpur": "JBP", "Jaipur": "JP", "Kanpur": "CNB", "Kochi": "ERS", "Kolkata": "HWH", "Lucknow": "LKO", "Madurai": "MDU", "Mangalore": "MAQ", "Mumbai": "CSTM", "Nagpur": "NGP", "Nashik": "NK", "Noida": "NDLS", "Patna": "PNBE", "Pune": "PUNE", "Raipur": "R", "Rajkot": "RJT", "Ranchi": "RNC", "Surat": "ST", "Thiruvananthapuram": "TVC", "Vadodara": "BRC", "Varanasi": "BSB", "Vijayawada": "BZA", "Visakhapatnam": "VSKP"}

CARRIERS = {
    "roadways": [{"id":"BlueDart Surface","rate":2.1,"surcharge":0.10}, {"id":"GATI","rate":1.8,"surcharge":0.12}, {"id":"Delhivery","rate":1.6,"surcharge":0.08}, {"id":"VRL","rate":1.5,"surcharge":0.08}, {"id":"TCI","rate":1.6,"surcharge":0.10}, {"id":"Safexpress","rate":2.2,"surcharge":0.10}],
    "railways": [{"id":"CONCOR","rate":1.6,"surcharge":0}, {"id":"DFC","rate":1.5,"surcharge":0}, {"id":"KRIBHCO","rate":1.4,"surcharge":0}, {"id":"Indian Railways Goods","rate":1.3,"surcharge":0}, {"id":"TransRail","rate":1.8,"surcharge":0}, {"id":"Hind Terminals","rate":2.1,"surcharge":0}],
    "airways": [{"id":"IndiGo","rate":22,"surcharge":0.15}, {"id":"Air India","rate":20,"surcharge":0.14}, {"id":"SpiceJet","rate":18,"surcharge":0.14}, {"id":"BlueDart Aviation","rate":28,"surcharge":0.20}, {"id":"StarAir","rate":19,"surcharge":0.14}, {"id":"Quikjet","rate":25,"surcharge":0.16}]
}

# In-memory route cache
route_cache = {}

# Build NetworkX graph once
rail_graph = nx.Graph()
_rail_segments = [
    ("NDLS", "AGC", 200), ("AGC", "JP", 240), ("NDLS", "CDG", 260), ("CDG", "ASR", 230),
    ("NDLS", "CNB", 440), ("CNB", "LKO", 75), ("LKO", "BSB", 320), ("BSB", "PNBE", 230), ("PNBE", "HWH", 530),
    ("HWH", "BBS", 440), ("BBS", "VSKP", 440), ("VSKP", "BZA", 350), ("BZA", "MAS", 430),
    ("MAS", "SBC", 360), ("SBC", "CBE", 380), ("CBE", "ERS", 200), ("ERS", "TVC", 220),
    ("MAS", "MDU", 460), ("SBC", "MAQ", 350), ("CSTM", "PUNE", 192), ("PUNE", "SBC", 840),
    ("CSTM", "ST", 260), ("ST", "BRC", 130), ("BRC", "ADI", 100), ("ADI", "RJT", 220),
    ("CSTM", "NK", 170), ("NK", "INDB", 420), ("INDB", "BPL", 190), ("BPL", "AGC", 510),
    ("BPL", "NGP", 350), ("NGP", "SC", 500), ("SC", "BZA", 350), ("SC", "SBC", 600),
    ("NGP", "R", 290), ("R", "BBS", 560), ("NDLS", "DDN", 250), ("HWH", "GHY", 960),
    ("PNBE", "RNC", 330), ("RNC", "HWH", 410), ("JBP", "BPL", 320), ("JBP", "CNB", 430)
]
for u, v, d in _rail_segments:
    rail_graph.add_edge(u, v, weight=d)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_route(origin, dest, mode):
    cache_key = f"{origin}|{dest}|{mode}"
    if cache_key in route_cache:
        return route_cache[cache_key]
    
    if origin not in CITY_COORDS or dest not in CITY_COORDS:
        return None
        
    lat1, lon1 = CITY_COORDS[origin]
    lat2, lon2 = CITY_COORDS[dest]
    crow_fly = haversine(lat1, lon1, lat2, lon2)
    
    res = {}
    if mode == "roadways":
        # Check if Google Maps key exists
        gmaps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if gmaps_key:
            try:
                url = f"https://maps.googleapis.com/maps/api/directions/json?origin={lat1},{lon1}&destination={lat2},{lon2}&key={gmaps_key}"
                data = requests.get(url, timeout=4).json()
                if data.get("status") == "OK":
                    route = data["routes"][0]["legs"][0]
                    dist_km = route["distance"]["value"] / 1000
                    duration_sec = route["duration"]["value"]
                    res = {
                        "distance_km": dist_km,
                        "transit_days": max(1.0, round(duration_sec / 86400 * 1.5, 1)), # Added buffer
                        "path_nodes": [origin, dest],
                        "via_hubs": [],
                        "source": "google_maps"
                    }
                    route_cache[cache_key] = res
                    return res
            except:
                pass
                
        # Fallback
        dist_km = crow_fly * 1.32
        res = {
            "distance_km": dist_km,
            "transit_days": max(1.0, round(dist_km / 500, 1)), # approx 500km per day
            "path_nodes": [origin, dest],
            "via_hubs": [],
            "source": "haversine_estimate"
        }
        
    elif mode == "railways":
        src_code = CITY_RAIL_CODE.get(origin)
        dst_code = CITY_RAIL_CODE.get(dest)
        if src_code in rail_graph and dst_code in rail_graph:
            try:
                path = nx.shortest_path(rail_graph, src_code, dst_code, weight="weight")
                dist_km = nx.shortest_path_length(rail_graph, src_code, dst_code, weight="weight")
                hours = (dist_km / 45.0) + (0.5 * max(0, len(path)-2)) + 8.0 # buffer loading
                res = {
                    "distance_km": dist_km,
                    "transit_days": round(hours / 24.0, 1),
                    "path_nodes": path,
                    "via_hubs": path[1:-1],
                    "source": "networkx_dijkstra"
                }
            except nx.NetworkXNoPath:
                res = {"distance_km": crow_fly * 1.2, "transit_days": round((crow_fly * 1.2)/45/24 + 1, 1), "path_nodes": [src_code, dst_code], "via_hubs": [], "source": "rail_estimate"}
        else:
            res = {"distance_km": crow_fly * 1.2, "transit_days": round((crow_fly * 1.2)/45/24 + 1, 1), "path_nodes": [src_code, dst_code], "via_hubs": [], "source": "rail_estimate"}
            
    elif mode == "airways":
        dist_km = crow_fly
        hours = (dist_km / 750.0) + 5.0 # buffer handling
        src_iata = CITY_IATA.get(origin)
        dst_iata = CITY_IATA.get(dest)
        res = {
            "distance_km": dist_km,
            "transit_days": round(hours / 24.0, 2),
            "path_nodes": [src_iata, dst_iata],
            "via_hubs": [],
            "source": "air_haversine"
        }
        
    if mode != "roadways" and res.get("transit_days", 0) < 0.1:
         res["transit_days"] = 0.5

    route_cache[cache_key] = res
    return res
