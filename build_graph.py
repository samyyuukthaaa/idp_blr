import gtfs_kit as gk
import pandas as pd
import networkx as nx
import pickle
from math import radians, sin, cos, sqrt, atan2

# --- Load feeds ---
feed = gk.read_feed("data/gtfs", dist_units="km")
metro_feed = gk.read_feed("data/gtfs_metro", dist_units="km")

combined_stops = pd.concat([feed.stops, metro_feed.stops], ignore_index=True)
combined_stop_times = pd.concat([feed.stop_times, metro_feed.stop_times], ignore_index=True)


def time_to_seconds(t):
    try:
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# --- Build the base graph from scheduled trips ---
print("Building base graph from trips...")
G = nx.DiGraph()
st_sorted = combined_stop_times.sort_values(["trip_id", "stop_sequence"])

for trip_id, group in st_sorted.groupby("trip_id"):
    rows = group[["stop_id", "arrival_time", "departure_time"]].values
    for i in range(len(rows) - 1):
        stop_a, arr_a, dep_a = rows[i]
        stop_b, arr_b, dep_b = rows[i + 1]
        dep_sec = time_to_seconds(dep_a)
        arr_sec = time_to_seconds(arr_b)
        if dep_sec is not None and arr_sec is not None and arr_sec > dep_sec:
            G.add_edge(stop_a, stop_b, weight=arr_sec - dep_sec, trip_id=trip_id)

print(f"Base graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Add walking transfer edges between nearby bus stops and metro stations ---
print("Computing walking transfers between bus stops and metro stations...")

bus_stops = feed.stops
metro_stops = metro_feed.stops

WALK_SPEED_MPS = 1.4  # average walking speed, meters/sec
MAX_TRANSFER_DIST = 500  # meters

transfer_edges = []
for _, bstop in bus_stops.iterrows():
    for _, mstop in metro_stops.iterrows():
        dist = haversine_distance(
            bstop["stop_lat"], bstop["stop_lon"],
            mstop["stop_lat"], mstop["stop_lon"]
        )
        if dist <= MAX_TRANSFER_DIST:
            walk_time = dist / WALK_SPEED_MPS
            transfer_edges.append((bstop["stop_id"], mstop["stop_id"], walk_time))
            transfer_edges.append((mstop["stop_id"], bstop["stop_id"], walk_time))

print(f"Found {len(transfer_edges)} walking transfer connections")

for stop_a, stop_b, walk_time in transfer_edges:
    G.add_edge(stop_a, stop_b, weight=walk_time, trip_id="WALK_TRANSFER")

print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Build lookup dicts ---
stop_names = combined_stops.set_index("stop_id")["stop_name"].to_dict()
stop_coords = combined_stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
name_to_id = {v: k for k, v in stop_names.items()}

# --- Save everything ---
with open("graph_cache.pkl", "wb") as f:
    pickle.dump((G, stop_names, stop_coords, name_to_id), f)

print("Graph built and saved to graph_cache.pkl")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")