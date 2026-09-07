# build_time_expanded_graph.py
import gtfs_kit as gk
import pandas as pd
import pickle
from collections import defaultdict

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

# edges_by_stop[stop_a][stop_b] = {"scheduled": [(dep_sec, arr_sec, trip_id), ...], "walk_duration": None}
edges_by_stop = defaultdict(lambda: defaultdict(lambda: {"scheduled": [], "walk_duration": None}))

st_sorted = combined_stop_times.sort_values(["trip_id", "stop_sequence"])
for trip_id, group in st_sorted.groupby("trip_id"):
    rows = group[["stop_id", "arrival_time", "departure_time"]].values
    for i in range(len(rows) - 1):
        stop_a, arr_a, dep_a = rows[i]
        stop_b, arr_b, dep_b = rows[i + 1]
        dep_sec = time_to_seconds(dep_a)
        arr_sec = time_to_seconds(arr_b)
        if dep_sec is not None and arr_sec is not None and arr_sec > dep_sec:
            edges_by_stop[stop_a][stop_b]["scheduled"].append((dep_sec, arr_sec, trip_id))

# Sort each stop-pair's departures by time so we can binary-search later
for stop_a in edges_by_stop:
    for stop_b in edges_by_stop[stop_a]:
        edges_by_stop[stop_a][stop_b]["scheduled"].sort(key=lambda x: x[0])

# --- Add fixed-duration walking transfers between very close stops ---
# (reuse whatever logic you already had generating WALK_TRANSFER edges —
#  set walk_duration in seconds, always available regardless of time)
# Example placeholder — replace with your real walk-transfer generation logic:
# edges_by_stop[stop_a][stop_b]["walk_duration"] = 180  # 3 min walk

stop_names = combined_stops.set_index("stop_id")["stop_name"].to_dict()
stop_coords = combined_stops.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
name_to_id = {v: k for k, v in stop_names.items()}

# Convert defaultdicts to plain dicts before pickling
edges_by_stop = {k: dict(v) for k, v in edges_by_stop.items()}

with open("time_expanded_graph.pkl", "wb") as f:
    pickle.dump((edges_by_stop, stop_names, stop_coords, name_to_id), f)

print(f"Saved schedule-aware graph: {len(edges_by_stop)} stops with outgoing connections")