# test_query.py
import pickle
from time_dependent_routing import time_dependent_shortest_path

with open("time_expanded_graph.pkl", "rb") as f:
    edges_by_stop, stop_names, stop_coords, name_to_id = pickle.load(f)

start = name_to_id["Hosahalli"]
end = name_to_id["Whitefield (Kadugodi)"]

# Try querying at, say, 11:30 PM (should trigger the "wait till 5am" scenario if metro shuts down)
query_time = 23 * 3600 + 30 * 60  # 23:30:00 in seconds

arrival, legs = time_dependent_shortest_path(edges_by_stop, start, end, query_time)

if legs:
    for leg in legs:
        h1, h2 = leg["dep"] // 3600, leg["arr"] // 3600
        print(f"{stop_names[leg['from']]} → {stop_names[leg['to']]} via {leg['trip_id']} "
              f"(dep {leg['dep']//3600:02d}:{(leg['dep']%3600)//60:02d}, "
              f"arr {leg['arr']//3600:02d}:{(leg['arr']%3600)//60:02d})")
else:
    print("No route found.")