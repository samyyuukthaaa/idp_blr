# check_graph.py
import pickle

with open("time_expanded_graph.pkl", "rb") as f:
    edges_by_stop, stop_names, stop_coords, name_to_id = pickle.load(f)

walk_count = 0
for stop_a, connections in edges_by_stop.items():
    for stop_b, options in connections.items():
        if options.get("walk_duration"):
            walk_count += 1

print(f"Total walk-transfer edges: {walk_count}")

start = name_to_id.get("Hosahalli")
end = name_to_id.get("Whitefield (Kadugodi)")
print(f"Hosahalli found: {start is not None}")
print(f"Whitefield (Kadugodi) found: {end is not None}")

# Check if Hosahalli has ANY outgoing edges at all
if start:
    print(f"Hosahalli outgoing connections: {len(edges_by_stop.get(start, {}))}")