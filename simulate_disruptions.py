import pickle
import random
import pandas as pd
import networkx as nx

with open("graph_cache.pkl", "rb") as f:
    G, stop_names, stop_coords, name_to_id = pickle.load(f)

def get_mode(G, a, b):
    edge_data = G.get_edge_data(a, b)
    trip_id = edge_data.get("trip_id", "")
    if trip_id == "WALK_TRANSFER":
        return "walk"
    elif str(trip_id).startswith("metro"):
        return "metro"
    return "bus"

def simulate_journey(G, path, hour, day_of_week):
    """Walk a planned path leg by leg, injecting random delays per mode."""
    total_actual_time = 0
    total_scheduled_time = 0
    num_transfers = 0
    min_buffer = float("inf")
    bus_legs = 0
    total_legs = len(path) - 1
    prev_mode = None
    missed_connection = False

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edge_data = G.get_edge_data(a, b)
        scheduled = edge_data["weight"]
        mode = get_mode(G, a, b)

        if mode == "bus":
            bus_legs += 1
            p_delay, mean_delay = 0.35, 240   # buses: 35% chance, avg 4 min delay
        elif mode == "metro":
            p_delay, mean_delay = 0.08, 90    # metro: more reliable
        else:
            p_delay, mean_delay = 0.0, 0      # walking legs don't get "delayed"

        delay = random.expovariate(1 / mean_delay) if random.random() < p_delay and mean_delay > 0 else 0
        actual = scheduled + delay

        if mode != prev_mode and prev_mode is not None:
            num_transfers += 1
            buffer = scheduled - delay  # rough proxy: how much slack this leg had
            min_buffer = min(min_buffer, buffer)
            if delay > 300:  # over 5 min late risks a missed connection
                missed_connection = True

        total_actual_time += actual
        total_scheduled_time += scheduled
        prev_mode = mode

    return {
        "num_transfers": num_transfers,
        "total_scheduled_time": total_scheduled_time,
        "total_actual_time": total_actual_time,
        "min_buffer": min_buffer if min_buffer != float("inf") else 999,
        "pct_bus": bus_legs / total_legs if total_legs else 0,
        "hour": hour,
        "day_of_week": day_of_week,
        "missed_connection": missed_connection,
    }

def generate_training_data(G, name_to_id, n_samples=3000):
    rows = []
    all_ids = list(name_to_id.values())

    while len(rows) < n_samples:
        start, end = random.sample(all_ids, 2)
        try:
            path = nx.shortest_path(G, source=start, target=end, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if len(path) < 2:
            continue

        hour = random.randint(6, 22)
        day_of_week = random.randint(0, 6)
        deadline_slack = random.uniform(1.0, 1.5)  # user gives themselves 0-50% buffer

        result = simulate_journey(G, path, hour, day_of_week)
        deadline = result["total_scheduled_time"] * deadline_slack
        reached_on_time = int(result["total_actual_time"] <= deadline)

        rows.append({**result, "deadline_slack": deadline_slack, "reached_on_time": reached_on_time})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_training_data(G, name_to_id, n_samples=3000)
    df.to_csv("simulated_journeys.csv", index=False)
    print(df["reached_on_time"].value_counts())
    print(f"Saved {len(df)} simulated journeys to simulated_journeys.csv")