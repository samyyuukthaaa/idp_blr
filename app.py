import streamlit as st
import pickle
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from datetime import datetime
from itertools import islice

st.set_page_config(page_title="Bengaluru Transit Planner", layout="wide")
st.title("Bengaluru Multimodal Transit Journey Planner")

# --- Initialize session state ---
for key in ["path", "total_time", "error", "ranked_routes", "disrupted_path", "disrupted_leg", "disrupted_candidates"]:
    if key not in st.session_state:
        st.session_state[key] = None


# --- Load pre-built graph from disk ---
@st.cache_resource
def load_data():
    with open("graph_cache.pkl", "rb") as f:
        G, stop_names, stop_coords, name_to_id = pickle.load(f)
    return G, stop_names, stop_coords, name_to_id


@st.cache_resource
def load_deadline_model():
    with open("deadline_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


G, stop_names, stop_coords, name_to_id = load_data()
deadline_model, deadline_features = load_deadline_model()

st.caption(f"Network loaded: {G.number_of_nodes():,} stops, {G.number_of_edges():,} connections")


# --- Mode + weight helpers ---
def get_mode(graph, a, b):
    edge_data = graph.get_edge_data(a, b)
    if not edge_data:
        return "🚌 Bus"
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        best_edge = min(edge_data.values(), key=lambda d: d.get("weight", float("inf")))
        trip_id = str(best_edge.get("trip_id", ""))
    else:
        trip_id = str(edge_data.get("trip_id", ""))
    if trip_id == "WALK_TRANSFER":
        return "🚶 Walk"
    elif trip_id.startswith("metro"):
        return "🚇 Metro"
    return "🚌 Bus"


def get_edge_weight(graph, a, b):
    edge_data = graph.get_edge_data(a, b)
    if not edge_data:
        return 0
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        best_edge = min(edge_data.values(), key=lambda d: d.get("weight", float("inf")))
        return best_edge.get("weight", 0)
    return edge_data.get("weight", 0)


def path_total_time(graph, path):
    return sum(get_edge_weight(graph, path[i], path[i + 1]) for i in range(len(path) - 1))


def get_legs(graph, path):
    """Group a path into (mode, start_node, end_node) legs."""
    legs = []
    current_mode = get_mode(graph, path[0], path[1])
    leg_start = path[0]
    for i in range(1, len(path) - 1):
        mode = get_mode(graph, path[i], path[i + 1])
        if mode != current_mode:
            legs.append((current_mode, leg_start, path[i]))
            current_mode = mode
            leg_start = path[i]
    legs.append((current_mode, leg_start, path[-1]))
    return legs


# --- ML feature extraction + prediction ---
def extract_route_features(graph, path):
    num_transfers = 0
    total_scheduled_time = 0
    min_buffer = float("inf")
    bus_legs = 0
    total_legs = len(path) - 1
    prev_mode = None

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        scheduled = get_edge_weight(graph, a, b)
        mode = get_mode(graph, a, b)
        mode_key = "bus" if "Bus" in mode else ("metro" if "Metro" in mode else "walk")

        if mode_key == "bus":
            bus_legs += 1
        if prev_mode is not None and mode_key != prev_mode:
            num_transfers += 1
            min_buffer = min(min_buffer, scheduled)

        total_scheduled_time += scheduled
        prev_mode = mode_key

    return {
        "num_transfers": num_transfers,
        "total_scheduled_time": total_scheduled_time,
        "min_buffer": min_buffer if min_buffer != float("inf") else 999,
        "pct_bus": bus_legs / total_legs if total_legs else 0,
    }


def predict_success_probability(graph, path, deadline_slack, hour, day_of_week):
    feats = extract_route_features(graph, path)
    row = pd.DataFrame([{
        "num_transfers": feats["num_transfers"],
        "total_scheduled_time": feats["total_scheduled_time"],
        "min_buffer": feats["min_buffer"],
        "pct_bus": feats["pct_bus"],
        "hour": hour,
        "day_of_week": day_of_week,
        "deadline_slack": deadline_slack,
    }])[deadline_features]
    return deadline_model.predict_proba(row)[0][1]


# --- Resilience-based ranking over k candidate routes ---
def get_ranked_routes(graph, start_id, end_id, deadline_slack, hour, day_of_week, k=5):
    try:
        path_generator = nx.shortest_simple_paths(graph, source=start_id, target=end_id, weight="weight")
        candidates = list(islice(path_generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    if not candidates:
        return []

    scored = []
    times = [path_total_time(graph, p) for p in candidates]
    max_time = max(times) if max(times) > 0 else 1

    for path, total_time in zip(candidates, times):
        feats = extract_route_features(graph, path)
        success_prob = predict_success_probability(graph, path, deadline_slack, hour, day_of_week)

        # Weighted multi-criteria score (lower is better). Weights are a design choice — documented in report.
        norm_time = total_time / max_time
        norm_transfers = feats["num_transfers"] / 5  # assume 5+ transfers is "bad"
        risk = 1 - success_prob

        score = (0.4 * norm_time) + (0.25 * norm_transfers) + (0.35 * risk)

        scored.append({
            "path": path,
            "total_time": total_time,
            "num_transfers": feats["num_transfers"],
            "success_prob": success_prob,
            "score": score,
        })

    scored.sort(key=lambda r: r["score"])
    return scored


# --- ML-driven disruption-aware recovery ---
def apply_disruption_and_reroute(graph, path, leg_index, end_id, deadline_slack, hour, day_of_week, k=3):
    """Cancel the leg at leg_index, then use the ML model to pick the BEST
    of several reroute candidates — not just the shortest one."""
    legs = get_legs(graph, path)
    if leg_index >= len(legs):
        return None, None, "Invalid leg selected."

    mode, leg_start, leg_end = legs[leg_index]

    # Build a disrupted copy of the graph with that connection removed
    G_disrupted = graph.copy()
    if G_disrupted.has_edge(leg_start, leg_end):
        G_disrupted.remove_edge(leg_start, leg_end)

    disruption_point_idx = path.index(leg_start)
    prefix = path[:disruption_point_idx + 1]

    # Generate multiple reroute candidates from the disruption point onward
    try:
        path_generator = nx.shortest_simple_paths(G_disrupted, source=leg_start, target=end_id, weight="weight")
        suffix_candidates = list(islice(path_generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None, (
            f"No alternative route found after cancelling the {mode} leg "
            f"from {stop_names.get(leg_start)} to {stop_names.get(leg_end)}."
        )

    if not suffix_candidates:
        return None, None, "No alternative route found after this cancellation."

    # Score each candidate full path using the trained ML model
    scored_candidates = []
    for suffix in suffix_candidates:
        full_path = prefix[:-1] + suffix
        success_prob = predict_success_probability(graph, full_path, deadline_slack, hour, day_of_week)
        total_time = path_total_time(graph, full_path)
        scored_candidates.append({
            "path": full_path,
            "success_prob": success_prob,
            "total_time": total_time,
        })

    # Pick the reroute the MODEL rates as most likely to succeed, not just the fastest
    best = max(scored_candidates, key=lambda c: c["success_prob"])
    return best["path"], scored_candidates, None


# --- Input UI ---
all_names = sorted(name_to_id.keys())

col1, col2 = st.columns(2)
with col1:
    start_name = st.selectbox("Start Stop", all_names)
with col2:
    end_default_idx = 1 if len(all_names) > 1 else 0
    end_name = st.selectbox("End Stop", all_names, index=end_default_idx)

deadline_buffer_pct = st.slider(
    "How much extra buffer time do you want to give yourself? (%)",
    0, 50, 20,
    help="A higher buffer means a more relaxed deadline for arrival."
)

if st.button("Find Route"):
    if start_name == end_name:
        st.session_state["path"] = None
        st.session_state["ranked_routes"] = None
        st.session_state["disrupted_path"] = None
        st.session_state["disrupted_candidates"] = None
        st.session_state["error"] = "Start and destination stops must be different."
    else:
        start_id = name_to_id[start_name]
        end_id = name_to_id[end_name]
        now = datetime.now()
        deadline_slack = 1 + (deadline_buffer_pct / 100)

        ranked = get_ranked_routes(
            G, start_id, end_id,
            deadline_slack=deadline_slack, hour=now.hour, day_of_week=now.weekday(), k=5
        )

        if not ranked:
            st.session_state["path"] = None
            st.session_state["ranked_routes"] = None
            st.session_state["error"] = "No path found between these two stops."
        else:
            st.session_state["path"] = ranked[0]["path"]
            st.session_state["total_time"] = ranked[0]["total_time"]
            st.session_state["ranked_routes"] = ranked
            st.session_state["disrupted_path"] = None
            st.session_state["disrupted_candidates"] = None
            st.session_state["error"] = None


# --- Display result ---
if st.session_state.get("path"):
    path = st.session_state["path"]
    total_time = st.session_state["total_time"]
    ranked_routes = st.session_state["ranked_routes"]

    st.success(f"Best route found — {total_time / 60:.1f} minutes total")

    now = datetime.now()
    deadline_slack = 1 + (deadline_buffer_pct / 100)
    success_prob = ranked_routes[0]["success_prob"]

    prob_col, time_col = st.columns(2)
    with prob_col:
        st.metric("Probability of arriving on time", f"{success_prob * 100:.0f}%")
    with time_col:
        adjusted_deadline_min = (total_time * deadline_slack) / 60
        st.metric("Effective deadline (with buffer)", f"{adjusted_deadline_min:.0f} min")

    if success_prob < 0.5:
        st.warning("This route has a high risk of missing your deadline given typical delays.")

    # --- Journey legs ---
    legs = get_legs(G, path)
    st.markdown("### Your Journey")
    for i, (mode, start_node, end_node) in enumerate(legs, 1):
        s_name = stop_names.get(start_node, str(start_node))
        e_name = stop_names.get(end_node, str(end_node))
        st.markdown(f"**{i}. {mode}** from *{s_name}* to *{e_name}*")

    with st.expander("View full stop-by-stop list"):
        st.write(" → ".join(stop_names.get(s, str(s)) for s in path))

    # --- Resilience-ranked alternatives ---
    if ranked_routes and len(ranked_routes) > 1:
        with st.expander(f"View {len(ranked_routes)} ranked alternative routes"):
            for i, route in enumerate(ranked_routes, 1):
                st.markdown(
                    f"**Option {i}** — {route['total_time']/60:.1f} min, "
                    f"{route['num_transfers']} transfers, "
                    f"{route['success_prob']*100:.0f}% on-time probability "
                    f"(resilience score: {route['score']:.2f}, lower is better)"
                )

    # --- Disruption simulation (ML-driven recovery) ---
    st.markdown("### Simulate a Disruption")
    leg_options = [f"{i+1}. {mode} from {stop_names.get(s)} to {stop_names.get(e)}"
                   for i, (mode, s, e) in enumerate(legs)]
    selected_leg = st.selectbox("Cancel this leg and see the alternative route:", leg_options, key="leg_picker")
    leg_idx = leg_options.index(selected_leg)

    if st.button("Simulate Cancellation & Reroute"):
        end_id = name_to_id[end_name]
        now = datetime.now()
        deadline_slack = 1 + (deadline_buffer_pct / 100)

        new_path, candidates, recovery_error = apply_disruption_and_reroute(
            G, path, leg_idx, end_id,
            deadline_slack=deadline_slack, hour=now.hour, day_of_week=now.weekday(), k=3
        )
        if recovery_error:
            st.session_state["disrupted_path"] = None
            st.session_state["disrupted_candidates"] = None
            st.error(recovery_error)
        else:
            st.session_state["disrupted_path"] = new_path
            st.session_state["disrupted_candidates"] = candidates

    if st.session_state.get("disrupted_path"):
        disrupted_path = st.session_state["disrupted_path"]
        disrupted_time = path_total_time(G, disrupted_path)
        disrupted_legs = get_legs(G, disrupted_path)
        disrupted_candidates = st.session_state.get("disrupted_candidates") or []

        chosen_prob = next((c["success_prob"] for c in disrupted_candidates if c["path"] == disrupted_path), None)

        st.info(
            f"**ML-selected recovery route** — {disrupted_time/60:.1f} minutes total "
            f"({(disrupted_time - total_time)/60:+.1f} min vs. original)"
            + (f", model-predicted success: {chosen_prob*100:.0f}%" if chosen_prob is not None else "")
        )

        if len(disrupted_candidates) > 1:
            with st.expander(f"Why this route? Compared {len(disrupted_candidates)} candidates"):
                for i, c in enumerate(sorted(disrupted_candidates, key=lambda x: -x["success_prob"]), 1):
                    marker = "✅ **Selected**" if c["path"] == disrupted_path else "—"
                    st.markdown(f"{marker} Option {i}: {c['total_time']/60:.1f} min, {c['success_prob']*100:.0f}% predicted success")

        for i, (mode, s, e) in enumerate(disrupted_legs, 1):
            st.markdown(f"**{i}. {mode}** from *{stop_names.get(s)}* to *{stop_names.get(e)}*")

    # --- Map Visualization ---
    display_path = st.session_state.get("disrupted_path") or path
    valid_points = [
        (s, (stop_coords[s]["stop_lat"], stop_coords[s]["stop_lon"]))
        for s in display_path
        if s in stop_coords and "stop_lat" in stop_coords[s] and "stop_lon" in stop_coords[s]
    ]

    if valid_points:
        route_coords = [pt[1] for pt in valid_points]
        start_stop, start_coord = valid_points[0]
        end_stop, end_coord = valid_points[-1]

        mid = route_coords[len(route_coords) // 2]
        m = folium.Map(location=mid, zoom_start=12)
        color = "#dc2626" if st.session_state.get("disrupted_path") else "#2563eb"
        folium.PolyLine(route_coords, color=color, weight=5, opacity=0.85).add_to(m)

        folium.Marker(start_coord, popup=f"Start: {stop_names.get(start_stop)}",
                      icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(end_coord, popup=f"End: {stop_names.get(end_stop)}",
                      icon=folium.Icon(color="red", icon="flag")).add_to(m)

        for stop_id, coord in valid_points[1:-1]:
            folium.CircleMarker(coord, radius=3, color="#f97316", fill=True,
                                 fill_opacity=0.9, popup=stop_names.get(stop_id)).add_to(m)

        m.fit_bounds([min(route_coords), max(route_coords)], padding=(30, 30))
        st_folium(m, width=1000, height=500)

elif st.session_state.get("error"):
    st.error(st.session_state["error"])