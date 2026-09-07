# time_dependent_routing.py
import heapq
from bisect import bisect_left

def find_next_departure(scheduled, current_time):
    """scheduled is sorted by dep_time. Find the earliest (dep, arr, trip_id) with dep >= current_time."""
    if not scheduled:
        return None
    idx = bisect_left(scheduled, (current_time, -1, ""))
    if idx < len(scheduled):
        return scheduled[idx]
    return None  # nothing available today — this is your "5am next metro" case

def time_dependent_shortest_path(edges_by_stop, start_id, end_id, start_time_sec):
    dist = {start_id: start_time_sec}
    prev = {}
    visited = set()
    pq = [(start_time_sec, start_id)]

    while pq:
        current_time, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end_id:
            break

        for next_stop, options in edges_by_stop.get(node, {}).items():
            # Option A: scheduled trip — respects real timetable, including "next one is hours away"
            next_dep = find_next_departure(options["scheduled"], current_time)
            if next_dep:
                dep_sec, arr_sec, trip_id = next_dep
                if arr_sec < dist.get(next_stop, float("inf")):
                    dist[next_stop] = arr_sec
                    prev[next_stop] = (node, trip_id, dep_sec, arr_sec)
                    heapq.heappush(pq, (arr_sec, next_stop))

            # Option B: walking transfer — always available, no wait
            if options.get("walk_duration"):
                arr_sec = current_time + options["walk_duration"]
                if arr_sec < dist.get(next_stop, float("inf")):
                    dist[next_stop] = arr_sec
                    prev[next_stop] = (node, "WALK_TRANSFER", current_time, arr_sec)
                    heapq.heappush(pq, (arr_sec, next_stop))

    if end_id not in prev:
        return None, None

    # Reconstruct the path as a list of legs
    legs = []
    node = end_id
    while node != start_id:
        p_node, trip_id, dep_sec, arr_sec = prev[node]
        legs.append({"from": p_node, "to": node, "trip_id": trip_id, "dep": dep_sec, "arr": arr_sec})
        node = p_node
    legs.reverse()

    return dist[end_id], legs