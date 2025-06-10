import argparse
import csv
import os
import sys
import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
import networkx as nx

# Allow running this file directly without installing the package
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trail_route_ai import planner_utils

# Type aliases
Edge = planner_utils.Edge

def midpoint(edge: Edge) -> Tuple[float, float]:
    sx, sy = edge.start
    ex, ey = edge.end
    return ((sx + ex) / 2.0, (sy + ey) / 2.0)

def total_time(
    edges: List[Edge], pace: float, grade: float, road_pace: float
) -> float:
    return sum(
        planner_utils.estimate_time(e, pace, grade, road_pace) for e in edges
    )

def build_nx_graph(edges: List[Edge], pace: float, grade: float, road_pace: float) -> nx.Graph:
    G = nx.Graph()
    for e in edges:
        w = planner_utils.estimate_time(e, pace, grade, road_pace)
        G.add_edge(e.start, e.end, weight=w, edge=e)
    return G


def identify_macro_clusters(all_trail_segments: List[Edge], all_road_segments: List[Edge], pace: float, grade: float, road_pace: float) -> List[List[Edge]]:
    """
    Identifies geographically distinct clusters of trail segments based on graph connectivity.
    Segments within a cluster are connectable by foot (trails or permissible roads).
    Segments in different clusters would typically require a vehicle transfer.
    """
    graph_edges = all_trail_segments + all_road_segments
    G = build_nx_graph(graph_edges, pace, grade, road_pace)
    macro_clusters: List[List[Edge]] = []
    assigned_segment_ids: set[str | int] = set()

    for component_nodes in nx.connected_components(G):
        current_cluster_segments: List[Edge] = []
        for seg in all_trail_segments:
            if seg.seg_id is not None and seg.seg_id in assigned_segment_ids:
                continue
            if seg.start in component_nodes or seg.end in component_nodes:
                current_cluster_segments.append(seg)
                if seg.seg_id is not None:
                    assigned_segment_ids.add(seg.seg_id)

        if current_cluster_segments:
            macro_clusters.append(current_cluster_segments)

    return macro_clusters


def nearest_node(nodes: List[Tuple[float, float]], point: Tuple[float, float]):
    return min(nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2)


def edges_from_path(G: nx.Graph, path: List[Tuple[float, float]]) -> List[Edge]:
    out = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if data:
            out.append(data[0]["edge"] if 0 in data else data["edge"])
    return out


def plan_route(
    G: nx.Graph,
    edges: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_road: float,
    road_threshold: float,
) -> List[Edge]:
    """Return a continuous route connecting ``edges`` starting from ``start``.

    ``max_road`` limits road mileage for any connector. ``road_threshold``
    expresses the additional time we're willing to spend to stay on trail.
    If a trail connector is within ``road_threshold`` of the best road option
    (in terms of time), the trail is chosen.
    """
    remaining = edges[:]
    route: List[Edge] = []
    cur = start
    while remaining:
        candidates = []
        for e in remaining:
            for end in [e.start, e.end]:
                try:
                    path = nx.shortest_path(G, cur, end, weight="weight")
                    edges_path = edges_from_path(G, path)
                    road_dist = sum(ed.length_mi for ed in edges_path if ed.kind == "road")
                    if road_dist > max_road:
                        continue
                    time = sum(
                        planner_utils.estimate_time(ed, pace, grade, road_pace)
                        for ed in edges_path
                    )
                    time += planner_utils.estimate_time(e, pace, grade, road_pace)
                    uses_road = any(ed.kind == "road" for ed in edges_path)
                    candidates.append((time, uses_road, e, end, edges_path))
                except nx.NetworkXNoPath:
                    continue

        if not candidates:
            # Get details for error message
            current_last_segment_name = route[-1].name if route and hasattr(route[-1], 'name') and route[-1].name else (str(route[-1].seg_id) if route and hasattr(route[-1], 'seg_id') else "the route start")
            remaining_segment_names = [s.name or str(s.seg_id) for s in remaining]

            print(
                f"Error in plan_route: Could not find a valid path from '{current_last_segment_name}' "
                f"to any of the remaining segments: {remaining_segment_names} "
                f"within the given constraints (e.g., max_road for connector). "
                f"This cluster cannot be routed continuously.",
                file=sys.stderr
            )
            return [] # Signify failure to route this cluster

        best = min(candidates, key=lambda c: c[0])
        trail_candidates = [c for c in candidates if not c[1]]
        if trail_candidates:
            best_trail = min(trail_candidates, key=lambda c: c[0])
            if best_trail[0] <= best[0] * (1 + road_threshold):
                chosen = best_trail
            else:
                chosen = best
        else:
            chosen = best

        time, uses_road, e, end, best_path_edges = chosen
        route.extend(best_path_edges)
        if end == e.start:
            route.append(e)
            cur = e.end
        else:
            # reverse orientation
            rev = Edge(e.seg_id, e.name, e.end, e.start, e.length_mi, e.elev_gain_ft, list(reversed(e.coords)))
            route.append(rev)
            cur = rev.end
        remaining.remove(e)

    if cur == start:
        return route

    G_for_path_back = G.copy()
    for edge_obj in edges: # 'edges' is the original list of segments for this cluster
        if G_for_path_back.has_edge(edge_obj.start, edge_obj.end):
            # Ensure the edge data is directly accessible; MultiDiGraph might have a list
            # For simple Graph, this should be fine. If it's a MultiGraph, one might need to iterate G_for_path_back.get_edge_data
            edge_data = G_for_path_back[edge_obj.start][edge_obj.end]
            if isinstance(edge_data, list): # Should not happen with G = nx.Graph()
                 # This case is more complex if multiple edges connect same nodes.
                 # Assuming build_nx_graph creates simple graph where one edge = one set of attributes.
                 # For now, if it's a list, we might be modifying the wrong one or need to find the specific one.
                 # However, current build_nx_graph adds one edge.
                 pass # Or log a warning if this structure is unexpected.


            # Check if 'weight' exists, add if not (though build_nx_graph should add it)
            if 'weight' not in edge_data:
                 edge_data['weight'] = planner_utils.estimate_time(edge_obj, pace, grade, road_pace) # Re-estimate if missing

            current_weight = edge_data['weight']
            edge_data['weight'] = current_weight * 10.0
            # For MultiGraph, one would do: G_for_path_back[edge_obj.start][edge_obj.end][key]['weight'] *= 10.0

    try:
        path_back_nodes = nx.shortest_path(G_for_path_back, cur, start, weight="weight")
        path_back_edges = edges_from_path(G, path_back_nodes) # Use original G for edge objects
        route.extend(path_back_edges)
    except nx.NetworkXNoPath:
        # Fallback to original graph if modified graph yields no path
        try:
            path_back_nodes_orig = nx.shortest_path(G, cur, start, weight="weight")
            route.extend(edges_from_path(G, path_back_nodes_orig))
        except nx.NetworkXNoPath:
            # No path back found even on original graph, or cur == start
            pass

    return route


def parse_remaining(value: str) -> List[str]:
    if os.path.exists(value):
        with open(value) as f:
            text = f.read()
    else:
        text = value
    items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
    return items


def cluster_segments(
    edges: List[Edge],
    pace: float,
    grade: float,
    budget: float,
    max_clusters: int,
    road_pace: float,
) -> List[List[Edge]]:
    if not edges:
        return []
    pts = np.array([midpoint(e) for e in edges])
    k = min(max_clusters, len(edges))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(pts)
    initial: Dict[int, List[Edge]] = defaultdict(list)
    for lbl, e in zip(labels, edges):
        initial[lbl].append(e)
    clusters: List[List[Edge]] = []
    for group in initial.values():
        group = sorted(
            group,
            key=lambda e: planner_utils.estimate_time(e, pace, grade, road_pace),
            reverse=True,
        )
        cur: List[Edge] = []
        t = 0.0
        for e in group:
            et = planner_utils.estimate_time(e, pace, grade, road_pace)
            if t + et > budget and cur:
                clusters.append(cur)
                cur = [e]
                t = et
            else:
                cur.append(e)
                t += et
        if cur:
            clusters.append(cur)
    while len(clusters) < max_clusters:
        clusters.append([])
    while len(clusters) > max_clusters:
        clusters.sort(key=lambda c: total_time(c, pace, grade, road_pace))
        small = clusters.pop(0)
        merged = False
        for i, other in enumerate(clusters):
            if total_time(other, pace, grade, road_pace) + total_time(small, pace, grade, road_pace) <= budget:
                clusters[i] = other + small
                merged = True
                break
        if not merged:
            clusters.append(small)
            break
    return clusters[:max_clusters]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Challenge route planner")
    parser.add_argument("--start-date", required=True, help="Challenge start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Challenge end date YYYY-MM-DD")
    parser.add_argument("--time", required=True, help="Daily time budget")
    parser.add_argument("--pace", required=True, type=float, help="Base running pace (min/mi)")
    parser.add_argument("--grade", type=float, default=0.0, help="Seconds per 100ft climb")
    parser.add_argument("--segments", default="data/traildata/trail.json")
    parser.add_argument("--roads", help="Optional road connector GeoJSON")
    parser.add_argument("--max-road", type=float, default=1.0, help="Max road distance per connector (mi)")
    parser.add_argument(
        "--road-threshold",
        type=float,
        default=0.1,
        help="Choose road connector only if it is this fraction faster than trail",
    )
    parser.add_argument("--road-pace", type=float, default=18.0, help="Pace on roads (min/mi)")
    parser.add_argument("--perf", default="data/segment_perf.csv")
    parser.add_argument("--year", type=int)
    parser.add_argument("--remaining")
    parser.add_argument("--output", default="challenge_plan.csv")
    parser.add_argument("--gpx-dir", default="gpx")
    parser.add_argument(
        "--mark-road-transitions",
        action="store_true",
        help="Annotate GPX files with waypoints and track extensions for road sections",
    )
    parser.add_argument("--average-driving-speed-mph", type=float, default=30.0, help="Average driving speed in mph for estimating travel time between activity clusters.")
    parser.add_argument("--max-drive-minutes-per-transfer", type=float, default=30.0, help="Maximum allowed driving time in minutes for a single transfer between activity clusters on the same day.")
    args = parser.parse_args(argv)

    start_date = datetime.date.fromisoformat(args.start_date)
    end_date = datetime.date.fromisoformat(args.end_date)
    if end_date < start_date:
        parser.error("--end-date must not be before --start-date")
    num_days = (end_date - start_date).days + 1

    budget = planner_utils.parse_time_budget(args.time)
    all_trail_segments = planner_utils.load_segments(args.segments)
    all_road_segments: List[Edge] = []
    if args.roads:
        all_road_segments = planner_utils.load_roads(args.roads)

    # This graph is used for on-foot routing *within* macro-clusters
    on_foot_routing_graph_edges = all_trail_segments + all_road_segments
    G = build_nx_graph(on_foot_routing_graph_edges, args.pace, args.grade, args.road_pace)

    completed_segment_ids = planner_utils.load_completed(args.perf, args.year or 0)

    current_challenge_segment_ids = None
    if args.remaining:
        current_challenge_segment_ids = set(parse_remaining(args.remaining))
    if current_challenge_segment_ids is None:
        current_challenge_segment_ids = {str(e.seg_id) for e in all_trail_segments} - completed_segment_ids

    current_challenge_segments = [e for e in all_trail_segments if str(e.seg_id) in current_challenge_segment_ids]

    # nodes list might be useful later for starting points, keep it around
    # nodes = list({e.start for e in on_foot_routing_graph_edges} | {e.end for e in on_foot_routing_graph_edges})

    potential_macro_clusters = identify_macro_clusters(
        current_challenge_segments, # Only uncompleted trail segments
        all_road_segments,          # All road segments
        args.pace,
        args.grade,
        args.road_pace
    )
    unplanned_macro_clusters = [mc for mc in potential_macro_clusters if mc] # Filter out empty lists

    all_on_foot_nodes = list(G.nodes()) # Get all nodes from the on-foot routing graph

    os.makedirs(args.gpx_dir, exist_ok=True)
    # summary_rows = [] # This will be populated by the new planning loop (or rather, daily_plans will be used)
    daily_plans = []

    for day_idx in range(num_days):
        cur_date = start_date + datetime.timedelta(days=day_idx)
        todays_total_budget_minutes = budget # budget is args.time parsed
        activities_for_this_day = []
        time_spent_on_activities_today = 0.0
        time_spent_on_drives_today = 0.0
        last_activity_end_coord = None

        while True:
            best_cluster_to_add_info = None

            eligible_clusters_indices = [] # Not strictly used yet, but good for future refinement

            for cluster_idx, cluster_candidate in enumerate(unplanned_macro_clusters):
                if not cluster_candidate: # Should have been filtered by list comprehension above
                    continue

                if not all_on_foot_nodes:
                    print("Warning: No nodes in on_foot_routing_graph. Cannot determine start for cluster.", file=sys.stderr)
                    continue

                cluster_centroid = (
                    sum(midpoint(e)[0] for e in cluster_candidate) / len(cluster_candidate),
                    sum(midpoint(e)[1] for e in cluster_candidate) / len(cluster_candidate),
                )
                current_cluster_start_node = nearest_node(all_on_foot_nodes, cluster_centroid)

                route_edges = plan_route(
                    G, # This is the on_foot_routing_graph
                    cluster_candidate,
                    current_cluster_start_node,
                    args.pace, args.grade, args.road_pace, args.max_road, args.road_threshold
                )
                if not route_edges:
                    continue

                estimated_activity_time = total_time(route_edges, args.pace, args.grade, args.road_pace)
                current_drive_time = 0.0
                drive_from_coord_for_this_candidate = None
                drive_to_coord_for_this_candidate = None

                if last_activity_end_coord:
                    drive_from_coord_for_this_candidate = last_activity_end_coord
                    drive_to_coord_for_this_candidate = route_edges[0].start
                    current_drive_time = planner_utils.estimate_drive_time_minutes(
                        drive_from_coord_for_this_candidate,
                        drive_to_coord_for_this_candidate,
                        all_road_segments,
                        args.average_driving_speed_mph
                    )
                    if current_drive_time > args.max_drive_minutes_per_transfer:
                        continue

                if (time_spent_on_activities_today + estimated_activity_time +
                    time_spent_on_drives_today + current_drive_time) <= todays_total_budget_minutes:
                    best_cluster_to_add_info = {
                        "cluster_original_index": cluster_idx,
                        "route_edges": route_edges,
                        "activity_time": estimated_activity_time,
                        "drive_time": current_drive_time,
                        "drive_from": drive_from_coord_for_this_candidate,
                        "drive_to": drive_to_coord_for_this_candidate
                    }
                    break

            if best_cluster_to_add_info:
                if best_cluster_to_add_info["drive_time"] > 0 and best_cluster_to_add_info["drive_from"] and best_cluster_to_add_info["drive_to"]:
                    activities_for_this_day.append({
                        "type": "drive",
                        "minutes": best_cluster_to_add_info["drive_time"],
                        "from_coord": best_cluster_to_add_info["drive_from"],
                        "to_coord": best_cluster_to_add_info["drive_to"]
                    })
                    time_spent_on_drives_today += best_cluster_to_add_info["drive_time"]

                act_route_edges = best_cluster_to_add_info["route_edges"]
                activities_for_this_day.append({
                    "type": "activity",
                    "route_edges": act_route_edges,
                    "name": f"Activity Part {len([a for a in activities_for_this_day if a['type'] == 'activity']) + 1}"
                })
                time_spent_on_activities_today += best_cluster_to_add_info["activity_time"]
                last_activity_end_coord = act_route_edges[-1].end

                unplanned_macro_clusters.pop(best_cluster_to_add_info["cluster_original_index"])
            else:
                break

        if activities_for_this_day:
            daily_plans.append({
                "date": cur_date,
                "activities": activities_for_this_day,
                "total_activity_time": time_spent_on_activities_today,
                "total_drive_time": time_spent_on_drives_today
            })

    # Placeholder for checking daily_plans structure
    # for dp in daily_plans:
    #     print(f"Date: {dp['date']}")
    #     for act in dp['activities']:
    #         if act['type'] == 'drive':
    #             print(f"  Drive: {act['minutes']:.1f} mins From {act['from_coord']} To {act['to_coord']}")
    #         elif act['type'] == 'activity':
    #             act_time = total_time(act['route_edges'], args.pace, args.grade, args.road_pace)
    #             act_dist = sum(e.length_mi for e in act['route_edges'])
    #             print(f"  {act['name']}: Segments: {len(act['route_edges'])}, Dist: {act_dist:.2f} mi, Time: {act_time:.1f} min")
    #     print(f"  Total Activity Time: {dp['total_activity_time']:.1f} min, Total Drive Time: {dp['total_drive_time']:.1f} min")

    summary_rows = []
    for day_plan in daily_plans:
        day_date_str = day_plan["date"].isoformat()
        gpx_part_counter = 1
        day_description_parts = []
        current_day_total_trail_distance = 0.0
        current_day_total_trail_gain = 0.0
        num_activities_this_day = 0
        num_drives_this_day = 0

        # Need to check if day_plan["activities"] exists and is not empty
        # The previous loop structure for daily_plans ensures 'activities' exists
        # and only adds to daily_plans if activities_for_this_day is non-empty.
        activities_for_this_day_in_plan = day_plan.get("activities", [])


        for activity_or_drive in activities_for_this_day_in_plan:
            if activity_or_drive["type"] == "activity":
                num_activities_this_day += 1
                route = activity_or_drive["route_edges"]
                activity_name = activity_or_drive["name"]

                dist = sum(e.length_mi for e in route)
                gain = sum(e.elev_gain_ft for e in route)
                est_activity_time = total_time(route, args.pace, args.grade, args.road_pace)

                current_day_total_trail_distance += dist
                current_day_total_trail_gain += gain

                gpx_file_name = f"{day_plan['date'].strftime('%Y%m%d')}_part{gpx_part_counter}.gpx"
                gpx_path = os.path.join(args.gpx_dir, gpx_file_name)
                planner_utils.write_gpx(
                    gpx_path, route, mark_road_transitions=args.mark_road_transitions
                )

                trail_segment_ids_in_route = sorted(list(set(str(e.seg_id) for e in route if e.kind == 'trail' and e.seg_id)))
                part_desc = f"{activity_name} (Segs: {', '.join(trail_segment_ids_in_route)}; {dist:.2f}mi; {gain:.0f}ft; {est_activity_time:.1f}min)"
                day_description_parts.append(part_desc)
                gpx_part_counter += 1

            elif activity_or_drive["type"] == "drive":
                num_drives_this_day += 1
                drive_minutes = activity_or_drive["minutes"]
                day_description_parts.append(f"Drive ({drive_minutes:.1f} min)")

        if activities_for_this_day_in_plan:
            summary_rows.append({
                "date": day_date_str,
                "plan_description": " >> ".join(day_description_parts),
                "total_trail_distance_mi": round(current_day_total_trail_distance, 2),
                "total_trail_elev_gain_ft": round(current_day_total_trail_gain, 0),
                "total_activity_time_min": round(day_plan["total_activity_time"], 1),
                "total_drive_time_min": round(day_plan["total_drive_time"], 1),
                "num_activities": num_activities_this_day,
                "num_drives": num_drives_this_day
            })

    # The old loop is commented out as it will be replaced:
    # for idx, cluster in enumerate(clusters):
    #     if not cluster:
    #         continue
    #     cur_date = start_date + datetime.timedelta(days=idx)
    #     centroid = (
    #         sum(midpoint(e)[0] for e in cluster) / len(cluster),
    #         sum(midpoint(e)[1] for e in cluster) / len(cluster),
    #     )
    #     start = nearest_node(nodes, centroid)
    #     route = plan_route(
    #         G,
    #         cluster,
    #         start,
    #         args.pace,
    #         args.grade,
    #         args.road_pace,
    #         args.max_road,
    #         args.road_threshold,
    #     )
    #     dist = sum(e.length_mi for e in route)
    #     gain = sum(e.elev_gain_ft for e in route)
    #     est_time = total_time(route, args.pace, args.grade, args.road_pace)
    #     gpx_path = os.path.join(
    #         args.gpx_dir, f"{cur_date.strftime('%Y%m%d')}.gpx"
    #     )
    #     planner_utils.write_gpx(gpx_path, route)
    #     summary_rows.append({
    #         "date": cur_date.isoformat(),
    #         "segments": " ".join(str(e.seg_id) for e in cluster),
    #         "plan": " > ".join(e.name or str(e.seg_id) for e in route),
    #         "distance_mi": round(dist, 2),
    #         "elev_gain_ft": round(gain, 0),
    #         "time_min": round(est_time, 1),
    #     })

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        default_fieldnames = ["date", "plan_description", "total_trail_distance_mi", "total_trail_elev_gain_ft", "total_activity_time_min", "total_drive_time_min", "num_activities", "num_drives"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=default_fieldnames)
            writer.writeheader()
            # Optionally, write a row indicating no plan:
            # writer.writerow({field: "N/A" for field in default_fieldnames})
            # writer.writerow({"date": "N/A", "plan_description": "No activities planned"})


    print(f"Challenge plan written to {args.output}")
    if not daily_plans or not any(dp.get("activities") for dp in daily_plans) : # Check if any activities were actually planned
        # More robust check if any GPX would have been generated
        gpx_files_present = False
        if os.path.exists(args.gpx_dir):
            gpx_files_present = any(f.endswith(".gpx") for f in os.listdir(args.gpx_dir))

        if not gpx_files_present:
            print(f"No GPX files generated as no activities were planned.")
        else:
            # This case might occur if GPX files from a previous run exist but current run planned nothing
            print(f"GPX files may exist in {args.gpx_dir} from previous runs, but no new activities were planned in this run.")
    else:
        print(f"GPX files written to {args.gpx_dir}")


if __name__ == "__main__":
    main()
