import argparse
import csv
import os
import sys
import datetime
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from tqdm.auto import tqdm

import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import math

# Allow running this file directly without installing the package
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trail_route_ai import planner_utils, plan_review

# Type aliases
Edge = planner_utils.Edge


@dataclass
class ClusterInfo:
    edges: List[Edge]
    nodes: Set[Tuple[float, float]]
    start_candidates: List[Tuple[Tuple[float, float], Optional[str]]]


@dataclass
class PlannerConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    time: str = "3h"
    daily_hours_file: Optional[str] = None
    pace: Optional[float] = None
    grade: float = 0.0
    segments: str = "data/traildata/trail.json"
    dem: Optional[str] = None
    roads: Optional[str] = None
    trailheads: Optional[str] = None
    home_lat: Optional[float] = None
    home_lon: Optional[float] = None
    max_road: float = 1.0
    road_threshold: float = 0.1
    road_pace: float = 18.0
    perf: str = "data/segment_perf.csv"
    year: Optional[int] = None
    remaining: Optional[str] = None
    output: str = "challenge_plan.csv"
    gpx_dir: str = "gpx"
    mark_road_transitions: bool = False
    average_driving_speed_mph: float = 30.0
    max_drive_minutes_per_transfer: float = 30.0
    review: bool = False


def load_config(path: str) -> PlannerConfig:
    """Load a :class:`PlannerConfig` from a JSON or YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        if path.lower().endswith(".json"):
            data = json.load(f)
        else:
            import yaml

            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping")
    return PlannerConfig(**data)

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


def identify_macro_clusters(
    all_trail_segments: List[Edge],
    all_road_segments: List[Edge],
    pace: float,
    grade: float,
    road_pace: float,
) -> List[Tuple[List[Edge], Set[Tuple[float, float]]]]:
    """Identify geographically distinct clusters of trail segments.

    Returns a list where each item contains the trail segments in the cluster
    and the set of nodes that make up the connected component used for the
    clustering graph.
    """

    graph_edges = all_trail_segments + all_road_segments
    G = build_nx_graph(graph_edges, pace, grade, road_pace)

    macro_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    assigned_segment_ids: set[str | int] = set()

    for component_nodes in nx.connected_components(G):
        nodes_set = set(component_nodes)
        current_cluster_segments: List[Edge] = []
        for seg in all_trail_segments:
            if seg.seg_id is not None and seg.seg_id in assigned_segment_ids:
                continue
            if seg.start in nodes_set or seg.end in nodes_set:
                current_cluster_segments.append(seg)
                if seg.seg_id is not None:
                    assigned_segment_ids.add(seg.seg_id)

        if current_cluster_segments:
            macro_clusters.append((current_cluster_segments, nodes_set))

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
            # fallback attempt ignoring max_road constraint
            for e in remaining:
                for end in [e.start, e.end]:
                    try:
                        path = nx.shortest_path(G, cur, end, weight="weight")
                        edges_path = edges_from_path(G, path)
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
                    file=sys.stderr,
                )
                return []  # Signify failure to route this cluster

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
    if argv is None:
        argv = sys.argv[1:]

    # Determine configuration file location before parsing full args
    config_path = None
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            break
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break
    if config_path is None:
        default_yaml = os.path.join("config", "planner_config.yaml")
        default_json = os.path.join("config", "planner_config.json")
        if os.path.exists(default_yaml):
            config_path = default_yaml
        elif os.path.exists(default_json):
            config_path = default_json

    config_defaults: Dict[str, object] = {}
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            config_defaults = asdict(cfg)
        except Exception:
            config_defaults = {}

    parser = argparse.ArgumentParser(description="Challenge route planner")
    parser.set_defaults(**config_defaults)

    parser.add_argument("--config", default=config_path, help="Path to config YAML or JSON file")
    parser.add_argument("--start-date", required="start_date" not in config_defaults, help="Challenge start date YYYY-MM-DD")
    parser.add_argument("--end-date", required="end_date" not in config_defaults, help="Challenge end date YYYY-MM-DD")
    parser.add_argument(
        "--time",
        default=config_defaults.get("time", "3h"),
        help="Default daily time budget when --daily-hours-file is not provided"
    )
    parser.add_argument(
        "--daily-hours-file",
        default=config_defaults.get("daily_hours_file"),
        help="JSON mapping YYYY-MM-DD dates to available hours for that day"
    )
    parser.add_argument("--pace", required="pace" not in config_defaults, type=float, help="Base running pace (min/mi)")
    parser.add_argument("--grade", type=float, default=config_defaults.get("grade", 0.0), help="Seconds added per 100ft climb")
    parser.add_argument("--segments", default=config_defaults.get("segments", "data/traildata/trail.json"), help="Trail segment JSON file")
    parser.add_argument(
        "--dem",
        help="Optional DEM GeoTIFF from clip_srtm.py for segment elevation",
    )
    parser.add_argument("--roads", help="Optional road connector GeoJSON or .pbf")
    parser.add_argument("--trailheads", help="Optional trailhead JSON or CSV file")
    parser.add_argument("--home-lat", type=float, help="Home latitude for drive time estimates")
    parser.add_argument("--home-lon", type=float, help="Home longitude for drive time estimates")
    parser.add_argument("--max-road", type=float, default=config_defaults.get("max_road", 1.0), help="Max road distance per connector (mi)")
    parser.add_argument(
        "--road-threshold",
        type=float,
        default=config_defaults.get("road_threshold", 0.1),
        help="Fractional speed advantage required to choose a road connector",
    )
    parser.add_argument("--road-pace", type=float, default=config_defaults.get("road_pace", 18.0), help="Pace on roads (min/mi)")
    parser.add_argument("--perf", default=config_defaults.get("perf", "data/segment_perf.csv"), help="CSV of previous segment completions")
    parser.add_argument("--year", type=int, help="Filter completions to this year")
    parser.add_argument("--remaining", help="Comma-separated list or file of segments to include")
    parser.add_argument("--output", default=config_defaults.get("output", "challenge_plan.csv"), help="Output CSV summary file")
    parser.add_argument("--gpx-dir", default=config_defaults.get("gpx_dir", "gpx"), help="Directory for GPX output")
    parser.add_argument(
        "--mark-road-transitions",
        action="store_true",
        help="Annotate GPX files with waypoints and track extensions for road sections",
    )
    parser.add_argument("--average-driving-speed-mph", type=float, default=config_defaults.get("average_driving_speed_mph", 30.0), help="Average driving speed in mph for estimating travel time between activity clusters")
    parser.add_argument("--max-drive-minutes-per-transfer", type=float, default=config_defaults.get("max_drive_minutes_per_transfer", 30.0), help="Maximum allowed driving time between clusters on the same day")
    parser.add_argument("--review", action="store_true", default=config_defaults.get("review", False), help="Send final plan for AI review")

    args = parser.parse_args(argv)

    home_coord = None
    if args.home_lat is not None and args.home_lon is not None:
        home_coord = (args.home_lon, args.home_lat)

    start_date = datetime.date.fromisoformat(args.start_date)
    end_date = datetime.date.fromisoformat(args.end_date)
    if end_date < start_date:
        parser.error("--end-date must not be before --start-date")
    num_days = (end_date - start_date).days + 1

    budget = planner_utils.parse_time_budget(args.time)

    daily_budget_minutes: Dict[datetime.date, float] = {}
    user_hours: Dict[datetime.date, float] = {}
    daily_hours_file = args.daily_hours_file
    if daily_hours_file is None and os.path.exists(os.path.join("config", "daily_hours.json")):
        daily_hours_file = os.path.join("config", "daily_hours.json")
    if daily_hours_file and os.path.exists(daily_hours_file):
        with open(daily_hours_file) as f:
            raw = json.load(f)
        for k, v in raw.items():
            try:
                d = datetime.date.fromisoformat(k)
            except ValueError:
                continue
            if d < start_date or d > end_date:
                continue
            try:
                hours = float(v)
            except (TypeError, ValueError):
                continue
            if hours < 0:
                hours = 0
            user_hours[d] = hours

    default_daily_minutes = 180.0 if daily_hours_file else budget
    for i in range(num_days):
        day = start_date + datetime.timedelta(days=i)
        hours = user_hours.get(day, default_daily_minutes / 60.0)
        daily_budget_minutes[day] = hours * 60.0
    all_trail_segments = planner_utils.load_segments(args.segments)
    if args.dem:
        planner_utils.add_elevation_from_dem(all_trail_segments, args.dem)
    all_road_segments: List[Edge] = []
    if args.roads:
        bbox = None
        if args.roads.lower().endswith(".pbf"):
            bbox = planner_utils.bounding_box_from_edges(all_trail_segments)
        all_road_segments = planner_utils.load_roads(args.roads, bbox=bbox)
    road_node_set: Set[Tuple[float, float]] = {e.start for e in all_road_segments} | {e.end for e in all_road_segments}

    trailhead_lookup: Dict[Tuple[float, float], str] = {}
    if args.trailheads and os.path.exists(args.trailheads):
        trailhead_lookup = planner_utils.load_trailheads(args.trailheads)

    # This graph is used for on-foot routing *within* macro-clusters
    on_foot_routing_graph_edges = all_trail_segments + all_road_segments
    G = build_nx_graph(on_foot_routing_graph_edges, args.pace, args.grade, args.road_pace)

    tracking = planner_utils.load_segment_tracking(os.path.join("config", "segment_tracking.json"), args.segments)
    completed_segment_ids = {sid for sid, done in tracking.items() if done}
    completed_segment_ids |= planner_utils.load_completed(args.perf, args.year or 0)

    current_challenge_segment_ids = None
    if args.remaining:
        current_challenge_segment_ids = set(parse_remaining(args.remaining))
    if current_challenge_segment_ids is None:
        current_challenge_segment_ids = {str(e.seg_id) for e in all_trail_segments} - completed_segment_ids

    current_challenge_segments = [e for e in all_trail_segments if str(e.seg_id) in current_challenge_segment_ids]

    # nodes list might be useful later for starting points, keep it around
    # nodes = list({e.start for e in on_foot_routing_graph_edges} | {e.end for e in on_foot_routing_graph_edges})

    potential_macro_clusters = identify_macro_clusters(
        current_challenge_segments,  # Only uncompleted trail segments
        all_road_segments,           # All road segments
        args.pace,
        args.grade,
        args.road_pace,
    )

    # Further split any macro-clusters that appear too large for a single day's
    # budget.  The "cluster_segments" helper uses a spatial KMeans followed by
    # a greedy time-based split which keeps each resulting cluster under the
    # provided budget whenever possible.
    expanded_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    for cluster_edges, cluster_nodes in potential_macro_clusters:
        if not cluster_edges:
            continue
        naive_time = total_time(cluster_edges, args.pace, args.grade, args.road_pace)
        oversized_threshold = 1.5 * budget
        if naive_time > oversized_threshold:
            max_parts = max(1, int(np.ceil(naive_time / budget)))
            subclusters = cluster_segments(
                cluster_edges,
                pace=args.pace,
                grade=args.grade,
                budget=budget,
                max_clusters=max_parts,
                road_pace=args.road_pace,
            )
            for sub in subclusters:
                if not sub:
                    continue
                sub_nodes = {pt for e in sub for pt in (e.start, e.end)}
                expanded_clusters.append((sub, sub_nodes))
        else:
            expanded_clusters.append((cluster_edges, cluster_nodes))

    unplanned_macro_clusters = [mc for mc in expanded_clusters if mc[0]]

    # Ensure each cluster can be routed; if not, break it into simpler pieces
    processed_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    for cluster_segs, cluster_nodes in unplanned_macro_clusters:
        cluster_centroid = (
            sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
            sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
        )
        start_node = nearest_node(list(cluster_nodes), cluster_centroid)
        initial_route = plan_route(
            G,
            cluster_segs,
            start_node,
            args.pace,
            args.grade,
            args.road_pace,
            args.max_road,
            args.road_threshold,
        )
        if initial_route:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue
        if len(cluster_segs) == 1:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue
        extended_route = plan_route(
            G,
            cluster_segs,
            start_node,
            args.pace,
            args.grade,
            args.road_pace,
            args.max_road * 3,
            args.road_threshold,
        )
        if extended_route:
            processed_clusters.append((cluster_segs, cluster_nodes))
        else:
            for seg in cluster_segs:
                processed_clusters.append(([seg], {seg.start, seg.end}))

    unplanned_macro_clusters: List[ClusterInfo] = []
    for cluster_segs, cluster_nodes in processed_clusters:
        start_candidates: List[Tuple[Tuple[float, float], Optional[str]]] = []
        for n in cluster_nodes:
            if n in trailhead_lookup:
                start_candidates.append((n, trailhead_lookup[n]))
        if not start_candidates:
            for n in cluster_nodes:
                if n in road_node_set:
                    start_candidates.append((n, None))
        if not start_candidates:
            centroid = (
                sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
                sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
            )
            start_candidates.append((nearest_node(list(cluster_nodes), centroid), None))
        unplanned_macro_clusters.append(ClusterInfo(cluster_segs, cluster_nodes, start_candidates))

    all_on_foot_nodes = list(G.nodes()) # Get all nodes from the on-foot routing graph

    os.makedirs(args.gpx_dir, exist_ok=True)
    # summary_rows = [] # This will be populated by the new planning loop (or rather, daily_plans will be used)
    daily_plans = []
    failed_cluster_signatures: Set[Tuple[str, ...]] = set()

    day_iter = tqdm(range(num_days), desc="Planning days", unit="day")
    for day_idx in day_iter:
        if not unplanned_macro_clusters:
            break
        cur_date = start_date + datetime.timedelta(days=day_idx)
        todays_total_budget_minutes = daily_budget_minutes.get(cur_date, budget)
        activities_for_this_day = []
        time_spent_on_activities_today = 0.0
        time_spent_on_drives_today = 0.0
        last_activity_end_coord = None

        while True:
            best_cluster_to_add_info = None
            candidate_pool = []

            # Compute a simple isolation score for each remaining cluster
            cluster_centroids: List[Tuple[float, float]] = []
            for cluster in unplanned_macro_clusters:
                segs = cluster.edges
                cx = sum(midpoint(e)[0] for e in segs) / len(segs)
                cy = sum(midpoint(e)[1] for e in segs) / len(segs)
                cluster_centroids.append((cx, cy))

            isolation_lookup = {}
            for idx, (cx, cy) in enumerate(cluster_centroids):
                if len(cluster_centroids) == 1:
                    isolation_lookup[idx] = math.inf
                else:
                    min_dist = min(
                        math.hypot(cx - ox, cy - oy)
                        for j, (ox, oy) in enumerate(cluster_centroids)
                        if j != idx
                    )
                    isolation_lookup[idx] = min_dist

            cluster_iter = tqdm(
                enumerate(unplanned_macro_clusters),
                desc=f"Day {day_idx+1} candidates",
                unit="cluster",
                leave=False,
            )
            for cluster_idx, cluster_candidate in cluster_iter:
                cluster_segs = cluster_candidate.edges
                cluster_nodes = cluster_candidate.nodes
                start_candidates = cluster_candidate.start_candidates
                if not cluster_segs:
                    continue

                if not all_on_foot_nodes:
                    tqdm.write(
                        "Warning: No nodes in on_foot_routing_graph. Cannot determine start for cluster.",
                        file=sys.stderr,
                    )
                    continue

                cluster_centroid = (
                    sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
                    sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
                )

                drive_origin = last_activity_end_coord if last_activity_end_coord else home_coord
                best_start_node = None
                best_start_name = None
                best_drive_time_to_start = float("inf")
                for cand_node, cand_name in start_candidates:
                    drive_time_tmp = 0.0
                    if drive_origin and all_road_segments:
                        drive_time_tmp = planner_utils.estimate_drive_time_minutes(
                            drive_origin,
                            cand_node,
                            all_road_segments,
                            args.average_driving_speed_mph,
                        )
                    if drive_time_tmp < best_drive_time_to_start:
                        best_drive_time_to_start = drive_time_tmp
                        best_start_node = cand_node
                        best_start_name = cand_name

                if best_start_node is None:
                    best_start_node = nearest_node(list(cluster_nodes), cluster_centroid)
                    best_start_name = None
                    best_drive_time_to_start = 0.0

                cluster_sig = tuple(sorted(str(e.seg_id) for e in cluster_segs))
                if cluster_sig in failed_cluster_signatures:
                    continue

                route_edges = plan_route(
                    G,  # This is the on_foot_routing_graph
                    cluster_segs,
                    best_start_node,
                    args.pace,
                    args.grade,
                    args.road_pace,
                    args.max_road,
                    args.road_threshold,
                )
                if not route_edges:
                    if len(cluster_segs) == 1:
                        seg = cluster_segs[0]
                        rev = Edge(
                            seg.seg_id,
                            seg.name,
                            seg.end,
                            seg.start,
                            seg.length_mi,
                            seg.elev_gain_ft,
                            list(reversed(seg.coords)),
                            seg.kind,
                        )
                        route_edges = [seg, rev]
                    else:
                        extended_route = plan_route(
                            G,
                            cluster_segs,
                            best_start_node,
                            args.pace,
                            args.grade,
                            args.road_pace,
                            args.max_road * 3,
                            args.road_threshold,
                        )
                        if extended_route:
                            route_edges = extended_route
                        else:
                            failed_cluster_signatures.add(cluster_sig)
                            tqdm.write(
                                f"Skipping unroutable cluster with segments {[e.seg_id for e in cluster_segs]}",
                                file=sys.stderr,
                            )
                            continue

                estimated_activity_time = total_time(route_edges, args.pace, args.grade, args.road_pace)
                current_drive_time = best_drive_time_to_start
                drive_from_coord_for_this_candidate = drive_origin
                drive_to_coord_for_this_candidate = best_start_node

                if current_drive_time > args.max_drive_minutes_per_transfer:
                    continue

                if (
                    time_spent_on_activities_today
                    + estimated_activity_time
                    + time_spent_on_drives_today
                    + current_drive_time
                ) <= todays_total_budget_minutes:
                    candidate_pool.append(
                        {
                            "cluster_original_index": cluster_idx,
                            "route_edges": route_edges,
                            "activity_time": estimated_activity_time,
                            "drive_time": current_drive_time,
                            "drive_from": drive_from_coord_for_this_candidate,
                            "drive_to": drive_to_coord_for_this_candidate,
                            "start_name": best_start_name,
                            "start_coord": best_start_node,
                            "ignored_budget": False,
                            "isolation_score": isolation_lookup.get(cluster_idx, 0.0),
                        }
                    )

            if candidate_pool:
                candidate_pool.sort(
                    key=lambda c: (
                        c["drive_time"],
                        -(c["activity_time"] + c["drive_time"]),
                        -c.get("isolation_score", 0.0),
                    )
                )
                best_cluster_to_add_info = candidate_pool[0]


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
                activities_for_this_day.append(
                    {
                        "type": "activity",
                        "route_edges": act_route_edges,
                        "name": f"Activity Part {len([a for a in activities_for_this_day if a['type'] == 'activity']) + 1}",
                        "ignored_budget": best_cluster_to_add_info.get("ignored_budget", False),
                        "start_name": best_cluster_to_add_info.get("start_name"),
                        "start_coord": best_cluster_to_add_info.get("start_coord"),
                    }
                )
                time_spent_on_activities_today += best_cluster_to_add_info["activity_time"]
                last_activity_end_coord = act_route_edges[-1].end

                unplanned_macro_clusters.pop(best_cluster_to_add_info["cluster_original_index"])
            else:
                break

        if activities_for_this_day:
            total_day_time = time_spent_on_activities_today + time_spent_on_drives_today
            note_parts = []
            if total_day_time > todays_total_budget_minutes:
                note_parts.append(f"over budget by {total_day_time - todays_total_budget_minutes:.1f} min")
            notes = "; ".join(note_parts)
            daily_plans.append({
                "date": cur_date,
                "activities": activities_for_this_day,
                "total_activity_time": time_spent_on_activities_today,
                "total_drive_time": time_spent_on_drives_today,
                "notes": notes
            })
            day_iter.set_postfix(note=notes)
        else:
            daily_plans.append({
                "date": cur_date,
                "activities": [],
                "total_activity_time": 0.0,
                "total_drive_time": 0.0,
                "notes": ""
            })
            day_iter.set_postfix(note="no activities")

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
        start_names_for_day: List[str] = []

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
                    gpx_path,
                    route,
                    mark_road_transitions=args.mark_road_transitions,
                    start_name=activity_or_drive.get("start_name"),
                )
                if activity_or_drive.get("start_name"):
                    start_names_for_day.append(activity_or_drive.get("start_name"))

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
                "num_drives": num_drives_this_day,
                "notes": day_plan.get("notes", ""),
                "start_trailheads": "; ".join(start_names_for_day),
            })
        else:
            summary_rows.append({
                "date": day_date_str,
                "plan_description": "Unable to complete",
                "total_trail_distance_mi": 0.0,
                "total_trail_elev_gain_ft": 0.0,
                "total_activity_time_min": 0.0,
                "total_drive_time_min": 0.0,
                "num_activities": 0,
                "num_drives": 0,
                "notes": day_plan.get("notes", ""),
                "start_trailheads": ""
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

    if args.review and summary_rows:
        plan_text = "\n".join(f"{r['date']}: {r['plan_description']}" for r in summary_rows)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            plan_review.review_plan(plan_text, run_id)
            print(f"Review saved to reviews/{run_id}.jsonl")
        except Exception as e:
            print(f"Review failed: {e}")

    # --- Check for unplanned segments -------------------------------------
    planned_segment_ids: Set[str] = set()
    for dp in daily_plans:
        for item in dp.get("activities", []):
            if item.get("type") == "activity":
                for ed in item.get("route_edges", []):
                    if ed.kind == "trail" and ed.seg_id is not None:
                        planned_segment_ids.add(str(ed.seg_id))

    remaining_segments = current_challenge_segment_ids - planned_segment_ids
    if remaining_segments:
        avg_hours = sum(daily_budget_minutes.values()) / 60.0 / len(daily_budget_minutes)
        msg = (
            f"With {avg_hours:.1f} hours/day from {start_date.isoformat()} to {end_date.isoformat()}, "
            "it's impossible to complete all trails. Extend the timeframe or increase daily budget."
        )
        raise SystemExit(msg)

    if daily_plans and any(dp.get("activities") for dp in daily_plans):
        colors = ["Red", "Blue", "Green", "Magenta", "Cyan", "Orange", "Purple", "Brown"]
        full_gpx_path = os.path.join(args.gpx_dir, "full_timespan.gpx")
        planner_utils.write_multiday_gpx(
            full_gpx_path,
            daily_plans,
            mark_road_transitions=args.mark_road_transitions,
            colors=colors,
        )


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
