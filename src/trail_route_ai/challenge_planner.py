import argparse
import csv
import os
import sys
import os  # Ensure os is imported for path operations
import datetime
import json

# Allow running this file directly without installing the package.
# This ensures that 'trail_route_ai' can be found in sys.path
# when the script is run directly, by adding its parent ('src') to sys.path.
if __package__ in (None, ""):
    # __file__ is src/trail_route_ai/challenge_planner.py when run from /app
    # os.path.dirname(os.path.dirname(os.path.abspath(__file__))) is /app/src
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)  # Insert at the beginning for precedence
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

# When executed as a script, ``__package__`` is not set, which breaks relative
# imports. Import ``cache_utils`` using its absolute name so the script works
# both as part of the package and when run standalone.
from trail_route_ai import cache_utils
import logging
import signal

logger = logging.getLogger(__name__)


class DijkstraTimeoutError(Exception):
    pass


from tqdm.auto import tqdm

import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import math

try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None
    _HAVE_SCIPY = False

from trail_route_ai import planner_utils, plan_review
from trail_route_ai import optimizer

# Type aliases
Edge = planner_utils.Edge

# Thresholds for when to prefer driving between activities
DRIVE_FASTER_FACTOR = 2.0  # drive must be at least this many times faster
MIN_DRIVE_TIME_SAVINGS_MIN = 5.0
MIN_DRIVE_DISTANCE_MI = 1.0
# Extra time assumed for each drive (parking, transition, etc.)
DRIVE_PARKING_OVERHEAD_MIN = 10.0
# Additional bias toward walking when doing so would complete a segment
COMPLETE_SEGMENT_BONUS = 1.0


def build_kdtree(nodes: List[Tuple[float, float]]):
    """Return a KDTree for ``nodes`` if SciPy is available."""
    if not nodes:
        return []
    if _HAVE_SCIPY:
        return cKDTree(np.array(nodes))
    return list(nodes)


_nearest_cache: dict[tuple[int, float, float], tuple[float, float]] = {}


def nearest_node(index, point: Tuple[float, float]):
    """Return the nearest node to ``point`` using ``index``.

    Results are cached to avoid repeated tree queries when the same point is
    looked up multiple times on the same index.
    """
    if not index:
        return point
    if _HAVE_SCIPY and hasattr(index, "query"):
        key = (id(index), round(point[0], 6), round(point[1], 6))
        if key in _nearest_cache:
            return _nearest_cache[key]
        _, idx = index.query(point)
        result = tuple(index.data[idx])
        if len(_nearest_cache) > 10_000:
            _nearest_cache.clear()
        _nearest_cache[key] = result
        return result
    nodes = index
    return min(nodes, key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2)


@dataclass
class ClusterInfo:
    edges: List[Edge]
    nodes: Set[Tuple[float, float]]
    start_candidates: List[Tuple[Tuple[float, float], Optional[str]]]
    node_tree: object | None = None

    def __post_init__(self) -> None:
        if self.node_tree is None:
            self.node_tree = build_kdtree(list(self.nodes))


@dataclass
class ClusterScore:
    drive_time: float
    activity_time: float
    isolation_score: float
    completion_bonus: float
    effort_distribution: float

    @property
    def total_score(self) -> tuple:
        """Return a composite score tuple for sorting candidates."""
        return (
            self.drive_time,
            -(self.activity_time + self.drive_time),
            -self.isolation_score,
            -self.completion_bonus,
            self.effort_distribution,
        )


@dataclass
class PlannerConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    time: str = "4h"
    daily_hours_file: Optional[str] = None
    pace: Optional[float] = None
    grade: float = 0.0
    segments: str = "data/traildata/trail.json"
    dem: Optional[str] = None
    roads: Optional[str] = None
    trailheads: Optional[str] = None
    home_lat: Optional[float] = 43.635278
    home_lon: Optional[float] = -116.205
    max_foot_road: float = 3.0
    road_threshold: float = 0.25
    road_pace: float = 12.0
    perf: str = "data/segment_perf.csv"
    year: Optional[int] = None
    remaining: Optional[str] = None
    output: str = "challenge_plan.csv"
    gpx_dir: str = "gpx"
    output_dir: Optional[str] = None
    auto_output_dir: bool = False
    mark_road_transitions: bool = True
    average_driving_speed_mph: float = 30.0
    max_drive_minutes_per_transfer: float = 30.0
    review: bool = False
    redundancy_threshold: float = 0.2
    allow_connector_trails: bool = True
    rpp_timeout: float = 5.0
    spur_length_thresh: float = 0.3
    spur_road_bonus: float = 0.25
    use_advanced_optimizer: bool = False
    strict_max_foot_road: bool = False
    first_day_segment: Optional[str] = None
    optimizer: str = "greedy2opt"
    postman_timeout: float = 30.0
    postman_max_odd: int = 40


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
    if "max_foot_road" not in data and "max_road" in data:
        data["max_foot_road"] = data.pop("max_road")
    return PlannerConfig(**data)


def midpoint(edge: Edge) -> Tuple[float, float]:
    sx, sy = edge.start
    ex, ey = edge.end
    return ((sx + ex) / 2.0, (sy + ey) / 2.0)


def total_time(edges: List[Edge], pace: float, grade: float, road_pace: float) -> float:
    return sum(planner_utils.estimate_time(e, pace, grade, road_pace) for e in edges)


def debug_log(args: argparse.Namespace | None, message: str) -> None:
    """Append ``message`` to the debug log and optionally echo to stdout."""
    if args is None:
        return
    if getattr(args, "verbose", False):
        tqdm.write(message)
    path = getattr(args, "debug", None)
    if not path:
        return
    try:
        with open(path, "a") as df:
            df.write(f"{message}\n")
    except Exception:
        pass


def build_nx_graph(
    edges: List[Edge], pace: float, grade: float, road_pace: float
) -> nx.DiGraph:
    """Return a routing graph built from ``edges``."""

    G = nx.DiGraph()
    for e in tqdm(edges, desc="Building routing graph", unit="edge"):
        w = planner_utils.estimate_time(e, pace, grade, road_pace)
        G.add_edge(e.start, e.end, weight=w, edge=e)
        if e.direction == "both":
            rev = planner_utils.Edge(
                e.seg_id,
                e.name,
                e.end,
                e.start,
                e.length_mi,
                e.elev_gain_ft,
                list(reversed(e.coords)),
                e.kind,
                e.direction,
                e.access_from,
            )
            G.add_edge(e.end, e.start, weight=w, edge=rev)
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
    tqdm.write("Building macro-cluster graph...")
    G = build_nx_graph(graph_edges, pace, grade, road_pace)

    macro_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    assigned_segment_ids: set[str | int] = set()

    components = list(nx.weakly_connected_components(G))
    for component_nodes in tqdm(components, desc="Analyzing macro clusters"):
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


def edges_from_path(
    G: nx.DiGraph,
    path: List[Tuple[float, float]],
    required_ids: Optional[Set[str]] = None,
) -> List[Edge]:
    """Return ``Edge`` objects along ``path``.

    When ``required_ids`` is provided, trail segments whose ``seg_id`` is not in
    the set are marked as ``kind="connector"`` to indicate they were added only
    to connect required trails.
    """

    out: List[Edge] = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if not data:
            continue
        ed = data[0]["edge"] if 0 in data else data["edge"]
        if required_ids is not None and ed.kind == "trail":
            sid = str(ed.seg_id) if ed.seg_id is not None else None
            if sid is None or sid not in required_ids:
                ed = Edge(
                    ed.seg_id,
                    ed.name,
                    ed.start,
                    ed.end,
                    ed.length_mi,
                    ed.elev_gain_ft,
                    ed.coords,
                    "connector",
                    ed.direction,
                    ed.access_from,
                )
        out.append(ed)
    return out


def _plan_route_greedy(
    G: nx.DiGraph,
    edges: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    path_back_penalty: float = 1.2,
    redundancy_threshold: float | None = None,
    strict_max_foot_road: bool = False,
    debug_args: Optional[argparse.Namespace] = None,
    dijkstra_timeout_override: Optional[float] = None,
) -> List[Edge]:
    """Return a continuous route connecting ``edges`` starting from ``start``
    using a greedy nearest-neighbor strategy.

    ``max_foot_road`` limits road mileage for any connector. ``road_threshold``
    expresses the additional time we're willing to spend to stay on trail.
    If a trail connector is within ``road_threshold`` of the best road option
    (in terms of time), the trail is chosen.
    """
    remaining = edges[:]
    route: List[Edge] = []
    order: List[Edge] = []
    cur = start
    degree: Dict[Tuple[float, float], int] = defaultdict(int)
    for seg in edges:
        degree[seg.start] += 1
        degree[seg.end] += 1
    last_seg: Optional[Edge] = None
    # Existing progress bar, description "Routing segments"
    progress = tqdm(
        total=len(edges), desc="Routing segments", unit="segment", leave=False
    )
    # New progress bar for greedy segment selection
    greedy_selection_progress = tqdm(
        total=len(edges), desc="Greedy segment selection", unit="segment", leave=False
    )

    def dijkstra_timeout_handler(signum, frame):
        raise DijkstraTimeoutError("Dijkstra pathfinding timed out")

    signal.signal(signal.SIGALRM, dijkstra_timeout_handler)
    DIJKSTRA_TIMEOUT_SECONDS = 60  # Default timeout in seconds

    effective_dijkstra_timeout = DIJKSTRA_TIMEOUT_SECONDS
    if dijkstra_timeout_override is not None:
        effective_dijkstra_timeout = dijkstra_timeout_override

    debug_log(
        debug_args,
        f"_plan_route_greedy: Using Dijkstra timeout of {effective_dijkstra_timeout} seconds."
    )

    iteration_count = 0
    max_iterations = len(edges) * 2

    try:
        while remaining:
            iteration_count += 1
            if iteration_count > max_iterations:
                debug_log(
                    debug_args,
                    f"_plan_route_greedy: Exceeded maximum iterations ({max_iterations}). Aborting greedy search for this cluster.",
                )
                break
            debug_log(
                debug_args,
                f"_plan_route_greedy: starting iteration {iteration_count}/{max_iterations} with {len(remaining)} remaining segments from {cur}",
            )
            paths = None
            if dist_cache is not None and cur in dist_cache:
                paths = dist_cache[cur]
            else:
                try:
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Attempting Dijkstra from node {cur}",
                    )
                    signal.alarm(int(effective_dijkstra_timeout)) # Ensure it's an int for signal.alarm
                    # Ensure 'cur' is actually in G before calling Dijkstra
                    if cur not in G:
                        raise nx.NodeNotFound(f"Node {cur} not in graph for Dijkstra.")
                    _, paths = nx.single_source_dijkstra(G, cur, weight="weight")
                    signal.alarm(0)  # Disable the alarm
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra successful from node {cur}. Found {len(paths)} paths.",
                    )
                    if dist_cache is not None:
                        dist_cache[cur] = paths
                except DijkstraTimeoutError as e:
                    signal.alarm(0)  # Ensure alarm is disabled
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra timed out from node {cur} for a cluster of {len(edges)} segments using a timeout of {effective_dijkstra_timeout}s. Error: {e}"
                    )
                    paths = {}
                except (
                    nx.NetworkXNoPath,
                    nx.NodeNotFound,
                ) as e:  # Catch if Dijkstra itself says no path or cur is invalid
                    signal.alarm(0)
                    paths = {}
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra error (NoPath or NodeNotFound) for node {cur}. paths set to empty.",
                    )
                    debug_log(debug_args, f"Dijkstra error for node {cur}: {e}")
            candidate_info = []
            for e in remaining:
                for end in [e.start, e.end]:
                    if end == e.end and e.direction != "both":
                        continue
                    if end not in paths:
                        continue
                    path_nodes = paths[end]
                    edges_path = edges_from_path(G, path_nodes)
                    road_dist = sum(
                        ed.length_mi for ed in edges_path if ed.kind == "road"
                    )
                    time = sum(
                        planner_utils.estimate_time(ed, pace, grade, road_pace)
                        for ed in edges_path
                    )
                    time += planner_utils.estimate_time(e, pace, grade, road_pace)
                    uses_road = any(ed.kind == "road" for ed in edges_path)
                    candidate_info.append(
                        (time, uses_road, e, end, edges_path, road_dist)
                    )

            debug_log(
                debug_args,
                f"_plan_route_greedy: evaluated {len(candidate_info)} candidate connectors from {cur}",
            )

            allowed_max_road = max_foot_road
            if (
                last_seg is not None
                and degree.get(cur, 0) == 1
                and last_seg.length_mi <= spur_length_thresh
            ):
                allowed_max_road += spur_road_bonus

            valid_candidates = [c for c in candidate_info if c[5] <= allowed_max_road]
            if valid_candidates:
                candidate_info = valid_candidates
            elif strict_max_foot_road:
                candidate_info = []

            if not candidate_info:
                # Enhanced debugging information
                if last_seg:
                    last_seg_info_str = (
                        f"segment '{last_seg.name or last_seg.seg_id}' (node {cur})"
                    )
                else:
                    last_seg_info_str = f"starting node {cur}"

                fail_details_list = []
                fail_details_dict_for_original_structure: Dict[str, str] = {}

                for e_rem in remaining:
                    e_rem_name = e_rem.name or str(e_rem.seg_id)
                    reasons_for_segment = []

                    # Check path to e_rem.start
                    if e_rem.start not in paths:
                        reasons_for_segment.append(
                            f"no path from {cur} to {e_rem_name}.start {e_rem.start}"
                        )
                    else:
                        path_to_start_nodes = paths[e_rem.start]
                        edges_to_start = edges_from_path(G, path_to_start_nodes)
                        road_dist_to_start = sum(
                            ed.length_mi for ed in edges_to_start if ed.kind == "road"
                        )
                        if road_dist_to_start > allowed_max_road:
                            reasons_for_segment.append(
                                f"connector to {e_rem_name}.start requires {road_dist_to_start:.2f}mi road > allowed {allowed_max_road:.2f}mi"
                            )
                        # If a valid path to start exists, this would have been a candidate.
                        # If it's not a candidate, it means it was filtered by other logic or this e_rem was not evaluated.
                        # For this block, we are concerned about why *no* candidates were found. So if a path was valid, it should have been a candidate.

                    # Check path to e_rem.end (if not one-way in the wrong direction)
                    if (
                        e_rem.direction == "both" or e_rem.start == cur
                    ):  # simplified, if it's one way and start is not cur, end might be target
                        if e_rem.end not in paths:
                            reasons_for_segment.append(
                                f"no path from {cur} to {e_rem_name}.end {e_rem.end}"
                            )
                        else:
                            path_to_end_nodes = paths[e_rem.end]
                            edges_to_end = edges_from_path(G, path_to_end_nodes)
                            road_dist_to_end = sum(
                                ed.length_mi for ed in edges_to_end if ed.kind == "road"
                            )
                            if road_dist_to_end > allowed_max_road:
                                reasons_for_segment.append(
                                    f"connector to {e_rem_name}.end requires {road_dist_to_end:.2f}mi road > allowed {allowed_max_road:.2f}mi"
                                )
                    elif (
                        e_rem.direction != "both" and e_rem.end == cur
                    ):  # Trying to go from e_rem.end to e_rem.start but it's one way
                        reasons_for_segment.append(
                            f"segment {e_rem_name} is one-way and cannot be traversed from its end"
                        )

                    if (
                        not reasons_for_segment
                    ):  # Should not happen if candidate_info is empty, means there was a valid path
                        reasons_for_segment.append(
                            f"was considered connectable but not chosen (e.g. strict_max_foot_road filtered it or other logic). This indicates a potential logic flaw if no other reasons present."
                        )

                    fail_details_list.append(
                        f"- {e_rem_name}: {'; '.join(reasons_for_segment)}"
                    )
                    fail_details_dict_for_original_structure[e_rem_name] = "; ".join(
                        reasons_for_segment
                    )

                error_message_details = "\n".join(fail_details_list)
                full_error_message = (
                    f"Error in _plan_route_greedy: Could not find a valid path from {last_seg_info_str} "
                    f"to any of the remaining {len(remaining)} segments. Details:\n{error_message_details}\n"
                    f"This cluster cannot be routed continuously."
                )

                debug_log(
                    debug_args,
                    "_plan_route_greedy: no valid connectors found. Detailed failure reasons:\n"
                    + full_error_message,
                )
                print(full_error_message, file=sys.stderr)

                # Populate the original fail_details if it's used elsewhere, though the new message is more comprehensive
                # For now, this populates it based on the new detailed reasons.
                # fail_details = fail_details_dict_for_original_structure # This line was in comments, but seems useful.
                # However, the original `fail_details` was used to build `details` string.
                # The new `full_error_message` replaces that.
                # If other parts of the system expect `fail_details` to be populated,
                # this dict should be assigned to a variable accessible by them.
                # For now, let's stick to the prompt's request of modifying the print and debug_log.

                return [], []  # No viable connector at all

            best = min(candidate_info, key=lambda c: c[0])
            trail_candidates = [c for c in candidate_info if not c[1]]

            if best[5] > allowed_max_road and trail_candidates:
                best_trail = min(trail_candidates, key=lambda c: c[0])
                if best_trail[0] <= best[0] * (1 + road_threshold):
                    chosen = best_trail
                else:
                    chosen = best
            else:
                if trail_candidates:
                    best_trail = min(trail_candidates, key=lambda c: c[0])
                    if best_trail[0] <= best[0] * (1 + road_threshold):
                        chosen = best_trail
                    else:
                        chosen = best
                else:
                    chosen = best

            time, uses_road, e, end, best_path_edges, _ = chosen
            debug_log(
                debug_args,
                f"_plan_route_greedy: chose segment {e.seg_id or e.name} via {len(best_path_edges)} connector edges, uses_road={uses_road}",
            )
            route.extend(best_path_edges)
            if end == e.start:
                route.append(e)
                order.append(e)
                cur = e.end
                last_seg = e
            else:
                # reverse orientation if allowed
                if e.direction != "both":
                    return [], []
                rev = Edge(
                    e.seg_id,
                    e.name,
                    e.end,
                    e.start,
                    e.length_mi,
                    e.elev_gain_ft,
                    list(reversed(e.coords)),
                    e.kind,
                    e.direction,
                    e.access_from,
                )
                route.append(rev)
                order.append(e)
                cur = rev.end
                last_seg = e
            remaining.remove(e)
            progress.update(1)  # Update for the original "Routing segments" bar
            greedy_selection_progress.update(
                1
            )  # Update for the new "Greedy segment selection" bar
            debug_log(
                debug_args,
                f"_plan_route_greedy: completed iteration, {len(remaining)} segments remaining",
            )

            if cur == start and not remaining:
                debug_log(
                    debug_args,
                    "_plan_route_greedy: all segments routed and returned to start",
                )
                return route, order
    finally:

        progress.close()
        greedy_selection_progress.close()

    # If any segments remain unrouted, treat the attempt as a failure
    if remaining:
        debug_log(
            debug_args,
            f"_plan_route_greedy: unable to route {len(remaining)} segments; returning failure",
        )
        return [], []

    if cur == start:
        debug_log(debug_args, "_plan_route_greedy: finished routing, already at start")
        return route, order

    G_for_path_back = G.copy()
    for edge_obj in edges:  # 'edges' is the original list of segments for this cluster
        if G_for_path_back.has_edge(edge_obj.start, edge_obj.end):
            # Ensure the edge data is directly accessible; MultiDiGraph might have a list
            # For simple Graph, this should be fine. If it's a MultiGraph, one might need to iterate G_for_path_back.get_edge_data
            edge_data = G_for_path_back[edge_obj.start][edge_obj.end]
            if isinstance(edge_data, list):  # Should not happen with G = nx.Graph()
                # This case is more complex if multiple edges connect same nodes.
                # Assuming build_nx_graph creates simple graph where one edge = one set of attributes.
                # For now, if it's a list, we might be modifying the wrong one or need to find the specific one.
                # However, current build_nx_graph adds one edge.
                pass  # Or log a warning if this structure is unexpected.

            # Check if 'weight' exists, add if not (though build_nx_graph should add it)
            if "weight" not in edge_data:
                edge_data["weight"] = planner_utils.estimate_time(
                    edge_obj, pace, grade, road_pace
                )  # Re-estimate if missing

            current_weight = edge_data["weight"]
            edge_data["weight"] = current_weight * path_back_penalty
            # For MultiGraph, one would do:
            #   G_for_path_back[edge_obj.start][edge_obj.end][key]['weight'] *= path_back_penalty

    try:
        path_pen_nodes = nx.shortest_path(G_for_path_back, cur, start, weight="weight")
        path_pen_edges = edges_from_path(G, path_pen_nodes)
        time_pen = total_time(path_pen_edges, pace, grade, road_pace)
    except nx.NetworkXNoPath:
        path_pen_edges = None
        time_pen = float("inf")

    try:
        if dist_cache is not None:
            if cur in dist_cache and start in dist_cache[cur]:
                path_unpen_nodes = dist_cache[cur][start]
            else:
                path_unpen_nodes = nx.shortest_path(G, cur, start, weight="weight")
                dist_cache.setdefault(cur, {})[start] = path_unpen_nodes
        else:
            path_unpen_nodes = nx.shortest_path(G, cur, start, weight="weight")
        path_unpen_edges = edges_from_path(G, path_unpen_nodes)
        time_unpen = total_time(path_unpen_edges, pace, grade, road_pace)
    except nx.NetworkXNoPath:
        path_unpen_edges = None
        time_unpen = float("inf")

    if time_pen <= time_unpen and path_pen_edges is not None:
        debug_log(
            debug_args, "_plan_route_greedy: returning to start via penalized path"
        )
        route.extend(path_pen_edges)
    elif path_unpen_edges is not None:
        debug_log(
            debug_args, "_plan_route_greedy: returning to start via unpenalized path"
        )
        route.extend(path_unpen_edges)

    return route, order


def _plan_route_for_sequence(
    G: nx.DiGraph,
    sequence: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    strict_max_foot_road: bool = False,
) -> List[Edge]:
    """Plan a route following ``sequence`` of segments in the given order."""

    cur = start
    route: List[Edge] = []
    degree: Dict[Tuple[float, float], int] = defaultdict(int)
    for seg in sequence:
        degree[seg.start] += 1
        degree[seg.end] += 1
    last_seg: Optional[Edge] = None
    for seg in sequence:
        candidates = []
        for end in [seg.start, seg.end]:
            if end == seg.end and seg.direction != "both":
                continue
            try:
                if dist_cache is not None:
                    if cur in dist_cache and end in dist_cache[cur]:
                        path_nodes = dist_cache[cur][end]
                    else:
                        path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                        dist_cache.setdefault(cur, {})[end] = path_nodes
                else:
                    path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                edges_path = edges_from_path(G, path_nodes)
                road_dist = sum(e.length_mi for e in edges_path if e.kind == "road")
                allowed_max_road = max_foot_road
                if (
                    last_seg is not None
                    and degree.get(cur, 0) == 1
                    and last_seg.length_mi <= spur_length_thresh
                ):
                    allowed_max_road += spur_road_bonus
                if road_dist > allowed_max_road:
                    continue
                t = sum(
                    planner_utils.estimate_time(e, pace, grade, road_pace)
                    for e in edges_path
                )
                t += planner_utils.estimate_time(seg, pace, grade, road_pace)
                uses_road = any(e.kind == "road" for e in edges_path)
                candidates.append((t, uses_road, end, edges_path))
            except nx.NetworkXNoPath:
                continue

        if not candidates:
            # fallback ignoring max_foot_road when not strict
            if strict_max_foot_road:
                return []
            # fallback ignoring max_foot_road
            for end in [seg.start, seg.end]:
                if end == seg.end and seg.direction != "both":
                    continue
                try:
                    if dist_cache is not None:
                        if cur in dist_cache and end in dist_cache[cur]:
                            path_nodes = dist_cache[cur][end]
                        else:
                            path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                            dist_cache.setdefault(cur, {})[end] = path_nodes
                    else:
                        path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                    edges_path = edges_from_path(G, path_nodes)
                    t = sum(
                        planner_utils.estimate_time(e, pace, grade, road_pace)
                        for e in edges_path
                    )
                    t += planner_utils.estimate_time(seg, pace, grade, road_pace)
                    uses_road = any(e.kind == "road" for e in edges_path)
                    candidates.append((t, uses_road, end, edges_path))
                except nx.NetworkXNoPath:
                    continue
            if not candidates:
                return []

        best = min(candidates, key=lambda c: c[0])
        trail_cands = [c for c in candidates if not c[1]]
        if trail_cands:
            best_trail = min(trail_cands, key=lambda c: c[0])
            if best_trail[0] <= best[0] * (1 + road_threshold):
                chosen = best_trail
            else:
                chosen = best
        else:
            chosen = best

        t, uses_road, end, path_edges = chosen
        route.extend(path_edges)
        if end == seg.start:
            route.append(seg)
            cur = seg.end
            last_seg = seg
        else:
            if seg.direction != "both":
                return []
            rev = Edge(
                seg.seg_id,
                seg.name,
                seg.end,
                seg.start,
                seg.length_mi,
                seg.elev_gain_ft,
                list(reversed(seg.coords)),
                seg.kind,
                seg.direction,
                seg.access_from,
            )
            route.append(rev)
            cur = rev.end
            last_seg = seg

    if cur != start:
        try:
            if dist_cache is not None:
                if cur in dist_cache and start in dist_cache[cur]:
                    path_back_nodes = dist_cache[cur][start]
                else:
                    path_back_nodes = nx.shortest_path(G, cur, start, weight="weight")
                    dist_cache.setdefault(cur, {})[start] = path_back_nodes
            else:
                path_back_nodes = nx.shortest_path(G, cur, start, weight="weight")
            route.extend(edges_from_path(G, path_back_nodes))
        except nx.NetworkXNoPath:
            pass

    return route


def plan_route_rpp(
    G: nx.DiGraph,
    edges: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    *,
    allow_connectors: bool = True,
    road_threshold: float = 0.25,
    rpp_timeout: float | None = None,
    debug_args: argparse.Namespace | None = None,
) -> List[Edge]:
    """Approximate Rural Postman solution using Steiner tree and Eulerization.

    Parameters
    ----------
    road_threshold : float, optional
        Maximum road mileage allowed when connecting disjoint components.
    rpp_timeout : float | None, optional
        Time limit in seconds. If exceeded, the best route found so far is
        returned instead of raising an error.
    """

    if not edges:
        return []

    start_time = time.perf_counter()

    def timed_out() -> bool:
        return (
            rpp_timeout is not None
            and rpp_timeout > 0
            and (time.perf_counter() - start_time >= rpp_timeout)
        )

    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        UG.add_edge(u, v, weight=data.get("weight", 0.0), edge=data.get("edge"))

    required_ids: Set[str] = {str(e.seg_id) for e in edges if e.seg_id is not None}
    required_nodes = {e.start for e in edges} | {e.end for e in edges}

    sub = nx.Graph()
    for e in edges:
        if UG.has_edge(e.start, e.end):
            sub.add_edge(e.start, e.end, **UG.get_edge_data(e.start, e.end))
        elif UG.has_edge(e.end, e.start):
            sub.add_edge(e.end, e.start, **UG.get_edge_data(e.end, e.start))

    if allow_connectors and not timed_out():
        debug_log(debug_args, "RPP: Calculating Steiner tree...")
        valid_required_nodes = set()
        for node in required_nodes:
            if node in UG.nodes():
                valid_required_nodes.add(node)
            else:
                debug_log(
                    debug_args,
                    f"RPP: Required node {node} not in UG. Removing from Steiner tree calculation.",
                )

        if len(valid_required_nodes) < 2:
            debug_log(
                debug_args,
                "RPP: Not enough valid required nodes for Steiner tree calculation after filtering. Returning empty list.",
            )
            return []

            # Check if UG has any nodes. If not, and valid_required_nodes exist, it's an inconsistency.
            # If UG is empty and valid_required_nodes is also empty, it's fine to proceed or return early.
            if not UG.nodes():
                if (
                    valid_required_nodes
                ):  # Should not happen if valid_required_nodes are filtered by UG.nodes()
                    debug_log(
                        debug_args,
                        "RPP: UG graph is empty, but valid_required_nodes is not. Inconsistency found. Returning empty list.",
                    )
                    return []
                else:  # UG is empty, and valid_required_nodes is also empty.
                    debug_log(
                        debug_args,
                        "RPP: UG graph is empty and no valid required nodes. Returning empty list.",
                    )
                    return []

            # Check if all valid_required_nodes are in the same connected component of UG
            if (
                valid_required_nodes
            ):  # Only proceed if there are nodes to check connectivity for
                components = list(nx.connected_components(UG))

                # This case (UG has nodes but components list is empty) should be extremely rare with NetworkX.
                if not components and UG.nodes():
                    debug_log(
                        debug_args,
                        "RPP: UG has nodes, but no connected components were found. This is unexpected. Returning empty list.",
                    )
                    return []

                # If valid_required_nodes is not empty, at least one component should exist if nodes are in UG.
                # And if components list is empty, valid_required_nodes must also be empty (or nodes are not in UG).
                if not components and valid_required_nodes:  # Defensive check
                    debug_log(
                        debug_args,
                        "RPP: No components found in UG, but valid_required_nodes exist. Nodes might not be in UG. Returning empty list.",
                    )
                    return []

                first_node = next(iter(valid_required_nodes))
                target_component_set = None
                node_found_in_any_component = False
                for comp_set in components:
                    if first_node in comp_set:
                        target_component_set = comp_set
                        node_found_in_any_component = True
                        break

                if not node_found_in_any_component:
                    # This implies first_node (which is in valid_required_nodes, meaning it was in UG.nodes())
                    # was not found in any component. This is theoretically impossible if nx.connected_components works as expected for non-empty graphs.
                    debug_log(
                        debug_args,
                        f"RPP: Critical error - Required node {first_node} (known to be in UG) was not found in any connected component. Returning empty list.",
                    )
                    return []

                for node in valid_required_nodes:
                    if node not in target_component_set:
                        debug_log(
                            debug_args,
                            f"RPP: Required nodes span multiple disconnected components in UG. Node {node} is not in the component of {first_node}. Cannot compute Steiner tree. Returning empty list.",
                        )
                        return []
                debug_log(
                    debug_args,
                    "RPP: All valid required nodes are confirmed to be in the same connected component of UG.",
                )

        try:
            steiner = nx.algorithms.approximation.steiner_tree(
                UG, valid_required_nodes, weight="weight"
            )
            sub.add_edges_from(steiner.edges(data=True))
            debug_log(debug_args, "RPP: Steiner tree calculation complete.")
        except (KeyError, ValueError) as e:
            debug_log(
                debug_args,
                f"RPP: Error during Steiner tree calculation: {e}. Returning empty list.",
            )
            return []

    if not nx.is_connected(sub) and not timed_out():
        debug_log(debug_args, "RPP: Connecting disjoint components...")
        components = list(nx.connected_components(sub))
        for i in range(len(components) - 1):
            c1 = components[i]
            c2 = components[i + 1]
            best_path = None
            best_road = float("inf")
            for u in c1:
                for v in c2:
                    try:
                        path = nx.shortest_path(UG, u, v, weight="weight")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    cand = edges_from_path(G, path, required_ids=required_ids)
                    road_dist = sum(e.length_mi for e in cand if e.kind == "road")
                    if road_dist < best_road:
                        best_road = road_dist
                        best_path = cand
            if best_path and best_road <= road_threshold:
                for e in best_path:
                    if UG.has_edge(e.start, e.end):
                        sub.add_edge(e.start, e.end, **UG.get_edge_data(e.start, e.end))
                    elif UG.has_edge(e.end, e.start):
                        sub.add_edge(e.end, e.start, **UG.get_edge_data(e.end, e.start))
        debug_log(debug_args, "RPP: Component connection phase complete.")

    if not nx.is_connected(sub):
        debug_log(
            debug_args,
            "RPP: Graph not connected after component connection attempt, returning empty list.",
        )
        return []

    if timed_out():
        debug_log(
            debug_args, "RPP: Timed out before Eulerization, returning empty list."
        )
        return []

    try:
        debug_log(debug_args, "RPP: Eulerizing graph...")
        eulerized = nx.euler.eulerize(sub)
        debug_log(debug_args, "RPP: Eulerization complete.")
    except (nx.NetworkXError, TimeoutError) as e:
        debug_log(debug_args, f"RPP: Eulerization failed: {e}, returning empty list.")
        return []

    original_start_node_for_eulerian = start  # For logging
    was_start_adjusted_for_eulerian = False  # For logging

    if start not in eulerized:
        debug_log(
            debug_args,
            f"RPP: Original start node {start} not in eulerized graph after eulerization. Finding nearest node in eulerized graph.",
        )
        if not list(eulerized.nodes()):  # Check if eulerized graph has any nodes
            debug_log(
                debug_args,
                "RPP: Eulerized graph has no nodes. Cannot find a start node. Returning empty list.",
            )
            return []
        tree_tmp = build_kdtree(list(eulerized.nodes()))
        start = nearest_node(tree_tmp, start)
        was_start_adjusted_for_eulerian = True
        debug_log(
            debug_args,
            f"RPP: Adjusted start node for Eulerian circuit to {start} (was {original_start_node_for_eulerian}).",
        )

    # ==> START NEW CODE BLOCK <==
    if not list(eulerized.nodes()):
        debug_log(
            debug_args,
            "RPP: Eulerized graph is empty. Cannot generate circuit. Returning empty list.",
        )
        return []

    is_start_valid = False
    if start in eulerized:
        if eulerized.degree(start) > 0:
            is_start_valid = True
        else:
            debug_log(
                debug_args,
                f"RPP: Current start node {start} has degree 0 in eulerized graph.",
            )
    else:
        debug_log(
            debug_args,
            f"RPP: Current start node {start} is not in the eulerized graph (nodes: {list(eulerized.nodes())[:5]}...).",
        )

    if not is_start_valid:
        debug_log(
            debug_args,
            f"RPP: Initial/adjusted start node {start} is invalid for Eulerian circuit. Attempting to find an alternative.",
        )
        alternative_start_found = False
        for node in eulerized.nodes():
            if eulerized.degree(node) > 0:
                start = node
                alternative_start_found = True
                debug_log(
                    debug_args,
                    f"RPP: Selected alternative start node {start} with degree {eulerized.degree(start)}.",
                )
                break
        if not alternative_start_found:
            debug_log(
                debug_args,
                "RPP: No valid alternative start node found in eulerized graph with degree > 0. Returning empty list.",
            )
            return []
    else:
        debug_log(
            debug_args,
            f"RPP: Using start node {start} (degree {eulerized.degree(start) if start in eulerized else 'N/A'}) for Eulerian circuit. Original was {original_start_node_for_eulerian}, adjusted: {was_start_adjusted_for_eulerian}.",
        )
    # ==> END NEW CODE BLOCK <==

    debug_log(debug_args, "RPP: Generating Eulerian circuit...")
    try:
        circuit = list(nx.eulerian_circuit(eulerized, source=start))
        debug_log(debug_args, "RPP: Eulerian circuit generation complete.")
    except nx.NetworkXError as e:  # Handle cases where eulerian_circuit might fail
        debug_log(
            debug_args,
            f"RPP: Eulerian circuit generation failed: {e}, returning empty list.",
        )
        return []

    if timed_out():
        debug_log(
            debug_args,
            "RPP: Timed out after Eulerian circuit generation, returning partial or empty route.",
        )
        return []  # Or route, if partial routes are acceptable on timeout
    route: List[Edge] = []
    for u, v in circuit:
        data = eulerized.get_edge_data(u, v)
        if data:
            if isinstance(data, dict) and 0 in data:
                ed = data[0].get("edge")
            else:
                ed = data.get("edge")
        else:
            ed = None
        if ed is not None:
            sid = str(ed.seg_id) if ed.seg_id is not None else None
            if sid is None or sid not in required_ids:
                if ed.kind == "trail":
                    ed = Edge(
                        ed.seg_id,
                        ed.name,
                        ed.start,
                        ed.end,
                        ed.length_mi,
                        ed.elev_gain_ft,
                        ed.coords,
                        "connector",
                        ed.direction,
                        ed.access_from,
                    )
            route.append(ed)
        else:
            try:
                path = nx.shortest_path(G, u, v, weight="weight")
                route.extend(edges_from_path(G, path, required_ids=required_ids))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if timed_out():
            break

    return route


def _reverse_edge(e: Edge) -> Edge:
    """Return a copy of ``e`` reversed in direction."""
    return Edge(
        e.seg_id,
        e.name,
        e.end,
        e.start,
        e.length_mi,
        e.elev_gain_ft,
        list(reversed(e.coords)),
        e.kind,
        e.direction,
        e.access_from,
    )


def _edges_form_tree(edges: List[Edge]) -> bool:
    """Return ``True`` if ``edges`` form a tree (no cycles)."""
    UG = nx.Graph()
    for e in edges:
        UG.add_edge(e.start, e.end)
    return nx.is_tree(UG)


def _plan_route_tree(edges: List[Edge], start: Tuple[float, float]) -> List[Edge]:
    """Plan a depth-first traversal of a tree network."""

    if not edges:
        return []

    UG = nx.Graph()
    for e in edges:
        UG.add_edge(e.start, e.end, edge=e)

    visited: set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
    route: List[Edge] = []

    def dfs(u: Tuple[float, float], parent: Tuple[float, float] | None) -> None:
        for v in UG.neighbors(u):
            if parent is not None and v == parent:
                continue
            key = tuple(sorted((u, v)))
            if key in visited:
                continue
            visited.add(key)
            e = UG[u][v]["edge"]
            forward = e if e.start == u else _reverse_edge(e)
            back = _reverse_edge(forward)
            route.append(forward)
            dfs(v, u)
            route.append(back)

    dfs(start, None)
    return route


def plan_route(
    G: nx.DiGraph,
    edges: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    use_rpp: bool = True,
    allow_connectors: bool = True,
    rpp_timeout: float = 5.0,
    debug_args: argparse.Namespace | None = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    path_back_penalty: float = 1.2,
    redundancy_threshold: float | None = None,
    use_advanced_optimizer: bool = False,
    strict_max_foot_road: bool = False,
    optimizer_name: str = "greedy2opt",
    postman_timeout: float = 30.0,
    postman_max_odd: int = 40,
) -> List[Edge]:
    """Plan an efficient loop through ``edges`` starting and ending at ``start``."""

    debug_log(
        debug_args,
        f"plan_route: Initiating for {len(edges)} segments, start_node={start}, use_rpp={use_rpp}",
    )

    cluster_nodes = {e.start for e in edges} | {e.end for e in edges}
    if start not in cluster_nodes:
        tree_tmp = build_kdtree(list(cluster_nodes))
        start = nearest_node(tree_tmp, start)
        debug_log(debug_args, f"plan_route: Adjusted start node to {start}")

    if optimizer_name == "postman":
        try:
            from . import postman

            return postman.solve_rpp(
                G,
                edges,
                start,
                pace=pace,
                grade=grade,
                road_pace=road_pace,
                timeout=postman_timeout,
                max_odd=postman_max_odd,
            )
        except Exception as e:  # pragma: no cover - fallback path
            debug_log(
                debug_args, f"postman optimizer failed: {e}; falling back to greedy"
            )

    debug_log(debug_args, "plan_route: Checking for tree traversal applicability.")
    if _edges_form_tree(edges) and all(e.direction == "both" for e in edges):
        debug_log(
            debug_args, "plan_route: Attempting tree traversal with _plan_route_tree."
        )
        try:
            tree_route = _plan_route_tree(edges, start)
            if tree_route:
                debug_log(
                    debug_args,
                    "plan_route: Tree traversal successful, returning route.",
                )
                return tree_route
            else:
                debug_log(
                    debug_args, "plan_route: Tree traversal did not return a route."
                )
        except Exception as e:
            debug_log(debug_args, f"plan_route: Tree traversal failed: {e}")
            pass
    else:
        debug_log(
            debug_args,
            "plan_route: Conditions for tree traversal not met or it failed.",
        )

    debug_log(debug_args, "plan_route: Entering RPP stage check.")
    if use_rpp:
        debug_log(debug_args, "plan_route: RPP is enabled. Checking connectivity.")
        connectivity_subs = split_cluster_by_connectivity(
            edges, G, max_foot_road, debug_args=debug_args
        )
        if len(connectivity_subs) == 1:
            debug_log(
                debug_args, "plan_route: Cluster is connected for RPP. Attempting RPP."
            )
            try:
                route_rpp = plan_route_rpp(
                    G,
                    edges,
                    start,
                    pace,
                    grade,
                    road_pace,
                    allow_connectors=allow_connectors,
                    road_threshold=road_threshold,
                    rpp_timeout=rpp_timeout,
                    debug_args=debug_args,
                )
                if route_rpp:
                    debug_log(
                        debug_args,
                        f"plan_route: RPP successful, route found with {len(route_rpp)} edges.",
                    )
                    return route_rpp
                else:
                    debug_log(
                        debug_args,
                        "plan_route: RPP attempted but returned an empty route. Proceeding to greedy.",
                    )
            except (nx.NodeNotFound, nx.NetworkXNoPath) as e:
                debug_log(
                    debug_args,
                    f"plan_route: RPP connectivity issue: {e}. Retrying with new start.",
                )
                try:
                    start = next(iter(cluster_nodes))  # type: ignore
                    debug_log(
                        debug_args, f"plan_route: RPP retry with start_node={start}."
                    )
                    route_rpp = plan_route_rpp(
                        G,
                        edges,
                        start,
                        pace,
                        grade,
                        road_pace,
                        allow_connectors=allow_connectors,
                        road_threshold=road_threshold,
                        rpp_timeout=rpp_timeout,
                        debug_args=debug_args,
                    )
                    if route_rpp:
                        debug_log(
                            debug_args,
                            f"plan_route: RPP retry successful, route found with {len(route_rpp)} edges.",
                        )
                        return route_rpp
                    else:
                        debug_log(
                            debug_args,
                            "plan_route: RPP retry returned an empty route. Proceeding to greedy.",
                        )
                except Exception as e2:
                    debug_log(
                        debug_args,
                        f"plan_route: RPP retry failed: {e2}. Proceeding to greedy.",
                    )
            except Exception as e:
                debug_log(
                    debug_args,
                    f"plan_route: RPP failed with exception: {e}. Proceeding to greedy.",
                )
        else:
            debug_log(
                debug_args,
                f"plan_route: RPP skipped, split_cluster_by_connectivity resulted in {len(connectivity_subs)} sub-clusters. Proceeding to greedy.",
            )
    else:
        debug_log(
            debug_args,
            "plan_route: RPP was disabled by use_rpp=False. Proceeding to greedy.",
        )

    debug_log(
        debug_args, f"plan_route_greedy: Graph G has {G.number_of_nodes()} nodes."
    )
    debug_log(
        debug_args, f"plan_route_greedy: Graph G has {G.number_of_edges()} edges."
    )
    debug_log(debug_args, f"plan_route_greedy: Routing {len(edges)} segments.")
    debug_log(
        debug_args, "plan_route: Entering greedy search stage with _plan_route_greedy."
    )
    dijkstra_timeout_for_greedy = None # Default, _plan_route_greedy uses its own default
    if len(edges) <= 2: # If it's a small cluster
        dijkstra_timeout_for_greedy = min(rpp_timeout * 2, 10.0) # e.g., 10 seconds, or related to rpp_timeout but capped
        # Ensure it's at least a small positive value, e.g. 1 second.
        dijkstra_timeout_for_greedy = max(dijkstra_timeout_for_greedy, 1.0)
        debug_log(
            debug_args,
            f"plan_route: Small cluster (len {len(edges)}), setting Dijkstra timeout for greedy to {dijkstra_timeout_for_greedy}s."
        )
    initial_route, seg_order = _plan_route_greedy(
        G,
        edges,
        start,
        pace,
        grade,
        road_pace,
        max_foot_road,
        road_threshold,
        dist_cache,
        spur_length_thresh=spur_length_thresh,
        spur_road_bonus=spur_road_bonus,
        path_back_penalty=path_back_penalty,
        strict_max_foot_road=strict_max_foot_road,
        dijkstra_timeout_override=dijkstra_timeout_for_greedy, # New parameter
        debug_args=debug_args,
    )
    if initial_route:
        debug_log(
            debug_args,
            f"plan_route: Greedy search completed. Initial route found with {len(initial_route)} edges. Segment order length: {len(seg_order)}. Route time: {total_time(initial_route, pace, grade, road_pace):.2f} min.",
        )
    else:
        debug_log(
            debug_args,
            "plan_route: Greedy search failed to find an initial route.",
        )
        return initial_route  # Return empty list if greedy failed

    if len(seg_order) <= 2:  # seg_order can't be empty if initial_route is not
        debug_log(
            debug_args,
            "plan_route: Route has too few segments for 2-Opt, returning greedy route.",
        )
        return initial_route

    best_route = initial_route
    best_order = seg_order
    best_time = total_time(best_route, pace, grade, road_pace)

    debug_log(debug_args, "plan_route: Entering 2-Opt optimization loop.")
    improved = True
    while improved:
        improved = False
        n = len(best_order)
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                new_order = best_order[:i] + best_order[i:j][::-1] + best_order[j:]
                candidate_route = _plan_route_for_sequence(
                    G,
                    new_order,
                    start,
                    pace,
                    grade,
                    road_pace,
                    max_foot_road,
                    road_threshold,
                    dist_cache,
                    spur_length_thresh=spur_length_thresh,
                    spur_road_bonus=spur_road_bonus,
                    strict_max_foot_road=strict_max_foot_road,
                )
                if not candidate_route:
                    continue
                cand_time = total_time(candidate_route, pace, grade, road_pace)
                if cand_time < best_time:
                    debug_log(
                        debug_args,
                        f"2-opt improvement {best_time:.2f} -> {cand_time:.2f}",
                    )
                    best_time = cand_time
                    best_route = candidate_route
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
    debug_log(
        debug_args,
        f"plan_route: 2-Opt optimization completed. Final route time after 2-Opt: {best_time:.2f} min.",
    )

    debug_log(debug_args, "plan_route: Entering advanced optimizer stage check.")
    if use_advanced_optimizer:
        debug_log(
            debug_args,
            "plan_route: Advanced optimizer is enabled. Attempting advanced optimization.",
        )
        ctx_adv = planner_utils.PlanningContext(
            graph=G,
            pace=pace,
            grade=grade,
            road_pace=road_pace,
            dist_cache=dist_cache,
        )
        required_ids = {str(e.seg_id) for e in edges if e.seg_id is not None}
        adv_route, adv_best_order = optimizer.advanced_2opt_optimization(
            ctx_adv,
            best_order,
            start,
            required_ids,
            max_foot_road,
            road_threshold,
            strict_max_foot_road=strict_max_foot_road,
        )
        if adv_route:
            best_route = adv_route
            best_order = adv_best_order  # Update best_order as well
            best_time = total_time(best_route, pace, grade, road_pace)
            debug_log(
                debug_args,
                f"plan_route: Advanced optimizer completed. New route time: {best_time:.2f} min.",
            )
        else:
            debug_log(
                debug_args, "plan_route: Advanced optimizer did not return a new route."
            )
    else:
        debug_log(debug_args, "plan_route: Advanced optimizer is disabled.")

    debug_log(debug_args, "plan_route: Entering redundancy optimization stage check.")
    if redundancy_threshold is not None:
        debug_log(
            debug_args,
            "plan_route: Redundancy optimization is enabled. Attempting optimization.",
        )
        ctx = planner_utils.PlanningContext(
            graph=G,
            pace=pace,
            grade=grade,
            road_pace=road_pace,
            dist_cache=dist_cache,
        )
        required = {str(e.seg_id) for e in edges if e.seg_id is not None}
        optimized_route_redundancy = planner_utils.optimize_route_for_redundancy(
            ctx, best_route, required, redundancy_threshold
        )
        if optimized_route_redundancy:
            best_route = optimized_route_redundancy
            debug_log(debug_args, "plan_route: Redundancy optimization completed.")
        else:
            debug_log(
                debug_args,
                "plan_route: Redundancy optimization did not change the route or failed.",
            )
    else:
        debug_log(
            debug_args,
            "plan_route: Redundancy optimization is disabled (threshold is None).",
        )

    debug_log(
        debug_args,
        f"plan_route: Finalizing route. Chosen route has {len(best_route)} edges.",
    )
    return best_route


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
    # Use fewer initializations for better performance on large datasets.
    try:
        km = KMeans(n_clusters=k, n_init=1, algorithm="elkan", random_state=0)
    except TypeError:
        km = KMeans(n_clusters=k, n_init=1, random_state=0)
    print("Starting KMeans fitting and prediction...")
    labels = km.fit_predict(pts)
    print("Finished KMeans fitting and prediction.")
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
    print(
        f"Cluster merging: Starting with {len(clusters)} clusters, target max_clusters {max_clusters}"
    )
    while len(clusters) > max_clusters:
        print(
            f"Cluster merging: Current clusters {len(clusters)}, attempting to reduce..."
        )
        clusters.sort(key=lambda c: total_time(c, pace, grade, road_pace))
        small = clusters.pop(0)
        merged = False
        for i, other in enumerate(clusters):
            print(
                f"Cluster merging: Considering merging small cluster (size {len(small)}) with other cluster (size {len(other)})"
            )
            if (
                total_time(other, pace, grade, road_pace)
                + total_time(small, pace, grade, road_pace)
                <= budget
            ):
                clusters[i] = other + small
                merged = True
                print(
                    f"Cluster merging: Successfully merged. New cluster size {len(clusters[i])}. Clusters remaining: {len(clusters)}"
                )
                break
        if not merged:
            print(
                "Cluster merging: Smallest cluster could not be merged with any other. Exiting merge loop."
            )
            clusters.append(small)
            break
    print(f"Cluster merging: Finished. Clusters count: {len(clusters)}")

    def _repartition_cluster(edges: List[Edge]) -> List[List[Edge]]:
        """Partition ``edges`` into routable subclusters.

        The previous implementation used recursion which could exceed Python's
        call stack limit when dealing with very large clusters.  This version
        mimics that logic iteratively using an explicit stack.
        """

        result: List[List[Edge]] = []
        stack: List[List[Edge]] = [edges]

        while stack:
            part = stack.pop()
            print(
                f"Repartitioning: stack size {len(stack)}, current part edges {len(part)}"
            )
            if not part:
                result.append(part)
                continue

            Gc = nx.Graph()
            for ed in part:
                w = planner_utils.estimate_time(ed, pace, grade, road_pace)
                Gc.add_edge(ed.start, ed.end, weight=w, edge=ed)

            # First attempt a simple tree traversal
            if _edges_form_tree(part):
                try:
                    _plan_route_tree(part, part[0].start)
                    result.append(part)
                    continue
                except Exception:
                    pass

            mst = (
                nx.maximum_spanning_tree(Gc, weight="weight")
                if Gc.number_of_edges()
                else Gc
            )
            if mst.number_of_edges() == 0:
                result.append(part)
                continue

            u, v, data = max(mst.edges(data=True), key=lambda t: t[2]["weight"])
            mst.remove_edge(u, v)
            comps = list(nx.connected_components(mst))
            if len(comps) <= 1:
                result.append(part)
                continue

            part_seg_ids = {e.seg_id for e in part}

            skip_subs = False
            for comp in comps:
                sub = [e for e in part if e.start in comp or e.end in comp]
                sub_seg_ids = {e.seg_id for e in sub}
                if sub_seg_ids == part_seg_ids and len(sub) == len(part):
                    print(
                        f"Repartitioning: Detected potential loop with part size {len(part)}. Adding part directly to results."
                    )
                    result.append(part)
                    skip_subs = True
                    break
                if len(sub) > 1:
                    stack.append(sub)
                else:
                    result.append(sub)

            if skip_subs:
                continue

        return result

    refined: List[List[Edge]] = []
    for c in tqdm(clusters, desc="Repartitioning sub-clusters"):
        for part in _repartition_cluster(c):
            if part:
                refined.append(part)

    # return refined[:max_clusters] # Allow all refined clusters to be returned
    return refined


def split_cluster_by_connectivity(
    cluster_edges: List[Edge],
    G: nx.DiGraph,
    max_foot_road: float,
    debug_args: argparse.Namespace | None = None,
) -> List[List[Edge]]:
    """Split ``cluster_edges`` into subclusters based on walkable connectivity.

    Two segments are considered connected if a path exists between any of their
    endpoints using at most ``max_foot_road`` miles of road. Trail mileage along the
    connector does not count against this limit. The returned list will contain
    one or more groups of edges that are internally connected according to this
    rule. The original ordering of segments is not preserved.
    """
    debug_log(
        debug_args,
        f"split_cluster_by_connectivity: Input cluster_edges: {[e.seg_id for e in cluster_edges]}, max_foot_road: {max_foot_road}",
    )

    def road_weight(
        u: Tuple[float, float], v: Tuple[float, float], data: dict
    ) -> float:
        edge = data.get("edge") if isinstance(data, dict) else data[0]["edge"]
        return edge.length_mi if edge.kind == "road" else 0.0

    remaining = list(cluster_edges)
    subclusters: List[List[Edge]] = []

    while remaining:
        seed = remaining.pop(0)
        debug_log(
            debug_args,
            f"split_cluster_by_connectivity: Starting new group with seed: {seed.seg_id}",
        )
        group = [seed]
        group_nodes = {seed.start, seed.end}

        changed = True
        while changed:
            changed = False
            for seg in remaining[:]:
                reachable = False
                for gn in list(group_nodes):
                    for node in (seg.start, seg.end):
                        try:
                            dist = nx.shortest_path_length(
                                G, gn, node, weight=road_weight
                            )
                            debug_log(
                                debug_args,
                                f"split_cluster_by_connectivity: Path check from {gn} to {node}, dist={dist}, reachable={dist <= max_foot_road}",
                            )
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            debug_log(
                                debug_args,
                                f"split_cluster_by_connectivity: No path found between {gn} and {node}",
                            )
                            continue
                        if dist <= max_foot_road:
                            reachable = True
                            break
                    if reachable:
                        break
                if reachable:
                    group.append(seg)
                    group_nodes.update([seg.start, seg.end])
                    remaining.remove(seg)
                    changed = True

        subclusters.append(group)

    debug_log(
        debug_args,
        f"split_cluster_by_connectivity: Found {len(subclusters)} subclusters.",
    )
    for i, sc in enumerate(subclusters):
        debug_log(
            debug_args,
            f"split_cluster_by_connectivity: Subcluster {i+1} segment IDs: {[e.seg_id for e in sc]}",
        )
    return subclusters


def smooth_daily_plans(
    daily_plans: List[Dict[str, object]],
    remaining_clusters: List[ClusterInfo],
    daily_budget_minutes: Dict[datetime.date, float],
    G: nx.Graph,
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    allow_connector_trails: bool = True,
    rpp_timeout: float = 5.0,
    road_graph: Optional[nx.Graph] = None,
    average_driving_speed_mph: float = 30.0,
    home_coord: Optional[Tuple[float, float]] = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    use_advanced_optimizer: bool = False,
    redundancy_threshold: float | None = None,
    debug_args: argparse.Namespace | None = None,
    strict_max_foot_road: bool = False,
) -> None:
    """Fill underutilized days with any remaining clusters."""

    if not remaining_clusters:
        return

    def remaining_time(day_plan: Dict[str, object]) -> float:
        budget = daily_budget_minutes.get(day_plan["date"], 240.0)
        used = day_plan["total_activity_time"] + day_plan["total_drive_time"]
        return budget - used

    # sort days by available time descending
    days_sorted = sorted(daily_plans, key=remaining_time, reverse=True)

    for cluster in tqdm(
        list(remaining_clusters),
        desc="Smoothing daily plans",
        unit="cluster",
        leave=False,
    ):
        # choose a starting node
        start_candidates = cluster.start_candidates
        if start_candidates:
            # Choose the start candidate with the shortest estimated drive
            # time from ``home_coord`` when possible.  This helps minimize
            # driving distance and encourages more efficient loops.
            start_node = start_candidates[0][0]
            start_name = start_candidates[0][1]
            if (
                home_coord is not None
                and road_graph is not None
                and average_driving_speed_mph > 0
            ):
                best_time = float("inf")
                for cand_node, cand_name in start_candidates:
                    drive_time = planner_utils.estimate_drive_time_minutes(
                        home_coord,
                        cand_node,
                        road_graph,
                        average_driving_speed_mph,
                    )
                    if drive_time < best_time:
                        best_time = drive_time
                        start_node = cand_node
                        start_name = cand_name
        else:
            centroid = (
                sum(midpoint(e)[0] for e in cluster.edges) / len(cluster.edges),
                sum(midpoint(e)[1] for e in cluster.edges) / len(cluster.edges),
            )
            start_node = nearest_node(cluster.node_tree, centroid)
            start_name = None

        route_edges = plan_route(
            G,
            cluster.edges,
            start_node,
            pace,
            grade,
            road_pace,
            max_foot_road,
            road_threshold,
            dist_cache,
            use_rpp=True,
            allow_connectors=allow_connector_trails,
            rpp_timeout=rpp_timeout,
            debug_args=debug_args,
            spur_length_thresh=spur_length_thresh,
            spur_road_bonus=spur_road_bonus,
            use_advanced_optimizer=use_advanced_optimizer,
            strict_max_foot_road=strict_max_foot_road,
            redundancy_threshold=redundancy_threshold,
        )
        if not route_edges:
            continue
        est_time = total_time(route_edges, pace, grade, road_pace)

        max_avail = max(remaining_time(d) for d in days_sorted) if days_sorted else 0
        if est_time > max_avail and len(cluster.edges) > 1 and max_avail > 0:
            num_parts = max(1, int(math.ceil(est_time / max_avail)))
            subclusters = cluster_segments(
                cluster.edges,
                pace=pace,
                grade=grade,
                budget=max_avail,
                max_clusters=num_parts,
                road_pace=road_pace,
            )
            for sub in subclusters:
                if not sub:
                    continue
                remaining_clusters.append(
                    ClusterInfo(
                        sub,
                        {pt for e in sub for pt in (e.start, e.end)},
                        start_candidates,
                    )
                )
            remaining_clusters.remove(cluster)
            days_sorted = sorted(daily_plans, key=remaining_time, reverse=True)
            continue

        # try to find a day with enough remaining time
        placed = False
        for day_plan in days_sorted:
            avail = remaining_time(day_plan)
            extra_drive = 0.0
            drive_from = home_coord
            if road_graph is not None and average_driving_speed_mph > 0:
                acts = [
                    a
                    for a in day_plan.get("activities", [])
                    if a.get("type") == "activity"
                ]
                if acts:
                    drive_from = acts[-1]["route_edges"][-1].end
                if drive_from is not None:
                    extra_drive = planner_utils.estimate_drive_time_minutes(
                        drive_from,
                        start_node,
                        road_graph,
                        average_driving_speed_mph,
                    )
                    if (
                        home_coord is not None
                        and drive_from == home_coord
                        and planner_utils._haversine_mi(drive_from, start_node)
                        <= MIN_DRIVE_DISTANCE_MI
                    ):
                        extra_drive = 0.0
                    extra_drive += DRIVE_PARKING_OVERHEAD_MIN

            if avail >= est_time + extra_drive:
                if extra_drive > 0:
                    day_plan.setdefault("activities", []).append(
                        {
                            "type": "drive",
                            "minutes": extra_drive,
                            "from_coord": drive_from,
                            "to_coord": start_node,
                        }
                    )
                    day_plan["total_drive_time"] += extra_drive
                day_plan.setdefault("activities", []).append(
                    {
                        "type": "activity",
                        "route_edges": route_edges,
                        "name": f"Activity Part {len([a for a in day_plan['activities'] if a['type']=='activity']) + 1}",
                        "ignored_budget": False,
                        "start_name": start_name,
                        "start_coord": start_node,
                    }
                )
                day_plan["total_activity_time"] += est_time
                placed = True
                break
        if placed:
            remaining_clusters.remove(cluster)


def force_schedule_remaining_clusters(
    daily_plans: List[Dict[str, object]],
    remaining_clusters: List[ClusterInfo],
    daily_budget_minutes: Dict[datetime.date, float],
    G: nx.Graph,
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    allow_connector_trails: bool = True,
    rpp_timeout: float = 5.0,
    road_graph: Optional[nx.Graph] = None,
    average_driving_speed_mph: float = 30.0,
    home_coord: Optional[Tuple[float, float]] = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    use_advanced_optimizer: bool = False,
    redundancy_threshold: float | None = None,
    debug_args: argparse.Namespace | None = None,
    strict_max_foot_road: bool = False,
) -> None:
    """Force schedule any remaining clusters ignoring daily budgets.

    Only unscheduled segments are considered. Segments already placed in
    ``daily_plans`` are skipped so that completed trails are not duplicated.
    """

    if not remaining_clusters:
        if debug_args:
            debug_log(
                debug_args,
                "force_schedule_remaining_clusters: Called with no remaining_clusters.",
            )
        return

    if debug_args:
        debug_log(
            debug_args,
            f"force_schedule_remaining_clusters: Entered with {len(remaining_clusters)} clusters.",
        )

    scheduled_ids: set[str] = set()
    for plan in daily_plans:
        for act in plan.get("activities", []):
            if act.get("type") != "activity":
                continue
            for e_act in act.get("route_edges", []):  # Renamed 'e' to 'e_act'
                if e_act.seg_id is not None:
                    scheduled_ids.add(str(e_act.seg_id))

    initial_len_remaining_clusters = len(remaining_clusters)
    # Existing filtering loop for cluster.edges:
    for cluster in list(remaining_clusters):
        cluster.edges = [
            e_filter
            for e_filter in cluster.edges
            if str(e_filter.seg_id) not in scheduled_ids
        ]  # Renamed 'e' to 'e_filter'
        if not cluster.edges:
            remaining_clusters.remove(cluster)
            continue
        cluster.nodes = {pt for ed in cluster.edges for pt in (ed.start, ed.end)}
        cluster.node_tree = build_kdtree(
            list(cluster.nodes)
        )  # Added this line from original code

    if debug_args:
        debug_log(
            debug_args,
            f"force_schedule_remaining_clusters: Initial num clusters: {initial_len_remaining_clusters}. After filtering already scheduled segments, {len(remaining_clusters)} clusters have segments remaining.",
        )

    if not remaining_clusters:
        if debug_args:
            debug_log(
                debug_args,
                "force_schedule_remaining_clusters: No clusters left after filtering.",
            )
        return

    segments_in_force_schedule = set()
    mileage_in_force_schedule = 0.0
    temp_seen_ids_for_mileage_log_fs = set()
    for cluster_info_fs in remaining_clusters:
        for edge_fs in cluster_info_fs.edges:
            if edge_fs.seg_id is not None:
                segments_in_force_schedule.add(str(edge_fs.seg_id))
                if str(edge_fs.seg_id) not in temp_seen_ids_for_mileage_log_fs:
                    mileage_in_force_schedule += edge_fs.length_mi
                    temp_seen_ids_for_mileage_log_fs.add(str(edge_fs.seg_id))
    if debug_args:
        debug_log(
            debug_args,
            f"force_schedule_remaining_clusters: Processing {len(remaining_clusters)} clusters with {len(segments_in_force_schedule)} unique segment IDs, totaling {mileage_in_force_schedule:.2f} mi.",
        )

    huge_budget = {
        d: daily_budget_minutes.get(d, 240.0) + 1_000_000 for d in daily_budget_minutes
    }

    smooth_daily_plans(
        daily_plans,
        remaining_clusters,
        huge_budget,
        G,
        pace,
        grade,
        road_pace,
        max_foot_road,
        road_threshold,
        dist_cache,
        allow_connector_trails=allow_connector_trails,
        rpp_timeout=rpp_timeout,
        road_graph=road_graph,
        average_driving_speed_mph=average_driving_speed_mph,
        home_coord=home_coord,
        spur_length_thresh=spur_length_thresh,
        spur_road_bonus=spur_road_bonus,
        use_advanced_optimizer=use_advanced_optimizer,
        redundancy_threshold=redundancy_threshold,
        debug_args=debug_args,
        strict_max_foot_road=strict_max_foot_road,
    )


def even_out_budgets(
    daily_plans: List[Dict[str, object]],
    daily_budget_minutes: Dict[datetime.date, float],
) -> float:
    """Increase all budgets equally so no day exceeds its allotment."""

    max_over = 0.0
    for dp in daily_plans:
        used = dp["total_activity_time"] + dp["total_drive_time"]
        b = daily_budget_minutes.get(dp["date"], 240.0)
        over = used - b
        if over > max_over:
            max_over = over

    if max_over > 0:
        for d in daily_budget_minutes:
            daily_budget_minutes[d] += max_over

    return max_over


def update_plan_notes(
    daily_plans: List[Dict[str, object]],
    daily_budget_minutes: Dict[datetime.date, float],
) -> None:
    """Refresh plan notes after budget adjustments."""

    for dp in daily_plans:
        budget = daily_budget_minutes.get(dp["date"], 240.0)
        total_day_time = dp["total_activity_time"] + dp["total_drive_time"]

        note_parts = []
        if total_day_time > budget:
            note_parts.append(f"over budget by {total_day_time - budget:.1f} min")
        else:
            note_parts.append("fits budget")

        if budget <= 120:
            note_parts.append("night run \u2013 kept easy")

        dp["notes"] = "; ".join(note_parts)


def _human_join(items: List[str]) -> str:
    """Join a list of strings with commas and 'and'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_route_description(
    activities: List[Dict[str, object]],
    total_distance_mi: float,
    total_time_min: float,
    drive_time_min: float,
    total_gain_ft: float,
    challenge_ids: Optional[Set[str]] = None,
) -> str:
    """Return a concise human-friendly description for a day's plan."""

    if not activities:
        return "Unable to complete"

    start_name = activities[0].get("start_name") or ""
    route_edges: List[Edge] = []
    for act in activities:
        if act.get("type") == "activity":
            route_edges.extend(act.get("route_edges", []))

    trail_names: List[str] = []
    prev = None
    for e in route_edges:
        if e.kind != "trail":
            continue
        if challenge_ids is not None and str(e.seg_id) not in challenge_ids:
            continue
        name = e.name or str(e.seg_id)
        if name != prev:
            trail_names.append(name)
            prev = name

    trails_part = _human_join(trail_names)
    if start_name:
        desc = (
            f"Starting at {start_name} via {trails_part}"
            if trails_part
            else f"Starting at {start_name}"
        )
    else:
        desc = f"Route covers {trails_part}" if trails_part else "Route"

    dist_part = f"~{total_distance_mi:.1f} miles"
    if total_time_min >= 90:
        time_part = f"~{total_time_min/60:.1f} hours"
    else:
        time_part = f"~{round(total_time_min)} min"
    desc += f", {dist_part} in {time_part}"
    if drive_time_min:
        desc += f" (plus {round(drive_time_min)} min drive)"

    difficulty = "Easy"
    if total_distance_mi > 7 or total_gain_ft > 1500:
        difficulty = "Hard"
    elif total_distance_mi > 3 or total_gain_ft > 500:
        difficulty = "Moderate"

    mid = len(route_edges) // 2 if route_edges else 0
    gain_first = sum(e.elev_gain_ft for e in route_edges[:mid])
    gain_second = sum(e.elev_gain_ft for e in route_edges[mid:])
    note = ""
    if total_gain_ft < 200:
        note = "mostly flat"
    elif gain_first > gain_second * 1.5 and gain_first - gain_second > 250:
        note = "steep start"
    elif gain_second > gain_first * 1.5 and gain_second - gain_first > 250:
        note = "steep finish"

    desc += f". Difficulty: {difficulty}"
    if note:
        desc += f" ({note})"
    return desc


def write_plan_html(
    path: str,
    daily_plans: List[Dict[str, object]],
    image_dir: str,
    dem_path: Optional[str] = None,
    routing_failed: bool = False,
) -> None:
    """Write an HTML overview of ``daily_plans`` with maps and elevation."""

    os.makedirs(image_dir, exist_ok=True)

    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<style>img{max-width:100%;height:auto;} body{font-family:sans-serif;} .day{margin-bottom:2em;}</style>",
        "<title>Challenge Plan</title>",
        "</head>",
        "<body>",
        "<h1>Daily Plan</h1>",
    ]
    if routing_failed:
        lines.append(
            '<p style="color: red; font-weight: bold;">'
            "NOTE: Some segments could not be routed successfully. Please check the logs for details. "
            "The following plan represents the successfully routed portions."
            "</p>"
        )

    for idx, day in enumerate(daily_plans, start=1):
        date_str = day["date"].isoformat()
        lines.append(f"<div class='day'><h2>Day {idx} - {date_str}</h2>")
        notes = day.get("notes")
        for part_idx, act in enumerate(day.get("activities", []), start=1):
            if act.get("type") != "activity":
                continue
            lines.append(f"<h3>Part {part_idx}: {act.get('start_name','Route')}</h3>")
            lines.append("<ul>")
            for e in act.get("route_edges", []):
                seg_name = e.name or str(e.seg_id)
                direction_note = f" ({e.direction})" if e.direction != "both" else ""
                lines.append(
                    f"<li><b>{seg_name}</b> – {e.length_mi:.1f} mi{direction_note}</li>"
                )
            lines.append("</ul>")

        if notes:
            lines.append(f"<p><em>{notes}</em></p>")
        rationale = day.get("rationale")
        if rationale:
            lines.append(f"<p><em>Rationale: {rationale}</em></p>")

        metrics = day.get("metrics")
        if metrics:
            lines.append("<ul>")
            lines.append(
                f"<li>Total Distance: {metrics['total_distance_mi']:.1f} mi</li>"
            )
            lines.append(f"<li>New Distance: {metrics['new_distance_mi']:.1f} mi</li>")
            lines.append(
                f"<li>Redundant Distance: {metrics['redundant_distance_mi']:.1f} mi ({metrics['redundant_distance_pct']:.0f}% )</li>"
            )
            if "redundant_distance_post_mi" in metrics:
                lines.append(
                    f"<li>Redundant miles (post-optimization): {metrics['redundant_distance_post_mi']:.1f} mi</li>"
                )
            lines.append(
                f"<li>Total Elevation Gain: {metrics['total_elev_gain_ft']:.0f} ft</li>"
            )
            lines.append(
                f"<li>Redundant Elevation Gain: {metrics['redundant_elev_gain_ft']:.0f} ft ({metrics['redundant_elev_pct']:.0f}% )</li>"
            )
            lines.append(f"<li>Drive Time: {metrics['drive_time_min']:.0f} min</li>")
            lines.append(f"<li>Run Time: {metrics['run_time_min']:.0f} min</li>")
            lines.append(f"<li>Total Time: {metrics['total_time_min']:.0f} min</li>")
            lines.append(
                f"<li>Number of Activities: {metrics.get('num_activities', 'N/A')}</li>"
            )
            lines.append(
                f"<li>Number of Drives: {metrics.get('num_drives', 'N/A')}</li>"
            )
            lines.append("</ul>")

        coords: List[Tuple[float, float]] = []
        for act in day.get("activities", []):
            if act.get("type") == "activity":
                coords.extend(
                    planner_utils.collect_route_coords(act.get("route_edges", []))
                )

        map_img = os.path.join(image_dir, f"map_day_{idx:02d}.png")
        planner_utils.plot_route_map(coords, map_img)
        rel_map = os.path.relpath(map_img, os.path.dirname(path))
        lines.append(f"<img src='{rel_map}' alt='Day {idx} map'>")

        if dem_path:
            elev_img = os.path.join(image_dir, f"elev_day_{idx:02d}.png")
            planner_utils.plot_elevation_profile(coords, dem_path, elev_img)
            rel_elev = os.path.relpath(elev_img, os.path.dirname(path))
            lines.append(f"<img src='{rel_elev}' alt='Day {idx} elevation'>")

        lines.append("</div>")

    totals = {
        "total_distance_mi": 0.0,
        "new_distance_mi": 0.0,
        "redundant_distance_mi": 0.0,
        "redundant_distance_post_mi": 0.0,
        "total_elev_gain_ft": 0.0,
        "redundant_elev_gain_ft": 0.0,
        "drive_time_min": 0.0,
        "run_time_min": 0.0,
        "total_time_min": 0.0,
    }
    for day in daily_plans:
        m = day.get("metrics")
        if not m:
            continue
        totals["total_distance_mi"] += m["total_distance_mi"]
        totals["new_distance_mi"] += m["new_distance_mi"]
        totals["redundant_distance_mi"] += m["redundant_distance_mi"]
        totals["redundant_distance_post_mi"] += m.get(
            "redundant_distance_post_mi", m["redundant_distance_mi"]
        )
        totals["total_elev_gain_ft"] += m["total_elev_gain_ft"]
        totals["redundant_elev_gain_ft"] += m["redundant_elev_gain_ft"]
        totals["drive_time_min"] += m["drive_time_min"]
        totals["run_time_min"] += m["run_time_min"]
        totals["total_time_min"] += m["total_time_min"]

    redundant_pct = (
        (totals["redundant_distance_mi"] / totals["total_distance_mi"]) * 100.0
        if totals["total_distance_mi"] > 0
        else 0.0
    )
    redundant_elev_pct = (
        (totals["redundant_elev_gain_ft"] / totals["total_elev_gain_ft"]) * 100.0
        if totals["total_elev_gain_ft"] > 0
        else 0.0
    )

    lines.append("<h2>Totals</h2>")
    if routing_failed:  # Add this line to indicate failure in HTML
        lines.append(
            "<h2 style='color:red;'>NOTE: ROUTING FAILED - THIS PLAN IS INCOMPLETE OR POTENTIALLY INCORRECT</h2>"
        )
    lines.append("<ul>")
    lines.append(f"<li>Total Distance: {totals['total_distance_mi']:.1f} mi</li>")
    lines.append(f"<li>New Distance: {totals['new_distance_mi']:.1f} mi</li>")
    lines.append(
        f"<li>Redundant Distance: {totals['redundant_distance_mi']:.1f} mi ({redundant_pct:.0f}% )</li>"
    )
    lines.append(
        f"<li>Redundant miles (post-optimization): {totals['redundant_distance_post_mi']:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total Elevation Gain: {totals['total_elev_gain_ft']:.0f} ft</li>"
    )
    lines.append(
        f"<li>Redundant Elevation Gain: {totals['redundant_elev_gain_ft']:.0f} ft ({redundant_elev_pct:.0f}% )</li>"
    )
    lines.append(f"<li>Drive Time: {totals['drive_time_min']:.0f} min</li>")
    lines.append(f"<li>Run Time: {totals['run_time_min']:.0f} min</li>")
    lines.append(f"<li>Total Time: {totals['total_time_min']:.0f} min</li>")
    lines.append("</ul>")

    lines.append("</body></html>")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def export_plan_files(
    daily_plans: List[Dict[str, object]],
    args: argparse.Namespace,
    *,
    csv_path: Optional[str] = None,
    write_gpx: bool = True,
    review: Optional[bool] = None,
    challenge_ids: Optional[Set[str]] = None,
    routing_failed: bool = False,
) -> None:
    """Write CSV and HTML outputs for ``daily_plans``.

    ``csv_path`` overrides ``args.output``. GPX files and plan review are
    skipped when ``write_gpx`` is ``False`` or ``review`` is ``False``.
    If ``routing_failed`` is True, output filenames are prefixed with "failed-".
    """

    orig_output_arg = args.output  # Store original args.output
    orig_review_arg = args.review  # Store original args.review

    # Determine effective base paths
    current_csv_path = csv_path if csv_path is not None else args.output
    current_gpx_dir = args.gpx_dir
    # Note: args.output and args.gpx_dir are not modified here yet,
    # allowing their original values to be restored if needed.

    if getattr(args, "debug", None):  # This uses args.debug, which is not prefixed
        open(args.debug, "w").close()

    # Apply "failed-" prefix if routing_failed is True
    if routing_failed:
        csv_dir_name, csv_base_name = os.path.split(current_csv_path)
        current_csv_path = os.path.join(csv_dir_name, f"failed-{csv_base_name}")

        gpx_dir_parent, gpx_folder_name = os.path.split(current_gpx_dir)
        if not gpx_folder_name:  # Handle cases like "gpx/" vs "gpx"
            gpx_folder_name = os.path.basename(gpx_dir_parent)
            gpx_dir_parent = os.path.dirname(gpx_dir_parent)
        current_gpx_dir = os.path.join(gpx_dir_parent, f"failed-{gpx_folder_name}")

    # HTML path is derived from the (potentially prefixed) CSV path
    current_html_path = os.path.splitext(current_csv_path)[0] + ".html"

    # Ensure the (potentially prefixed) GPX directory exists
    # This needs to be done *before* writing GPX files.
    if write_gpx:  # Only make dir if we intend to write GPX files
        os.makedirs(current_gpx_dir, exist_ok=True)

    # If review is explicitly passed, use that value. Otherwise, use args.review.
    # This is separate from path prefixing.
    effective_review_setting = review if review is not None else args.review

    summary_rows = []
    for day_plan in daily_plans:
        day_date_str = day_plan["date"].isoformat()
        gpx_part_counter = 1
        day_description_parts = []
        current_day_total_trail_distance = 0.0
        current_day_total_trail_gain = 0.0
        current_day_unique_trail_distance = 0.0
        current_day_unique_trail_gain = 0.0
        seen_segment_ids: Set[str] = set()
        num_activities_this_day = 0
        num_drives_this_day = 0
        start_names_for_day: List[str] = []

        activities_for_this_day_in_plan = day_plan.get("activities", [])

        for activity_or_drive in activities_for_this_day_in_plan:
            if activity_or_drive["type"] == "activity":
                num_activities_this_day += 1
                route = activity_or_drive["route_edges"]
                activity_name = activity_or_drive["name"]

                dist = sum(e.length_mi for e in route)
                gain = sum(e.elev_gain_ft for e in route)
                est_activity_time = total_time(
                    route, args.pace, args.grade, args.road_pace
                )

                current_day_total_trail_distance += dist
                current_day_total_trail_gain += gain
                for e in route:
                    if (
                        e.kind == "trail"
                        and e.seg_id is not None
                        and (challenge_ids is None or str(e.seg_id) in challenge_ids)
                        and e.seg_id not in seen_segment_ids
                    ):
                        current_day_unique_trail_distance += e.length_mi
                        current_day_unique_trail_gain += e.elev_gain_ft
                        seen_segment_ids.add(e.seg_id)

                if write_gpx:
                    gpx_file_name = f"{day_plan['date'].strftime('%Y%m%d')}_part{gpx_part_counter}.gpx"
                    # Use current_gpx_dir for the path
                    gpx_path = os.path.join(current_gpx_dir, gpx_file_name)
                    planner_utils.write_gpx(
                        gpx_path,
                        route,
                        mark_road_transitions=args.mark_road_transitions,
                        start_name=activity_or_drive.get("start_name"),
                    )
                if activity_or_drive.get("start_name"):
                    start_names_for_day.append(activity_or_drive.get("start_name"))

                ids = []
                for e in route:
                    if (
                        e.kind != "trail"
                        or not e.seg_id
                        or (
                            challenge_ids is not None
                            and str(e.seg_id) not in challenge_ids
                        )
                    ):
                        continue
                    sid = str(e.seg_id)
                    if e.direction != "both":
                        arrow = "↑" if e.direction == "ascent" else f"({e.direction})"
                        sid += arrow
                    if sid not in ids:
                        ids.append(sid)
                trail_segment_ids_in_route = sorted(ids)
                part_desc = f"{activity_name} (Segs: {', '.join(trail_segment_ids_in_route)}; {dist:.2f}mi; {gain:.0f}ft; {est_activity_time:.1f}min)"
                day_description_parts.append(part_desc)
                gpx_part_counter += 1

            elif activity_or_drive["type"] == "drive":
                num_drives_this_day += 1
                drive_minutes = activity_or_drive["minutes"]
                day_description_parts.append(f"Drive ({drive_minutes:.1f} min)")

        if activities_for_this_day_in_plan:
            route_desc = build_route_description(
                activities_for_this_day_in_plan,
                current_day_total_trail_distance,
                day_plan["total_activity_time"],
                day_plan["total_drive_time"],
                current_day_total_trail_gain,
                challenge_ids=challenge_ids,
            )
            notes_final = day_plan.get("notes", "")
            idx = daily_plans.index(day_plan)
            if idx > 0:
                prev_time = daily_plans[idx - 1]["total_activity_time"]
                cur_time = day_plan["total_activity_time"]
                if prev_time > cur_time * 1.5:
                    extra = "easier day to recover after yesterday’s long run"
                    notes_final = f"{notes_final}; {extra}" if notes_final else extra
                elif cur_time > prev_time * 1.5:
                    extra = "big effort after easier day"
                    notes_final = f"{notes_final}; {extra}" if notes_final else extra
            day_plan["notes"] = notes_final

            start_set = {
                a.get("start_name")
                for a in activities_for_this_day_in_plan
                if a.get("type") == "activity" and a.get("start_name")
            }
            rationale_parts: List[str] = []
            if len(start_set) == 1:
                only = next(iter(start_set))
                rationale_parts.append(
                    f"Trails grouped around {only} to minimize driving."
                )
            elif len(start_set) > 1:
                rationale_parts.append(
                    "Trails from nearby areas combined for efficiency."
                )
            if num_drives_this_day:
                rationale_parts.append("Includes drive transfers between trail groups.")
            if idx > 0:
                prev_time = daily_plans[idx - 1]["total_activity_time"]
                cur_time = day_plan["total_activity_time"]
                if prev_time > cur_time * 1.5:
                    rationale_parts.append("Shorter day planned for recovery.")
                elif cur_time > prev_time * 1.5:
                    rationale_parts.append(
                        "Longer effort scheduled after an easier day."
                    )
            if not rationale_parts:
                rationale_parts.append(
                    "Routes selected based on proximity and time budget."
                )
            day_plan["rationale"] = " ".join(rationale_parts)

            redundant_miles = (
                current_day_total_trail_distance - current_day_unique_trail_distance
            )
            redundant_pct = (
                (redundant_miles / current_day_total_trail_distance) * 100.0
                if current_day_total_trail_distance > 0
                else 0.0
            )
            redundant_elev = (
                current_day_total_trail_gain - current_day_unique_trail_gain
            )
            redundant_elev_pct = (
                (redundant_elev / current_day_total_trail_gain) * 100.0
                if current_day_total_trail_gain > 0
                else 0.0
            )

            summary_rows.append(
                {
                    "date": day_date_str,
                    "plan_description": " >> ".join(day_description_parts),
                    "route_description": route_desc,
                    "total_trail_distance_mi": round(
                        current_day_total_trail_distance, 2
                    ),
                    "unique_trail_miles": round(current_day_unique_trail_distance, 2),
                    "redundant_miles": round(redundant_miles, 2),
                    "redundant_pct": round(redundant_pct, 1),
                    "total_trail_elev_gain_ft": round(current_day_total_trail_gain, 0),
                    "unique_trail_elev_gain_ft": round(
                        current_day_unique_trail_gain, 0
                    ),
                    "redundant_elev_gain_ft": round(redundant_elev, 0),
                    "redundant_elev_pct": round(redundant_elev_pct, 1),
                    "total_activity_time_min": round(
                        day_plan["total_activity_time"], 1
                    ),
                    "total_drive_time_min": round(day_plan["total_drive_time"], 1),
                    "total_time_min": round(
                        day_plan["total_activity_time"] + day_plan["total_drive_time"],
                        1,
                    ),
                    "num_activities": num_activities_this_day,
                    "num_drives": num_drives_this_day,
                    "notes": notes_final,
                    "start_trailheads": "; ".join(start_names_for_day),
                }
            )
            day_plan["metrics"] = {
                "total_distance_mi": round(current_day_total_trail_distance, 2),
                "new_distance_mi": round(current_day_unique_trail_distance, 2),
                "redundant_distance_mi": round(redundant_miles, 2),
                "redundant_distance_post_mi": round(redundant_miles, 2),
                "redundant_distance_pct": round(redundant_pct, 1),
                "total_elev_gain_ft": round(current_day_total_trail_gain, 0),
                "redundant_elev_gain_ft": round(redundant_elev, 0),
                "redundant_elev_pct": round(redundant_elev_pct, 1),
                "drive_time_min": round(day_plan["total_drive_time"], 1),
                "run_time_min": round(day_plan["total_activity_time"], 1),
                "total_time_min": round(
                    day_plan["total_activity_time"] + day_plan["total_drive_time"], 1
                ),
                "num_activities": num_activities_this_day,
                "num_drives": num_drives_this_day,
            }
            debug_log(args, f"{day_date_str}: {day_plan['rationale']}")
        else:
            day_plan["rationale"] = ""
            summary_rows.append(
                {
                    "date": day_date_str,
                    "plan_description": "Unable to complete",
                    "route_description": "Unable to complete",
                    "total_trail_distance_mi": 0.0,
                    "unique_trail_miles": 0.0,
                    "redundant_miles": 0.0,
                    "redundant_pct": 0.0,
                    "total_trail_elev_gain_ft": 0.0,
                    "unique_trail_elev_gain_ft": 0.0,
                    "redundant_elev_gain_ft": 0.0,
                    "redundant_elev_pct": 0.0,
                    "total_activity_time_min": 0.0,
                    "total_drive_time_min": 0.0,
                    "total_time_min": 0.0,
                    "num_activities": 0,
                    "num_drives": 0,
                    "notes": day_plan.get("notes", ""),
                    "start_trailheads": "",
                }
            )
            day_plan["metrics"] = {
                "total_distance_mi": 0.0,
                "new_distance_mi": 0.0,
                "redundant_distance_mi": 0.0,
                "redundant_distance_pct": 0.0,
                "total_elev_gain_ft": 0.0,
                "redundant_elev_gain_ft": 0.0,
                "redundant_elev_pct": 0.0,
                "drive_time_min": 0.0,
                "run_time_min": 0.0,
                "total_time_min": 0.0,
                "num_activities": 0,
                "num_drives": 0,
            }
            debug_log(args, f"{day_date_str}: {day_plan['rationale']}")

    if summary_rows:
        # Calculate plan-wide unique metrics
        plan_wide_seen_challenge_segment_ids: Set[str] = set()
        plan_wide_unique_trail_miles = 0.0
        plan_wide_unique_trail_gain_ft = 0.0
        if (
            daily_plans and challenge_ids
        ):  # challenge_ids is current_challenge_segment_ids
            for day_plan_item in daily_plans:
                for activity_item in day_plan_item.get("activities", []):
                    if activity_item.get("type") == "activity":
                        for e_item in activity_item.get("route_edges", []):
                            if (
                                e_item.kind == "trail"
                                and e_item.seg_id is not None
                                and str(e_item.seg_id) in challenge_ids
                                and str(e_item.seg_id)
                                not in plan_wide_seen_challenge_segment_ids
                            ):
                                plan_wide_unique_trail_miles += e_item.length_mi
                                plan_wide_unique_trail_gain_ft += e_item.elev_gain_ft
                                plan_wide_seen_challenge_segment_ids.add(
                                    str(e_item.seg_id)
                                )

        totals = {
            "total_trail_distance_mi": 0.0,
            "unique_trail_miles": 0.0,  # This will be updated by plan_wide_unique_trail_miles later
            "redundant_miles": 0.0,
            "total_trail_elev_gain_ft": 0.0,
            "unique_trail_elev_gain_ft": 0.0,  # This will be updated by plan_wide_unique_trail_gain_ft later
            "redundant_elev_gain_ft": 0.0,
            "total_activity_time_min": 0.0,
            "total_drive_time_min": 0.0,
            "total_time_min": 0.0,
        }
        for row in summary_rows:
            if row.get("plan_description") == "Unable to complete":
                continue
            totals["total_trail_distance_mi"] += row["total_trail_distance_mi"]
            # totals["unique_trail_miles"] += row["unique_trail_miles"] # Old way, summing daily uniques
            totals["redundant_miles"] += row[
                "redundant_miles"
            ]  # This will be recalculated
            totals["total_trail_elev_gain_ft"] += row["total_trail_elev_gain_ft"]
            # totals["unique_trail_elev_gain_ft"] += row["unique_trail_elev_gain_ft"] # Old way
            totals["redundant_elev_gain_ft"] += row[
                "redundant_elev_gain_ft"
            ]  # This will be recalculated
            totals["total_activity_time_min"] += row["total_activity_time_min"]
            totals["total_drive_time_min"] += row["total_drive_time_min"]
            totals["total_time_min"] += row["total_time_min"]

        # Use plan-wide unique values for totals
        totals["unique_trail_miles"] = plan_wide_unique_trail_miles
        totals["unique_trail_elev_gain_ft"] = plan_wide_unique_trail_gain_ft

        # Recalculate redundant miles and elevation gain for totals
        totals_redundant_miles = (
            totals["total_trail_distance_mi"] - totals["unique_trail_miles"]
        )
        totals_redundant_elev_gain = (
            totals["total_trail_elev_gain_ft"] - totals["unique_trail_elev_gain_ft"]
        )

        totals["redundant_miles"] = totals_redundant_miles
        totals["redundant_elev_gain_ft"] = totals_redundant_elev_gain

        total_pct = (
            (totals["redundant_miles"] / totals["total_trail_distance_mi"]) * 100.0
            if totals["total_trail_distance_mi"] > 0
            else 0.0
        )
        total_elev_pct = (
            (totals["redundant_elev_gain_ft"] / totals["total_trail_elev_gain_ft"])
            * 100.0
            if totals["total_trail_elev_gain_ft"] > 0
            else 0.0
        )

        summary_rows.append(
            {
                "date": "Totals",
                "plan_description": "",
                "route_description": "",
                "total_trail_distance_mi": round(totals["total_trail_distance_mi"], 2),
                "unique_trail_miles": round(
                    totals["unique_trail_miles"], 2
                ),  # Now uses plan-wide
                "redundant_miles": round(
                    totals["redundant_miles"], 2
                ),  # Now recalculated
                "redundant_pct": round(total_pct, 1),
                "total_trail_elev_gain_ft": round(
                    totals["total_trail_elev_gain_ft"], 0
                ),
                "unique_trail_elev_gain_ft": round(
                    totals["unique_trail_elev_gain_ft"], 0  # Now uses plan-wide
                ),
                "redundant_elev_gain_ft": round(
                    totals["redundant_elev_gain_ft"], 0
                ),  # Now recalculated
                "redundant_elev_pct": round(total_elev_pct, 1),
                "total_activity_time_min": round(totals["total_activity_time_min"], 1),
                "total_drive_time_min": round(totals["total_drive_time_min"], 1),
                "total_time_min": round(totals["total_time_min"], 1),
                "num_activities": "",
                "num_drives": "",
                "notes": "",
                "start_trailheads": "",
            }
        )

    fieldnames = list(summary_rows[0].keys()) if summary_rows else default_fieldnames
    if summary_rows:
        debug_log(
            args,  # This debug_log uses args, which might be an issue if args.output was expected to be prefixed for logging.
            # However, the core task is about output file paths, not modifying args for logging.
            f"export_plan_files: Calculated plan_wide_unique_trail_miles: {plan_wide_unique_trail_miles:.2f} mi from {len(plan_wide_seen_challenge_segment_ids)} unique segment IDs for the CSV Totals row. Outputting to {current_csv_path}",
        )
    else:
        debug_log(
            args,
            f"export_plan_files: No summary rows to process. Outputting to {current_csv_path}",
        )

    # Use current_csv_path for writing
    with open(current_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Use effective_review_setting for conditional review
    if effective_review_setting and summary_rows:
        plan_text = "\n".join(
            f"{r['date']}: {r['plan_description']}" for r in summary_rows
        )
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            plan_review.review_plan(plan_text, run_id)
            print(f"Review saved to reviews/{run_id}.jsonl")
        except Exception as e:
            print(f"Review failed: {e}")

    if write_gpx and daily_plans and any(dp.get("activities") for dp in daily_plans):
        colors = [
            "Red",
            "Blue",
            "Green",
            "Magenta",
            "Cyan",
            "Orange",
            "Purple",
            "Brown",
        ]
        # Use current_gpx_dir for the full timespan GPX path
        full_gpx_path = os.path.join(current_gpx_dir, "full_timespan.gpx")
        planner_utils.write_multiday_gpx(
            full_gpx_path,
            daily_plans,
            mark_road_transitions=args.mark_road_transitions,  # This uses args.mark_road_transitions
            colors=colors,
        )

    # current_html_path is already determined based on current_csv_path
    # img_dir is derived from current_html_path, so it will also be in the "failed-" prefixed structure if applicable
    img_dir = os.path.join(os.path.dirname(current_html_path), "plan_images")
    # Ensure the image directory exists, especially if it's a "failed-" prefixed one
    os.makedirs(img_dir, exist_ok=True)

    write_plan_html(
        current_html_path,  # Use current_html_path
        daily_plans,
        img_dir,  # This img_dir is now correctly based on current_html_path
        dem_path=args.dem,  # This uses args.dem
        routing_failed=routing_failed,  # Pass the flag here
    )
    print(f"HTML plan written to {current_html_path}")

    print(f"Challenge plan written to {current_csv_path}")
    if write_gpx:
        if not daily_plans or not any(dp.get("activities") for dp in daily_plans):
            gpx_files_present = False
            # Check the potentially prefixed GPX directory
            if os.path.exists(current_gpx_dir):
                gpx_files_present = any(
                    f.endswith(".gpx") for f in os.listdir(current_gpx_dir)
                )

            if not gpx_files_present:
                print(
                    f"No GPX files generated as no activities were planned (checked: {current_gpx_dir})."
                )
            else:  # Should ideally not happen if no activities planned, unless old files exist
                print(
                    f"GPX files may exist in {current_gpx_dir} from previous runs or other issues, but no new activities were planned in this run."
                )
        else:
            print(f"GPX files written to {current_gpx_dir}")

    # Restore original args values that might have been temporarily changed for csv_path logic
    # The first hunk already changed orig_output to orig_output_arg and orig_review to orig_review_arg.
    args.output = orig_output_arg
    args.review = orig_review_arg


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

    parser.add_argument(
        "--config", default=config_path, help="Path to config YAML or JSON file"
    )
    parser.add_argument(
        "--start-date",
        required="start_date" not in config_defaults,
        help="Challenge start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        required="end_date" not in config_defaults,
        help="Challenge end date YYYY-MM-DD",
    )
    parser.add_argument(
        "--time",
        default=config_defaults.get("time", "4h"),
        help="Default daily time budget when --daily-hours-file is not provided",
    )
    parser.add_argument(
        "--daily-hours-file",
        default=config_defaults.get("daily_hours_file"),
        help="JSON mapping YYYY-MM-DD dates to available hours for that day",
    )
    parser.add_argument(
        "--pace",
        required="pace" not in config_defaults,
        type=float,
        help="Base running pace (min/mi)",
    )
    parser.add_argument(
        "--grade",
        type=float,
        default=config_defaults.get("grade", 0.0),
        help="Seconds added per 100ft climb",
    )
    parser.add_argument(
        "--segments",
        default=config_defaults.get("segments", "data/traildata/trail.json"),
        help="Trail segment JSON file",
    )
    parser.add_argument(
        "--dem",
        help="Optional DEM GeoTIFF from clip_srtm.py for segment elevation",
    )
    parser.add_argument("--roads", help="Optional road connector GeoJSON or .pbf")
    parser.add_argument("--trailheads", help="Optional trailhead JSON or CSV file")
    parser.add_argument(
        "--home-lat",
        type=float,
        default=config_defaults.get("home_lat", 43.635278),
        help="Home latitude for drive time estimates",
    )
    parser.add_argument(
        "--home-lon",
        type=float,
        default=config_defaults.get("home_lon", -116.205),
        help="Home longitude for drive time estimates",
    )
    parser.add_argument(
        "--max-foot-road",
        "--max-road",
        dest="max_foot_road",
        type=float,
        default=config_defaults.get(
            "max_foot_road", config_defaults.get("max_road", 3.0)
        ),
        help="Maximum road distance allowed while walking (mi)",
    )
    parser.add_argument(
        "--road-threshold",
        type=float,
        default=config_defaults.get("road_threshold", 0.25),
        help="Fractional speed advantage required to choose a road connector",
    )
    parser.add_argument(
        "--spur-length-thresh",
        type=float,
        default=config_defaults.get("spur_length_thresh", 0.3),
        help="Trail length in miles below which spur detours are considered",
    )
    parser.add_argument(
        "--spur-road-bonus",
        type=float,
        default=config_defaults.get("spur_road_bonus", 0.25),
        help="Additional road miles allowed when exiting a short spur",
    )
    parser.add_argument(
        "--road-pace",
        type=float,
        default=config_defaults.get("road_pace", 12.0),
        help="Pace on roads (min/mi)",
    )
    parser.add_argument(
        "--perf",
        default=config_defaults.get("perf", "data/segment_perf.csv"),
        help="CSV of previous segment completions",
    )
    parser.add_argument("--year", type=int, help="Filter completions to this year")
    parser.add_argument(
        "--remaining", help="Comma-separated list or file of segments to include"
    )
    parser.add_argument(
        "--output",
        default=config_defaults.get("output", "challenge_plan.csv"),
        help="Output CSV summary file",
    )
    parser.add_argument(
        "--gpx-dir",
        default=config_defaults.get("gpx_dir", "gpx"),
        help="Directory for GPX output",
    )
    parser.add_argument(
        "--output-dir",
        default=config_defaults.get("output_dir"),
        help="Directory to store all outputs for this run",
    )
    parser.add_argument(
        "--auto-output-dir",
        action="store_true",
        default=config_defaults.get("auto_output_dir", False),
        help="Create a timestamped directory under 'outputs/' when --output-dir is not provided",
    )
    parser.add_argument(
        "--no-mark-road-transitions",
        dest="mark_road_transitions",
        action="store_false",
        help="Do not annotate GPX files with waypoints and track extensions for road sections",
    )
    parser.set_defaults(mark_road_transitions=True)
    parser.add_argument(
        "--average-driving-speed-mph",
        type=float,
        default=config_defaults.get("average_driving_speed_mph", 30.0),
        help="Average driving speed in mph for estimating travel time between activity clusters",
    )
    parser.add_argument(
        "--max-drive-minutes-per-transfer",
        type=float,
        default=config_defaults.get("max_drive_minutes_per_transfer", 30.0),
        help="Maximum allowed driving time between clusters on the same day",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        default=config_defaults.get("review", False),
        help="Send final plan for AI review",
    )
    parser.add_argument(
        "--debug",
        metavar="PATH",
        help="Write per-day route rationale to this file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug log messages to the console",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=config_defaults.get("redundancy_threshold", 0.2),
        help="Maximum acceptable redundant distance ratio",
    )
    parser.add_argument(
        "--no-connector-trails",
        dest="allow_connector_trails",
        action="store_false",
        help="Disallow using non-challenge trail connectors",
    )
    parser.set_defaults(
        allow_connector_trails=config_defaults.get("allow_connector_trails", True)
    )
    parser.add_argument(
        "--rpp-timeout",
        type=float,
        default=config_defaults.get("rpp_timeout", 5.0),
        help="Time limit in seconds for RPP solver",
    )
    parser.add_argument(
        "--advanced-optimizer",
        dest="use_advanced_optimizer",
        action="store_true",
        default=config_defaults.get("use_advanced_optimizer", False),
        help="Use experimental multi-objective route optimizer",
    )
    parser.add_argument(
        "--optimizer",
        choices=["greedy2opt", "postman"],
        default=config_defaults.get("optimizer", "greedy2opt"),
        help="Routing optimizer to use",
    )
    parser.add_argument(
        "--postman-timeout",
        type=float,
        default=config_defaults.get("postman_timeout", 30.0),
        help="Time limit in seconds for the postman optimizer",
    )
    parser.add_argument(
        "--postman-max-odd",
        type=int,
        default=config_defaults.get("postman_max_odd", 40),
        help="Abort postman if more than this many odd nodes",
    )
    parser.add_argument(
        "--draft-every",
        type=int,
        metavar="N",
        default=0,
        help="Write draft CSV and HTML files every N days",
    )
    parser.add_argument(
        "--strict-max-foot-road",
        action="store_true",
        default=config_defaults.get("strict_max_foot_road", False),
        help="Disallow walking connectors that exceed --max-foot-road",
    )
    parser.add_argument(
        "--first-day-seg",
        dest="first_day_segment",
        help="Segment ID to start the first day with",
        default=config_defaults.get("first_day_segment"),
    )
    parser.add_argument(
        "--force-recompute-apsp",
        action="store_true",
        default=False,
        help="Force re-computation of All-Pairs Shortest Paths (APSP) cache",
    )

    args = parser.parse_args(argv)
    overall_routing_status_ok = True  # Initialize routing status

    if getattr(args, "debug", None):
        debug_log_path = getattr(args, "debug")
        if os.path.exists(debug_log_path):
            try:
                open(debug_log_path, "w").close()
                print(
                    f"Debug log '{debug_log_path}' cleared at start of run.",
                    file=sys.stderr,
                )
            except IOError as e:
                print(
                    f"Warning: Could not clear debug log '{debug_log_path}': {e}",
                    file=sys.stderr,
                )
        else:
            debug_log_dir = os.path.dirname(debug_log_path)
            if debug_log_dir and not os.path.exists(debug_log_dir):
                os.makedirs(debug_log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if "--time" in argv and "--daily-hours-file" not in argv:
        args.daily_hours_file = None

    output_dir = args.output_dir
    if output_dir is None and args.auto_output_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", f"plan_{ts}")
        args.output_dir = output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, os.path.basename(args.output))
        args.gpx_dir = os.path.join(output_dir, os.path.basename(args.gpx_dir))

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

    default_daily_minutes = 240.0 if daily_hours_file else budget
    for i in range(num_days):
        day = start_date + datetime.timedelta(days=i)
        hours = user_hours.get(day, default_daily_minutes / 60.0)
        daily_budget_minutes[day] = hours * 60.0
    all_trail_segments = planner_utils.load_segments(args.segments)
    if args.dem:
        planner_utils.add_elevation_from_dem(all_trail_segments, args.dem)
    access_coord_lookup: Dict[str, Tuple[float, float]] = {}
    tmp_access: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for e in all_trail_segments:
        if e.access_from:
            tmp_access[e.access_from].extend([e.start, e.end])
    for name, nodes in tmp_access.items():
        if not nodes:
            continue
        counts: Dict[Tuple[float, float], int] = defaultdict(int)
        for n in nodes:
            counts[n] += 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        access_coord_lookup[name] = best
    all_road_segments: List[Edge] = []
    if args.roads:
        bbox = None
        if args.roads.lower().endswith(".pbf"):
            bbox = planner_utils.bounding_box_from_edges(all_trail_segments)
        all_road_segments = planner_utils.load_roads(args.roads, bbox=bbox)
    road_graph_for_drive = planner_utils.build_road_graph(all_road_segments)
    road_node_set: Set[Tuple[float, float]] = {e.start for e in all_road_segments} | {
        e.end for e in all_road_segments
    }

    trailhead_lookup: Dict[Tuple[float, float], str] = {}
    if args.trailheads and os.path.exists(args.trailheads):
        trailhead_lookup = planner_utils.load_trailheads(args.trailheads)

    # This graph is used for on-foot routing *within* macro-clusters
    on_foot_routing_graph_edges = all_trail_segments + all_road_segments
    G = build_nx_graph(
        on_foot_routing_graph_edges, args.pace, args.grade, args.road_pace
    )

    cache_key = f"{args.pace}:{args.grade}:{args.road_pace}"
    path_cache: dict | None = cache_utils.load_cache("dist_cache", cache_key) or {}
    logger.info("Loaded path cache with %d start nodes", len(path_cache))

    # APSP pre-computation
    num_nodes_in_g = G.number_of_nodes()
    # Ensure path_cache is not None before checking its length or force recompute
    if path_cache is None: # Should not happen if cache_utils.load_cache returns {} on miss
        path_cache = {}

    needs_apsp_recompute = (
        args.force_recompute_apsp
        or not path_cache
        or (num_nodes_in_g > 0 and len(path_cache) < 0.1 * num_nodes_in_g)
    )

    if needs_apsp_recompute:
        if args.force_recompute_apsp:
            tqdm.write("Forcing APSP pre-computation due to --force-recompute-apsp flag.")
        elif not path_cache:
            tqdm.write("APSP cache is empty, starting pre-computation.")
        else:
            tqdm.write(
                f"APSP cache has {len(path_cache)} entries, less than 10% of graph nodes ({num_nodes_in_g}). Starting pre-computation."
            )

        path_cache.clear() # Clear existing entries if we are recomputing

        # Using all_pairs_dijkstra instead of all_pairs_dijkstra_path to get both distances and paths
        # The path_cache is expected to store paths, so we'll use the path part of the result.
        # nx.all_pairs_dijkstra returns (distances, paths)
        # We are interested in paths.
        # The prompt specified all_pairs_dijkstra_path, which returns an iterator of (source, {target: path})

        apsp_iterator = nx.all_pairs_dijkstra_path(G, weight="weight")

        # Wrap the iterator with tqdm for progress.
        # The iterator yields (source_node, dictionary_of_paths_from_source)
        # The total number of iterations for tqdm will be the number of nodes in G.
        for source, targets in tqdm(
            apsp_iterator,
            total=num_nodes_in_g,
            desc="Pre-calculating APSP",
            unit="node",
        ):
            path_cache[source] = targets

        tqdm.write(f"APSP pre-computation complete. Cache populated with {len(path_cache)} entries.")
        # The path_cache is updated in place and will be saved later.

    tracking = planner_utils.load_segment_tracking(
        os.path.join("config", "segment_tracking.json"), args.segments
    )
    completed_segment_ids = {sid for sid, done in tracking.items() if done}
    completed_segment_ids |= planner_utils.load_completed(args.perf, args.year or 0)

    current_challenge_segment_ids = None
    if args.remaining:
        current_challenge_segment_ids = set(parse_remaining(args.remaining))
    if current_challenge_segment_ids is None:
        current_challenge_segment_ids = {
            str(e.seg_id) for e in all_trail_segments
        } - completed_segment_ids

    current_challenge_segments = [
        e for e in all_trail_segments if str(e.seg_id) in current_challenge_segment_ids
    ]

    debug_log(
        args,
        f"Initial current_challenge_segments: {len(current_challenge_segments)} segments, Total mileage: {sum(e.length_mi for e in current_challenge_segments):.2f} mi",
    )

    # nodes list might be useful later for starting points, keep it around
    # nodes = list({e.start for e in on_foot_routing_graph_edges} | {e.end for e in on_foot_routing_graph_edges})

    potential_macro_clusters = identify_macro_clusters(
        current_challenge_segments,  # Only uncompleted trail segments
        all_road_segments,  # All road segments
        args.pace,
        args.grade,
        args.road_pace,
    )

    # Further split any macro-clusters that appear too large for a single day's
    # budget.  The "cluster_segments" helper uses a spatial KMeans followed by
    # a greedy time-based split which keeps each resulting cluster under the
    # provided budget whenever possible.
    expanded_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    expansion_iter = tqdm(
        potential_macro_clusters,
        desc="Expanding clusters",
        unit="cluster",
    )
    for cluster_edges, cluster_nodes in expansion_iter:
        if not cluster_edges:
            continue
        naive_time = total_time(cluster_edges, args.pace, args.grade, args.road_pace)
        oversized_threshold = 1.5 * budget
        if naive_time > oversized_threshold:
            debug_log(
                args, f"splitting large cluster of {naive_time:.1f} min into parts"
            )
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

    debug_log(
        args, f"Deduplicating clusters: Initial count {len(unplanned_macro_clusters)}"
    )
    unique_unplanned_macro_clusters_map = {}
    temp_clusters_for_deduplication = list(
        unplanned_macro_clusters
    )  # Create a copy to iterate over if needed, though direct iteration should be fine.

    for (
        cluster_tuple
    ) in (
        temp_clusters_for_deduplication
    ):  # Assuming items are (cluster_segs, cluster_nodes)
        cluster_segs_item, cluster_nodes_item = cluster_tuple  # Unpack the tuple

        if (
            not cluster_segs_item
        ):  # Should be filtered by `if mc[0]` but good for safety
            continue

        # Create a unique signature for the cluster based on segment IDs
        # Ensuring seg_id is string and handling None by excluding them from signature,
        # or by using a placeholder if None IDs are significant and distinguishable by other means.
        # For now, non-None segment IDs define the signature primarily.
        cluster_sig_list = sorted(
            [str(e.seg_id) for e in cluster_segs_item if e.seg_id is not None]
        )

        # To make signature more robust if all seg_ids are None, or if different clusters might have same set of None seg_ids
        # we can add count of segments, or hash of coordinates if seg_ids are not reliable.
        # For now, focusing on seg_ids. If cluster_sig_list is empty, all seg_ids were None.
        # Add total number of segments to distinguish clusters if all seg_ids are None or empty list.
        cluster_sig = tuple(cluster_sig_list + [f"count:{len(cluster_segs_item)}"])

        if cluster_sig not in unique_unplanned_macro_clusters_map:
            unique_unplanned_macro_clusters_map[cluster_sig] = (
                cluster_segs_item,
                cluster_nodes_item,
            )

    deduplicated_unplanned_macro_clusters = list(
        unique_unplanned_macro_clusters_map.values()
    )
    debug_log(
        args,
        f"Deduplicating clusters: Final count {len(deduplicated_unplanned_macro_clusters)}",
    )

    # Ensure each cluster can be routed; if not, break it into simpler pieces
    processed_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    for idx, (cluster_segs, cluster_nodes) in enumerate(tqdm(
        deduplicated_unplanned_macro_clusters, desc="Initial cluster processing"
    )):
        debug_log(args, f"MainLoop: Start processing cluster {idx}. Segments: {len(cluster_segs)}. First segment ID: {cluster_segs[0].seg_id if cluster_segs else 'N/A'}")
        cluster_centroid = (
            sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
            sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
        )
        node_tree_tmp = build_kdtree(list(cluster_nodes))
        start_node = nearest_node(node_tree_tmp, cluster_centroid)
        initial_route = plan_route(
            G,
            cluster_segs,
            start_node,
            args.pace,
            args.grade,
            args.road_pace,
            args.max_foot_road,
            args.road_threshold,
            path_cache,
            use_rpp=True,
            allow_connectors=args.allow_connector_trails,
            rpp_timeout=args.rpp_timeout,
            debug_args=args,
            spur_length_thresh=args.spur_length_thresh,
            spur_road_bonus=args.spur_road_bonus,
            use_advanced_optimizer=args.use_advanced_optimizer,
            strict_max_foot_road=args.strict_max_foot_road,
            redundancy_threshold=args.redundancy_threshold,
            optimizer_name=args.optimizer,
            postman_timeout=args.postman_timeout,
            postman_max_odd=args.postman_max_odd,
        )
        debug_log(args, f"MainLoop: Finished processing cluster {idx}. Route found: {bool(initial_route)}. Route length: {len(initial_route) if initial_route else 0}")
        if initial_route:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue

        if len(cluster_segs) == 1:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue

        connectivity_subs = split_cluster_by_connectivity(
            cluster_segs,
            G,
            args.max_foot_road,
            debug_args=args,
        )

        if len(connectivity_subs) > 1:
            debug_log(
                args,
                f"split cluster into {len(connectivity_subs)} parts due to connectivity",
            )
            if args.max_foot_road <= 0.01:
                debug_log(
                    args,
                    "Connectivity split with max_foot_road too small; segments will remain unscheduled",
                )
                # Skip adding these segments so they appear as unscheduled later
                continue
            for sub in connectivity_subs:
                nodes = {pt for e in sub for pt in (e.start, e.end)}
                processed_clusters.append((sub, nodes))
            continue

        extended_route = plan_route(
            G,
            cluster_segs,
            start_node,
            args.pace,
            args.grade,
            args.road_pace,
            args.max_foot_road * 3,
            args.road_threshold,
            path_cache,
            use_rpp=True,
            allow_connectors=args.allow_connector_trails,
            rpp_timeout=args.rpp_timeout,
            debug_args=args,
            spur_length_thresh=args.spur_length_thresh,
            spur_road_bonus=args.spur_road_bonus,
            use_advanced_optimizer=args.use_advanced_optimizer,
            strict_max_foot_road=args.strict_max_foot_road,
            redundancy_threshold=args.redundancy_threshold,
            optimizer_name=args.optimizer,
            postman_timeout=args.postman_timeout,
            postman_max_odd=args.postman_max_odd,
        )
        if extended_route:
            debug_log(args, "extended route successful")
            processed_clusters.append((cluster_segs, cluster_nodes))
        else:
            debug_log(args, "extended route failed; splitting segments")
            for seg in cluster_segs:
                processed_clusters.append(([seg], {seg.start, seg.end}))

    unplanned_macro_clusters: List[ClusterInfo] = []
    for cluster_segs, cluster_nodes in processed_clusters:
        start_candidates: List[Tuple[Tuple[float, float], Optional[str]]] = []
        access_counts: Dict[str, int] = defaultdict(int)
        for seg in cluster_segs:
            if seg.access_from:
                access_counts[seg.access_from] += 1
        if access_counts:
            for name, _ in sorted(access_counts.items(), key=lambda kv: -kv[1]):
                if name in access_coord_lookup:
                    start_candidates.append((access_coord_lookup[name], name))
        for n in cluster_nodes:
            if n in trailhead_lookup:
                start_candidates.append((n, trailhead_lookup[n]))
        if not start_candidates:
            best_node = None
            best_dist = float("inf")
            road_tree = build_kdtree(list(road_node_set)) if road_node_set else None
            for seg in cluster_segs:
                for cand in (seg.start, seg.end):
                    if road_tree:
                        nearest_road = nearest_node(road_tree, cand)
                        dist = planner_utils._haversine_mi(nearest_road, cand)
                    else:
                        dist = float("inf")
                    if dist < best_dist:
                        best_dist = dist
                        best_node = cand
            if best_node is not None:
                start_candidates.append((best_node, None))
        if not start_candidates:
            centroid = (
                sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
                sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
            )
            tree_tmp = build_kdtree(list(cluster_nodes))
            start_candidates.append((nearest_node(tree_tmp, centroid), None))
        unplanned_macro_clusters.append(
            ClusterInfo(cluster_segs, cluster_nodes, start_candidates)
        )

    all_on_foot_nodes = list(G.nodes())  # Get all nodes from the on-foot routing graph

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

        # Optionally force the first day's initial cluster
        if day_idx == 0 and args.first_day_segment:
            forced_cluster = None
            for c in unplanned_macro_clusters:
                if any(str(e.seg_id) == str(args.first_day_segment) for e in c.edges):
                    forced_cluster = c
                    break
            if forced_cluster is not None:
                cluster_segs = forced_cluster.edges
                cluster_nodes = forced_cluster.nodes
                start_candidates = forced_cluster.start_candidates
                cluster_centroid = (
                    sum(midpoint(e)[0] for e in cluster_segs) / len(cluster_segs),
                    sum(midpoint(e)[1] for e in cluster_segs) / len(cluster_segs),
                )
                drive_origin = home_coord
                best_start_node = None
                best_start_name = None
                best_drive_time = float("inf")
                for cand_node, cand_name in start_candidates:
                    drive_time_tmp = 0.0
                    if drive_origin and all_road_segments:
                        drive_time_tmp = planner_utils.estimate_drive_time_minutes(
                            drive_origin,
                            cand_node,
                            road_graph_for_drive,
                            args.average_driving_speed_mph,
                        )
                        drive_time_tmp += DRIVE_PARKING_OVERHEAD_MIN
                    if drive_time_tmp < best_drive_time:
                        best_drive_time = drive_time_tmp
                        best_start_node = cand_node
                        best_start_name = cand_name
                if best_start_node is None:
                    tmp_tree = build_kdtree(list(cluster_nodes))
                    best_start_node = nearest_node(tmp_tree, cluster_centroid)
                    best_start_name = None
                    best_drive_time = 0.0
                route_edges = plan_route(
                    G,
                    cluster_segs,
                    best_start_node,
                    args.pace,
                    args.grade,
                    args.road_pace,
                    args.max_foot_road,
                    args.road_threshold,
                    path_cache,
                    use_rpp=True,
                    allow_connectors=args.allow_connector_trails,
                    rpp_timeout=args.rpp_timeout,
                    debug_args=args,
                    spur_length_thresh=args.spur_length_thresh,
                    spur_road_bonus=args.spur_road_bonus,
                    use_advanced_optimizer=args.use_advanced_optimizer,
                    strict_max_foot_road=args.strict_max_foot_road,
                    redundancy_threshold=args.redundancy_threshold,
                    optimizer=args.optimizer,
                    postman_timeout=args.postman_timeout,
                    postman_max_odd=args.postman_max_odd,
                )
                if route_edges:
                    if best_drive_time > 0:
                        activities_for_this_day.append(
                            {
                                "type": "drive",
                                "minutes": best_drive_time,
                                "from_coord": drive_origin,
                                "to_coord": best_start_node,
                            }
                        )
                        time_spent_on_drives_today += best_drive_time
                    activities_for_this_day.append(
                        {
                            "type": "activity",
                            "route_edges": route_edges,
                            "name": f"Activity Part {len([a for a in activities_for_this_day if a['type'] == 'activity']) + 1}",
                            "ignored_budget": False,
                            "start_name": best_start_name,
                            "start_coord": best_start_node,
                        }
                    )
                    time_spent_on_activities_today += total_time(
                        route_edges, args.pace, args.grade, args.road_pace
                    )
                    last_activity_end_coord = route_edges[-1].end
                    unplanned_macro_clusters.remove(forced_cluster)
                args.first_day_segment = None

        while True:
            best_cluster_to_add_info = None
            candidate_pool = []

            past_efforts = [
                d["total_activity_time"] for d in daily_plans if d["activities"]
            ]
            mean_effort = (
                sum(past_efforts) / len(past_efforts)
                if past_efforts
                else time_spent_on_activities_today
            )

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

                drive_origin = (
                    last_activity_end_coord if last_activity_end_coord else home_coord
                )
                best_start_node = None
                best_start_name = None
                best_drive_time_to_start = float("inf")
                for cand_node, cand_name in start_candidates:
                    drive_time_tmp = 0.0
                    if drive_origin and all_road_segments:
                        drive_time_tmp = planner_utils.estimate_drive_time_minutes(
                            drive_origin,
                            cand_node,
                            road_graph_for_drive,
                            args.average_driving_speed_mph,
                        )
                        drive_time_tmp += DRIVE_PARKING_OVERHEAD_MIN
                    if drive_time_tmp < best_drive_time_to_start:
                        best_drive_time_to_start = drive_time_tmp
                        best_start_node = cand_node
                        best_start_name = cand_name

                if (
                    drive_origin == home_coord
                    and best_start_node is not None
                    and planner_utils._haversine_mi(drive_origin, best_start_node)
                    <= MIN_DRIVE_DISTANCE_MI
                ):
                    best_drive_time_to_start = 0.0

                if best_start_node is None:
                    tmp_tree = build_kdtree(list(cluster_nodes))
                    best_start_node = nearest_node(tmp_tree, cluster_centroid)
                    best_start_name = None
                    best_drive_time_to_start = 0.0

                cluster_sig = tuple(sorted(str(e.seg_id) for e in cluster_segs))
                if cluster_sig in failed_cluster_signatures:
                    continue

                debug_log(
                    args,
                    f"Attempting plan_route for cluster with segments: {[s.seg_id for s in cluster_segs]}",
                )
                route_edges = plan_route(
                    G,  # This is the on_foot_routing_graph
                    cluster_segs,
                    best_start_node,
                    args.pace,
                    args.grade,
                    args.road_pace,
                    args.max_foot_road,
                    args.road_threshold,
                    path_cache,
                    use_rpp=True,
                    allow_connectors=args.allow_connector_trails,
                    rpp_timeout=args.rpp_timeout,
                    debug_args=args,
                    spur_length_thresh=args.spur_length_thresh,
                    spur_road_bonus=args.spur_road_bonus,
                    use_advanced_optimizer=args.use_advanced_optimizer,
                    strict_max_foot_road=args.strict_max_foot_road,
                    redundancy_threshold=args.redundancy_threshold,
                    optimizer=args.optimizer,
                    postman_timeout=args.postman_timeout,
                    postman_max_odd=args.postman_max_odd,
                )
                if not route_edges:
                    if len(cluster_segs) == 1:
                        seg = cluster_segs[0]
                        if seg.direction == "both":
                            rev = Edge(
                                seg.seg_id,
                                seg.name,
                                seg.end,
                                seg.start,
                                seg.length_mi,
                                seg.elev_gain_ft,
                                list(reversed(seg.coords)),
                                seg.kind,
                                seg.direction,
                                seg.access_from,
                            )
                            route_edges = [seg, rev]
                        else:
                            route_edges = [seg]
                    else:
                        debug_log(
                            args,
                            f"Attempting extended plan_route for cluster with segments: {[s.seg_id for s in cluster_segs]}",
                        )
                        extended_route = plan_route(
                            G,
                            cluster_segs,
                            best_start_node,
                            args.pace,
                            args.grade,
                            args.road_pace,
                            args.max_foot_road * 3,
                            args.road_threshold,
                            path_cache,
                            use_rpp=True,
                            allow_connectors=args.allow_connector_trails,
                            rpp_timeout=args.rpp_timeout,
                            debug_args=args,
                            spur_length_thresh=args.spur_length_thresh,
                            spur_road_bonus=args.spur_road_bonus,
                            use_advanced_optimizer=args.use_advanced_optimizer,
                            strict_max_foot_road=args.strict_max_foot_road,
                            redundancy_threshold=args.redundancy_threshold,
                            optimizer=args.optimizer,
                            postman_timeout=args.postman_timeout,
                            postman_max_odd=args.postman_max_odd,
                        )
                        if extended_route:
                            debug_log(args, "extended route successful")
                            route_edges = extended_route
                        else:
                            failed_cluster_signatures.add(cluster_sig)
                            debug_log(
                                args,
                                f"Added {cluster_sig} to failed_cluster_signatures. Current size: {len(failed_cluster_signatures)}",
                            )
                            tqdm.write(
                                f"Skipping unroutable cluster with segments {[e.seg_id for e in cluster_segs]}",
                                file=sys.stderr,
                            )
                            debug_log(args, "extended route failed; skipping cluster")
                            continue

                estimated_activity_time = total_time(
                    route_edges, args.pace, args.grade, args.road_pace
                )
                current_drive_time = best_drive_time_to_start
                drive_from_coord_for_this_candidate = drive_origin
                drive_to_coord_for_this_candidate = best_start_node

                if last_activity_end_coord and best_drive_time_to_start < float("inf"):
                    walk_time = float("inf")
                    walk_edges: List[Edge] = []
                    walk_road_dist = float("inf")
                    try:
                        walk_path = nx.shortest_path(
                            G, drive_origin, best_start_node, weight="weight"
                        )
                        walk_edges = edges_from_path(G, walk_path)
                        walk_time = total_time(
                            walk_edges, args.pace, args.grade, args.road_pace
                        )
                        walk_road_dist = sum(
                            e.length_mi for e in walk_edges if e.kind == "road"
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

                    drive_time_tmp, drive_dist_tmp = (
                        planner_utils.estimate_drive_time_minutes(
                            drive_origin,
                            best_start_node,
                            road_graph_for_drive,
                            args.average_driving_speed_mph,
                            return_distance=True,
                        )
                    )
                    drive_time_tmp += DRIVE_PARKING_OVERHEAD_MIN
                    if (
                        drive_from_coord_for_this_candidate == home_coord
                        and drive_dist_tmp <= MIN_DRIVE_DISTANCE_MI
                    ):
                        drive_time_tmp = 0.0

                    walk_completion_time = sum(
                        planner_utils.estimate_time(
                            e, args.pace, args.grade, args.road_pace
                        )
                        for e in walk_edges
                        if e.seg_id is not None
                        and str(e.seg_id) in current_challenge_segment_ids
                    )
                    adjusted_walk = walk_time - walk_completion_time
                    factor = DRIVE_FASTER_FACTOR + (
                        COMPLETE_SEGMENT_BONUS if walk_completion_time > 0 else 0.0
                    )

                    if (
                        walk_time < float("inf")
                        and walk_road_dist <= args.max_foot_road
                        and (
                            adjusted_walk <= drive_time_tmp * factor
                            or adjusted_walk - drive_time_tmp
                            <= MIN_DRIVE_TIME_SAVINGS_MIN
                            or drive_dist_tmp <= MIN_DRIVE_DISTANCE_MI
                        )
                    ):
                        route_edges = walk_edges + route_edges
                        estimated_activity_time += walk_time
                        current_drive_time = 0.0

                if current_drive_time > args.max_drive_minutes_per_transfer:
                    continue

                if (
                    time_spent_on_activities_today
                    + estimated_activity_time
                    + time_spent_on_drives_today
                    + current_drive_time
                ) <= todays_total_budget_minutes:
                    access_names = {
                        e.access_from for e in cluster_segs if e.access_from
                    }
                    completion_bonus = 0.0
                    for name in access_names:
                        remaining = any(
                            any(e.access_from == name for e in other.edges)
                            for j, other in enumerate(unplanned_macro_clusters)
                            if j != cluster_idx
                        )
                        if not remaining:
                            completion_bonus += 1.0

                    projected_effort = (
                        time_spent_on_activities_today + estimated_activity_time
                    )
                    effort_dist = abs(projected_effort - mean_effort)
                    score = ClusterScore(
                        drive_time=current_drive_time,
                        activity_time=estimated_activity_time,
                        isolation_score=isolation_lookup.get(cluster_idx, 0.0),
                        completion_bonus=completion_bonus,
                        effort_distribution=effort_dist,
                    )
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
                            "score": score,
                            "cluster_ref": cluster_candidate,  # Store the ClusterInfo object
                        }
                    )

            if candidate_pool:
                candidate_pool.sort(key=lambda c: c["score"].total_score)
                best_cluster_to_add_info = candidate_pool[0]

            if best_cluster_to_add_info:
                chosen_cluster_object_to_remove = best_cluster_to_add_info[
                    "cluster_ref"
                ]
                if (
                    best_cluster_to_add_info["drive_time"] > 0
                    and best_cluster_to_add_info["drive_from"]
                    and best_cluster_to_add_info["drive_to"]
                ):
                    activities_for_this_day.append(
                        {
                            "type": "drive",
                            "minutes": best_cluster_to_add_info["drive_time"],
                            "from_coord": best_cluster_to_add_info["drive_from"],
                            "to_coord": best_cluster_to_add_info["drive_to"],
                        }
                    )
                    time_spent_on_drives_today += best_cluster_to_add_info["drive_time"]

                act_route_edges = best_cluster_to_add_info["route_edges"]
                activities_for_this_day.append(
                    {
                        "type": "activity",
                        "route_edges": act_route_edges,
                        "name": f"Activity Part {len([a for a in activities_for_this_day if a['type'] == 'activity']) + 1}",
                        "ignored_budget": best_cluster_to_add_info.get(
                            "ignored_budget", False
                        ),
                        "start_name": best_cluster_to_add_info.get("start_name"),
                        "start_coord": best_cluster_to_add_info.get("start_coord"),
                    }
                )
                time_spent_on_activities_today += best_cluster_to_add_info[
                    "activity_time"
                ]
                last_activity_end_coord = act_route_edges[-1].end

                try:
                    unplanned_macro_clusters.remove(chosen_cluster_object_to_remove)
                    debug_log(
                        args,
                        f"Successfully removed cluster (first seg: {chosen_cluster_object_to_remove.edges[0].seg_id if chosen_cluster_object_to_remove.edges else 'EMPTY'}) by object reference.",
                    )
                except ValueError:
                    debug_log(
                        args,
                        f"Warning: Cluster object (first seg: {chosen_cluster_object_to_remove.edges[0].seg_id if chosen_cluster_object_to_remove.edges else 'EMPTY'}) not found in unplanned_macro_clusters for removal via .remove(). It might have been popped by fallback or already processed.",
                    )
            else:
                if unplanned_macro_clusters and todays_total_budget_minutes > 0:
                    debug_log(
                        args,
                        f"Fallback pop(0): removing cluster with segments: "
                        f"{[s.seg_id for s in unplanned_macro_clusters[0].edges] if unplanned_macro_clusters else 'EMPTY_LIST'}",
                    )
                    fallback_cluster = unplanned_macro_clusters.pop(0)
                    if fallback_cluster.start_candidates:
                        start_node, start_name = fallback_cluster.start_candidates[0]
                    else:
                        start_node = fallback_cluster.edges[0].start
                        start_name = None
                    debug_log(
                        args,
                        f"Attempting fallback plan_route for cluster with segments: {[s.seg_id for s in fallback_cluster.edges]}",
                    )
                    act_route_edges = plan_route(
                        G,
                        fallback_cluster.edges,
                        start_node,
                        args.pace,
                        args.grade,
                        args.road_pace,
                        args.max_foot_road,
                        args.road_threshold,
                        path_cache,
                        use_rpp=True,
                        allow_connectors=args.allow_connector_trails,
                        rpp_timeout=args.rpp_timeout,
                        debug_args=args,
                        spur_length_thresh=args.spur_length_thresh,
                        spur_road_bonus=args.spur_road_bonus,
                        use_advanced_optimizer=args.use_advanced_optimizer,
                        strict_max_foot_road=args.strict_max_foot_road,
                        redundancy_threshold=args.redundancy_threshold,
            optimizer_name=args.optimizer,
                        postman_timeout=args.postman_timeout,
                        postman_max_odd=args.postman_max_odd,
                    )
                    if act_route_edges:
                        activities_for_this_day.append(
                            {
                                "type": "activity",
                                "route_edges": act_route_edges,
                                "name": f"Activity Part {len([a for a in activities_for_this_day if a['type'] == 'activity']) + 1}",
                                "ignored_budget": True,
                                "start_name": start_name,
                                "start_coord": start_node,
                            }
                        )
                        time_spent_on_activities_today += total_time(
                            act_route_edges,
                            args.pace,
                            args.grade,
                            args.road_pace,
                        )
                        last_activity_end_coord = act_route_edges[-1].end
                    else:
                        debug_log(
                            args,
                            "Fallback plan_route failed; returning cluster to remaining list.",
                        )
                        # Reinsert the cluster so it can be attempted later
                        unplanned_macro_clusters.insert(0, fallback_cluster)
                break

        if activities_for_this_day:
            total_day_time = time_spent_on_activities_today + time_spent_on_drives_today
            note_parts = []
            if total_day_time > todays_total_budget_minutes:
                note_parts.append(
                    f"over budget by {total_day_time - todays_total_budget_minutes:.1f} min"
                )
            else:
                note_parts.append("fits budget")
            if todays_total_budget_minutes <= 120:
                note_parts.append("night run \u2013 kept easy")
            notes = "; ".join(note_parts)
            daily_plans.append(
                {
                    "date": cur_date,
                    "activities": activities_for_this_day,
                    "total_activity_time": time_spent_on_activities_today,
                    "total_drive_time": time_spent_on_drives_today,
                    "notes": notes,
                }
            )
            day_iter.set_postfix(note=notes)
        else:
            daily_plans.append(
                {
                    "date": cur_date,
                    "activities": [],
                    "total_activity_time": 0.0,
                    "total_drive_time": 0.0,
                    "notes": "",
                }
            )
            day_iter.set_postfix(note="no activities")

        if (
            args.draft_every
            and args.draft_every > 0
            and (day_idx + 1) % args.draft_every == 0
        ):
            draft_csv = os.path.splitext(args.output)[0] + f"_draft_{day_idx+1}.csv"
            export_plan_files(
                daily_plans,
                args,
                csv_path=draft_csv,
                write_gpx=True,
                review=False,
                challenge_ids=current_challenge_segment_ids,
            )

    # Smooth the schedule if we have lightly used days and remaining clusters
    segments_in_unplanned_before_smooth = set()
    mileage_in_unplanned_before_smooth = 0.0
    temp_seen_ids_for_mileage_log_smooth = set()
    for cluster_info_smooth in unplanned_macro_clusters:
        for edge_smooth in cluster_info_smooth.edges:
            if edge_smooth.seg_id is not None:
                segments_in_unplanned_before_smooth.add(str(edge_smooth.seg_id))
                if str(edge_smooth.seg_id) not in temp_seen_ids_for_mileage_log_smooth:
                    mileage_in_unplanned_before_smooth += edge_smooth.length_mi
                    temp_seen_ids_for_mileage_log_smooth.add(str(edge_smooth.seg_id))
    debug_log(
        args,
        f"Before first smooth_daily_plans: {len(unplanned_macro_clusters)} clusters remain, with {len(segments_in_unplanned_before_smooth)} unique segment IDs, totaling {mileage_in_unplanned_before_smooth:.2f} mi.",
    )

    smooth_daily_plans(
        daily_plans,
        unplanned_macro_clusters,
        daily_budget_minutes,
        G,
        args.pace,
        args.grade,
        args.road_pace,
        args.max_foot_road,
        args.road_threshold,
        path_cache,
        allow_connector_trails=args.allow_connector_trails,
        rpp_timeout=args.rpp_timeout,
        road_graph=road_graph_for_drive,
        average_driving_speed_mph=args.average_driving_speed_mph,
        home_coord=home_coord,
        spur_length_thresh=args.spur_length_thresh,
        spur_road_bonus=args.spur_road_bonus,
        use_advanced_optimizer=args.use_advanced_optimizer,
        redundancy_threshold=args.redundancy_threshold,
        debug_args=args,
        strict_max_foot_road=args.strict_max_foot_road,
    )

    # Force insert any remaining clusters even if it exceeds the budget
    segments_in_unplanned_before_force = set()
    mileage_in_unplanned_before_force = 0.0
    temp_seen_ids_for_mileage_log_force = set()
    for cluster_info_force in unplanned_macro_clusters:
        for edge_force in cluster_info_force.edges:
            if edge_force.seg_id is not None:
                segments_in_unplanned_before_force.add(str(edge_force.seg_id))
                if str(edge_force.seg_id) not in temp_seen_ids_for_mileage_log_force:
                    mileage_in_unplanned_before_force += edge_force.length_mi
                    temp_seen_ids_for_mileage_log_force.add(str(edge_force.seg_id))
    debug_log(
        args,
        f"Before force_schedule_remaining_clusters: {len(unplanned_macro_clusters)} clusters remain, with {len(segments_in_unplanned_before_force)} unique segment IDs, totaling {mileage_in_unplanned_before_force:.2f} mi.",
    )

    force_schedule_remaining_clusters(
        daily_plans,
        unplanned_macro_clusters,
        daily_budget_minutes,
        G,
        args.pace,
        args.grade,
        args.road_pace,
        args.max_foot_road,
        args.road_threshold,
        path_cache,
        allow_connector_trails=args.allow_connector_trails,
        rpp_timeout=args.rpp_timeout,
        road_graph=road_graph_for_drive,
        average_driving_speed_mph=args.average_driving_speed_mph,
        home_coord=home_coord,
        spur_length_thresh=args.spur_length_thresh,
        spur_road_bonus=args.spur_road_bonus,
        use_advanced_optimizer=args.use_advanced_optimizer,
        redundancy_threshold=args.redundancy_threshold,
        debug_args=args,
        strict_max_foot_road=args.strict_max_foot_road,
    )

    # Increase all budgets evenly if any day is now over budget
    if even_out_budgets(daily_plans, daily_budget_minutes) > 0:
        update_plan_notes(daily_plans, daily_budget_minutes)

    # After smoothing, ensure all segments have been scheduled. If any
    # clusters remain unscheduled the plan is infeasible.
    if unplanned_macro_clusters:
        overall_routing_status_ok = False  # Set status to False if clusters remain
        remaining_ids: Set[str] = set()
        for cluster in unplanned_macro_clusters:
            for seg in cluster.edges:
                if seg.seg_id is not None:
                    remaining_ids.add(str(seg.seg_id))

        unscheduled_mileage = 0.0
        segment_lengths_lookup = {
            str(seg.seg_id): seg.length_mi
            for seg in all_trail_segments
            if seg.seg_id is not None
        }
        for seg_id_str in remaining_ids:
            unscheduled_mileage += segment_lengths_lookup.get(seg_id_str, 0.0)

        debug_log(
            args,
            f"Final unscheduled segments: {len(remaining_ids)} unique segment IDs, Total unique mileage of these segments: {unscheduled_mileage:.2f} mi. IDs: {', '.join(sorted(list(remaining_ids)))}",
        )

        avg_hours = (
            sum(daily_budget_minutes.values()) / len(daily_budget_minutes) / 60.0
            if daily_budget_minutes
            else 0.0
        )
        tqdm_msg = (
            f"With {avg_hours:.1f} hours/day from {start_date} to {end_date}, "
            "it's impossible to complete all trails. Extend the timeframe or "
            "increase daily budget."
        )
        if remaining_ids:
            tqdm_msg += f" Failed to schedule {len(remaining_ids)} unique segments ({unscheduled_mileage:.2f} mi). See debug log for IDs."
        tqdm.write(tqdm_msg, file=sys.stderr)

    # Verify that all current_challenge_segment_ids are actually scheduled
    scheduled_segment_ids: Set[str] = set()
    for day_plan in daily_plans:
        for activity in day_plan.get("activities", []):
            if activity.get("type") == "activity":
                for edge in activity.get("route_edges", []):
                    if edge.seg_id is not None:
                        scheduled_segment_ids.add(str(edge.seg_id))

    if current_challenge_segment_ids:  # Only check if there were segments to schedule
        missing_segment_ids = current_challenge_segment_ids - scheduled_segment_ids
        if missing_segment_ids:
            sorted_missing_ids = sorted(list(missing_segment_ids))
            error_message = (
                f"Error: The following {len(sorted_missing_ids)} challenge segments were not scheduled: "
                f"{sorted_missing_ids}. This may indicate an issue with the "
                "planning logic or insufficient time/budget."
            )
            # Optionally, log to debug log as well
            debug_log(args, error_message)
            raise ValueError(error_message)

    export_plan_files(
        daily_plans,
        args,
        challenge_ids=current_challenge_segment_ids,
        routing_failed=not overall_routing_status_ok,
    )

    cache_utils.save_cache("dist_cache", cache_key, path_cache)
    logger.info("Saved path cache with %d start nodes", len(path_cache))


if __name__ == "__main__":
    main()
