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

try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None
    _HAVE_SCIPY = False

# Allow running this file directly without installing the package
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trail_route_ai import planner_utils, plan_review

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
    if _HAVE_SCIPY:
        return cKDTree(np.array(nodes))
    return list(nodes)


def nearest_node(index, point: Tuple[float, float]):
    """Return the nearest node to ``point`` using ``index``."""
    if _HAVE_SCIPY and hasattr(index, "query"):
        _, idx = index.query(point)
        return tuple(index.data[idx])
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
    max_road: float = 3.0
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
    precompute_paths: bool = False
    redundancy_threshold: float = 0.2
    allow_connector_trails: bool = True
    rpp_timeout: float = 5.0
    spur_length_thresh: float = 0.3
    spur_road_bonus: float = 0.25


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


def total_time(edges: List[Edge], pace: float, grade: float, road_pace: float) -> float:
    return sum(planner_utils.estimate_time(e, pace, grade, road_pace) for e in edges)


def debug_log(args: argparse.Namespace | None, message: str) -> None:
    """Append ``message`` to the debug log if ``args.debug`` is set."""
    if args is None:
        return
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
    G = nx.DiGraph()
    for e in edges:
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
    G = build_nx_graph(graph_edges, pace, grade, road_pace)

    macro_clusters: List[Tuple[List[Edge], Set[Tuple[float, float]]]] = []
    assigned_segment_ids: set[str | int] = set()

    for component_nodes in nx.weakly_connected_components(G):
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
    max_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
) -> List[Edge]:
    """Return a continuous route connecting ``edges`` starting from ``start``
    using a greedy nearest-neighbor strategy.

    ``max_road`` limits road mileage for any connector. ``road_threshold``
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
    while remaining:
        paths = None
        if dist_cache and cur in dist_cache:
            paths = dist_cache[cur]
        else:
            _, paths = nx.single_source_dijkstra(G, cur, weight="weight")
        candidate_info = []
        for e in remaining:
            for end in [e.start, e.end]:
                if end == e.end and e.direction != "both":
                    continue
                if end not in paths:
                    continue
                path_nodes = paths[end]
                edges_path = edges_from_path(G, path_nodes)
                road_dist = sum(ed.length_mi for ed in edges_path if ed.kind == "road")
                time = sum(
                    planner_utils.estimate_time(ed, pace, grade, road_pace)
                    for ed in edges_path
                )
                time += planner_utils.estimate_time(e, pace, grade, road_pace)
                uses_road = any(ed.kind == "road" for ed in edges_path)
                candidate_info.append((time, uses_road, e, end, edges_path, road_dist))

        allowed_max_road = max_road
        if last_seg is not None and degree.get(cur, 0) == 1 and last_seg.length_mi <= spur_length_thresh:
            allowed_max_road += spur_road_bonus
        candidates = [c[:5] for c in candidate_info if c[5] <= allowed_max_road]

        if not candidates:
            if not candidate_info:
                current_last_segment_name = (
                    route[-1].name
                    if route and hasattr(route[-1], "name") and route[-1].name
                    else (
                        str(route[-1].seg_id)
                        if route and hasattr(route[-1], "seg_id")
                        else "the route start"
                    )
                )
                remaining_segment_names = [s.name or str(s.seg_id) for s in remaining]

                print(
                    f"Error in plan_route: Could not find a valid path from '{current_last_segment_name}' "
                    f"to any of the remaining segments: {remaining_segment_names} "
                    f"within the given constraints (e.g., max_road for connector). "
                    f"This cluster cannot be routed continuously.",
                    file=sys.stderr,
                )
            return [], []  # No viable connector under max_road

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

    if cur == start:
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
            edge_data["weight"] = current_weight * 10.0
            # For MultiGraph, one would do: G_for_path_back[edge_obj.start][edge_obj.end][key]['weight'] *= 10.0

    try:
        path_back_nodes = nx.shortest_path(G_for_path_back, cur, start, weight="weight")
        path_back_edges = edges_from_path(
            G, path_back_nodes
        )  # Use original G for edge objects
        route.extend(path_back_edges)
    except nx.NetworkXNoPath:
        # Fallback to original graph if modified graph yields no path
        try:
            path_back_nodes_orig = nx.shortest_path(G, cur, start, weight="weight")
            route.extend(edges_from_path(G, path_back_nodes_orig))
        except nx.NetworkXNoPath:
            # No path back found even on original graph, or cur == start
            pass

    return route, order


def _plan_route_for_sequence(
    G: nx.DiGraph,
    sequence: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
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
                if dist_cache and cur in dist_cache and end in dist_cache[cur]:
                    path_nodes = dist_cache[cur][end]
                else:
                    path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                edges_path = edges_from_path(G, path_nodes)
                road_dist = sum(e.length_mi for e in edges_path if e.kind == "road")
                allowed_max_road = max_road
                if last_seg is not None and degree.get(cur, 0) == 1 and last_seg.length_mi <= spur_length_thresh:
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
            # fallback ignoring max_road
            for end in [seg.start, seg.end]:
                if end == seg.end and seg.direction != "both":
                    continue
                try:
                    if dist_cache and cur in dist_cache and end in dist_cache[cur]:
                        path_nodes = dist_cache[cur][end]
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
) -> List[Edge]:
    """Approximate Rural Postman solution using Steiner tree and Eulerization."""

    if not edges:
        return []

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

    if allow_connectors:
        steiner = nx.algorithms.approximation.steiner_tree(UG, required_nodes, weight="weight")
        sub.add_edges_from(steiner.edges(data=True))

    if not nx.is_connected(sub):
        # connect components using shortest paths in UG
        components = list(nx.connected_components(sub))
        for i in range(len(components) - 1):
            c1 = next(iter(components[i]))
            c2 = next(iter(components[i + 1]))
            try:
                path = nx.shortest_path(UG, c1, c2, weight="weight")
            except nx.NetworkXNoPath:
                continue
            path_edges = edges_from_path(G, path, required_ids=required_ids)
            for e in path_edges:
                if UG.has_edge(e.start, e.end):
                    sub.add_edge(e.start, e.end, **UG.get_edge_data(e.start, e.end))
                elif UG.has_edge(e.end, e.start):
                    sub.add_edge(e.end, e.start, **UG.get_edge_data(e.end, e.start))

    try:
        eulerized = nx.euler.eulerize(sub)
    except nx.NetworkXError:
        return []

    if start not in eulerized:
        start = next(iter(eulerized.nodes))

    circuit = list(nx.eulerian_circuit(eulerized, source=start))
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
            except nx.NetworkXNoPath:
                continue

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
    max_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    use_rpp: bool = False,
    allow_connectors: bool = True,
    rpp_timeout: float = 5.0,
    debug_args: argparse.Namespace | None = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
) -> List[Edge]:
    """Plan an efficient loop through ``edges`` starting and ending at ``start``."""

    debug_log(debug_args, f"plan_route: {len(edges)} segs from {start}, use_rpp={use_rpp}")
    if _edges_form_tree(edges) and all(e.direction == "both" for e in edges):
        try:
            tree_route = _plan_route_tree(edges, start)
            if tree_route:
                return tree_route
        except Exception:
            pass


    if use_rpp:
        try:
            debug_log(debug_args, "attempting RPP")
            route_rpp = plan_route_rpp(
                G,
                edges,
                start,
                pace,
                grade,
                road_pace,
                allow_connectors=allow_connectors,
            )
            if route_rpp:
                debug_log(debug_args, "RPP successful")
                return route_rpp
        except Exception as e:
            debug_log(debug_args, f"RPP failed: {e}")

    debug_log(debug_args, "falling back to greedy search")

    initial_route, seg_order = _plan_route_greedy(
        G,
        edges,
        start,
        pace,
        grade,
        road_pace,
        max_road,
        road_threshold,
        dist_cache,
        spur_length_thresh=spur_length_thresh,
        spur_road_bonus=spur_road_bonus,
    )
    debug_log(debug_args, f"greedy initial route time {total_time(initial_route, pace, grade, road_pace):.2f} min")
    if not initial_route or len(seg_order) <= 2:
        return initial_route

    best_route = initial_route
    best_order = seg_order
    best_time = total_time(best_route, pace, grade, road_pace)

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
                    max_road,
                    road_threshold,
                    dist_cache,
                    spur_length_thresh=spur_length_thresh,
                    spur_road_bonus=spur_road_bonus,
                )
                if not candidate_route:
                    continue
                cand_time = total_time(candidate_route, pace, grade, road_pace)
                if cand_time < best_time:
                    debug_log(debug_args, f"2-opt improvement {best_time:.2f} -> {cand_time:.2f}")
                    best_time = cand_time
                    best_route = candidate_route
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break

    debug_log(debug_args, f"final route time {best_time:.2f}")
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
            if (
                total_time(other, pace, grade, road_pace)
                + total_time(small, pace, grade, road_pace)
                <= budget
            ):
                clusters[i] = other + small
                merged = True
                break
        if not merged:
            clusters.append(small)
            break
    return clusters[:max_clusters]


def split_cluster_by_connectivity(
    cluster_edges: List[Edge],
    G: nx.DiGraph,
    max_road: float,
) -> List[List[Edge]]:
    """Split ``cluster_edges`` into subclusters based on walkable connectivity.

    Two segments are considered connected if a path exists between any of their
    endpoints using at most ``max_road`` miles of road. Trail mileage along the
    connector does not count against this limit. The returned list will contain
    one or more groups of edges that are internally connected according to this
    rule. The original ordering of segments is not preserved.
    """

    def road_weight(
        u: Tuple[float, float], v: Tuple[float, float], data: dict
    ) -> float:
        edge = data.get("edge") if isinstance(data, dict) else data[0]["edge"]
        return edge.length_mi if edge.kind == "road" else 0.0

    remaining = list(cluster_edges)
    subclusters: List[List[Edge]] = []

    while remaining:
        seed = remaining.pop(0)
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
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        if dist <= max_road:
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

    return subclusters


def smooth_daily_plans(
    daily_plans: List[Dict[str, object]],
    remaining_clusters: List[ClusterInfo],
    daily_budget_minutes: Dict[datetime.date, float],
    G: nx.Graph,
    pace: float,
    grade: float,
    road_pace: float,
    max_road: float,
    road_threshold: float,
    dist_cache: Optional[dict] = None,
    *,
    allow_connector_trails: bool = True,
    rpp_timeout: float = 5.0,
    road_graph: Optional[nx.Graph] = None,
    average_driving_speed_mph: float = 30.0,
    home_coord: Optional[Tuple[float, float]] = None,
    debug_args: argparse.Namespace | None = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
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

    for cluster in list(remaining_clusters):
        # choose a starting node
        start_candidates = cluster.start_candidates
        if start_candidates:
            start_node = start_candidates[0][0]
            start_name = start_candidates[0][1]
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
            max_road,
            road_threshold,
            dist_cache,
            use_rpp=True,
            allow_connectors=allow_connector_trails,
            rpp_timeout=rpp_timeout,
            debug_args=debug_args,
            spur_length_thresh=spur_length_thresh,
            spur_road_bonus=spur_road_bonus,
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
                acts = [a for a in day_plan.get("activities", []) if a.get("type") == "activity"]
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
    hit_list: Optional[List[Dict[str, object]]] = None,
) -> None:
    """Write an HTML overview of ``daily_plans`` with maps and elevation.

    When ``hit_list`` is provided, an additional section summarizing quick
    segments is appended to the report.
    """

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
            lines.append(
                f"<li>Total Elevation Gain: {metrics['total_elev_gain_ft']:.0f} ft</li>"
            )
            lines.append(
                f"<li>Redundant Elevation Gain: {metrics['redundant_elev_gain_ft']:.0f} ft ({metrics['redundant_elev_pct']:.0f}% )</li>"
            )
            lines.append(f"<li>Drive Time: {metrics['drive_time_min']:.0f} min</li>")
            lines.append(f"<li>Run Time: {metrics['run_time_min']:.0f} min</li>")
            lines.append(f"<li>Total Time: {metrics['total_time_min']:.0f} min</li>")
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
    lines.append("<ul>")
    lines.append(f"<li>Total Distance: {totals['total_distance_mi']:.1f} mi</li>")
    lines.append(f"<li>New Distance: {totals['new_distance_mi']:.1f} mi</li>")
    lines.append(
        f"<li>Redundant Distance: {totals['redundant_distance_mi']:.1f} mi ({redundant_pct:.0f}% )</li>"
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

    if hit_list:
        lines.append("<h2>Hit List: Quick Segments</h2>")
        lines.append("<ul>")
        for item in hit_list:
            name = item.get("name", "Trail")
            dist = item.get("distance_mi", 0.0)
            gain = item.get("elev_gain_ft", 0.0)
            t = item.get("time_min", 0.0)
            lines.append(
                f"<li><b>{name}</b> – {dist:.1f} mi, {gain:.0f} ft (est. {t:.0f} min)</li>"
            )
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
    hit_list: Optional[List[Dict[str, object]]] = None,
    challenge_ids: Optional[Set[str]] = None,
) -> None:
    """Write CSV and HTML outputs for ``daily_plans``.

    ``csv_path`` overrides ``args.output``. GPX files and plan review are
    skipped when ``write_gpx`` is ``False`` or ``review`` is ``False``.
    When ``hit_list`` is provided, a ``hit_list.csv`` file is written and the
    items are included in the HTML report.
    """

    orig_output = args.output
    orig_review = args.review
    if getattr(args, "debug", None):
        open(args.debug, "w").close()
    if csv_path is not None:
        args.output = csv_path
    if review is not None:
        args.review = review

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
                    gpx_path = os.path.join(args.gpx_dir, gpx_file_name)
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
                        or (challenge_ids is not None and str(e.seg_id) not in challenge_ids)
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
                "redundant_distance_pct": round(redundant_pct, 1),
                "total_elev_gain_ft": round(current_day_total_trail_gain, 0),
                "redundant_elev_gain_ft": round(redundant_elev, 0),
                "redundant_elev_pct": round(redundant_elev_pct, 1),
                "drive_time_min": round(day_plan["total_drive_time"], 1),
                "run_time_min": round(day_plan["total_activity_time"], 1),
                "total_time_min": round(
                    day_plan["total_activity_time"] + day_plan["total_drive_time"], 1
                ),
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
            }
            debug_log(args, f"{day_date_str}: {day_plan['rationale']}")

    if summary_rows:
        totals = {
            "total_trail_distance_mi": 0.0,
            "unique_trail_miles": 0.0,
            "redundant_miles": 0.0,
            "total_trail_elev_gain_ft": 0.0,
            "unique_trail_elev_gain_ft": 0.0,
            "redundant_elev_gain_ft": 0.0,
            "total_activity_time_min": 0.0,
            "total_drive_time_min": 0.0,
            "total_time_min": 0.0,
        }
        for row in summary_rows:
            if row.get("plan_description") == "Unable to complete":
                continue
            totals["total_trail_distance_mi"] += row["total_trail_distance_mi"]
            totals["unique_trail_miles"] += row["unique_trail_miles"]
            totals["redundant_miles"] += row["redundant_miles"]
            totals["total_trail_elev_gain_ft"] += row["total_trail_elev_gain_ft"]
            totals["unique_trail_elev_gain_ft"] += row["unique_trail_elev_gain_ft"]
            totals["redundant_elev_gain_ft"] += row["redundant_elev_gain_ft"]
            totals["total_activity_time_min"] += row["total_activity_time_min"]
            totals["total_drive_time_min"] += row["total_drive_time_min"]
            totals["total_time_min"] += row["total_time_min"]

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
                "unique_trail_miles": round(totals["unique_trail_miles"], 2),
                "redundant_miles": round(totals["redundant_miles"], 2),
                "redundant_pct": round(total_pct, 1),
                "total_trail_elev_gain_ft": round(
                    totals["total_trail_elev_gain_ft"], 0
                ),
                "unique_trail_elev_gain_ft": round(
                    totals["unique_trail_elev_gain_ft"], 0
                ),
                "redundant_elev_gain_ft": round(totals["redundant_elev_gain_ft"], 0),
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

        fieldnames = list(summary_rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        if hit_list is not None:
            hit_path = os.path.join(
                os.path.dirname(args.output),
                "hit_list.csv",
            )
            with open(hit_path, "w", newline="") as hf:
                hw = csv.DictWriter(
                    hf,
                    fieldnames=[
                        "name",
                        "segments",
                        "distance_mi",
                        "elev_gain_ft",
                        "time_min",
                    ],
                )
                hw.writeheader()
                hw.writerows(hit_list)
    else:
        default_fieldnames = [
            "date",
            "plan_description",
            "route_description",
            "total_trail_distance_mi",
            "unique_trail_miles",
            "redundant_miles",
            "redundant_pct",
            "total_trail_elev_gain_ft",
            "unique_trail_elev_gain_ft",
            "redundant_elev_gain_ft",
            "redundant_elev_pct",
            "total_activity_time_min",
            "total_drive_time_min",
            "total_time_min",
            "num_activities",
            "num_drives",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=default_fieldnames)
            writer.writeheader()
        if hit_list is not None:
            hit_path = os.path.join(
                os.path.dirname(args.output),
                "hit_list.csv",
            )
            with open(hit_path, "w", newline="") as hf:
                hw = csv.DictWriter(
                    hf,
                    fieldnames=[
                        "name",
                        "segments",
                        "distance_mi",
                        "elev_gain_ft",
                        "time_min",
                    ],
                )
                hw.writeheader()
                hw.writerows(hit_list)

    if args.review and summary_rows:
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
        full_gpx_path = os.path.join(args.gpx_dir, "full_timespan.gpx")
        planner_utils.write_multiday_gpx(
            full_gpx_path,
            daily_plans,
            mark_road_transitions=args.mark_road_transitions,
            colors=colors,
        )

    html_out = os.path.splitext(args.output)[0] + ".html"
    img_dir = os.path.join(os.path.dirname(html_out), "plan_images")

    write_plan_html(
        html_out,
        daily_plans,
        img_dir,
        dem_path=args.dem,
        hit_list=hit_list,
    )
    print(f"HTML plan written to {html_out}")

    print(f"Challenge plan written to {args.output}")
    if write_gpx:
        if not daily_plans or not any(dp.get("activities") for dp in daily_plans):
            gpx_files_present = False
            if os.path.exists(args.gpx_dir):
                gpx_files_present = any(
                    f.endswith(".gpx") for f in os.listdir(args.gpx_dir)
                )

            if not gpx_files_present:
                print(f"No GPX files generated as no activities were planned.")
            else:
                print(
                    f"GPX files may exist in {args.gpx_dir} from previous runs, but no new activities were planned in this run."
                )
        else:
            print(f"GPX files written to {args.gpx_dir}")

    args.output = orig_output
    args.review = orig_review


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
        "--max-road",
        type=float,
        default=config_defaults.get("max_road", 3.0),
        help="Max road distance per connector (mi)",
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
        help="Automatically create dated output directory when --output-dir is not given",
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
        "--precompute-paths",
        action="store_true",
        default=config_defaults.get("precompute_paths", False),
        help="Precompute all-pairs shortest paths between key graph nodes",
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
    parser.set_defaults(allow_connector_trails=config_defaults.get("allow_connector_trails", True))
    parser.add_argument(
        "--rpp-timeout",
        type=float,
        default=config_defaults.get("rpp_timeout", 5.0),
        help="Time limit in seconds for RPP solver",
    )
    parser.add_argument(
        "--draft-every",
        type=int,
        metavar="N",
        default=0,
        help="Write draft CSV and HTML files every N days",
    )

    args = parser.parse_args(argv)

    if "--time" in argv and "--daily-hours-file" not in argv:
        args.daily_hours_file = None

    output_dir = args.output_dir
    if output_dir is None and args.auto_output_dir:
        today = datetime.date.today().isoformat()
        output_dir = os.path.join("outputs", f"plan_{today}")
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

    path_cache: dict | None = None
    if args.precompute_paths:
        key_nodes = {e.start for e in on_foot_routing_graph_edges} | {
            e.end for e in on_foot_routing_graph_edges
        }
        path_cache = {}
        for n in key_nodes:
            _, paths = nx.single_source_dijkstra(G, n, weight="weight")
            path_cache[n] = {t: paths[t] for t in key_nodes if t in paths}

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

    quick_hits = planner_utils.identify_quick_hits(
        current_challenge_segments,
        args.pace,
        args.grade,
        args.road_pace,
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
    for cluster_edges, cluster_nodes in potential_macro_clusters:
        if not cluster_edges:
            continue
        naive_time = total_time(cluster_edges, args.pace, args.grade, args.road_pace)
        oversized_threshold = 1.5 * budget
        if naive_time > oversized_threshold:
            debug_log(args, f"splitting large cluster of {naive_time:.1f} min into parts")
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
        node_tree_tmp = build_kdtree(list(cluster_nodes))
        start_node = nearest_node(node_tree_tmp, cluster_centroid)
        initial_route = plan_route(
            G,
            cluster_segs,
            start_node,
            args.pace,
            args.grade,
            args.road_pace,
            args.max_road,
            args.road_threshold,
            path_cache,
            use_rpp=False,
            allow_connectors=args.allow_connector_trails,
            rpp_timeout=args.rpp_timeout,
            debug_args=args,
            spur_length_thresh=args.spur_length_thresh,
            spur_road_bonus=args.spur_road_bonus,
        )
        if initial_route:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue

        if len(cluster_segs) == 1:
            processed_clusters.append((cluster_segs, cluster_nodes))
            continue

        connectivity_subs = split_cluster_by_connectivity(
            cluster_segs,
            G,
            args.max_road,
        )

        if len(connectivity_subs) > 1:
            debug_log(args, f"split cluster into {len(connectivity_subs)} parts due to connectivity")
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
            args.max_road * 3,
            args.road_threshold,
            path_cache,
            use_rpp=False,
            allow_connectors=args.allow_connector_trails,
            rpp_timeout=args.rpp_timeout,
            debug_args=args,
            spur_length_thresh=args.spur_length_thresh,
            spur_road_bonus=args.spur_road_bonus,
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

                route_edges = plan_route(
                    G,  # This is the on_foot_routing_graph
                    cluster_segs,
                    best_start_node,
                    args.pace,
                    args.grade,
                    args.road_pace,
                    args.max_road,
                    args.road_threshold,
                    path_cache,
                    use_rpp=True,
                    allow_connectors=args.allow_connector_trails,
                    rpp_timeout=args.rpp_timeout,
                    debug_args=args,
                    spur_length_thresh=args.spur_length_thresh,
                    spur_road_bonus=args.spur_road_bonus,
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
                        extended_route = plan_route(
                            G,
                            cluster_segs,
                            best_start_node,
                            args.pace,
                            args.grade,
                            args.road_pace,
                            args.max_road * 3,
                            args.road_threshold,
                            path_cache,
                            use_rpp=True,
                            allow_connectors=args.allow_connector_trails,
                            rpp_timeout=args.rpp_timeout,
                            debug_args=args,
                            spur_length_thresh=args.spur_length_thresh,
                            spur_road_bonus=args.spur_road_bonus,
                        )
                        if extended_route:
                            debug_log(args, "extended route successful")
                            route_edges = extended_route
                        else:
                            failed_cluster_signatures.add(cluster_sig)
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
                    try:
                        walk_path = nx.shortest_path(
                            G, drive_origin, best_start_node, weight="weight"
                        )
                        walk_edges = edges_from_path(G, walk_path)
                        walk_time = total_time(
                            walk_edges, args.pace, args.grade, args.road_pace
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

                    drive_time_tmp, drive_dist_tmp = planner_utils.estimate_drive_time_minutes(
                        drive_origin,
                        best_start_node,
                        road_graph_for_drive,
                        args.average_driving_speed_mph,
                        return_distance=True,
                    )
                    drive_time_tmp += DRIVE_PARKING_OVERHEAD_MIN
                    if (
                        drive_from_coord_for_this_candidate == home_coord
                        and drive_dist_tmp <= MIN_DRIVE_DISTANCE_MI
                    ):
                        drive_time_tmp = 0.0

                    walk_completion_time = sum(
                        planner_utils.estimate_time(e, args.pace, args.grade, args.road_pace)
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
                        and (
                            adjusted_walk <= drive_time_tmp * factor
                            or adjusted_walk - drive_time_tmp <= MIN_DRIVE_TIME_SAVINGS_MIN
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

                unplanned_macro_clusters.pop(
                    best_cluster_to_add_info["cluster_original_index"]
                )
            else:
                if unplanned_macro_clusters and todays_total_budget_minutes > 0:
                    fallback_cluster = unplanned_macro_clusters.pop(0)
                    if fallback_cluster.start_candidates:
                        start_node, start_name = fallback_cluster.start_candidates[0]
                    else:
                        start_node = fallback_cluster.edges[0].start
                        start_name = None
                    act_route_edges = plan_route(
                        G,
                        fallback_cluster.edges,
                        start_node,
                        args.pace,
                        args.grade,
                        args.road_pace,
                        args.max_road,
                        args.road_threshold,
                        path_cache,
                        use_rpp=True,
                        allow_connectors=args.allow_connector_trails,
                        debug_args=args,
                        spur_length_thresh=args.spur_length_thresh,
                        spur_road_bonus=args.spur_road_bonus,
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
            if todays_total_budget_minutes >= 240 and total_day_time > 0:
                note_parts.append(
                    "long route scheduled on weekend to utilize extra time"
                )
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
                hit_list=quick_hits,
                challenge_ids=current_challenge_segment_ids,
            )

    # Smooth the schedule if we have lightly used days and remaining clusters
    smooth_daily_plans(
        daily_plans,
        unplanned_macro_clusters,
        daily_budget_minutes,
        G,
        args.pace,
        args.grade,
        args.road_pace,
        args.max_road,
        args.road_threshold,
        path_cache,
        allow_connector_trails=args.allow_connector_trails,
        rpp_timeout=args.rpp_timeout,
        road_graph=road_graph_for_drive,
        average_driving_speed_mph=args.average_driving_speed_mph,
        home_coord=home_coord,
        debug_args=args,
        spur_length_thresh=args.spur_length_thresh,
        spur_road_bonus=args.spur_road_bonus,
    )

    # After smoothing, ensure all segments have been scheduled. If any
    # clusters remain unscheduled the plan is infeasible.
    if unplanned_macro_clusters:
        remaining_ids: Set[str] = set()
        for cluster in unplanned_macro_clusters:
            for seg in cluster.edges:
                if seg.seg_id is not None:
                    remaining_ids.add(str(seg.seg_id))

        avg_hours = (
            sum(daily_budget_minutes.values()) / len(daily_budget_minutes) / 60.0
            if daily_budget_minutes
            else 0.0
        )
        msg = (
            f"With {avg_hours:.1f} hours/day from {start_date} to {end_date}, "
            "it's impossible to complete all trails. Extend the timeframe or "
            "increase daily budget."
        )
        if remaining_ids:
            msg += " Unscheduled segment IDs: " + ", ".join(sorted(remaining_ids))
        tqdm.write(msg, file=sys.stderr)

    export_plan_files(
        daily_plans,
        args,
        hit_list=quick_hits,
        challenge_ids=current_challenge_segment_ids,
    )


if __name__ == "__main__":
    main()
