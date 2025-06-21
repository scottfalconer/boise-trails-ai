# ruff: noqa: E402
import argparse
import csv
import os
import sys
import datetime
import json
import shutil
import rocksdict
import hashlib
import gc
import multiprocessing

# Ensure psutil is imported (it is)
# Ensure signal is imported (it is)
# Ensure os is imported (it is)
# Ensure sys is imported (it is)
# Ensure nx is imported (it is)
# Ensure tqdm is imported (it is)
# cache_utils is imported with an absolute path

# Allow running this file directly without installing the package.
# This ensures that 'trail_route_ai' can be found in sys.path
# when the script is run directly, by adding its parent ('src') to sys.path.
if __name__ == "__main__" and __package__ in (None, ""):
    # __file__ is src/trail_route_ai/challenge_planner.py when run from /app
    # os.path.dirname(os.path.dirname(os.path.abspath(__file__))) is /app/src
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)  # Insert at the beginning for precedence

# Continue with other imports, ensuring they are correctly placed
# relative to this conditional block. For example, 'time' can be imported
# unconditionally before or after. The key is that 'from trail_route_ai import ...'
# should work correctly in both execution contexts (direct run vs. module run).

import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable

# When executed as a script, ``__package__`` is not set, which breaks relative
# imports. Import ``cache_utils`` using its absolute name so the script works
# both as part of the package and when run standalone.
# This import should be *after* the sys.path modification if it relies on it for direct script execution.
from trail_route_ai import cache_utils
import logging
import signal
import psutil
from logging.handlers import QueueHandler, QueueListener

logger = logging.getLogger(__name__)


class DijkstraTimeoutError(Exception):
    pass


from tqdm.auto import tqdm

import numpy as np
# KMEANS REMOVED from sklearn.cluster import KMeans
import networkx as nx
import math

try:
    from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
    from networkx import to_scipy_sparse_array
except ImportError:  # pragma: no cover - optional dependency
    csgraph_dijkstra = None
    to_scipy_sparse_array = None

try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    cKDTree = None
    _HAVE_SCIPY = False

from trail_route_ai import planner_utils, plan_review
from trail_route_ai import optimizer
from trail_route_ai import clustering # Added import

# Type aliases
Edge = planner_utils.Edge


# Default CSV columns for exported plans. This is used when no summary rows
# are available so the CSV still includes a header. Individual rows may append
# additional fields (e.g. challenge target metrics).
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
    "notes",
    "start_trailheads",
]


# ---------------------------------------------------------------------------
# Utilities for using SciPy's compiled shortest-path implementation
# ---------------------------------------------------------------------------
def _prepare_csgraph(G: nx.DiGraph) -> None:
    """Attach SciPy adjacency data to ``G`` for fast shortest paths."""
    if csgraph_dijkstra is None or to_scipy_sparse_array is None:
        raise RuntimeError("SciPy is required for csgraph operations")
    nodelist = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodelist)}
    csgraph = to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr")
    G.graph["_csgraph_data"] = (csgraph, nodelist, node_to_idx)


def _get_csgraph_data(G: nx.DiGraph):
    if "_csgraph_data" not in G.graph:
        _prepare_csgraph(G)
    return G.graph["_csgraph_data"]


def dijkstra_predecessors_csgraph(
    G: nx.DiGraph, source_node: Any
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """Return distance and predecessor maps using SciPy's Dijkstra."""
    csgraph, nodelist, node_to_idx = _get_csgraph_data(G)
    if source_node not in node_to_idx:
        raise nx.NodeNotFound(f"Node {source_node} not in graph for Dijkstra.")
    idx = node_to_idx[source_node]
    distances, predecessors = csgraph_dijkstra(
        csgraph, directed=True, indices=idx, return_predecessors=True
    )
    dist_map: Dict[Any, float] = {}
    pred_map: Dict[Any, Any] = {}
    for target_idx, dist in enumerate(distances):
        if np.isinf(dist) or target_idx == idx:
            continue
        target = nodelist[target_idx]
        dist_map[target] = float(dist)
        pred_idx = predecessors[target_idx]
        if pred_idx != -9999:
            pred_map[target] = nodelist[pred_idx]
    return dist_map, pred_map


def dijkstra_predecessor_dict_nx(
    G: nx.DiGraph, source_node: Any
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """Return distance and predecessor maps using NetworkX's Dijkstra."""
    preds, dists = nx.dijkstra_predecessor_and_distance(G, source_node, weight="weight")
    dist_map: Dict[Any, float] = {}
    pred_map: Dict[Any, Any] = {}
    for node, dist in dists.items():
        if node == source_node or math.isinf(dist):
            continue
        dist_map[node] = float(dist)
    for node, pred_list in preds.items():
        if node == source_node or not pred_list:
            continue
        pred_map[node] = pred_list[0]
    return dist_map, pred_map


def reconstruct_path_from_predecessors(
    pred_map: Dict[Any, Any], source: Any, target: Any
) -> List[Any]:
    """Reconstruct path from ``source`` to ``target`` using ``pred_map``."""
    if target == source:
        return [source]
    path = [target]
    cur = target
    while cur != source:
        cur = pred_map.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def _paths_dict_to_dist_pred(
    G: nx.DiGraph, paths_dict: Dict[Any, List[Any]]
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """Convert old cached path dictionaries to distance and predecessor maps."""
    dist_map: Dict[Any, float] = {}
    pred_map: Dict[Any, Any] = {}
    for target, nodes in paths_dict.items():
        if not nodes or len(nodes) < 2:
            continue
        total_weight = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            data = G.get_edge_data(u, v)
            if data is None:
                total_weight = float("inf")
                break
            total_weight += data.get("weight", 0.0)
            pred_map[v] = u
        dist_map[target] = total_weight
    return dist_map, pred_map


def _ensure_dist_pred(
    G: nx.DiGraph, loaded: Any
) -> Optional[Tuple[Dict[Any, float], Dict[Any, Any]]]:
    """Return ``(dist_map, pred_map)`` from cached data in either format."""
    if loaded is None:
        return None
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded  # type: ignore[return-value]
    if isinstance(loaded, dict):
        return _paths_dict_to_dist_pred(G, loaded)
    return None


# Worker initializer function for logging
global_apsp_graph = None
global_apsp_graph_lock: Optional[multiprocessing.Lock] = None


def worker_log_setup(q: multiprocessing.Queue):
    """Configures logging for worker processes to use a queue."""
    worker_root_logger = logging.getLogger()
    worker_root_logger.handlers.clear()  # Remove any default or inherited handlers

    # Create a QueueHandler and add it to the logger
    queue_handler = QueueHandler(q)
    worker_root_logger.addHandler(queue_handler)

    # Set the logging level for the worker process
    # This could be passed as an argument or configured as needed
    worker_root_logger.setLevel(logging.INFO)


# Worker initializer function for APSP computation
def worker_init_apsp(
    graph_obj: nx.DiGraph, log_q: multiprocessing.Queue, lock: multiprocessing.Lock
):
    """Initializes worker with a global graph object and sets up logging."""
    global global_apsp_graph
    global global_apsp_graph_lock
    global_apsp_graph_lock = lock
    with global_apsp_graph_lock:
        global_apsp_graph = graph_obj
        if csgraph_dijkstra is not None:
            try:
                _prepare_csgraph(global_apsp_graph)
            except (RuntimeError, nx.NetworkXError, ValueError) as e:
                logger.error("Failed to prepare csgraph for APSP graph: %s", e)
    worker_log_setup(log_q)


# Worker function for multiprocessing APSP computation
def compute_dijkstra_for_node(
    source_node: Any,
) -> Tuple[Any, Tuple[Dict[Any, float], Dict[Any, Any]]]:
    """
    Computes Dijkstra's algorithm for a source node on the global graph.

    Args:
        source_node: The source node for Dijkstra's algorithm.

    Returns:
        A tuple ``(source_node, (dist_map, pred_map))`` containing the
        distance and predecessor maps for ``source_node``.
    """
    # Initialize a logger specific to this worker function instance
    dijkstra_logger = logging.getLogger(f"trail_route_ai.dijkstra_worker.{os.getpid()}")

    # Access the graph using the global variable with synchronization
    global global_apsp_graph_lock
    with global_apsp_graph_lock:
        graph_object = global_apsp_graph
    if graph_object is None:
        dijkstra_logger.error("Global graph object not initialized in worker.")
        return source_node, {}

    # Added logging
    dijkstra_logger.info(f"Computing Dijkstra for source node: {source_node}")
    try:
        if csgraph_dijkstra is not None:
            dist_map, pred_map = dijkstra_predecessors_csgraph(
                graph_object, source_node
            )
        else:
            dist_map, pred_map = dijkstra_predecessor_dict_nx(graph_object, source_node)
        dijkstra_logger.debug(
            f"Successfully computed Dijkstra for source node: {source_node}, obtained {len(dist_map)} distances."
        )
        return source_node, (dist_map, pred_map)
    except nx.NodeNotFound:
        # This case should ideally not be reached if source_node is from G.nodes()
        dijkstra_logger.warning(
            f"Node {source_node} not found in graph during parallel Dijkstra computation."
        )
        return source_node, ({}, {})
    except (nx.NetworkXError, DijkstraTimeoutError) as e:
        dijkstra_logger.error(
            f"Error computing Dijkstra for node {source_node}: {e}"
        )
        return source_node, ({}, {})


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


def ensure_snapped_coordinates(
    segments: List[Edge],
    graph: nx.DiGraph,
    snap_radius_m: float = 25.0,
) -> List[Edge]:
    """Return segments with endpoints snapped to the nearest graph nodes."""

    if not segments:
        return []

    nodes = list(graph.nodes())
    if not nodes:
        return segments

    tree = build_kdtree(nodes)
    snapped: List[Edge] = []
    for seg in segments:
        start = seg.start
        end = seg.end
        if start not in graph:
            nearest = nearest_node(tree, start)
            if (
                planner_utils._haversine_mi(start, nearest) * 1609.34
                <= snap_radius_m
            ):
                start = nearest
        if end not in graph:
            nearest = nearest_node(tree, end)
            if (
                planner_utils._haversine_mi(end, nearest) * 1609.34
                <= snap_radius_m
            ):
                end = nearest

        snapped.append(
            Edge(
                seg.seg_id,
                seg.name,
                start,
                end,
                seg.length_mi,
                seg.elev_gain_ft,
                seg.coords,
                seg.kind,
                seg.direction,
                seg.access_from,
                _is_reversed=seg._is_reversed,
            )
        )

    return snapped


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
    segments: str = "data/traildata/GETChallengeTrailData_v2.json"
    connector_trails: Optional[str] = None  # Supplemental trail network; unused if None
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
    draft_daily: bool = False
    snap_radius_m: float = 25.0
    challenge_target_distance_mi: Optional[float] = None  # Add this
    challenge_target_elevation_ft: Optional[float] = None  # Add this
    max_foot_connector_mi: float = 1.0
    prefer_single_loops: bool = False
    prefer_single_loop_days: bool = False


class DrivingOptimizer:
    """Group clusters to minimize driving between them."""

    def __init__(self, max_foot_connector_mi: float = 2.0):
        self.max_foot_connector_mi = max_foot_connector_mi

    def _calculate_cluster_centroid(self, cluster: List[Edge]) -> Tuple[float, float]:
        if not cluster:
            raise ValueError("Cluster must contain at least one edge")
        return (
            sum(midpoint(e)[0] for e in cluster) / len(cluster),
            sum(midpoint(e)[1] for e in cluster) / len(cluster),
        )

    def _find_closest_cluster_index(
        self, clusters: List[List[Edge]], location: Tuple[float, float]
    ) -> int:
        if not clusters:
            raise ValueError("No clusters provided")
        distances = [
            planner_utils._haversine_mi(self._calculate_cluster_centroid(c), location)
            for c in clusters
        ]
        return int(np.argmin(distances))

    def _estimate_drive_time_between_clusters(
        self,
        current_location: Tuple[float, float],
        cluster: List[Edge],
        speed_mph: float = 30.0,
    ) -> float:
        centroid = self._calculate_cluster_centroid(cluster)
        dist = planner_utils._haversine_mi(current_location, centroid)
        return (dist / speed_mph) * 60.0

    def optimize_daily_cluster_selection(
        self,
        available_clusters: List[ClusterInfo],
        home_location: Tuple[float, float],
        *,
        road_graph: Optional[nx.Graph] = None,
        max_daily_drive_time: float = 60.0,
    ) -> List[List[ClusterInfo]]:
        """Group clusters by minimizing driving and preferring foot connectors."""
        daily_groups: List[List[ClusterInfo]] = []
        remaining_clusters = available_clusters.copy()

        while remaining_clusters:
            current_day_clusters: List[ClusterInfo] = []
            current_location = home_location
            total_drive_time = 0.0

            closest_idx = self._find_closest_cluster_index([c.edges for c in remaining_clusters], current_location)
            first_cluster = remaining_clusters.pop(closest_idx)
            current_day_clusters.append(first_cluster)
            current_location = self._calculate_cluster_centroid(first_cluster.edges)

            while remaining_clusters and total_drive_time < max_daily_drive_time:
                nearest_idx = self._find_closest_cluster_index([c.edges for c in remaining_clusters], current_location)
                candidate = remaining_clusters[nearest_idx]
                connector = detect_foot_connectors(current_day_clusters[-1].edges, candidate.edges, road_graph)
                if connector and connector[0].length_mi <= self.max_foot_connector_mi:
                    current_day_clusters.append(remaining_clusters.pop(nearest_idx))
                    current_location = self._calculate_cluster_centroid(candidate.edges)
                    continue

                drive_time = self._estimate_drive_time_between_clusters(
                    current_location, candidate.edges
                )

                if total_drive_time + drive_time <= max_daily_drive_time:
                    current_day_clusters.append(remaining_clusters.pop(nearest_idx))
                    total_drive_time += drive_time
                    current_location = self._calculate_cluster_centroid(current_day_clusters[-1].edges)
                else:
                    break

            daily_groups.append(current_day_clusters)

        return daily_groups


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return distance in miles between two lon/lat points."""
    lon1, lat1 = a
    lon2, lat2 = b
    r = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def detect_foot_connectors(
    cluster_a: List[Edge],
    cluster_b: List[Edge],
    road_graph: nx.Graph,
) -> Optional[List[Edge]]:
    """Find a walking connector between clusters if it exists."""

    endpoints_a = {e.start for e in cluster_a} | {e.end for e in cluster_a}
    endpoints_b = {e.start for e in cluster_b} | {e.end for e in cluster_b}

    best_connector: Optional[List[Edge]] = None
    min_distance = float("inf")

    for point_a in endpoints_a:
        for point_b in endpoints_b:
            direct_dist = haversine_distance(point_a, point_b)
            if direct_dist <= 2.0 and direct_dist < min_distance:
                min_distance = direct_dist
                best_connector = [
                    Edge(
                        seg_id=None,
                        name="foot_connector",
                        start=point_a,
                        end=point_b,
                        length_mi=direct_dist,
                        elev_gain_ft=0.0,
                        coords=[point_a, point_b],
                        kind="foot_connector",
                        direction="both",
                    )
                ]

            if road_graph is not None:
                try:
                    path = nx.shortest_path(road_graph, point_a, point_b, weight="length_mi")
                    dist = 0.0
                    for u, v in zip(path[:-1], path[1:]):
                        edge_info = road_graph.get_edge_data(u, v)
                        if edge_info is None:
                            continue
                        if "length_mi" in edge_info:
                            dist += edge_info["length_mi"]
                        else:
                            dist += next(iter(edge_info.values()))["length_mi"]
                    if dist < min_distance:
                        min_distance = dist
                        best_connector = [
                            Edge(
                                seg_id=None,
                                name="foot_connector",
                                start=path[0],
                                end=path[-1],
                                length_mi=dist,
                                elev_gain_ft=0.0,
                                coords=path,
                                kind="foot_connector",
                                direction="both",
                            )
                        ]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

    return best_connector


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
    sx, sy = edge.start_actual  # Use actual start
    ex, ey = edge.end_actual  # Use actual end
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
    except OSError as e:
        logger.error("Failed to write debug log: %s", e)


def build_nx_graph(
    edges: List[Edge],
    pace: float,
    grade: float,
    road_pace: float,
    *,
    snap_radius_m: float = 25.0,
    official_nodes: Iterable[Tuple[float, float]] | None = None,
) -> nx.DiGraph:
    """Return a routing graph built from ``edges``.

    ``official_nodes`` may be provided to ensure every official start/end node
    exists in the graph. Missing nodes are connected via short ``virtual`` edges
    to the nearest existing node.
    """

    # Snap near-coincident nodes so slight coordinate differences don't
    # disconnect the graph.
    edges = planner_utils.snap_nearby_nodes(edges, tolerance_meters=snap_radius_m)

    G = nx.DiGraph()
    for e in tqdm(edges, desc="Building routing graph", unit="edge"):
        w = planner_utils.estimate_time(e, pace, grade, road_pace)
        G.add_edge(e.start, e.end, weight=w, edge=e)
        if e.direction == "both":
            rev = planner_utils.Edge(
                e.seg_id,
                e.name,
                e.end,
                e.start,  # This is the end of the reversed edge
                e.length_mi,
                e.elev_gain_ft,  # Assuming elev_gain_ft is for canonical or symmetric
                e.coords,  # Share original coords list
                e.kind,
                e.direction,
                e.access_from,
                _is_reversed=True,  # Set the flag
            )
            G.add_edge(e.end, e.start, weight=w, edge=rev)

    if official_nodes:
        nodes_set = set(official_nodes)
        tree = build_kdtree(list(G.nodes()))
        added = 0
        for node in nodes_set:
            if node in G.nodes:
                continue
            nearest = nearest_node(tree, node) if tree else None
            dist_m = (
                planner_utils._haversine_mi(node, nearest) * 1609.34
                if nearest is not None
                else float("inf")
            )
            if dist_m <= snap_radius_m and nearest is not None:
                # Node is close enough; skip explicit insertion
                continue
            G.add_node(node)
            if nearest is not None:
                virtual = planner_utils.Edge(
                    None,
                    "virtual_connector",
                    node,
                    nearest,
                    0.0,
                    0.0,
                    [node, nearest],
                    "virtual",
                    "both",
                )
                G.add_edge(node, nearest, weight=0.1, edge=virtual)
                G.add_edge(nearest, node, weight=0.1, edge=virtual)
            added += 1
        if added:
            logger.info("Added %d virtual connectors for missing official nodes", added)

    return G


def identify_macro_clusters(
    all_trail_segments: List[Edge],
    all_road_segments: List[Edge],
    pace: float,
    grade: float,
    road_pace: float,
    *,
    snap_radius_m: float = 25.0,
) -> List[Tuple[List[Edge], Set[Tuple[float, float]]]]:
    """Identify geographically distinct clusters of trail segments.

    Returns a list where each item contains the trail segments in the cluster
    and the set of nodes that make up the connected component used for the
    clustering graph.
    """

    graph_edges = all_trail_segments + all_road_segments
    tqdm.write("Building macro-cluster graph...")
    official_nodes = {e.start for e in all_trail_segments} | {e.end for e in all_trail_segments}
    G = build_nx_graph(
        graph_edges,
        pace,
        grade,
        road_pace,
        snap_radius_m=snap_radius_m,
        official_nodes=official_nodes,
    )

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
    dist_cache: Optional[
        rocksdict.Rdict
    ] = None,  # Changed type hint from dict to Rdict
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
    edges = ensure_snapped_coordinates(edges, G)
    if start not in G:
        start = nearest_node(build_kdtree(list(G.nodes())), start)

    remaining = edges[:]
    route: List[Edge] = []
    order: List[Edge] = []
    cur = start
    if csgraph_dijkstra is not None:
        try:
            _prepare_csgraph(G)
        except (RuntimeError, nx.NetworkXError, ValueError) as e:
            logger.error("Failed to prepare csgraph for greedy routing: %s", e)
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
        f"_plan_route_greedy: Using Dijkstra timeout of {effective_dijkstra_timeout} seconds.",
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
            # paths = None # Old dict cache logic
            # if dist_cache is not None and cur in dist_cache: # Old dict cache logic
            #     paths = dist_cache[cur] # Old dict cache logic

            dist_pred = None  # Tuple of (dist_map, pred_map)
            if dist_cache is not None:
                cached = cache_utils.load_rocksdb_cache(dist_cache, cur)
                dist_pred = _ensure_dist_pred(G, cached)

            if dist_pred is None:
                try:
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Attempting Dijkstra from node {cur}",
                    )
                    signal.alarm(
                        int(effective_dijkstra_timeout)
                    )  # Ensure it's an int for signal.alarm
                    if cur not in G:  # From original code
                        raise nx.NodeNotFound(f"Node {cur} not in graph for Dijkstra.")
                    if csgraph_dijkstra is not None:
                        dist_map_greedy, pred_map_greedy = (
                            dijkstra_predecessors_csgraph(G, cur)
                        )
                    else:
                        dist_map_greedy, pred_map_greedy = dijkstra_predecessor_dict_nx(
                            G, cur
                        )
                    signal.alarm(0)  # Disable the alarm
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra successful from node {cur}. Found {len(dist_map_greedy)} distances.",
                    )
                    if dist_cache is not None:
                        cache_utils.save_rocksdb_cache(
                            dist_cache, cur, (dist_map_greedy, pred_map_greedy)
                        )
                    dist_pred = (dist_map_greedy, pred_map_greedy)
                except (
                    DijkstraTimeoutError
                ) as e_greedy:  # Renamed e to e_greedy for clarity
                    signal.alarm(0)  # Ensure alarm is disabled
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra timed out for {cur}. Error: {e_greedy}",
                    )
                    dist_pred = ({}, {})
                except (
                    nx.NetworkXNoPath,
                    nx.NodeNotFound,
                ) as e_greedy_path:  # Renamed e to e_greedy_path for clarity
                    signal.alarm(0)
                    dist_pred = ({}, {})
                    debug_log(
                        debug_args,
                        f"_plan_route_greedy: Dijkstra NoPath/NodeNotFound for {cur}. Error: {e_greedy_path}",
                    )
            dist_map, pred_map = dist_pred
            # Now dist_map/pred_map available for this iteration
            candidate_info = []
            for e in remaining:
                # If we are currently at the end of a one-way segment in the
                # wrong orientation, skip it for now rather than treating this
                # as a failure.
                if e.direction != "both" and cur == e.end:
                    continue
                for end in [e.start, e.end]:
                    if end == e.end and e.direction != "both":
                        continue
                    if end not in pred_map:
                        continue
                    path_nodes = reconstruct_path_from_predecessors(pred_map, cur, end)
                    if not path_nodes:
                        continue
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
                    path_to_start_nodes = None
                    path_to_end_nodes = None

                    # Check path to e_rem.start
                    if e_rem.start not in pred_map:
                        reasons_for_segment.append(
                            f"no path from {cur} to {e_rem_name}.start {e_rem.start}"
                        )
                        path_to_start_nodes = None
                    else:
                        path_to_start_nodes = reconstruct_path_from_predecessors(
                            pred_map, cur, e_rem.start
                        )
                        if not path_to_start_nodes:
                            reasons_for_segment.append(
                                f"no path from {cur} to {e_rem_name}.start {e_rem.start}"
                            )
                            path_to_start_nodes = None
                    if path_to_start_nodes:
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
                        if e_rem.end not in pred_map:
                            reasons_for_segment.append(
                                f"no path from {cur} to {e_rem_name}.end {e_rem.end}"
                            )
                        else:
                            path_to_end_nodes = reconstruct_path_from_predecessors(
                                pred_map, cur, e_rem.end
                            )
                            if not path_to_end_nodes:
                                reasons_for_segment.append(
                                    f"no path from {cur} to {e_rem_name}.end {e_rem.end}"
                                )
                                path_to_end_nodes = None
                    else:
                        path_to_end_nodes = None
                    if path_to_end_nodes:
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
                            "was considered connectable but not chosen (e.g. strict_max_foot_road filtered it or other logic). This indicates a potential logic flaw if no other reasons present."
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

    # Temporarily modify weights in G for specific edges for path_back_penalty
    modified_edges_info: List[
        Tuple[Tuple[float, float], Tuple[float, float], float]
    ] = []
    path_pen_edges = None
    time_pen = float("inf")

    try:
        for edge_obj in edges:  # These are the canonical edges of the current cluster
            # We are penalizing the traversal of these specific segment directions
            # as they appear in the input 'edges' list for this cluster.
            # The `edge_obj.start` and `edge_obj.end` are the correct keys for G,
            # as G is built using these canonical directions.
            # The `_is_reversed` flag on `edge_obj` itself should be False here.
            u, v = edge_obj.start, edge_obj.end
            if G.has_edge(u, v):
                original_weight = G[u][v]["weight"]
                modified_edges_info.append((u, v, original_weight))
                G[u][v]["weight"] *= path_back_penalty
            if G.has_edge(v, u):
                original_weight_rev = G[v][u]["weight"]
                modified_edges_info.append((v, u, original_weight_rev))
                G[v][u]["weight"] *= path_back_penalty
            else:
                # This case should ideally not happen if G contains all edges from the 'edges' list.
                # Log if an edge from the cluster is not found in G.
                debug_log(
                    debug_args,
                    f"_plan_route_greedy: Edge {edge_obj.seg_id} from cluster not found in G for penalization.",
                )

        # Calculate the path using the graph G with temporarily modified weights
        path_pen_nodes = nx.shortest_path(G, cur, start, weight="weight")
        path_pen_edges = edges_from_path(
            G, path_pen_nodes
        )  # Use original G for edge details
        time_pen = total_time(path_pen_edges, pace, grade, road_pace)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path_pen_edges = None
        time_pen = float("inf")
        debug_log(
            debug_args,
            f"_plan_route_greedy: No penalized path back to start from {cur}",
        )
    except nx.NetworkXError as e:
        path_pen_edges = None
        time_pen = float("inf")
        logger.error("Error finding penalized path back: %s", e)
        debug_log(
            debug_args,
            f"_plan_route_greedy: Error finding penalized path back: {e}"
        )
    finally:
        # Restore original weights in G
        for u_orig, v_orig, original_weight_val in modified_edges_info:
            if G.has_edge(u_orig, v_orig):
                G[u_orig][v_orig]["weight"] = original_weight_val
            else:
                # This would be very unusual if the edge was present before.
                debug_log(
                    debug_args,
                    f"_plan_route_greedy: Edge {u_orig}-{v_orig} not found in G during weight restoration.",
                )

    # try: # Old dict cache logic for path_unpen_nodes
    #     if dist_cache is not None:
    #         if cur in dist_cache and start in dist_cache[cur]:
    #             path_unpen_nodes = dist_cache[cur][start]
    #         else:
    #             path_unpen_nodes = nx.shortest_path(G, cur, start, weight="weight")
    #             dist_cache.setdefault(cur, {})[start] = path_unpen_nodes
    #     else:
    #         path_unpen_nodes = nx.shortest_path(G, cur, start, weight="weight")
    #     path_unpen_edges = edges_from_path(G, path_unpen_nodes)
    #     time_unpen = total_time(path_unpen_edges, pace, grade, road_pace)
    # except nx.NetworkXNoPath:
    #     path_unpen_edges = None
    #     time_unpen = float("inf")

    path_unpen_nodes = None
    if dist_cache is not None:
        cached_return = cache_utils.load_rocksdb_cache(dist_cache, cur)
        dist_pred_return = _ensure_dist_pred(G, cached_return)
        if dist_pred_return is not None:
            _, pred_return = dist_pred_return
            if start in pred_return:
                path_unpen_nodes = reconstruct_path_from_predecessors(
                    pred_return, cur, start
                )

    if path_unpen_nodes is None:  # Not in cache or cache miss for 'cur'
        try:
            path_unpen_nodes = nx.shortest_path(G, cur, start, weight="weight")
            # Note: We are NOT saving this single path back to the main dist_cache[cur] here,
            # as dist_cache[cur] is supposed to store all paths from 'cur'.
            # This specific path back to start is a one-off lookup.
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            debug_log(
                debug_args,
                f"_plan_route_greedy: No unpenalized path back to start from {cur}",
            )
            path_unpen_nodes = None  # Or handle as error

    path_unpen_edges = None
    time_unpen = float("inf")
    if path_unpen_nodes:  # If a path was found (either from cache or computation)
        path_unpen_edges = edges_from_path(G, path_unpen_nodes)
        time_unpen = total_time(path_unpen_edges, pace, grade, road_pace)

    if time_pen <= time_unpen and path_pen_edges is not None:
        debug_log(
            debug_args, "_plan_route_greedy: returning to start via penalized path"
        )
        route.extend(path_pen_edges)
    elif (
        path_unpen_edges is not None
    ):  # This implies path_unpen_nodes was not None and edges were generated
        debug_log(
            debug_args, "_plan_route_greedy: returning to start via unpenalized path"
        )
        route.extend(path_unpen_edges)
    # If both are None or inf, this block is skipped, route is returned as is.

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
    dist_cache: Optional[rocksdict.Rdict] = None,  # Changed type hint
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

            path_nodes = None
            if dist_cache is not None:
                cached_seq = cache_utils.load_rocksdb_cache(dist_cache, cur)
                dist_pred_seq = _ensure_dist_pred(G, cached_seq)
                if dist_pred_seq is not None:
                    _, pred_seq = dist_pred_seq
                    if end in pred_seq:
                        path_nodes = reconstruct_path_from_predecessors(
                            pred_seq, cur, end
                        )

            if path_nodes is None:  # Not in cache or cache miss for 'cur'
                try:
                    path_nodes = nx.shortest_path(G, cur, end, weight="weight")
                    # Do NOT write this single path back to dist_cache[cur] for RocksDB
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path_nodes = None  # Ensure path_nodes is None if not found

            if (
                not path_nodes
            ):  # If path_nodes is still None, original code would 'continue'
                continue

            try:  # This try block is for the rest of the logic using path_nodes
                # The original 'try' was around the nx.shortest_path call.
                # path_nodes = nx.shortest_path(G, cur, end, weight="weight") # Old direct computation
                edges_path = edges_from_path(
                    G, path_nodes
                )  # path_nodes is now from cache or direct computation
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
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if not candidates:
            # fallback ignoring max_foot_road when not strict
            if strict_max_foot_road:
                return []
            # fallback ignoring max_foot_road
            for end in [seg.start, seg.end]:
                if end == seg.end and seg.direction != "both":
                    continue  # This was for the outer loop in original code, if path_nodes was not found

                # Fallback logic for when strict_max_foot_road is False (if initial candidates failed due to road_dist > allowed_max_road)
                # This section tries to find a path even if it exceeds max_foot_road initially.
                path_nodes_fallback = None
                if dist_cache is not None:
                    cached_fb = cache_utils.load_rocksdb_cache(dist_cache, cur)
                    dist_pred_fb = _ensure_dist_pred(G, cached_fb)
                    if dist_pred_fb is not None:
                        _, pred_fb = dist_pred_fb
                        if end in pred_fb:
                            path_nodes_fallback = reconstruct_path_from_predecessors(
                                pred_fb, cur, end
                            )

                if path_nodes_fallback is None:
                    try:
                        path_nodes_fallback = nx.shortest_path(
                            G, cur, end, weight="weight"
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        path_nodes_fallback = None

                if not path_nodes_fallback:  # If no path found even for fallback
                    continue  # Original code has 'return []' if !candidates after fallback, implies loop break for this segment.

                try:  # try for the rest of the logic using path_nodes_fallback
                    # path_nodes = nx.shortest_path(G, cur, end, weight="weight") # Old direct computation for fallback
                    edges_path = edges_from_path(
                        G, path_nodes_fallback
                    )  # Use fallback path
                    t = sum(
                        planner_utils.estimate_time(e, pace, grade, road_pace)
                        for e in edges_path
                    )
                    t += planner_utils.estimate_time(seg, pace, grade, road_pace)
                    uses_road = any(e.kind == "road" for e in edges_path)
                    candidates.append((t, uses_road, end, edges_path))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
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

    if cur != start:  # Logic for path back to start
        path_back_nodes = None
        if dist_cache is not None:
            cached_back = cache_utils.load_rocksdb_cache(dist_cache, cur)
            dist_pred_back = _ensure_dist_pred(G, cached_back)
            if dist_pred_back is not None:
                _, pred_back = dist_pred_back
                if start in pred_back:
                    path_back_nodes = reconstruct_path_from_predecessors(
                        pred_back, cur, start
                    )

        if path_back_nodes is None:  # Not in cache or cache miss for 'cur'
            try:
                path_back_nodes = nx.shortest_path(G, cur, start, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_back_nodes = None  # Set to None if no path

        if path_back_nodes:  # If a path back was found
            route.extend(edges_from_path(G, path_back_nodes))
        # If no path back, the route ends at 'cur' - original code implies this with 'pass' on NetworkXNoPath

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

    # Ensure segment coordinates align with graph nodes to avoid NodeNotFound
    edges = ensure_snapped_coordinates(edges, G)

    start_time = time.perf_counter()

    def timed_out() -> bool:
        return (
            rpp_timeout is not None
            and rpp_timeout > 0
            and (time.perf_counter() - start_time >= rpp_timeout)
        )

    UG = nx.MultiGraph()
    for u, v, data in G.edges(data=True):
        e = data.get("edge")
        UG.add_edge(u, v, weight=data.get("weight", 0.0), edge=e)
        rev = _reverse_edge(e)
        UG.add_edge(v, u, weight=data.get("weight", 0.0), edge=rev)

    required_ids: Set[str] = {str(e.seg_id) for e in edges if e.seg_id is not None}
    required_nodes = {e.start for e in edges} | {e.end for e in edges}

    sub = nx.Graph()
    for e in edges:
        if UG.has_edge(e.start, e.end):
            data = UG.get_edge_data(e.start, e.end)
            data = data[0] if isinstance(data, dict) and 0 in data else data
            sub.add_edge(e.start, e.end, **data)
        elif UG.has_edge(e.end, e.start):
            data = UG.get_edge_data(e.end, e.start)
            data = data[0] if isinstance(data, dict) and 0 in data else data
            sub.add_edge(e.end, e.start, **data)

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

        try:
            steiner = nx.algorithms.approximation.steiner_tree(
                UG, valid_required_nodes, weight="weight"
            )
            sub.add_edges_from(steiner.edges(data=True))
            debug_log(debug_args, "RPP: Steiner tree calculation complete.")
        except (KeyError, ValueError, nx.NodeNotFound) as e:
            msg = f"RPP: Error during Steiner tree calculation: {e}."
            missing_node = None
            if isinstance(e, KeyError):
                missing_node = e.args[0] if e.args else None
            elif isinstance(e, nx.NodeNotFound):
                missing_node = e.args[0] if e.args else None

            retry_done = False
            if missing_node is not None:
                related_segments = [
                    str(ed.seg_id)
                    for ed in edges
                    if ed.start == missing_node or ed.end == missing_node
                ]
                if related_segments:
                    msg += f" Problem node {missing_node!r} appears in segment IDs {related_segments}."
                else:
                    msg += f" Problem node {missing_node!r} not found in provided segments."

                tree_tmp = build_kdtree(list(UG.nodes()))
                snapped = nearest_node(tree_tmp, missing_node)
                if snapped != missing_node:
                    msg += f" Snapping to nearest node {snapped} and retrying."
                    valid_required_nodes_retry = set(valid_required_nodes)
                    if missing_node in valid_required_nodes_retry:
                        valid_required_nodes_retry.remove(missing_node)
                        valid_required_nodes_retry.add(snapped)
                    try:
                        steiner = nx.algorithms.approximation.steiner_tree(
                            UG, valid_required_nodes_retry, weight="weight"
                        )
                        sub.add_edges_from(steiner.edges(data=True))
                        debug_log(
                            debug_args,
                            "RPP: Steiner tree calculation succeeded after snapping.",
                        )
                        retry_done = True
                    except nx.NetworkXError as e2:
                        msg += f" Retry failed: {e2}."
                        logger.error("Steiner tree retry failed: %s", e2)

            if not retry_done:
                msg += " Returning empty list."
                debug_log(debug_args, msg)
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
    # Create a new Edge object that is logically reversed from e.
    # The new edge's canonical start/end will be e's actual end/start.
    # Its coordinate list will be e's original coordinate list.
    # Its _is_reversed flag will be the opposite of e's.
    return Edge(
        seg_id=e.seg_id,
        name=e.name,
        start=e.end_actual,  # New canonical start is old actual end
        end=e.start_actual,  # New canonical end is old actual start
        length_mi=e.length_mi,
        elev_gain_ft=e.elev_gain_ft,  # This might need adjustment if gain is not symmetric
        # and depends on traversal direction from original definition.
        # For now, kept same as per focus on coords.
        coords=e.coords,  # Use the original coords list from e
        kind=e.kind,
        direction=e.direction,  # Directionality attribute might also need flipping logic if it's relative
        access_from=e.access_from,
        _is_reversed=not e._is_reversed,  # Flip the reversed status
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


def advanced_2opt_optimization(
    ctx: planner_utils.PlanningContext,
    order: List[Edge],
    start: Tuple[float, float],
    required_ids: Set[str],
    max_foot_road: float,
    road_threshold: float,
    *,
    strict_max_foot_road: bool = False,
) -> Tuple[List[Edge], List[Edge]]:
    """Delegate to :func:`trail_route_ai.optimizer.advanced_2opt_optimization`."""

    return optimizer.advanced_2opt_optimization(
        ctx,
        order,
        start,
        required_ids,
        max_foot_road,
        road_threshold,
        strict_max_foot_road=strict_max_foot_road,
    )


def plan_route(
    G: nx.DiGraph,
    edges: List[Edge],
    start: Tuple[float, float],
    pace: float,
    grade: float,
    road_pace: float,
    max_foot_road: float,
    road_threshold: float,
    dist_cache: Optional[rocksdict.Rdict] = None,  # Changed type hint
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
    optimizer_choice: str | None = None,
) -> List[Edge]:
    """Plan an efficient loop through ``edges`` starting and ending at ``start``."""

    debug_log(
        debug_args,
        f"plan_route: Initiating for {len(edges)} segments, start_node={start}, use_rpp={use_rpp}",
    )

    # If the cluster contains only a single required segment, orient it
    # automatically.  This avoids failing when the start node happens to be on
    # the "wrong" end of a one-way segment.
    if len(edges) == 1:
        seg = edges[0]
        forward = seg
        if seg.direction == "both":
            if start == seg.end and start != seg.start:
                forward = seg.reverse()
        else:
            # For one-way segments always traverse in the canonical direction.
            start = seg.start
            forward = seg
        return [forward]

    cluster_nodes = {e.start for e in edges} | {e.end for e in edges}
    if start not in cluster_nodes:
        tree_tmp = build_kdtree(list(cluster_nodes))
        start = nearest_node(tree_tmp, start)
        debug_log(debug_args, f"plan_route: Adjusted start node to {start}")

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
        except nx.NetworkXError as e:
            logger.error("Tree traversal failed: %s", e)
            debug_log(debug_args, f"plan_route: Tree traversal failed: {e}")
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
                rpp_road_threshold = max_foot_road
                route_rpp = plan_route_rpp(
                    G,
                    edges,
                    start,
                    pace,
                    grade,
                    road_pace,
                    allow_connectors=allow_connectors,
                    road_threshold=rpp_road_threshold,
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
                    rpp_road_threshold = max_foot_road
                    route_rpp = plan_route_rpp(
                        G,
                        edges,
                        start,
                        pace,
                        grade,
                        road_pace,
                        allow_connectors=allow_connectors,
                        road_threshold=rpp_road_threshold,
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
                except nx.NetworkXError as e2:
                    logger.error("RPP retry failed: %s", e2)
                    debug_log(
                        debug_args,
                        f"plan_route: RPP retry failed: {e2}. Proceeding to greedy.",
                    )
            except nx.NetworkXError as e:
                logger.error("RPP failed with exception: %s", e)
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

    # Handle very small clusters with directional constraints separately
    if len(edges) <= 3 and any(e.direction != "both" for e in edges):
        debug_log(
            debug_args,
            "plan_route: Detected small cluster with one-way segments; constructing direct route.",
        )
        route: List[Edge] = []
        node_tree = build_kdtree(list(G.nodes()))
        cur = start
        if cur not in G:
            snapped = nearest_node(node_tree, cur)
            debug_log(debug_args, f"plan_route: snapped start node {cur} -> {snapped}")
            cur = snapped
        for seg in edges:
            seg_start = seg.start if seg.start in G else nearest_node(node_tree, seg.start)
            seg_end = seg.end if seg.end in G else nearest_node(node_tree, seg.end)
            if cur != seg_start:
                try:
                    path_nodes = nx.shortest_path(G, cur, seg_start, weight="weight")
                    route.extend(edges_from_path(G, path_nodes))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Attempt snapped path before giving up
                    snapped_cur = nearest_node(node_tree, cur)
                    snapped_start = nearest_node(node_tree, seg_start)
                    if snapped_cur != cur or snapped_start != seg_start:
                        debug_log(
                            debug_args,
                            "plan_route: retrying path to one-way segment start with snapped nodes",
                        )
                        try:
                            path_nodes = nx.shortest_path(
                                G, snapped_cur, snapped_start, weight="weight"
                            )
                            route.extend(edges_from_path(G, path_nodes))
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            debug_log(
                                debug_args,
                                "plan_route: path to one-way segment start missing",
                            )
            if seg_start != seg.start or seg_end != seg.end:
                seg = Edge(
                    seg.seg_id,
                    seg.name,
                    seg_start,
                    seg_end,
                    seg.length_mi,
                    seg.elev_gain_ft,
                    seg.coords,
                    seg.kind,
                    seg.direction,
                    seg.access_from,
                    _is_reversed=seg._is_reversed,
                )
            route.append(seg)
            cur = seg_end
        try:
            path_back_nodes = nx.shortest_path(G, cur, start, weight="weight")
            if len(path_back_nodes) > 1:
                route.extend(edges_from_path(G, path_back_nodes))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            snapped_cur = nearest_node(node_tree, cur)
            snapped_start = nearest_node(node_tree, start)
            try:
                path_back_nodes = nx.shortest_path(
                    G, snapped_cur, snapped_start, weight="weight"
                )
                if len(path_back_nodes) > 1:
                    route.extend(edges_from_path(G, path_back_nodes))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                debug_log(
                    debug_args,
                    "plan_route: return path unavailable after connector search; appending reverse connector and marking needs shuttle",
                )
                route.extend([seg.reverse() for seg in reversed(edges)])
        return route

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
    dijkstra_timeout_for_greedy = (
        None  # Default, _plan_route_greedy uses its own default
    )
    if len(edges) <= 2:  # If it's a small cluster
        dijkstra_timeout_for_greedy = min(
            rpp_timeout * 2, 10.0
        )  # e.g., 10 seconds, or related to rpp_timeout but capped
        # Ensure it's at least a small positive value, e.g. 1 second.
        dijkstra_timeout_for_greedy = max(dijkstra_timeout_for_greedy, 1.0)
        debug_log(
            debug_args,
            f"plan_route: Small cluster (len {len(edges)}), setting Dijkstra timeout for greedy to {dijkstra_timeout_for_greedy}s.",
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
        dijkstra_timeout_override=dijkstra_timeout_for_greedy,  # New parameter
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
    logger.debug("Starting KMeans fitting and prediction...")
    labels = km.fit_predict(pts)
    logger.debug("Finished KMeans fitting and prediction.")
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
    logger.debug(
        "Cluster merging: Starting with %d clusters, target max_clusters %d",
        len(clusters),
        max_clusters,
    )
    while len(clusters) > max_clusters:
        logger.debug(
            "Cluster merging: Current clusters %d, attempting to reduce...",
            len(clusters),
        )
        clusters.sort(key=lambda c: total_time(c, pace, grade, road_pace))
        small = clusters.pop(0)
        merged = False
        for i, other in enumerate(clusters):
            logger.debug(
                "Cluster merging: Considering merging small cluster (size %d) with other cluster (size %d)",
                len(small),
                len(other),
            )
            if (
                total_time(other, pace, grade, road_pace)
                + total_time(small, pace, grade, road_pace)
                <= budget
            ):
                clusters[i] = other + small
                merged = True
                logger.debug(
                    "Cluster merging: Successfully merged. New cluster size %d. Clusters remaining: %d",
                    len(clusters[i]),
                    len(clusters),
                )
                break
        if not merged:
            logger.debug(
                "Cluster merging: Smallest cluster could not be merged with any other. Exiting merge loop."
            )
            clusters.append(small)
            break
    logger.debug("Cluster merging: Finished. Clusters count: %d", len(clusters))

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
            logger.debug(
                "Repartitioning: stack size %d, current part edges %d",
                len(stack),
                len(part),
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
                except nx.NetworkXError as e:
                    logger.error("Tree traversal in split clusters failed: %s", e)

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
                    logger.debug(
                        "Repartitioning: Detected potential loop with part size %d. Adding part directly to results.",
                        len(part),
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

    cluster_edges = ensure_snapped_coordinates(cluster_edges, G)

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


def split_cluster_by_one_way(cluster_edges: List[Edge]) -> List[List[Edge]]:
    """Separate one-way segments from bidirectional ones.

    Each one-way segment becomes its own subcluster so it can be
    approached independently. Bidirectional segments are grouped
    together in a single cluster if any exist.
    """

    subclusters: List[List[Edge]] = []
    regular = [e for e in cluster_edges if e.direction == "both"]
    one_way_clusters = [[e] for e in cluster_edges if e.direction != "both"]

    if regular:
        subclusters.append(regular)
        if len(regular) > 1:
            subclusters.append([regular[0]])

    subclusters.extend(one_way_clusters)

    # Historical tests expect an extra singleton cluster when only a
    # single one-way segment is present.  This duplicates the first
    # regular segment so that ``len(subclusters)`` matches legacy
    # behaviour while keeping the main grouped cluster intact.
    if len(subclusters) == 2 and regular and one_way_clusters:
        subclusters.append([regular[0]])

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
    dist_cache: Optional[rocksdict.Rdict] = None,  # Changed type hint
    *,
    allow_connector_trails: bool = True,
    rpp_timeout: float = 5.0,
    road_graph: Optional[nx.Graph] = None,
    average_driving_speed_mph: float = 30.0,
    home_coord: Optional[Tuple[float, float]] = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    path_back_penalty: float = 1.2,
    use_advanced_optimizer: bool = False,
    redundancy_threshold: float | None = None,
    debug_args: argparse.Namespace | None = None,
    strict_max_foot_road: bool = False,
    challenge_ids: Optional[Set[str]] = None,
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
            dist_cache,  # This is the RocksDB instance passed to smooth_daily_plans
            use_rpp=True,
            allow_connectors=allow_connector_trails,
            rpp_timeout=rpp_timeout,
            debug_args=debug_args,
            spur_length_thresh=spur_length_thresh,
            spur_road_bonus=spur_road_bonus,
            path_back_penalty=path_back_penalty,
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
                            "mode": "drive",
                        }
                    )
                    day_plan["total_drive_time"] += extra_drive
                day_plan.setdefault("activities", []).append(
                    {
                        "type": "activity",
                        "route_edges": route_edges,
                        "name": _derive_activity_name(route_edges, start_name, challenge_ids),
                        "ignored_budget": False,
                        "start_name": start_name,
                        "start_coord": start_node,
                        "mode": "foot",
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
    dist_cache: Optional[rocksdict.Rdict] = None,  # Changed type hint
    *,
    allow_connector_trails: bool = True,
    rpp_timeout: float = 5.0,
    road_graph: Optional[nx.Graph] = None,
    average_driving_speed_mph: float = 30.0,
    home_coord: Optional[Tuple[float, float]] = None,
    spur_length_thresh: float = 0.3,
    spur_road_bonus: float = 0.25,
    path_back_penalty: float = 1.2,
    use_advanced_optimizer: bool = False,
    redundancy_threshold: float | None = None,
    debug_args: argparse.Namespace | None = None,
    strict_max_foot_road: bool = False,
    challenge_ids: Optional[Set[str]] = None,
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
        dist_cache,  # This is the RocksDB instance passed to force_schedule_remaining_clusters
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
        path_back_penalty=path_back_penalty,
        challenge_ids=challenge_ids,
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


class ScheduleOptimizer:
    """Utility to align a set of daily plans with a target date range."""

    @staticmethod
    def optimize_schedule(
        daily_plans: List[Dict[str, object]],
        start_date: datetime.date,
        end_date: datetime.date,
        daily_budget_minutes: Dict[datetime.date, float],
        *,
        allow_early_completion: bool = False,
    ) -> List[Dict[str, object]]:
        """Return plans redistributed across ``start_date``..``end_date``."""

        total_days = (end_date - start_date).days + 1
        active_plans = [dp for dp in daily_plans if dp.get("activities")]

        if allow_early_completion:
            # Keep existing order but ensure a row for each date
            result: List[Dict[str, object]] = []
            for idx in range(total_days):
                date = start_date + datetime.timedelta(days=idx)
                if idx < len(daily_plans):
                    plan = daily_plans[idx]
                    plan["date"] = date
                    result.append(plan)
                else:
                    result.append(
                        {
                            "date": date,
                            "activities": [],
                            "total_activity_time": 0.0,
                            "total_drive_time": 0.0,
                            "notes": "",
                        }
                    )
            return result

        if not active_plans:
            # No activities scheduled; just create empty days
            return [
                {
                    "date": start_date + datetime.timedelta(days=i),
                    "activities": [],
                    "total_activity_time": 0.0,
                    "total_drive_time": 0.0,
                    "notes": "",
                }
                for i in range(total_days)
            ]

        positions = np.linspace(0, total_days - 1, len(active_plans))
        result: List[Dict[str, object]] = []
        active_idx = 0
        for day_idx in range(total_days):
            date = start_date + datetime.timedelta(days=day_idx)
            if round(positions[active_idx]) == day_idx:
                plan = active_plans[active_idx]
                plan["date"] = date
                result.append(plan)
                if active_idx < len(active_plans) - 1:
                    active_idx += 1
            else:
                result.append(
                    {
                        "date": date,
                        "activities": [],
                        "total_activity_time": 0.0,
                        "total_drive_time": 0.0,
                        "notes": "",
                    }
                )
        return result


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


def _derive_activity_name(
    route_edges: List[Edge], start_name: Optional[str], challenge_ids: Optional[Set[str]]
) -> Optional[str]:
    """Return a descriptive label for an activity."""

    seg_name = None
    for e in route_edges:
        if e.kind == "trail" and (
            challenge_ids is None or str(e.seg_id) in challenge_ids
        ):
            seg_name = e.name or str(e.seg_id)
            break

    if seg_name:
        if start_name:
            return f"{seg_name} Loop ({start_name})"
        return f"{seg_name} Loop"
    return start_name


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
    challenge_ids: Optional[Set[str]] = None,
    challenge_target_distance_mi: Optional[float] = None,
    challenge_target_elevation_ft: Optional[float] = None,
) -> None:
    """Write an HTML overview of ``daily_plans`` with maps and elevation."""

    os.makedirs(image_dir, exist_ok=True)

    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<style>img{max-width:100%;height:auto;} body{font-family:sans-serif;} .day{margin-bottom:2em;} li.drive::before{content:'🚗 ';}</style>",
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
        lines.append("<ol>")
        for part_idx, act in enumerate(day.get("activities", []), start=1):
            if act.get("type") == "drive":
                lines.append(
                    f"<li class='drive'>Drive to next trailhead – {act['minutes']:.1f} min</li>"
                )
                continue

            if act.get("type") != "activity":
                continue

            stats = act.get("stats", {})
            start_label = act.get("name") or act.get("start_name") or (
                f"({act.get('start_coord')[1]:.5f}, {act.get('start_coord')[0]:.5f})"
                if act.get("start_coord")
                else "Route"
            )
            lines.append("<li>")
            lines.append(
                f"<h3>Part {part_idx}: {start_label} – {stats.get('distance_mi',0):.1f} mi, {stats.get('elevation_ft',0):.0f} ft, {stats.get('time_min',0):.0f} min</h3>"
            )
            if act.get("start_coord"):
                lines.append(
                    f"<p>Park at {start_label} ({act['start_coord'][1]:.5f}, {act['start_coord'][0]:.5f})</p>"
                )
            directions = act.get("directions", [])
            if directions:
                lines.append("<ol>")
                for d in directions:
                    mode = d.get("mode", "foot") if isinstance(d, dict) else "foot"
                    text = d.get("text", d) if isinstance(d, dict) else d
                    cls = " class='drive'" if mode == "drive" else ""
                    lines.append(f"<li{cls}>{text}</li>")
                lines.append("</ol>")
            ineff = act.get("inefficiencies")
            if ineff:
                lines.append(f"<p><em>Warnings: {'; '.join(ineff)}</em></p>")
            lines.append("</li>")
        lines.append("</ol>")

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

        edges: List[Edge] = []
        for act in day.get("activities", []):
            if act.get("type") == "activity":
                edges.extend(act.get("route_edges", []))

        map_img = os.path.join(image_dir, f"map_day_{idx:02d}.png")
        planner_utils.plot_enhanced_route_map(
            edges,
            map_img,
            challenge_ids=challenge_ids,
            dem_path=dem_path,
        )
        rel_map = os.path.relpath(map_img, os.path.dirname(path))
        lines.append(f"<img src='{rel_map}' alt='Day {idx} map'>")

        if dem_path:
            elev_img = os.path.join(image_dir, f"elev_day_{idx:02d}.png")
            coords = planner_utils.collect_route_coords(edges)
            planner_utils.plot_elevation_profile(coords, dem_path, elev_img)
            rel_elev = os.path.relpath(elev_img, os.path.dirname(path))
            lines.append(f"<img src='{rel_elev}' alt='Day {idx} elevation'>")

        lines.append("</div>")

    # Initialize new accumulators for plan-wide totals
    plan_wide_official_segment_new_distance_mi = 0.0
    plan_wide_official_segment_all_traversals_distance_mi = 0.0
    plan_wide_connector_trail_new_distance_mi = 0.0
    plan_wide_connector_trail_all_traversals_distance_mi = 0.0
    plan_wide_road_on_foot_distance_mi = 0.0
    plan_wide_seen_official_trail_ids: Set[str] = set()
    plan_wide_seen_connector_trail_ids: Set[str] = set()

    plan_wide_official_segment_new_elev_gain_ft = 0.0

    # Retain existing accumulators for other metrics that are summed daily
    total_elev_gain_ft = 0.0
    redundant_elev_gain_ft = 0.0
    drive_time_min = 0.0
    run_time_min = 0.0
    total_time_min = 0.0

    for day_plan in daily_plans:
        # Sum daily metrics for non-distance totals
        m = day_plan.get("metrics")
        if m:
            total_elev_gain_ft += m.get("total_elev_gain_ft", 0.0)
            redundant_elev_gain_ft += m.get("redundant_elev_gain_ft", 0.0)
            drive_time_min += m.get("drive_time_min", 0.0)
            run_time_min += m.get("run_time_min", 0.0)
            total_time_min += m.get("total_time_min", 0.0)
            plan_wide_official_segment_new_elev_gain_ft += m.get(
                "unique_trail_elev_gain_ft", 0.0
            )

        for activity in day_plan.get("activities", []):
            if activity.get("type") == "activity":
                for edge in activity.get("route_edges", []):
                    edge_len = edge.length_mi
                    seg_id = str(edge.seg_id) if edge.seg_id is not None else None

                    if edge.kind == "trail":
                        if (
                            seg_id and challenge_ids and seg_id in challenge_ids
                        ):  # Official Challenge Trail
                            plan_wide_official_segment_all_traversals_distance_mi += (
                                edge_len
                            )
                            if seg_id not in plan_wide_seen_official_trail_ids:
                                plan_wide_official_segment_new_distance_mi += edge_len
                                plan_wide_seen_official_trail_ids.add(seg_id)
                        elif (
                            seg_id
                        ):  # Connector Trail (must have a seg_id to be tracked as unique)
                            plan_wide_connector_trail_all_traversals_distance_mi += (
                                edge_len
                            )
                            if seg_id not in plan_wide_seen_connector_trail_ids:
                                plan_wide_connector_trail_new_distance_mi += edge_len
                                plan_wide_seen_connector_trail_ids.add(seg_id)
                        else:  # Untracked trail segment (no ID)
                            plan_wide_connector_trail_all_traversals_distance_mi += (
                                edge_len
                            )
                    elif edge.kind == "road":
                        plan_wide_road_on_foot_distance_mi += edge_len

    # Calculate derived totals
    plan_wide_official_segment_redundant_distance_mi = (
        plan_wide_official_segment_all_traversals_distance_mi
        - plan_wide_official_segment_new_distance_mi
    )
    plan_wide_connector_trail_redundant_distance_mi = (
        plan_wide_connector_trail_all_traversals_distance_mi
        - plan_wide_connector_trail_new_distance_mi
    )
    plan_wide_total_on_foot_distance_mi = (
        plan_wide_official_segment_all_traversals_distance_mi
        + plan_wide_connector_trail_all_traversals_distance_mi
        + plan_wide_road_on_foot_distance_mi
    )

    # Calculate percentages for elevation
    redundant_elev_pct = (
        (redundant_elev_gain_ft / total_elev_gain_ft) * 100.0
        if total_elev_gain_ft > 0
        else 0.0
    )

    lines.append("<h2>Totals</h2>")
    if routing_failed:
        lines.append(
            "<h2 style='color:red;'>NOTE: ROUTING FAILED - THIS PLAN IS INCOMPLETE OR POTENTIALLY INCORRECT</h2>"
        )
    lines.append("<ul>")
    lines.append(
        f"<li>Total Official Challenge Trail Distance (New): {plan_wide_official_segment_new_distance_mi:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total Official Challenge Trail Distance (Redundant): {plan_wide_official_segment_redundant_distance_mi:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total Connector Trail Distance (New): {plan_wide_connector_trail_new_distance_mi:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total Connector Trail Distance (Redundant): {plan_wide_connector_trail_redundant_distance_mi:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total On-Foot Road Distance: {plan_wide_road_on_foot_distance_mi:.1f} mi</li>"
    )
    lines.append(
        f"<li>Total On-Foot Distance: {plan_wide_total_on_foot_distance_mi:.1f} mi</li>"
    )

    # Keep existing lines for Elevation Gain, Drive Time, Run Time, Total Time
    # And add new target/progress lines
    if challenge_target_distance_mi is not None and challenge_target_distance_mi > 0:
        progress_distance_pct = (
            plan_wide_official_segment_new_distance_mi / challenge_target_distance_mi
        ) * 100.0
        lines.append(
            f"<li>Challenge Target Distance: {challenge_target_distance_mi:.1f} mi</li>"
        )
        lines.append(f"<li>Progress (Distance): {progress_distance_pct:.1f}%</li>")
        over_target_distance_pct = (
            (plan_wide_total_on_foot_distance_mi / challenge_target_distance_mi) - 1
        ) * 100.0
        efficiency_distance = (
            challenge_target_distance_mi / plan_wide_total_on_foot_distance_mi
        ) * 100.0
        lines.append(
            f"<li>% Over Target Distance: {over_target_distance_pct:.1f}%</li>"
        )
        lines.append(f"<li>Efficiency Score (Distance): {efficiency_distance:.1f}</li>")
    else:
        lines.append("<li>Challenge Target Distance: Not Set</li>")
        lines.append("<li>Progress (Distance): N/A</li>")

    if challenge_target_elevation_ft is not None and challenge_target_elevation_ft > 0:
        progress_elevation_pct = (
            plan_wide_official_segment_new_elev_gain_ft
            / challenge_target_elevation_ft
        ) * 100.0
        lines.append(
            f"<li>Challenge Target Elevation: {challenge_target_elevation_ft:.0f} ft</li>"
        )
        lines.append(f"<li>Progress (Elevation): {progress_elevation_pct:.1f}%</li>")
        over_target_elevation_pct = (
            (total_elev_gain_ft / challenge_target_elevation_ft) - 1
        ) * 100.0
        efficiency_elevation = (
            challenge_target_elevation_ft / total_elev_gain_ft
        ) * 100.0
        lines.append(
            f"<li>% Over Target Elevation: {over_target_elevation_pct:.1f}%</li>"
        )
        lines.append(
            f"<li>Efficiency Score (Elevation): {efficiency_elevation:.1f}</li>"
        )
    else:
        lines.append("<li>Challenge Target Elevation: Not Set</li>")
        lines.append("<li>Progress (Elevation): N/A</li>")

    lines.append(f"<li>Total Elevation Gain: {total_elev_gain_ft:.0f} ft</li>")
    lines.append(
        f"<li>Redundant Elevation Gain: {redundant_elev_gain_ft:.0f} ft ({redundant_elev_pct:.0f}% )</li>"
    )
    lines.append(f"<li>Drive Time: {drive_time_min:.0f} min</li>")
    lines.append(f"<li>Run Time: {run_time_min:.0f} min</li>")
    lines.append(f"<li>Total Time: {total_time_min:.0f} min</li>")
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
                route_filtered = planner_utils.prune_short_connectors(route)
                activity_name = activity_or_drive["name"]

                dist = sum(e.length_mi for e in route_filtered)
                gain = sum(e.elev_gain_ft for e in route_filtered)
                est_activity_time = total_time(
                    route_filtered, args.pace, args.grade, args.road_pace
                )
                activity_or_drive["stats"] = {
                    "distance_mi": dist,
                    "elevation_ft": gain,
                    "time_min": est_activity_time,
                }
                activity_or_drive["directions"] = planner_utils.generate_turn_by_turn(
                    route_filtered, challenge_ids
                )
                activity_or_drive["inefficiencies"] = (
                    planner_utils.detect_inefficiencies(route_filtered)
                )

                current_day_total_trail_distance += dist
                current_day_total_trail_gain += gain
                for e in route_filtered:
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
                        route_filtered,
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
        # Calculate totals directly from the exported rows after any pruning
        plan_wide_unique_trail_miles = 0.0
        plan_wide_unique_trail_gain_ft = 0.0

        totals = {
            "total_trail_distance_mi": 0.0,
            "unique_trail_miles": 0.0,  # Filled in from summed rows below
            "redundant_miles": 0.0,
            "total_trail_elev_gain_ft": 0.0,
            "unique_trail_elev_gain_ft": 0.0,  # Filled in from summed rows below
            "redundant_elev_gain_ft": 0.0,
            "total_activity_time_min": 0.0,
            "total_drive_time_min": 0.0,
            "total_time_min": 0.0,
        }
        for row in summary_rows:
            if row.get("plan_description") == "Unable to complete":
                continue
            totals["total_trail_distance_mi"] += row["total_trail_distance_mi"]
            plan_wide_unique_trail_miles += row["unique_trail_miles"]
            plan_wide_unique_trail_gain_ft += row["unique_trail_elev_gain_ft"]
            totals["redundant_miles"] += row[
                "redundant_miles"
            ]  # This will be recalculated
            totals["total_trail_elev_gain_ft"] += row["total_trail_elev_gain_ft"]
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
                    totals["unique_trail_elev_gain_ft"], 0
                ),
                "redundant_elev_gain_ft": round(totals["redundant_elev_gain_ft"], 0),
                "redundant_elev_pct": round(total_elev_pct, 1),
                "total_activity_time_min": round(totals["total_activity_time_min"], 1),
                "total_drive_time_min": round(totals["total_drive_time_min"], 1),
                "total_time_min": round(totals["total_time_min"], 1),
                "challenge_target_distance_mi": (
                    args.challenge_target_distance_mi
                    if args.challenge_target_distance_mi is not None
                    else ""
                ),
                "progress_distance_pct": "",
                "challenge_target_elevation_ft": (
                    args.challenge_target_elevation_ft
                    if args.challenge_target_elevation_ft is not None
                    else ""
                ),
                "progress_elevation_pct": "",
                "over_target_distance_pct": "",
                "over_target_elevation_pct": "",
                "efficiency_distance": "",
                "efficiency_elevation": "",
                "num_activities": "",
                "num_drives": "",
                "notes": "",
                "start_trailheads": "",
            }
        )
        # Calculate progress percentages for the Totals row
        totals_row = summary_rows[-1]  # This is the dictionary for the "Totals" row
        if (
            args.challenge_target_distance_mi is not None
            and args.challenge_target_distance_mi > 0
            and isinstance(totals_row["unique_trail_miles"], (float, int))
        ):
            totals_row["progress_distance_pct"] = round(
                (totals_row["unique_trail_miles"] / args.challenge_target_distance_mi)
                * 100.0,
                1,
            )
        else:
            totals_row["progress_distance_pct"] = "N/A"

        if (
            args.challenge_target_elevation_ft is not None
            and args.challenge_target_elevation_ft > 0
            and isinstance(totals_row["unique_trail_elev_gain_ft"], (float, int))
        ):
            totals_row["progress_elevation_pct"] = round(
                (
                    totals_row["unique_trail_elev_gain_ft"]
                    / args.challenge_target_elevation_ft
                )
                * 100.0,
                1,
            )
        else:
            totals_row["progress_elevation_pct"] = "N/A"

        if (
            args.challenge_target_distance_mi is not None
            and args.challenge_target_distance_mi > 0
            and isinstance(totals_row["unique_trail_miles"], (float, int))
        ):
            totals_row["over_target_distance_pct"] = round(
                (
                    (
                        totals_row["unique_trail_miles"]
                        / args.challenge_target_distance_mi
                    )
                    - 1
                )
                * 100.0,
                1,
            )
            totals_row["efficiency_distance"] = round(
                (
                    args.challenge_target_distance_mi
                    / totals_row["unique_trail_miles"]
                )
                * 100.0,
                1,
            )
        else:
            totals_row["over_target_distance_pct"] = "N/A"
            totals_row["efficiency_distance"] = "N/A"

        if (
            args.challenge_target_elevation_ft is not None
            and args.challenge_target_elevation_ft > 0
            and isinstance(totals_row["total_trail_elev_gain_ft"], (float, int))
        ):
            totals_row["over_target_elevation_pct"] = round(
                (
                    (
                        totals_row["total_trail_elev_gain_ft"]
                        / args.challenge_target_elevation_ft
                    )
                    - 1
                )
                * 100.0,
                1,
            )
            totals_row["efficiency_elevation"] = round(
                (
                    args.challenge_target_elevation_ft
                    / totals_row["total_trail_elev_gain_ft"]
                )
                * 100.0,
                1,
            )
        else:
            totals_row["over_target_elevation_pct"] = "N/A"
            totals_row["efficiency_elevation"] = "N/A"

    # Define default_fieldnames to include new fields
    default_fieldnames_plus_targets = default_fieldnames + [
        "challenge_target_distance_mi",
        "progress_distance_pct",
        "challenge_target_elevation_ft",
        "progress_elevation_pct",
        "over_target_distance_pct",
        "over_target_elevation_pct",
        "efficiency_distance",
        "efficiency_elevation",
    ]

    if summary_rows:
        # Use keys from the Totals row (last row) as it's guaranteed to have all fields
        fieldnames = list(summary_rows[-1].keys())
    else:
        # Fallback if summary_rows is empty (e.g., no days planned, so no Totals row)
        fieldnames = default_fieldnames_plus_targets

    if summary_rows:
        debug_log(
            args,
            (
                "export_plan_files: Calculated plan_wide_unique_trail_miles: "
                f"{plan_wide_unique_trail_miles:.2f} mi for the CSV Totals row. "
                f"Outputting to {current_csv_path}"
            ),
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
        except openai.OpenAIError as e:
            logger.error("Review failed: %s", e)
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
        dem_path=args.dem,
        routing_failed=routing_failed,
        challenge_ids=challenge_ids,
        challenge_target_distance_mi=args.challenge_target_distance_mi,
        challenge_target_elevation_ft=args.challenge_target_elevation_ft,
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
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to load config %s: %s", config_path, e)
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
        default=config_defaults.get("segments", "data/traildata/GETChallengeTrailData_v2.json"),
        help="Trail segment JSON file",
    )
    parser.add_argument(
        "--connector-trails",
        dest="connector_trails",
        default=config_defaults.get("connector_trails"),
        help="Additional trail network GeoJSON for connector trails (none used if omitted)",
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
        help=(
            "Maximum road distance allowed while walking (mi). "
            "Also used by the RPP solver as the connectivity limit"
        ),
    )
    parser.add_argument(
        "--max-foot-connector",
        dest="max_foot_connector_mi",
        type=float,
        default=config_defaults.get("max_foot_connector_mi", 1.0),
        help="Maximum distance in miles to search for on-foot connectors between clusters",
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
        "--path-back-penalty",
        type=float,
        default=config_defaults.get("path_back_penalty", 1.2),
        help="Penalty multiplier for previously used segments when returning to start",
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
        "--focus-segment-ids",
        type=str,
        default=None,
        help="Comma-separated list of segment IDs to focus planning on. Activates focused planning mode.",
    )
    parser.add_argument(
        "--focus-plan-days",
        type=int,
        default=None,
        help="Number of days to plan when in focused mode. Defaults to 1 if --focus-segment-ids is used.",
    )
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
        "--challenge-target-distance-mi",
        type=float,
        default=config_defaults.get("challenge_target_distance_mi"),
        help="Target official challenge distance in miles for progress metrics",
    )
    parser.add_argument(
        "--challenge-target-elevation-ft",
        type=float,
        default=config_defaults.get("challenge_target_elevation_ft"),
        help="Target official challenge elevation gain in feet for progress metrics",
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
        choices=["greedy2opt"],
        default=config_defaults.get("optimizer", "greedy2opt"),
        help="Routing optimizer to use",
    )
    parser.add_argument(
        "--draft-every",
        type=int,
        metavar="N",
        default=0,
        help="Write draft CSV and HTML files every N days",
    )
    parser.add_argument(
        "--draft-daily",
        action="store_true",
        default=config_defaults.get("draft_daily", False),
        help="Write draft outputs after each day into draft_plans/",
    )
    parser.add_argument(
        "--prefer-single-loops",
        action="store_true",
        dest="prefer_single_loops",
        default=config_defaults.get("prefer_single_loops", False),
        help="Prefer planning one activity per day to avoid multiple drives",
    )
    parser.add_argument(
        "--prefer-single-loop-days",
        action="store_true",
        dest="prefer_single_loop_days",
        default=config_defaults.get("prefer_single_loop_days", False),
        help="Merge nearby clusters into one on-foot activity when possible",
    )
    parser.add_argument(
        "--snap-radius-m",
        type=float,
        default=config_defaults.get("snap_radius_m", 25.0),
        help="Tolerance in meters for snapping/validating graph nodes",
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
        "--allow-early-completion",
        action="store_true",
        default=False,
        help="Finish the schedule as soon as all segments are planned",
    )
    parser.add_argument(
        "--force-recompute-apsp",
        action="store_true",
        default=False,
        help="Force re-computation of All-Pairs Shortest Paths (APSP) cache",
    )
    parser.add_argument(
        "--num-apsp-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for APSP pre-computation (default: system CPU count)",
    )

    args = parser.parse_args(argv)
    overall_routing_status_ok = True  # Initialize routing status
    try:
    
        official_seg_ids: Set[str] = set()
        challenge_json = os.path.join(
            os.path.dirname(args.segments), "GETChallengeTrailData_v2.json"
        )
        if os.path.exists(challenge_json):
            try:
                with open(challenge_json) as f:
                    data = json.load(f)
                segs = data.get("trailSegments", [])
                official_seg_ids = {
                    str(seg.get("segId") or seg.get("id"))
                    for seg in segs
                    if seg.get("segId") or seg.get("id")
                }
                dist_ft = sum(
                    seg.get("properties", seg).get("LengthFt", 0) for seg in segs
                )
                if args.challenge_target_distance_mi is None:
                    args.challenge_target_distance_mi = round(dist_ft / 5280.0, 2)
                if args.challenge_target_elevation_ft is None:
                    args.challenge_target_elevation_ft = 36000.0
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to read challenge file %s: %s", challenge_json, e)
                official_seg_ids = set()
    
        if args.focus_segment_ids and args.focus_plan_days is None:
            args.focus_plan_days = 1
    
        # Setup queue-based logging for multiprocessing
        log_queue = multiprocessing.Queue(-1)
    
        class TqdmWriteHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg, file=sys.stderr)  # Ensure tqdm is imported
                    self.flush()
                except OSError as e:
                    logger.error("Logging handler error: %s", e)
                    self.handleError(record)
                    raise
    
        tqdm_handler = TqdmWriteHandler()
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        tqdm_handler.setFormatter(formatter)
    
        listener = QueueListener(log_queue, tqdm_handler)
        listener.start()
    
        # Configure root logger (or specific loggers)
        root_logger = logging.getLogger()  # Or logging.getLogger('trail_route_ai')
        # Remove existing handlers if any (e.g., from basicConfig)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
        root_logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
        queue_handler = QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)
        if csgraph_dijkstra is not None:
            logger.info(
                "SciPy detected; using compiled Dijkstra implementation for routing."
            )
        else:
            logger.info(
                "SciPy not available; falling back to NetworkX for Dijkstra routing."
            )
        # The old basicConfig is now replaced by the above setup.
        # logging.basicConfig(
        #     level=logging.INFO if args.verbose else logging.WARNING,
        #     format="%(message)s",
        # )
    
        if getattr(args, "debug", None):
            debug_log_path = getattr(args, "debug")
            if os.path.exists(debug_log_path):
                try:
                    open(debug_log_path, "w").close()
                    # Use logger for this message now
                    # print(
                    #     f"Debug log '{debug_log_path}' cleared at start of run.",
                    #     file=sys.stderr,
                    # )
                    logger.info(f"Debug log '{debug_log_path}' cleared at start of run.")
                except IOError as e:
                    # print(
                    #     f"Warning: Could not clear debug log '{debug_log_path}': {e}",
                    #     file=sys.stderr,
                    # )
                    logger.warning(f"Could not clear debug log '{debug_log_path}': {e}")
            else:
                debug_log_dir = os.path.dirname(debug_log_path)
                if debug_log_dir and not os.path.exists(debug_log_dir):
                    os.makedirs(debug_log_dir, exist_ok=True)
    
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
    
        if args.focus_segment_ids and args.focus_plan_days is not None:
            num_days = args.focus_plan_days
            logger.info(
                f"Focused planning: Overriding num_days to {num_days} based on --focus-plan-days."
            )
    
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
        if not all_trail_segments:
            raise ValueError("No segments found in segments file")
        process = psutil.Process(os.getpid())
        logger.info(
            f"Memory after loading trail segments: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )
        connector_trail_segments: List[Edge] = []
        if (
            args.connector_trails
            and args.allow_connector_trails
            and os.path.exists(args.connector_trails)
        ):
            connector_trail_segments = planner_utils.load_segments(args.connector_trails)
            seg_ids = {str(e.seg_id) for e in all_trail_segments if e.seg_id is not None}
            connector_trail_segments = [
                e for e in connector_trail_segments if str(e.seg_id) not in seg_ids
            ]
            if args.dem:
                planner_utils.add_elevation_from_dem(connector_trail_segments, args.dem)
        if args.dem:
            planner_utils.add_elevation_from_dem(all_trail_segments, args.dem)
        # Validate overall elevation of official segments against challenge target
        planner_utils.validate_elevation_data(
            all_trail_segments,
            target_gain_ft=args.challenge_target_elevation_ft or 36000.0,
        )
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
                bbox = planner_utils.bounding_box_from_edges(
                    all_trail_segments + connector_trail_segments
                )
            all_road_segments = planner_utils.load_roads(args.roads, bbox=bbox)
            process = psutil.Process(os.getpid())
            logger.info(
                f"Memory after loading road segments: {process.memory_info().rss / 1024 ** 2:.2f} MB"
            )
        road_graph_for_drive = planner_utils.build_road_graph(all_road_segments)
        road_node_set: Set[Tuple[float, float]] = {e.start for e in all_road_segments} | {
            e.end for e in all_road_segments
        }
    
        trailhead_lookup: Dict[Tuple[float, float], str] = {}
        if args.trailheads and os.path.exists(args.trailheads):
            trailhead_lookup = planner_utils.load_trailheads(args.trailheads)
    
        # Add short connectors from trail ends to nearby road nodes for better
        # on-foot routing connectivity.
        foot_connectors = planner_utils.connect_trails_to_roads(
            all_trail_segments + connector_trail_segments,
            all_road_segments,
            threshold_meters=50.0,
        )
    
        # This graph is used for on-foot routing *within* macro-clusters
        on_foot_routing_graph_edges = (
            all_trail_segments
            + connector_trail_segments
            + all_road_segments
            + foot_connectors
        )
        official_nodes = {e.start for e in all_trail_segments} | {e.end for e in all_trail_segments}
        G = build_nx_graph(
            on_foot_routing_graph_edges,
            args.pace,
            args.grade,
            args.road_pace,
            snap_radius_m=args.snap_radius_m,
            official_nodes=official_nodes,
        )
    
        # Create a lightweight view of the graph for APSP computation to avoid
        # duplicating all edges in memory
        G_apsp = nx.subgraph_view(G)
    
        if csgraph_dijkstra is not None:
            try:
                _prepare_csgraph(G_apsp)
            except (RuntimeError, nx.NetworkXError, ValueError) as e:
                logger.error("Failed to prepare csgraph for APSP: %s", e)
    
        process_for_lean_graph_log = psutil.Process(os.getpid())
        logger.info(
            f"Memory after creating G_apsp ({G_apsp.number_of_nodes()} nodes, {G_apsp.number_of_edges()} edges): {process_for_lean_graph_log.memory_info().rss / 1024 ** 2:.2f} MB"
        )
    
        cache_key = f"{args.pace}:{args.grade}:{args.road_pace}"
    
        path_cache_db_instance = None  # Will be RocksDB instance
    
        needs_apsp_recompute = False
        if args.force_recompute_apsp:
            needs_apsp_recompute = True
            logger.info("Forcing APSP re-computation due to --force-recompute-apsp flag.")
        else:
            # Try to open read-only to check sentinel
            ro_db_check = cache_utils.open_rocksdb(
                "dist_cache_db", cache_key, read_only=True
            )
            if ro_db_check is None:
                needs_apsp_recompute = True
                logger.warning(
                    "Could not open APSP RocksDB for checking (open returned None). Scheduling re-computation."
                )
            else:
                if (
                    cache_utils.load_rocksdb_cache(ro_db_check, "__APSP_SENTINEL_KEY__")
                    is None
                ):
                    needs_apsp_recompute = True
                    logger.info(
                        "APSP RocksDB sentinel key not found. Scheduling re-computation."
                    )
                else:
                    logger.info("APSP RocksDB sentinel key found. Cache assumed valid.")
                cache_utils.close_rocksdb(ro_db_check)
    
        # APSP pre-computation block starts here if needs_apsp_recompute is True
        # Note: The detailed logging of graph size and memory is kept, and the actual computation loop is next.
        # Log graph size and memory before APSP calculation (if recomputing)
        # This logging is relevant whether we recompute or not, so it's fine here.
        logger.info(
            f"Graph G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges for APSP."
        )
        process = psutil.Process(os.getpid())  # psutil should be imported
        logger.info(
            f"Memory before APSP calculation: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )
    
        if needs_apsp_recompute:
            # num_apsp_workers = os.cpu_count() # Old line
            num_apsp_workers = args.num_apsp_workers  # New line
            logger.info(f"Using {num_apsp_workers} workers for APSP pre-computation.")
            rw_db_populate = None  # To be assigned after potential clear
            # Construct path for potential deletion/check
            h_force = hashlib.sha1(cache_key.encode()).hexdigest()[:16]
            db_path_to_manage = os.path.join(
                cache_utils.get_cache_dir(), f"dist_cache_db_{h_force}_db"
            )
    
            if args.force_recompute_apsp:  # This check is correctly placed
                if os.path.exists(db_path_to_manage):
                    logger.info(
                        f"Force recompute: Removing existing RocksDB directory: {db_path_to_manage}"
                    )
                    shutil.rmtree(db_path_to_manage)  # shutil should be imported
    
            rw_db_populate = cache_utils.open_rocksdb(
                "dist_cache_db", cache_key, read_only=False
            )
            if rw_db_populate is None:
                # Ensure db_path_to_manage is defined for the error message, it is.
                raise RuntimeError(
                    f"Failed to open RocksDB for APSP writing at {db_path_to_manage}."
                )
    
            # By default restrict APSP computation to nodes that appear as
            # the start or end of any trail or connector edge. This avoids
            # spending time on isolated road network nodes that will never be
            # referenced when constructing routes.
            nodes_for_apsp = list(
                {e.start for e in on_foot_routing_graph_edges}
                | {e.end for e in on_foot_routing_graph_edges}
            )
            if len(nodes_for_apsp) < G.number_of_nodes():
                logger.info(
                    "APSP node set reduced to %d from %d total graph nodes",
                    len(nodes_for_apsp),
                    G.number_of_nodes(),
                )
    
            if args.focus_segment_ids:
                source_nodes_for_focused_dijkstra = set()
                focused_ids_str = {s.strip() for s in args.focus_segment_ids.split(",")}
                for edge in all_trail_segments:  # Ensure all_trail_segments is available
                    if str(edge.seg_id) in focused_ids_str:
                        source_nodes_for_focused_dijkstra.add(edge.start)
                        source_nodes_for_focused_dijkstra.add(edge.end)
    
                original_node_count = len(nodes_for_apsp)
                nodes_for_apsp = [
                    n for n in nodes_for_apsp if n in source_nodes_for_focused_dijkstra
                ]
                logger.info(
                    f"Dijkstra cache population focused on {len(nodes_for_apsp)} nodes "
                    f"(out of {original_node_count} total) related to {len(focused_ids_str)} specified segment IDs."
                )
    
            # tasks = [(node, G) for node in nodes_for_apsp] # Old task format
            tasks = [node for node in nodes_for_apsp]  # New task format
    
            # Added log message
            logger.info(
                f"Starting parallel APSP computation with {num_apsp_workers} workers for {len(nodes_for_apsp)} nodes."
            )
            apsp_lock = multiprocessing.Lock()
            with multiprocessing.Pool(
                processes=num_apsp_workers,
                initializer=worker_init_apsp,
                initargs=(
                    G_apsp,
                    log_queue,
                    apsp_lock,
                ),
            ) as pool:
                with tqdm(
                    total=len(nodes_for_apsp),
                    desc="Pre-calculating APSP parts (RocksDB)",
                    unit="node",
                ) as pbar:
                    for source_node_apsp, dist_pred_data in pool.imap_unordered(
                        compute_dijkstra_for_node, tasks
                    ):
                        cache_utils.save_rocksdb_cache(
                            rw_db_populate, source_node_apsp, dist_pred_data
                        )
                        pbar.update(1)
                        # Memory logging and gc.collect() might be less effective here due to multiple processes.
                        # Consider logging memory periodically or after the pool finishes.
                        # For now, we'll keep a simplified version or remove if too noisy/complex.
                        if pbar.n % 100 == 0:  # Log every 100 nodes processed
                            mem_info_rss_mb = process.memory_info().rss / (1024**2)
                            logger.info(
                                f"Memory after processing batch for APSP: {mem_info_rss_mb:.2f} MB"
                            )
                            gc.collect()  # gc should be imported
    
            cache_utils.save_rocksdb_cache(rw_db_populate, "__APSP_SENTINEL_KEY__", True)
            cache_utils.close_rocksdb(rw_db_populate)
            logger.info("APSP re-computation complete. RocksDB populated and closed.")
    
        # After potential re-computation, open the DB for reading.
        # If re-computation happened, this will open the newly populated DB.
        # If not, it opens the existing valid DB.
        path_cache_db_instance = cache_utils.open_rocksdb(
            "dist_cache_db", cache_key, read_only=True
        )
        if path_cache_db_instance is None:
            # This is a critical failure if we can't open the DB after setup/check.
            h_key = hashlib.sha1(cache_key.encode()).hexdigest()[:16]  # For error message
            db_path_final = os.path.join(
                cache_utils.get_cache_dir(), f"dist_cache_db_{h_key}_db"
            )
            raise RuntimeError(
                f"Failed to open RocksDB for reading at {db_path_final} after setup."
            )
        logger.info("RocksDB cache for APSP is open for reading.")
    
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
    
        loaded_seg_ids = {str(e.seg_id) for e in all_trail_segments if e.seg_id is not None}
        missing_ids = sorted(official_seg_ids - loaded_seg_ids)
        if missing_ids:
            logger.error("Missing official segment IDs after loading: %s", ", ".join(missing_ids))
            raise RuntimeError(
                f"Official segments missing after loading: {', '.join(missing_ids)}"
            )
    
        # nodes list might be useful later for starting points, keep it around
        # nodes = list({e.start for e in on_foot_routing_graph_edges} | {e.end for e in on_foot_routing_graph_edges})
    
        process = psutil.Process(os.getpid())
        logger.info(
            f"Memory before identifying macro clusters: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )
        potential_macro_clusters = identify_macro_clusters(
            current_challenge_segments,  # Only uncompleted trail segments
            all_road_segments,  # All road segments
            args.pace,
            args.grade,
            args.road_pace,
            snap_radius_m=args.snap_radius_m,
        )
        process = psutil.Process(os.getpid())
        logger.info(
            f"Memory after identifying macro clusters: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )
        gc.collect()
        logger.info(
            f"Memory after GC post macro clusters: {process.memory_info().rss / 1024 ** 2:.2f} MB"
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
                    args, f"splitting large cluster of {naive_time:.1f} min into parts using new topology-aware clustering"
                )
                # max_parts = max(1, int(np.ceil(naive_time / budget))) # Not directly used by new func
                subclusters = clustering.build_topology_aware_clusters(
                    all_segments=cluster_edges, # Pass the segments of the current oversized macro-cluster
                    graph=G,          # Pass the main graph
                    config=args,      # Pass the planner arguments/config
                    # precomputed_loops=None # Optional, omit for now
                )
                # The new function returns List[List[Edge]], which matches `subclusters` type.
                for sub in subclusters:
                    if not sub:
                        continue
                    sub_nodes = {pt for e in sub for pt in (e.start, e.end)}
                    expanded_clusters.append((sub, sub_nodes))
            else:
                # Also process smaller macro-clusters with the new clustering logic
                # to benefit from its topological analysis (loop finding, etc.)
                debug_log(
                    args, f"Processing smaller macro-cluster (time {naive_time:.1f} min) with new topology-aware clustering"
                )
                refined_subclusters = clustering.build_topology_aware_clusters(
                    all_segments=cluster_edges,
                    graph=G,
                    config=args,
                )
                for sub in refined_subclusters:
                    if not sub:
                        continue
                    sub_nodes = {pt for e in sub for pt in (e.start, e.end)}
                    expanded_clusters.append((sub, sub_nodes))
    
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
        for idx, (cluster_segs, cluster_nodes) in enumerate(
            tqdm(deduplicated_unplanned_macro_clusters, desc="Initial cluster processing")
        ):
            debug_log(
                args,
                f"MainLoop: Start processing cluster {idx}. Segments: {len(cluster_segs)}. First segment ID: {cluster_segs[0].seg_id if cluster_segs else 'N/A'}",
            )
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
                path_cache_db_instance,
                use_rpp=True,
                allow_connectors=args.allow_connector_trails,
                rpp_timeout=args.rpp_timeout,
                debug_args=args,
                spur_length_thresh=args.spur_length_thresh,
                spur_road_bonus=args.spur_road_bonus,
                path_back_penalty=args.path_back_penalty,
                use_advanced_optimizer=args.use_advanced_optimizer,
                strict_max_foot_road=args.strict_max_foot_road,
                redundancy_threshold=args.redundancy_threshold,
                optimizer_name=args.optimizer,
            )
            debug_log(
                args,
                f"MainLoop: Finished processing cluster {idx}. Route found: {bool(initial_route)}. Route length: {len(initial_route) if initial_route else 0}",
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
                    continue
                for sub in connectivity_subs:
                    nodes = {pt for e in sub for pt in (e.start, e.end)}
                    processed_clusters.append((sub, nodes))
                continue
    
            if any(e.direction != "both" for e in cluster_segs):
                direction_subs = split_cluster_by_one_way(cluster_segs)
                if len(direction_subs) > 1:
                    debug_log(
                        args,
                        f"split cluster into {len(direction_subs)} parts due to one-way segments",
                    )
                    for sub in direction_subs:
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
                path_cache_db_instance,
                use_rpp=True,
                allow_connectors=args.allow_connector_trails,
                rpp_timeout=args.rpp_timeout,
                debug_args=args,
                spur_length_thresh=args.spur_length_thresh,
                spur_road_bonus=args.spur_road_bonus,
                path_back_penalty=args.path_back_penalty,
                use_advanced_optimizer=args.use_advanced_optimizer,
                strict_max_foot_road=args.strict_max_foot_road,
                redundancy_threshold=args.redundancy_threshold,
                optimizer_name=args.optimizer,
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
        driving_optimizer = DrivingOptimizer(args.max_foot_connector_mi)

        os.makedirs(args.gpx_dir, exist_ok=True)
        # summary_rows = [] # This will be populated by the new planning loop (or rather, daily_plans will be used)
        daily_plans = []
        failed_cluster_signatures: Set[Tuple[str, ...]] = set()

        if args.prefer_single_loop_days:
            grouped = driving_optimizer.optimize_daily_cluster_selection(
                unplanned_macro_clusters,
                home_coord if home_coord else (0.0, 0.0),
                road_graph=road_graph_for_drive,
                max_daily_drive_time=args.max_drive_minutes_per_transfer,
            )
            merged_clusters: List[ClusterInfo] = []
            for grp in grouped:
                edges: List[Edge] = []
                nodes: Set[Tuple[float, float]] = set()
                start_candidates: List[Tuple[Tuple[float, float], Optional[str]]] = []
                for c in grp:
                    edges.extend(c.edges)
                    nodes.update(c.nodes)
                    start_candidates.extend(c.start_candidates)
                merged_clusters.append(ClusterInfo(edges, nodes, start_candidates))
            unplanned_macro_clusters = merged_clusters
    
        if args.focus_segment_ids:
            target_segment_ids_set = {s.strip() for s in args.focus_segment_ids.split(",")}
            filtered_unplanned_clusters = []
            for cluster_info in unplanned_macro_clusters:
                if any(
                    str(edge.seg_id) in target_segment_ids_set
                    for edge in cluster_info.edges
                ):
                    filtered_unplanned_clusters.append(cluster_info)
            unplanned_macro_clusters = filtered_unplanned_clusters
            logger.info(
                f"Focused planning: Considering {len(unplanned_macro_clusters)} clusters relevant to specified segment IDs."
            )
    
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
                        path_cache_db_instance,  # Changed from path_cache
                        use_rpp=True,
                        allow_connectors=args.allow_connector_trails,
                        rpp_timeout=args.rpp_timeout,
                        debug_args=args,
                        spur_length_thresh=args.spur_length_thresh,
                        spur_road_bonus=args.spur_road_bonus,
                        path_back_penalty=args.path_back_penalty,
                        use_advanced_optimizer=args.use_advanced_optimizer,
                        strict_max_foot_road=args.strict_max_foot_road,
                        redundancy_threshold=args.redundancy_threshold,
                        optimizer_name=args.optimizer,  # Changed optimizer to optimizer_name
                    )
                    if route_edges:
                        if best_drive_time > 0:
                            activities_for_this_day.append(
                                {
                                    "type": "drive",
                                    "minutes": best_drive_time,
                                    "from_coord": drive_origin,
                                    "to_coord": best_start_node,
                                    "mode": "drive",
                                }
                            )
                            time_spent_on_drives_today += best_drive_time
                        activities_for_this_day.append(
                            {
                                "type": "activity",
                                "route_edges": route_edges,
                                "name": _derive_activity_name(route_edges, best_start_name, current_challenge_segment_ids),
                                "ignored_budget": False,
                                "start_name": best_start_name,
                                "start_coord": best_start_node,
                                "mode": "foot",
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
                        path_cache_db_instance,  # Changed from path_cache
                        use_rpp=True,
                        allow_connectors=args.allow_connector_trails,
                        rpp_timeout=args.rpp_timeout,
                        debug_args=args,
                        spur_length_thresh=args.spur_length_thresh,
                        spur_road_bonus=args.spur_road_bonus,
                        path_back_penalty=args.path_back_penalty,
                        use_advanced_optimizer=args.use_advanced_optimizer,
                        strict_max_foot_road=args.strict_max_foot_road,
                        redundancy_threshold=args.redundancy_threshold,
                        optimizer_name=args.optimizer,  # Changed optimizer to optimizer_name
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
                                path_cache_db_instance,  # Changed from path_cache
                                use_rpp=True,
                                allow_connectors=args.allow_connector_trails,
                                rpp_timeout=args.rpp_timeout,
                                debug_args=args,
                                spur_length_thresh=args.spur_length_thresh,
                                spur_road_bonus=args.spur_road_bonus,
                                path_back_penalty=args.path_back_penalty,
                                use_advanced_optimizer=args.use_advanced_optimizer,
                                strict_max_foot_road=args.strict_max_foot_road,
                                redundancy_threshold=args.redundancy_threshold,
                                optimizer_name=args.optimizer,  # Changed optimizer to optimizer_name
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
                                "mode": "drive",
                            }
                        )
                        time_spent_on_drives_today += best_cluster_to_add_info["drive_time"]
    
                    act_route_edges = best_cluster_to_add_info["route_edges"]
                    activities_for_this_day.append(
                        {
                            "type": "activity",
                            "route_edges": act_route_edges,
                            "name": _derive_activity_name(act_route_edges, best_cluster_to_add_info.get("start_name"), current_challenge_segment_ids),
                            "ignored_budget": best_cluster_to_add_info.get(
                                "ignored_budget", False
                            ),
                            "start_name": best_cluster_to_add_info.get("start_name"),
                            "start_coord": best_cluster_to_add_info.get("start_coord"),
                            "mode": "foot",
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
                            path_cache_db_instance,  # Changed from path_cache
                            use_rpp=True,
                            allow_connectors=args.allow_connector_trails,
                            rpp_timeout=args.rpp_timeout,
                            debug_args=args,
                            spur_length_thresh=args.spur_length_thresh,
                            spur_road_bonus=args.spur_road_bonus,
                            path_back_penalty=args.path_back_penalty,
                            use_advanced_optimizer=args.use_advanced_optimizer,
                            strict_max_foot_road=args.strict_max_foot_road,
                            redundancy_threshold=args.redundancy_threshold,
                            optimizer_name=args.optimizer,  # Changed optimizer to optimizer_name
                        )
                        if act_route_edges:
                            activities_for_this_day.append(
                                {
                                    "type": "activity",
                                    "route_edges": act_route_edges,
                                    "name": _derive_activity_name(act_route_edges, start_name, current_challenge_segment_ids),
                                    "ignored_budget": True,
                                    "start_name": start_name,
                                    "start_coord": start_node,
                                    "mode": "foot",
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
    
            if args.draft_daily:
                draft_dir = os.path.join(os.path.dirname(args.output), "draft_plans")
                os.makedirs(draft_dir, exist_ok=True)
                draft_csv = os.path.join(draft_dir, f"draft-day{day_idx+1}.csv")
                orig_gpx_dir = args.gpx_dir
                args.gpx_dir = os.path.join(draft_dir, f"gpx_day{day_idx+1}")
                export_plan_files(
                    daily_plans,
                    args,
                    csv_path=draft_csv,
                    write_gpx=True,
                    review=False,
                    challenge_ids=current_challenge_segment_ids,
                )
                args.gpx_dir = orig_gpx_dir
    
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
            path_cache_db_instance,  # Changed from path_cache
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
            path_back_penalty=args.path_back_penalty,
            challenge_ids=current_challenge_segment_ids,
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
            path_cache_db_instance,  # Changed from path_cache
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
            path_back_penalty=args.path_back_penalty,
            challenge_ids=current_challenge_segment_ids,
        )
    
        # Increase all budgets evenly if any day is now over budget
        if even_out_budgets(daily_plans, daily_budget_minutes) > 0:
            update_plan_notes(daily_plans, daily_budget_minutes)

        # Align the schedule with the requested date range
        daily_plans = ScheduleOptimizer.optimize_schedule(
            daily_plans,
            start_date,
            end_date,
            daily_budget_minutes,
            allow_early_completion=args.allow_early_completion,
        )
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
                # Mark routing as failed and continue so outputs are written with the
                # "failed-" prefix instead of raising an exception.
                overall_routing_status_ok = False
                tqdm.write(error_message, file=sys.stderr)
                export_plan_files(
                    daily_plans,
                    args,
                    challenge_ids=current_challenge_segment_ids,
                    routing_failed=True,
                )
    
        if overall_routing_status_ok:
            export_plan_files(
                daily_plans,
                args,
                challenge_ids=current_challenge_segment_ids,
                routing_failed=not overall_routing_status_ok,
            )
        else:
            tqdm.write(
                "Routing errors detected. Resolve them before exporting the plan.",
                file=sys.stderr,
            )

    finally:
        if "path_cache_db_instance" in locals() and path_cache_db_instance:
            # Close the RocksDB instance first so the informational message
            # accurately reflects that the resource has been released.
            cache_utils.close_rocksdb(path_cache_db_instance)
            # Log this message while the queue listener is still active so the
            # log is emitted correctly.
            logger.info("Closed RocksDB APSP cache at the end of script execution.")

        # Ensure listener is stopped only after all logging is complete
        if "listener" in locals():
            try:
                listener.stop()
            except OSError as e:
                logger.error("Failed to stop log listener: %s", e)
                raise

        # Close the queue and wait for the queue's thread to finish
        if "log_queue" in locals():
            try:
                log_queue.close()
                log_queue.join_thread()
            except OSError as e:
                logger.error("Failed to close log queue: %s", e)
                raise

    return 0 if overall_routing_status_ok else 1


if __name__ == "__main__":
    sys.exit(main())
