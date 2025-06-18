from __future__ import annotations

import time
from typing import List, Tuple, Optional, Set
import networkx as nx
from tqdm import tqdm

from . import planner_utils
from .graph_utils import build_cluster_graph

Edge = planner_utils.Edge


def _edges_from_path(
    G: nx.DiGraph,
    path: List[Tuple[float, float]],
    required_ids: Optional[Set[str]] = None,
) -> List[Edge]:
    out: List[Edge] = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if not data:
            continue
        ed = data["edge"]
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


def solve_rpp(
    full_graph: nx.DiGraph,
    required_edges: List[Edge],
    start: Tuple[float, float],
    *,
    pace: float,
    grade: float,
    road_pace: float,
    timeout: float = 30.0,
    max_odd: int = 40,
) -> List[Edge]:
    """Solve the Rural Postman Problem for ``required_edges`` using ``full_graph``.

    The algorithm connects disjoint components with shortest paths, makes all
    degrees even using minimum-weight matching, and then extracts an Eulerian
    circuit. If the computation exceeds ``timeout`` seconds or ``max_odd`` is
    surpassed, a ``RuntimeError`` is raised so callers can fall back.
    """

    if not required_edges:
        return []

    start_time = time.perf_counter()

    def timed_out() -> bool:
        return timeout is not None and (time.perf_counter() - start_time) >= timeout

    G = build_cluster_graph(required_edges, pace=pace, grade=grade, road_pace=road_pace)
    required_ids: Set[str] = {
        str(e.seg_id) for e in required_edges if e.seg_id is not None
    }

    # Step B - ensure connectivity
    # This step can be computationally intensive if there are many components or if components are large,
    # as it may involve many shortest path calculations. Protected by timeout.
    components = list(nx.connected_components(G))
    if len(components) > 1:
        base = set(components[0])
        for comp in tqdm(components[1:], desc="Connecting components"):
            if timed_out():
                raise RuntimeError("postman timeout")
            best_path = None
            best_cost = float("inf")
            for u in comp:
                for v in base:
                    try:
                        path = nx.shortest_path(full_graph, u, v, weight="weight")
                        cost = nx.path_weight(full_graph, path, weight="weight")
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                    except nx.NetworkXNoPath:
                        continue
            if best_path is None:
                raise RuntimeError("components cannot be connected")
            for e in _edges_from_path(full_graph, best_path, required_ids=required_ids):
                w = planner_utils.estimate_time(e, pace, grade, road_pace)
                G.add_edge(e.start, e.end, weight=w, edge=e, required=False)
            base |= set(comp)

    # Step C - make degrees even
    odd_nodes = [n for n, d in G.degree() if d % 2 == 1]
    if len(odd_nodes) > max_odd:
        raise RuntimeError("too many odd nodes")
    if odd_nodes:
        # This step involves calculating all-pairs shortest paths and then a minimum weight matching,
        # which can be computationally intensive for large numbers of odd nodes. Protected by timeout and max_odd heuristic.

        all_pairs_iterator = nx.all_pairs_dijkstra_path_length(full_graph, weight="weight")
        num_nodes_for_progress = len(full_graph) # Assumes full_graph is connected and all nodes are sources

        all_pairs = {}
        # Only iterate and store paths if odd_nodes exist, otherwise, this is skipped.
        for source_node, paths in tqdm(all_pairs_iterator, total=num_nodes_for_progress, desc="Calculating all-pairs shortest paths for matching"):
            all_pairs[source_node] = paths

        metric = nx.Graph()
        for i, u in enumerate(odd_nodes):
            for v in odd_nodes[i + 1 :]:
                metric.add_edge(u, v, weight=all_pairs[u][v])
        matching = nx.algorithms.matching.min_weight_matching(
            metric, maxcardinality=True
        )
        for u, v in tqdm(matching, desc="Augmenting graph from matching"):
            if timed_out():
                raise RuntimeError("postman timeout")
            path = nx.shortest_path(full_graph, u, v, weight="weight")
            for e in _edges_from_path(full_graph, path, required_ids=required_ids):
                w = planner_utils.estimate_time(e, pace, grade, road_pace)
                G.add_edge(e.start, e.end, weight=w, edge=e, required=False)

    if start not in G:
        start = next(iter(G.nodes()))

    circuit = list(nx.eulerian_circuit(G, source=start, keys=True))
    result: List[Edge] = []
    for u, v, key in circuit:
        edge_data = G.get_edge_data(u, v)[key]
        result.append(edge_data["edge"])
    return result
