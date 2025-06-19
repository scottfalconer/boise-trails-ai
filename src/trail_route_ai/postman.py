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
    """Return a simple route covering ``required_edges`` using shortest paths."""

    if not required_edges:
        return []

    required_ids: Set[str] = {str(e.seg_id) for e in required_edges if e.seg_id is not None}

    route: List[Edge] = []
    current = start
    graph_no_required = full_graph.copy()
    for u, v, data in list(graph_no_required.edges(data=True)):
        ed = data.get("edge")
        if ed and ed.seg_id is not None and str(ed.seg_id) in required_ids and not ed._is_reversed:
            graph_no_required.remove_edge(u, v)

    for e in required_edges:
        if current != e.start:
            path = nx.shortest_path(graph_no_required, current, e.start, weight="weight")
            route.extend(_edges_from_path(full_graph, path, required_ids=required_ids))
        route.append(e)
        current = e.end

    return route
