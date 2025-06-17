from __future__ import annotations

import networkx as nx
from typing import List

from . import planner_utils

Edge = planner_utils.Edge


def build_cluster_graph(
    required_edges: List[Edge],
    *,
    pace: float,
    grade: float,
    road_pace: float,
) -> nx.MultiGraph:
    """Return an undirected graph for ``required_edges``.

    Each edge is marked with ``required=True``. We do not include connector
    edges here; those may be added later by the optimizer as needed.
    """
    G = nx.MultiGraph()
    for e in required_edges:
        w = planner_utils.estimate_time(e, pace, grade, road_pace)
        G.add_edge(
            e.start,
            e.end,
            weight=w,
            edge=e,
            required=True,
        )
    return G
