import sys

sys.path.append("src")

import networkx as nx
from trail_route_ai import planner_utils, challenge_planner, postman


def build_square():
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (1.0, 1.0)
    D = (0.0, 1.0)
    e1 = planner_utils.Edge("a", "a", A, B, 1.0, 0.0, [A, B], "trail", "both")
    e2 = planner_utils.Edge("b", "b", B, C, 1.0, 0.0, [B, C], "trail", "both")
    e3 = planner_utils.Edge("c", "c", C, D, 1.0, 0.0, [C, D], "trail", "both")
    e4 = planner_utils.Edge("d", "d", D, A, 1.0, 0.0, [D, A], "trail", "both")
    edges = [e1, e2, e3, e4]
    G = challenge_planner.build_nx_graph(edges, pace=10.0, grade=0.0, road_pace=10.0)
    return G, edges, A


def test_postman_exact_coverage():
    G, edges, start = build_square()
    route = postman.solve_rpp(G, edges, start, pace=10.0, grade=0.0, road_pace=10.0)
    ids = [e.seg_id for e in route]
    for e in edges:
        assert ids.count(e.seg_id) == 1


def test_postman_vs_greedy():
    G, edges, start = build_square()
    r_post = postman.solve_rpp(G, edges, start, pace=10.0, grade=0.0, road_pace=10.0)
    r_greedy = challenge_planner.plan_route(
        G,
        edges,
        start,
        10.0,
        0.0,
        10.0,
        0.5,
        0.1,
        optimizer="greedy2opt",
    )
    len_post = sum(e.length_mi for e in r_post)
    len_greedy = sum(e.length_mi for e in r_greedy)
    assert len_post <= len_greedy
