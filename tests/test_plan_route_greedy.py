import networkx as nx
from trail_route_ai import challenge_planner, planner_utils


def build_base_graph():
    """Graph with only road connectors between two trails."""
    t1 = planner_utils.Edge("T1", "T1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], "trail", "both")
    t2 = planner_utils.Edge("T2", "T2", (3.0, 0.0), (4.0, 0.0), 1.0, 0.0, [(3.0, 0.0), (4.0, 0.0)], "trail", "both")
    r1 = planner_utils.Edge("R1", "R1", (1.0, 0.0), (3.0, 0.0), 2.0, 0.0, [(1.0, 0.0), (3.0, 0.0)], "road", "both")
    G = challenge_planner.build_nx_graph([t1, t2, r1], pace=10.0, grade=0.0, road_pace=15.0)
    return G, [t1, t2]


def build_graph_with_trail(trail_len, road_len):
    t1 = planner_utils.Edge("T1", "T1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], "trail", "both")
    t2 = planner_utils.Edge("T2", "T2", (3.0, 0.0), (4.0, 0.0), 1.0, 0.0, [(3.0, 0.0), (4.0, 0.0)], "trail", "both")
    t_conn = planner_utils.Edge("TC", "TC", (1.0, 0.0), (3.0, 0.0), trail_len, 0.0, [(1.0, 0.0), (3.0, 0.0)], "trail", "both")
    r1 = planner_utils.Edge("R1", "R1", (1.0, 0.0), (3.0, 0.0), road_len, 0.0, [(1.0, 0.0), (3.0, 0.0)], "road", "both")
    G = challenge_planner.build_nx_graph([t1, t2, t_conn, r1], pace=10.0, grade=0.0, road_pace=15.0)
    return G, [t1, t2]


def test_over_limit_without_trail_option():
    G, trails = build_base_graph()
    params = dict(pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.25)
    route, order = challenge_planner._plan_route_greedy(G, trails, (0.0, 0.0), **params, dist_cache={})
    assert route  # route should be produced even though connector exceeds max_road
    assert order


def test_over_limit_prefers_trail_when_not_fast():
    G, trails = build_graph_with_trail(trail_len=2.0, road_len=2.5)
    params = dict(pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.25)
    route, order = challenge_planner._plan_route_greedy(G, trails, (0.0, 0.0), **params, dist_cache={})
    # The planner should choose the trail connector because the road is not sufficiently faster
    ids_in_route = [e.seg_id for e in order]
    assert "TC" in ids_in_route and "R1" not in ids_in_route


def test_over_limit_accepts_fast_road():
    G, trails = build_graph_with_trail(trail_len=2.0, road_len=1.2)
    params = dict(pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.25)
    route, order = challenge_planner._plan_route_greedy(G, trails, (0.0, 0.0), **params, dist_cache={})
    ids_in_route = [e.seg_id for e in order]
    assert "R1" in ids_in_route
