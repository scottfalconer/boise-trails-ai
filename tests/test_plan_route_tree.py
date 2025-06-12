import sys
sys.path.append('src')
from trail_route_ai import challenge_planner, planner_utils


def build_tree_edges():
    t1 = planner_utils.Edge('T1', 'T1', (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], 'trail', 'both')
    t2 = planner_utils.Edge('T2', 'T2', (1.0, 0.0), (2.0, 0.0), 1.0, 0.0, [(1.0, 0.0), (2.0, 0.0)], 'trail', 'both')
    b1 = planner_utils.Edge('B1', 'B1', (1.0, 0.0), (1.0, 1.0), 1.0, 0.0, [(1.0, 0.0), (1.0, 1.0)], 'trail', 'both')
    b2 = planner_utils.Edge('B2', 'B2', (2.0, 0.0), (2.0, 1.0), 1.0, 0.0, [(2.0, 0.0), (2.0, 1.0)], 'trail', 'both')
    return [t1, t2, b1, b2]


def test_tree_route_uses_trunk_twice():
    edges = build_tree_edges()
    G = challenge_planner.build_nx_graph(edges, pace=10.0, grade=0.0, road_pace=10.0)
    route = challenge_planner.plan_route(
        G,
        edges,
        (0.0, 0.0),
        pace=10.0,
        grade=0.0,
        road_pace=10.0,
        max_road=0.0,
        road_threshold=0.1,
        use_rpp=True,
    )
    seg_ids = [e.seg_id for e in route]
    assert seg_ids.count('T1') == 2
    assert seg_ids.count('T2') == 2

