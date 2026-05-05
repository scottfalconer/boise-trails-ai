import sys
sys.path.append('src')

from trail_route_ai import planner_utils, challenge_planner


def build_graph():
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (2.0, 0.0)
    D = (2.0, 1.0)
    t1 = planner_utils.Edge('T1', 'T1', A, B, 1.0, 0.0, [A, B], 'trail', 'both')
    t2 = planner_utils.Edge('T2', 'T2', B, C, 1.0, 0.0, [B, C], 'trail', 'both')
    t3 = planner_utils.Edge('T3', 'T3', B, D, 1.0, 0.0, [B, D], 'trail', 'both')
    conn = planner_utils.Edge('X', 'X', C, D, 0.5, 0.0, [C, D], 'trail', 'both')
    G = challenge_planner.build_nx_graph([t1, t2, t3, conn], pace=10.0, grade=0.0, road_pace=10.0)
    return G, t1, t2, t3, conn


def test_calculate_route_efficiency_score():
    e1 = planner_utils.Edge('A', 'A', (0, 0), (1, 0), 1.0, 0.0, [(0, 0), (1, 0)])
    e2 = planner_utils.Edge('B', 'B', (1, 0), (2, 0), 1.0, 0.0, [(1, 0), (2, 0)])
    score = planner_utils.calculate_route_efficiency_score([e1, e2, e1])
    assert round(score, 2) == 0.67


def test_optimize_route_for_redundancy():
    G, t1, t2, t3, conn = build_graph()
    ctx = planner_utils.PlanningContext(G, 10.0, 0.0, 10.0, None)
    t2_rev = planner_utils.Edge('T2', 'T2', (2.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(2.0, 0.0), (1.0, 0.0)], 'trail', 'both')
    t3_rev = planner_utils.Edge('T3', 'T3', (2.0, 1.0), (1.0, 0.0), 1.0, 0.0, [(2.0, 1.0), (1.0, 0.0)], 'trail', 'both')
    t1_rev = planner_utils.Edge('T1', 'T1', (1.0, 0.0), (0.0, 0.0), 1.0, 0.0, [(1.0, 0.0), (0.0, 0.0)], 'trail', 'both')
    route = [t1, t2, t2_rev, t3, t3_rev, t1_rev]
    optimized = planner_utils.optimize_route_for_redundancy(ctx, route, {'T1', 'T2', 'T3'}, 0.5)
    seg_ids = [e.seg_id for e in optimized]
    assert 'X' in seg_ids
    assert seg_ids.count('T2') == 1
    assert planner_utils.calculate_route_efficiency_score(optimized) >= planner_utils.calculate_route_efficiency_score(route)
