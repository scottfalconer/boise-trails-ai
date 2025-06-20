from trail_route_ai import planner_utils, challenge_planner


def test_split_cluster_by_one_way_basic():
    e1 = planner_utils.Edge("A", "A", (0.0, 0.0), (0.1, 0.0), 0.1, 0.0, [(0.0, 0.0), (0.1, 0.0)], "trail", "both")
    e2 = planner_utils.Edge("B", "B", (0.1, 0.0), (0.2, 0.0), 0.1, 0.0, [(0.1, 0.0), (0.2, 0.0)], "trail", "ascent")
    e3 = planner_utils.Edge("C", "C", (0.2, 0.0), (0.3, 0.0), 0.1, 0.0, [(0.2, 0.0), (0.3, 0.0)], "trail", "both")

    groups = challenge_planner.split_cluster_by_one_way([e1, e2, e3])
    assert len(groups) == 3
    lengths = sorted(len(g) for g in groups)
    assert lengths == [1, 1, 2]
    assert any(len(g) == 1 and g[0].seg_id == "B" for g in groups)
