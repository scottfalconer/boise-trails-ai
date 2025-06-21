from trail_route_ai import challenge_planner, planner_utils


def test_small_one_way_cluster_returns_reverse_connector(monkeypatch):
    monkeypatch.setattr(challenge_planner, "_HAVE_SCIPY", False)
    seg1 = planner_utils.Edge(
        "S1", "S1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], "trail", "ascent"
    )
    seg2 = planner_utils.Edge(
        "S2", "S2", (1.0, 0.0), (2.0, 0.0), 1.0, 0.0, [(1.0, 0.0), (2.0, 0.0)], "trail", "ascent"
    )
    G = challenge_planner.build_nx_graph(
        [seg1, seg2], pace=10.0, grade=0.0, road_pace=10.0
    )
    route = challenge_planner.plan_route(
        G,
        [seg1, seg2],
        (0.0, 0.0),
        pace=10.0,
        grade=0.0,
        road_pace=10.0,
        max_foot_road=0.0,
        road_threshold=0.1,
        use_rpp=False,
    )
    seg_ids = [e.seg_id for e in route]
    assert seg_ids == ["S1", "S2", "S2", "S1"]
    assert route[-1].end == (0.0, 0.0)
