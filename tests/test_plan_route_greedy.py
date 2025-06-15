import networkx as nx
from trail_route_ai import challenge_planner, planner_utils


def build_test_graph():
    t1 = planner_utils.Edge(
        "T1",
        "T1",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    t2 = planner_utils.Edge(
        "T2",
        "T2",
        (3.0, 0.0),
        (4.0, 0.0),
        1.0,
        0.0,
        [(3.0, 0.0), (4.0, 0.0)],
        "trail",
        "both",
    )
    r1 = planner_utils.Edge(
        "R1",
        "R1",
        (1.0, 0.0),
        (3.0, 0.0),
        2.0,
        0.0,
        [(1.0, 0.0), (3.0, 0.0)],
        "road",
        "both",
    )
    r2 = planner_utils.Edge(
        "R2",
        "R2",
        (4.0, 0.0),
        (0.0, 0.0),
        4.0,
        0.0,
        [(4.0, 0.0), (0.0, 0.0)],
        "road",
        "both",
    )
    G = challenge_planner.build_nx_graph(
        [t1, t2, r1, r2], pace=10.0, grade=0.0, road_pace=15.0
    )
    return G, [t1, t2]


def build_graph_with_trail_connector(trail_len: float) -> tuple[nx.DiGraph, list[planner_utils.Edge]]:
    """Graph with both a road and trail connector between ``t1`` and ``t2``."""
    t1 = planner_utils.Edge(
        "T1",
        "T1",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    t2 = planner_utils.Edge(
        "T2",
        "T2",
        (3.0, 0.0),
        (4.0, 0.0),
        1.0,
        0.0,
        [(3.0, 0.0), (4.0, 0.0)],
        "trail",
        "both",
    )
    road_conn_a = planner_utils.Edge(
        "R1a",
        "R1",
        (1.0, 0.0),
        (2.0, 0.0),
        1.0,
        0.0,
        [(1.0, 0.0), (2.0, 0.0)],
        "road",
        "both",
    )
    road_conn_b = planner_utils.Edge(
        "R1b",
        "R1",
        (2.0, 0.0),
        (3.0, 0.0),
        1.0,
        0.0,
        [(2.0, 0.0), (3.0, 0.0)],
        "road",
        "both",
    )
    trail_conn = planner_utils.Edge(
        "C1",
        "C1",
        (1.0, 0.0),
        (3.0, 0.0),
        trail_len,
        0.0,
        [(1.0, 0.0), (3.0, 0.0)],
        "trail",
        "both",
    )
    r2 = planner_utils.Edge(
        "R2",
        "R2",
        (4.0, 0.0),
        (0.0, 0.0),
        4.0,
        0.0,
        [(4.0, 0.0), (0.0, 0.0)],
        "road",
        "both",
    )
    G = challenge_planner.build_nx_graph(
        [
            t1,
            t2,
            road_conn_a,
            road_conn_b,
            trail_conn,
            r2,
        ],
        pace=10.0,
        grade=0.0,
        road_pace=15.0,
    )
    return G, [t1, t2]


def old_plan_route_greedy(
    G, edges, start, pace, grade, road_pace, max_road, road_threshold
):
    remaining = edges[:]
    route = []
    order = []
    cur = start
    while remaining:
        candidates = []
        for e in remaining:
            for end in [e.start, e.end]:
                if end == e.end and e.direction != "both":
                    continue
                try:
                    path = nx.shortest_path(G, cur, end, weight="weight")
                    edges_path = challenge_planner.edges_from_path(G, path)
                    road_dist = sum(
                        ed.length_mi for ed in edges_path if ed.kind == "road"
                    )
                    if road_dist > max_road:
                        continue
                    time = sum(
                        planner_utils.estimate_time(ed, pace, grade, road_pace)
                        for ed in edges_path
                    )
                    time += planner_utils.estimate_time(e, pace, grade, road_pace)
                    uses_road = any(ed.kind == "road" for ed in edges_path)
                    candidates.append((time, uses_road, e, end, edges_path))
                except nx.NetworkXNoPath:
                    continue
        if not candidates:
            for e in remaining:
                for end in [e.start, e.end]:
                    if end == e.end and e.direction != "both":
                        continue
                    try:
                        path = nx.shortest_path(G, cur, end, weight="weight")
                        edges_path = challenge_planner.edges_from_path(G, path)
                        time = sum(
                            planner_utils.estimate_time(ed, pace, grade, road_pace)
                            for ed in edges_path
                        )
                        time += planner_utils.estimate_time(e, pace, grade, road_pace)
                        uses_road = any(ed.kind == "road" for ed in edges_path)
                        candidates.append((time, uses_road, e, end, edges_path))
                    except nx.NetworkXNoPath:
                        continue
            if not candidates:
                return []
        best = min(candidates, key=lambda c: c[0])
        trail_candidates = [c for c in candidates if not c[1]]
        if trail_candidates:
            best_trail = min(trail_candidates, key=lambda c: c[0])
            if best_trail[0] <= best[0] * (1 + road_threshold):
                chosen = best_trail
            else:
                chosen = best
        else:
            chosen = best
        _, _, e, end, best_path_edges = chosen
        route.extend(best_path_edges)
        if end == e.start:
            route.append(e)
            order.append(e)
            cur = e.end
        else:
            if e.direction != "both":
                return []
            rev = planner_utils.Edge(
                e.seg_id,
                e.name,
                e.end,
                e.start,
                e.length_mi,
                e.elev_gain_ft,
                list(reversed(e.coords)),
                e.kind,
                e.direction,
            )
            route.append(rev)
            order.append(e)
            cur = rev.end
        remaining.remove(e)
    if cur == start:
        return route
    G_for_path_back = G.copy()
    for edge_obj in edges:
        if G_for_path_back.has_edge(edge_obj.start, edge_obj.end):
            edge_data = G_for_path_back[edge_obj.start][edge_obj.end]
            if "weight" not in edge_data:
                edge_data["weight"] = planner_utils.estimate_time(
                    edge_obj, pace, grade, road_pace
                )
            edge_data["weight"] = edge_data["weight"] * 1.2
    try:
        path_back_nodes = nx.shortest_path(G_for_path_back, cur, start, weight="weight")
        route.extend(challenge_planner.edges_from_path(G, path_back_nodes))
    except nx.NetworkXNoPath:
        try:
            path_back_nodes_orig = nx.shortest_path(G, cur, start, weight="weight")
            route.extend(challenge_planner.edges_from_path(G, path_back_nodes_orig))
        except nx.NetworkXNoPath:
            pass
    return route, order


def test_greedy_allows_overlimit_when_no_trail():
    G, trails = build_test_graph()
    params = dict(
        pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.1
    )
    route, order = challenge_planner._plan_route_greedy(
        G, trails, (0.0, 0.0), **params, dist_cache={}
    )

    # With no trail connector available, the planner should use the road
    # connector even though it exceeds ``max_road``.
    assert route
    assert order == trails


def test_overlimit_road_vs_trail_fallback():
    """When a trail connector is nearly as fast, the planner should prefer it."""
    G, trails = build_graph_with_trail_connector(trail_len=1.45)
    params = dict(
        pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.1
    )
    route, _ = challenge_planner._plan_route_greedy(
        G, trails, (0.0, 0.0), **params, dist_cache={}
    )

    # The trail connector should be used instead of the faster but over-limit road.
    names = [e.name for e in route]
    assert "C1" in names
    assert "R1" not in names


def test_overlimit_road_chosen_when_much_faster():
    """Road connector is used when significantly faster than trail alternative."""
    G, trails = build_graph_with_trail_connector(trail_len=5.0)
    params = dict(
        pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.1
    )
    route, _ = challenge_planner._plan_route_greedy(
        G, trails, (0.0, 0.0), **params, dist_cache={}
    )

    names = [e.name for e in route]
    assert "R1" in names


def test_greedy_fallback_handles_unreachable_segment():
    """Greedy fallback should exit cleanly when a segment cannot be reached."""
    seg_a = planner_utils.Edge(
        "A",
        "A",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    seg_b = planner_utils.Edge(
        "B",
        "B",
        (5.0, 0.0),
        (6.0, 0.0),
        1.0,
        0.0,
        [(5.0, 0.0), (6.0, 0.0)],
        "trail",
        "both",
    )
    G = challenge_planner.build_nx_graph(
        [seg_a, seg_b], pace=10.0, grade=0.0, road_pace=15.0
    )

    route = challenge_planner.plan_route(
        G,
        [seg_a, seg_b],
        (0.0, 0.0),
        pace=10.0,
        grade=0.0,
        road_pace=15.0,
        max_road=1.0,
        road_threshold=0.1,
        use_rpp=False,
    )

    # With B unreachable the planner should return an empty list quickly.
    assert route == []
