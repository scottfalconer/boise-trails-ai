import networkx as nx
from trail_route_ai import challenge_planner, planner_utils


def build_test_graph():
    t1 = planner_utils.Edge("T1", "T1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], "trail", "both")
    t2 = planner_utils.Edge("T2", "T2", (3.0, 0.0), (4.0, 0.0), 1.0, 0.0, [(3.0, 0.0), (4.0, 0.0)], "trail", "both")
    r1 = planner_utils.Edge("R1", "R1", (1.0, 0.0), (3.0, 0.0), 2.0, 0.0, [(1.0, 0.0), (3.0, 0.0)], "road", "both")
    r2 = planner_utils.Edge("R2", "R2", (4.0, 0.0), (0.0, 0.0), 4.0, 0.0, [(4.0, 0.0), (0.0, 0.0)], "road", "both")
    G = challenge_planner.build_nx_graph([t1, t2, r1, r2], pace=10.0, grade=0.0, road_pace=15.0)
    return G, [t1, t2]


def old_plan_route_greedy(G, edges, start, pace, grade, road_pace, max_road, road_threshold):
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
                    road_dist = sum(ed.length_mi for ed in edges_path if ed.kind == "road")
                    if road_dist > max_road:
                        continue
                    time = sum(
                        planner_utils.estimate_time(ed, pace, grade, road_pace) for ed in edges_path
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
                            planner_utils.estimate_time(ed, pace, grade, road_pace) for ed in edges_path
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
            if 'weight' not in edge_data:
                edge_data['weight'] = planner_utils.estimate_time(edge_obj, pace, grade, road_pace)
            edge_data['weight'] = edge_data['weight'] * 10.0
    try:
        path_back_nodes = nx.shortest_path(G_for_path_back, cur, start, weight='weight')
        route.extend(challenge_planner.edges_from_path(G, path_back_nodes))
    except nx.NetworkXNoPath:
        try:
            path_back_nodes_orig = nx.shortest_path(G, cur, start, weight='weight')
            route.extend(challenge_planner.edges_from_path(G, path_back_nodes_orig))
        except nx.NetworkXNoPath:
            pass
    return route, order


def test_greedy_identical_to_old():
    G, trails = build_test_graph()
    params = dict(pace=10.0, grade=0.0, road_pace=15.0, max_road=1.0, road_threshold=0.1)
    new_route, new_order = challenge_planner._plan_route_greedy(G, trails, (0.0, 0.0), **params)
    old_route, old_order = old_plan_route_greedy(G, trails, (0.0, 0.0), **params)

    assert [(e.seg_id, e.start, e.end) for e in new_route] == [
        (e.seg_id, e.start, e.end) for e in old_route
    ]
    assert [e.seg_id for e in new_order] == [e.seg_id for e in old_order]
