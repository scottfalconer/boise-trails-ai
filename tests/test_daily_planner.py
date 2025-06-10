from scripts import daily_planner


def build_sample_edges():
    # simple triangle A-B-C-A
    edges = [
        daily_planner.Edge('A', 'A-B', (-1.0, 0.0), (0.0, 0.0), 1.0, 0.0),
        daily_planner.Edge('B', 'B-C', (0.0, 0.0), (0.0, 1.0), 1.0, 0.0),
        daily_planner.Edge('C', 'C-A', (0.0, 1.0), (-1.0, 0.0), 1.0, 0.0),
    ]
    return edges


def test_simple_loop():
    edges = build_sample_edges()
    graph = daily_planner.build_graph(edges)
    result = daily_planner.search_loops(
        graph,
        edges[0].start,
        pace=10.0,
        grade=0.0,
        time_budget=40.0,
        completed=set(),
        max_depth=5,
    )
    assert result is not None
    assert len(result['path']) == 3
    assert result['new_count'] == 3
    assert abs(result['time'] - 30.0) < 1e-6

