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
        max_segments=5,
    )
    assert result is not None
    assert len(result['path']) == 3
    assert result['new_count'] == 3
    assert abs(result['time'] - 30.0) < 1e-6


def test_reuse_only_to_close():
    """Single segment should be allowed twice only when closing the loop."""
    edge = daily_planner.Edge('X', 'A-B', (0.0, 0.0), (1.0, 0.0), 1.0, 0.0)
    graph = daily_planner.build_graph([edge])
    result = daily_planner.search_loops(
        graph,
        edge.start,
        pace=10.0,
        grade=0.0,
        time_budget=25.0,
        completed=set(),
        max_segments=2,
    )
    assert result is not None
    seg_ids = [e.seg_id for e in result['path']]
    assert seg_ids == ['X', 'X']


def test_maximize_unique_segments():
    edges = build_sample_edges()
    # add reverse edge allowing a short loop with repeated ID
    edges.append(daily_planner.Edge('A', 'B-A', (0.0, 0.0), (-1.0, 0.0), 1.0, 0.0))
    graph = daily_planner.build_graph(edges)
    result = daily_planner.search_loops(
        graph,
        edges[0].start,
        pace=10.0,
        grade=0.0,
        time_budget=40.0,
        completed=set(),
        max_segments=5,
    )
    assert result is not None
    # best loop uses three unique segments rather than repeating 'A'
    seg_ids = [e.seg_id for e in result['path']]
    assert len(set(seg_ids)) == 3
    assert len(seg_ids) == 3

