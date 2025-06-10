import pytest
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


@pytest.fixture
def two_cluster_setup():
    edges = [
        daily_planner.Edge('th1a', '1A', (0.0, 0.0), (1.0, 0.0), 1.0, 0.0),
        daily_planner.Edge('th1b', '1B', (1.0, 0.0), (0.0, 0.0), 1.0, 0.0),
        daily_planner.Edge('th2a', '2A', (0.0, 1.0), (1.0, 1.0), 1.0, 0.0),
        daily_planner.Edge('th2b', '2B', (1.0, 1.0), (1.0, 2.0), 1.0, 0.0),
        daily_planner.Edge('th2c', '2C', (1.0, 2.0), (0.0, 1.0), 1.0, 0.0),
    ]
    graph = daily_planner.build_graph(edges)
    trailheads = {'th1': (0.0, 0.0), 'th2': (0.0, 1.0)}
    completed = {'th1a'}
    return graph, trailheads, completed


def test_select_best_trailhead(two_cluster_setup):
    graph, trailheads, completed = two_cluster_setup
    th, result = daily_planner.choose_trailhead(
        graph,
        trailheads,
        pace=10.0,
        grade=0.0,
        time_budget=60.0,
        completed=completed,
        max_segments=3,
    )
    assert th == 'th2'
    assert result['new_count'] == 3


def test_no_repeated_seg_ids():
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
    seg_ids = [e.seg_id for e in result['path']]
    assert len(seg_ids) == len(set(seg_ids))


def test_write_gpx(tmp_path):
    edges = build_sample_edges()[:2]
    path = [edges[0], edges[1]]
    out = tmp_path / 'route.gpx'
    daily_planner.write_gpx(path, out)
    import gpxpy

    with open(out) as fh:
        gpx = gpxpy.parse(fh)

    coords = [
        (p.longitude, p.latitude)
        for trk in gpx.tracks
        for seg in trk.segments
        for p in seg.points
    ]
    expected = [path[0].start, path[0].end, path[1].end]
    assert coords == expected

