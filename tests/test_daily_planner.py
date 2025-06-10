import json
import pytest
import gpxpy
from scripts import daily_planner


def build_sample_edges():
    # simple triangle A-B-C-A
    edges = [
        daily_planner.Edge('A', 'A-B', (-1.0, 0.0), (0.0, 0.0), 1.0, 0.0,
                          [(-1.0, 0.0), (0.0, 0.0)]),
        daily_planner.Edge('B', 'B-C', (0.0, 0.0), (0.0, 1.0), 1.0, 0.0,
                          [(0.0, 0.0), (0.0, 1.0)]),
        daily_planner.Edge('C', 'C-A', (0.0, 1.0), (-1.0, 0.0), 1.0, 0.0,
                          [(0.0, 1.0), (-1.0, 0.0)]),
    ]
    return edges


@pytest.fixture
def two_clusters(tmp_path):
    """Create a segments file with two trailhead clusters."""
    segments = {
        "segments": [
            # cluster 1 - single segment that must be repeated to form a loop
            {"id": "A", "name": "A", "coordinates": [[0.0, 0.0], [1.0, 0.0]]},
            # cluster 2 - triangle of three unique segments
            {"id": "B1", "name": "B1", "coordinates": [[10.0, 0.0], [11.0, 0.0]]},
            {"id": "B2", "name": "B2", "coordinates": [[11.0, 0.0], [11.0, 1.0]]},
            {"id": "B3", "name": "B3", "coordinates": [[11.0, 1.0], [10.0, 0.0]]},
        ]
    }
    seg_path = tmp_path / "segments.json"
    with open(seg_path, "w") as f:
        json.dump(segments, f)
    perf_path = tmp_path / "perf.csv"
    perf_path.write_text("seg_id,year\n")
    return seg_path, perf_path


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
    edge = daily_planner.Edge('X', 'A-B', (0.0, 0.0), (1.0, 0.0), 1.0, 0.0,
                              [(0.0, 0.0), (1.0, 0.0)])
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
    edges.append(daily_planner.Edge('A', 'B-A', (0.0, 0.0), (-1.0, 0.0), 1.0, 0.0,
                                   [(0.0, 0.0), (-1.0, 0.0)]))
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


def test_best_trailhead_selection(two_clusters, monkeypatch, capsys):
    seg_path, perf_path = two_clusters
    monkeypatch.chdir(seg_path.parent)
    daily_planner.main([
        "--time",
        "40",
        "--pace",
        "10",
        "--segments",
        str(seg_path),
        "--perf",
        str(perf_path),
    ])
    out = capsys.readouterr().out
    assert "Route Summary" in out
    # planner should choose a trailhead from the second cluster
    assert any(t in out for t in ["(10.0, 0.0)", "(11.0, 0.0)", "(11.0, 1.0)"])


def test_no_repeat_except_closure():
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
    seg_ids = [e.seg_id for e in result["path"]]
    if seg_ids[0] == seg_ids[-1]:
        assert len(seg_ids) - 1 == len(set(seg_ids))
    else:
        assert len(seg_ids) == len(set(seg_ids))


def test_write_gpx(tmp_path):
    edges = build_sample_edges()
    out_file = tmp_path / "out.gpx"
    daily_planner.write_gpx(out_file, edges)
    with open(out_file) as f:
        gpx = gpxpy.parse(f)
    pts = [
        (pt.longitude, pt.latitude)
        for trk in gpx.tracks
        for seg in trk.segments
        for pt in seg.points
    ]
    expected = []
    for i, e in enumerate(edges):
        seg_coords = [tuple(c) for c in e.coords]
        if i == 0:
            expected.extend(seg_coords)
        else:
            last = expected[-1]
            if daily_planner._close(last, seg_coords[0]):
                expected.extend(seg_coords[1:])
            elif daily_planner._close(last, seg_coords[-1]):
                expected.extend(list(reversed(seg_coords[:-1])))
            else:
                expected.extend(seg_coords)
    assert pts == expected

