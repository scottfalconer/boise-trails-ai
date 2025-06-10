import csv
import json
import gpxpy
import pytest

from trail_route_ai import planner_utils, challenge_planner


def build_edges(n=3):
    """Create n simple connected segments forming a chain."""
    edges = []
    for i in range(n):
        start = (float(i), 0.0)
        end = (float(i) + 1.0, 0.0)
        seg_id = f"S{i}"
        edges.append(
            planner_utils.Edge(seg_id, seg_id, start, end, 1.0, 0.0, [start, end])
        )
    return edges


def write_segments(path, edges):
    data = {"segments": []}
    for e in edges:
        data["segments"].append({"id": e.seg_id, "name": e.name, "coordinates": e.coords})
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.mark.parametrize("count", [1, 31])
def test_cluster_limit(count):
    edges = build_edges(count)
    clusters = challenge_planner.cluster_segments(
        edges, pace=10.0, grade=0.0, budget=30.0, max_clusters=30
    )
    assert len(clusters) <= 30


def test_planner_outputs(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-03",
            "--time",
            "30",
            "--pace",
            "10",
            "--segments",
            str(seg_path),
            "--perf",
            str(perf_path),
            "--year",
            "2024",
            "--output",
            str(out_csv),
            "--gpx-dir",
            str(gpx_dir),
        ]
    )

    rows = list(csv.DictReader(open(out_csv)))
    assert rows
    for row in rows:
        assert float(row["time_min"]) <= 30.0
        day_str = row["date"].replace("-", "")
        gpx_file = gpx_dir / f"{day_str}.gpx"
        assert gpx_file.exists()
        with open(gpx_file) as f:
            gpx = gpxpy.parse(f)
        pts = [
            (pt.longitude, pt.latitude)
            for trk in gpx.tracks
            for seg in trk.segments
            for pt in seg.points
        ]
        assert pts[0] == pts[-1]
        assert "plan" in row and row["plan"]


def test_completed_excluded(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    write_segments(seg_path, edges)
    with open(perf_path, "w") as f:
        f.write("seg_id,year\nS1,2024\n")

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-03",
            "--time",
            "30",
            "--pace",
            "10",
            "--segments",
            str(seg_path),
            "--perf",
            str(perf_path),
            "--year",
            "2024",
            "--output",
            str(out_csv),
            "--gpx-dir",
            str(gpx_dir),
        ]
    )

    rows = list(csv.DictReader(open(out_csv)))
    segs = {row["segments"] for row in rows}
    assert "S1" not in " ".join(segs)

