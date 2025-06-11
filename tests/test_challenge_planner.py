import csv
import json
import gpxpy
import pytest

from trail_route_ai import planner_utils, challenge_planner
import numpy as np
import rasterio
from rasterio.transform import from_origin


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


def create_dem(path):
    data = np.tile(np.arange(4, dtype=np.float32) * 10, (2, 1))
    transform = from_origin(0, 1, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.mark.parametrize("count", [1, 31])
def test_cluster_limit(count):
    edges = build_edges(count)
    clusters = challenge_planner.cluster_segments(
        edges, pace=10.0, grade=0.0, budget=30.0, max_clusters=30, road_pace=18.0
    )
    assert len(clusters) <= 30


def test_planner_outputs(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)
    create_dem(dem_path)

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
            "--dem",
            str(dem_path),
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
        assert float(row["total_activity_time_min"]) <= 30.0
        day_str = row["date"].replace("-", "")
        gpx_files = list(gpx_dir.glob(f"{day_str}_part*.gpx"))
        assert gpx_files
        with open(gpx_files[0]) as f:
            gpx = gpxpy.parse(f)
        pts = [
            (pt.longitude, pt.latitude)
            for trk in gpx.tracks
            for seg in trk.segments
            for pt in seg.points
        ]
        assert len(pts) >= 2
        assert "plan_description" in row and row["plan_description"]
        assert float(row["total_trail_elev_gain_ft"]) > 0
        assert "notes" in row


def test_completed_excluded(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    write_segments(seg_path, edges)
    create_dem(dem_path)
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
            "--dem",
            str(dem_path),
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
    text = " ".join(row["plan_description"] for row in rows)
    assert "S1" not in text
    for row in rows:
        assert "notes" in row


def test_write_gpx_marks_roads(tmp_path):
    edges = [
        planner_utils.Edge("T1", "T1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)]),
        planner_utils.Edge("R1", "R1", (1.0, 0.0), (2.0, 0.0), 1.0, 0.0, [(1.0, 0.0), (2.0, 0.0)], kind="road"),
        planner_utils.Edge("T2", "T2", (2.0, 0.0), (3.0, 0.0), 1.0, 0.0, [(2.0, 0.0), (3.0, 0.0)]),
    ]
    gpx_file = tmp_path / "out.gpx"
    planner_utils.write_gpx(gpx_file, edges, mark_road_transitions=True)
    with open(gpx_file) as f:
        gpx = gpxpy.parse(f)

    seg_kinds = [seg.extensions[0].text for seg in gpx.tracks[0].segments]
    assert seg_kinds == ["trail", "road", "trail"]
    assert len(gpx.waypoints) == 2
    assert gpx.waypoints[0].name == "Road start"
    assert gpx.waypoints[1].name == "Road end"


def test_multiday_gpx(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    # Include a length so the planner estimates non-zero time
    write_segments(seg_path, edges)
    create_dem(dem_path)

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
            "--dem",
            str(dem_path),
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

    full_gpx = gpx_dir / "full_timespan.gpx"
    assert full_gpx.exists()
    with open(full_gpx) as f:
        gpx = gpxpy.parse(f)
    dates_in_csv = [row["date"] for row in csv.DictReader(open(out_csv))]
    names = [trk.name for trk in gpx.tracks]
    assert names == dates_in_csv


def test_oversized_cluster_split(tmp_path):
    edges = build_edges(10)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    data = {"segments": []}
    for e in edges:
        data["segments"].append({"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280})
    with open(seg_path, "w") as f:
        json.dump(data, f)
    create_dem(dem_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-05",
            "--time",
            "30",
            "--pace",
            "10",
            "--segments",
            str(seg_path),
            "--dem",
            str(dem_path),
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
    assert len(rows) > 1


def test_daily_hours_file(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    data = {"segments": []}
    for e in edges:
        data["segments"].append({"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280})
    with open(seg_path, "w") as f:
        json.dump(data, f)
    create_dem(dem_path)

    hours_file = tmp_path / "daily_hours.json"
    json.dump({"2024-07-01": 0.0}, hours_file.open("w"))

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-01",
            "--time",
            "30",
            "--pace",
            "10",
            "--segments",
            str(seg_path),
            "--dem",
            str(dem_path),
            "--perf",
            str(perf_path),
            "--year",
            "2024",
            "--output",
            str(out_csv),
            "--gpx-dir",
            str(gpx_dir),
            "--daily-hours-file",
            str(hours_file),
        ]
    )

    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 1
    assert rows[0]["plan_description"] == "Unable to complete"

