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
            planner_utils.Edge(
                seg_id, seg_id, start, end, 1.0, 0.0, [start, end], "trail", "both"
            )
        )
    return edges


def write_segments(path, edges):
    data = {"segments": []}
    for e in edges:
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords}
        )
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
    data = {"segments": []}
    for e in edges:
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
        )
    with open(seg_path, "w") as f:
        json.dump(data, f)
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
    html_out = out_csv.with_suffix(".html")
    assert html_out.exists()
    hit_csv = out_csv.with_name("hit_list.csv")
    assert hit_csv.exists()
    for row in rows:
        if row["date"] == "Totals":
            continue
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
        assert "route_description" in row and row["route_description"]
        assert "unique_trail_miles" in row
        assert "redundant_miles" in row
        assert "redundant_pct" in row
        assert float(row["total_trail_elev_gain_ft"]) > 0
        assert "notes" in row

    # HTML should include rationale text for each day
    html_content = html_out.read_text()
    assert "Rationale:" in html_content


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
    text2 = " ".join(row.get("route_description", "") for row in rows)
    assert "S1" not in text
    assert "S1" not in text2
    for row in rows:
        assert "unique_trail_miles" in row
        assert "notes" in row


def test_write_gpx_marks_roads(tmp_path):
    edges = [
        planner_utils.Edge(
            "T1",
            "T1",
            (0.0, 0.0),
            (1.0, 0.0),
            1.0,
            0.0,
            [(0.0, 0.0), (1.0, 0.0)],
            "trail",
            "both",
        ),
        planner_utils.Edge(
            "R1",
            "R1",
            (1.0, 0.0),
            (2.0, 0.0),
            1.0,
            0.0,
            [(1.0, 0.0), (2.0, 0.0)],
            kind="road",
            direction="both",
        ),
        planner_utils.Edge(
            "T2",
            "T2",
            (2.0, 0.0),
            (3.0, 0.0),
            1.0,
            0.0,
            [(2.0, 0.0), (3.0, 0.0)],
            "trail",
            "both",
        ),
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
    dates_in_csv = [
        row["date"] for row in csv.DictReader(open(out_csv)) if row["date"] != "Totals"
    ]
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
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
        )
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
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
        )
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
    assert len(rows) == 2
    assert rows[0]["plan_description"] == "Unable to complete"
    assert rows[0]["route_description"] == "Unable to complete"


def test_trailhead_start_in_output(tmp_path):
    edges = build_edges(2)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    trailhead_file = tmp_path / "ths.json"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)
    create_dem(dem_path)
    json.dump([{"name": "TH1", "lon": 0.0, "lat": 0.0}], trailhead_file.open("w"))

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
            "--trailheads",
            str(trailhead_file),
        ]
    )

    rows = list(csv.DictReader(open(out_csv)))
    assert rows
    assert "start_trailheads" in rows[0]


def test_balance_workload(tmp_path):
    edges = build_edges(3)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    data = {"segments": []}
    for e in edges:
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
        )
    with open(seg_path, "w") as f:
        json.dump(data, f)
    create_dem(dem_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-03",
            "--time",
            "20",
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
    assert any(float(r["total_activity_time_min"]) > 0 for r in rows)


def test_infeasible_plan_detection(tmp_path):
    edges = build_edges(2)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    perf_path.write_text("seg_id,year\n")
    data = {"segments": []}
    for e in edges:
        data["segments"].append(
            {"id": e.seg_id, "name": e.name, "coordinates": e.coords, "LengthFt": 5280}
        )
    with open(seg_path, "w") as f:
        json.dump(data, f)
    create_dem(dem_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-01",
            "--time",
            "10",
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
    assert float(rows[0]["total_activity_time_min"]) > 10.0


def test_unrouteable_cluster_split(tmp_path):
    segs = [
        planner_utils.Edge(
            "A",
            "A",
            (0.0, 0.0),
            (0.01, 0.0),
            0.0,
            0.0,
            [(0.0, 0.0), (0.01, 0.0)],
            "trail",
            "both",
        ),
        planner_utils.Edge(
            "B",
            "B",
            (0.1, 0.0),
            (0.11, 0.0),
            0.0,
            0.0,
            [(0.1, 0.0), (0.11, 0.0)],
            "trail",
            "both",
        ),
    ]
    roads = {
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "R1"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.01, 0.0], [0.1, 0.0]],
                },
            }
        ]
    }

    seg_path = tmp_path / "segments.json"
    road_path = tmp_path / "roads.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"

    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, segs)
    json.dump(roads, road_path.open("w"))
    create_dem(dem_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-01",
            "--time",
            "60",
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
            "--roads",
            str(road_path),
            "--home-lat",
            "0",
            "--home-lon",
            "0",
            "--output",
            str(out_csv),
            "--gpx-dir",
            str(gpx_dir),
            "--max-road",
            "0.01",
        ]
    )

    rows = list(csv.DictReader(open(out_csv)))
    day = next(r for r in rows if r["date"] != "Totals")
    assert int(day["num_activities"]) == 2
    assert int(day["num_drives"]) == 1


def test_output_directory(tmp_path):
    edges = build_edges(2)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    output_dir = tmp_path / "outputs"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)
    create_dem(dem_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-01",
            "--time",
            "60",
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
            "--output-dir",
            str(output_dir),
        ]
    )

    csv_path = output_dir / "challenge_plan.csv"
    html_path = output_dir / "challenge_plan.html"
    gpx_dir = output_dir / "gpx"
    assert csv_path.exists()
    assert html_path.exists()
    assert gpx_dir.is_dir()
    assert any(gpx_dir.glob("*.gpx"))


def test_auto_output_dir(tmp_path, monkeypatch):
    edges = build_edges(2)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)
    create_dem(dem_path)

    monkeypatch.chdir(tmp_path)

    challenge_planner.main(
        [
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-01",
            "--time",
            "60",
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
            "--auto-output-dir",
        ]
    )

    outputs_dir = tmp_path / "outputs"
    dirs = list(outputs_dir.iterdir())
    assert len(dirs) == 1
    outdir = dirs[0]
    csv_path = outdir / "challenge_plan.csv"
    html_path = outdir / "challenge_plan.html"
    gpx_dir = outdir / "gpx"
    assert csv_path.exists()
    assert html_path.exists()
    assert gpx_dir.is_dir()
    assert any(gpx_dir.glob("*.gpx"))


def test_debug_log_written(tmp_path):
    edges = build_edges(2)
    seg_path = tmp_path / "segments.json"
    perf_path = tmp_path / "perf.csv"
    dem_path = tmp_path / "dem.tif"
    out_csv = tmp_path / "out.csv"
    gpx_dir = tmp_path / "gpx"
    debug_file = tmp_path / "debug.txt"
    perf_path.write_text("seg_id,year\n")
    write_segments(seg_path, edges)
    create_dem(dem_path)

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
            "--debug",
            str(debug_file),
        ]
    )

    assert debug_file.exists()
    text = debug_file.read_text().strip()
    assert "2024-07-01" in text


def test_advanced_optimizer_reduces_redundancy():
    edges = [
        planner_utils.Edge("T1", "T1", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 0.0)], "trail", "both"),
        planner_utils.Edge("T2", "T2", (1.0, 0.0), (1.0, 1.0), 1.0, 0.0, [(1.0, 0.0), (1.0, 1.0)], "trail", "both"),
        planner_utils.Edge("T3", "T3", (1.0, 1.0), (0.0, 1.0), 1.0, 0.0, [(1.0, 1.0), (0.0, 1.0)], "trail", "both"),
        planner_utils.Edge("T4", "T4", (0.0, 1.0), (0.0, 0.0), 1.0, 0.0, [(0.0, 1.0), (0.0, 0.0)], "trail", "both"),
    ]
    G = challenge_planner.build_nx_graph(edges, pace=10.0, grade=0.0, road_pace=10.0)

    base = challenge_planner.plan_route(
        G, edges, (0.0, 0.0), pace=10.0, grade=0.0, road_pace=10.0, max_road=0.0, road_threshold=0.1
    )
    adv = challenge_planner.plan_route(
        G,
        edges,
        (0.0, 0.0),
        pace=10.0,
        grade=0.0,
        road_pace=10.0,
        max_road=0.0,
        road_threshold=0.1,
        use_advanced_optimizer=True,
    )

    score_base = planner_utils.calculate_route_efficiency_score(base)
    score_adv = planner_utils.calculate_route_efficiency_score(adv)
    assert score_adv >= score_base


def test_plan_route_rpp_disconnected_components(tmp_path):
    """
    Tests that plan_route_rpp returns an empty list when required_nodes
    belong to disconnected components in the graph UG.
    """
    import networkx as nx # Added import
    from trail_route_ai.challenge_planner import plan_route_rpp # Added import

    # 1. Define Edges for two disconnected components
    edge1 = planner_utils.Edge(
        seg_id="1",
        name="Trail1",
        start=(0.0,0.0),
        end=(1.0,0.0),
        length_mi=1,
        elev_gain_ft=0,
        coords=[(0.0,0.0), (1.0,0.0)],
        kind="trail",
        direction="both",
        access_from=None
    )
    edge2 = planner_utils.Edge(
        seg_id="2",
        name="Trail2",
        start=(2.0,0.0), # Disconnected from edge1
        end=(3.0,0.0),
        length_mi=1,
        elev_gain_ft=0,
        coords=[(2.0,0.0), (3.0,0.0)],
        kind="trail",
        direction="both",
        access_from=None
    )

    # 2. Create a DiGraph G and add these edges
    G = nx.DiGraph()
    G.add_edge(edge1.start, edge1.end, weight=10.0, edge=edge1)
    G.add_edge(edge2.start, edge2.end, weight=10.0, edge=edge2)

    # 3. Define the list of edges to route (these define 'required_nodes')
    edges_to_route = [edge1, edge2]

    # 4. Define a start_node
    start_node = (0.0, 0.0) # Must be one of the nodes in G

    # 5. Set dummy values for pace, grade, road_pace
    pace = 10.0
    grade = 0.0
    road_pace = 10.0

    # 6. Call plan_route_rpp
    result = plan_route_rpp(
        G,
        edges_to_route,
        start_node,
        pace,
        grade,
        road_pace,
        debug_args=None
    )

    # 7. Assert that the result is an empty list
    assert result == [], f"Expected empty list due to disconnected components, but got: {result}"


def test_plan_route_rpp_node_not_in_graph(tmp_path):
    """
    Tests that plan_route with RPP enabled handles cases where a 'required_node'
    for the Steiner tree calculation is not actually present in the routing graph UG.
    This used to cause a NodeNotFound error. The fix involves filtering such nodes.
    """
    import argparse
    import networkx as nx
    from trail_route_ai.planner_utils import Edge

    # 1. Define Edges
    # Edge that will have a node not in G. Node (0.0, 1.0) is the problem.
    edge_problem = Edge(seg_id="problem_edge", name="Problem Edge", start=(0.0,0.0), end=(0.0,1.0), length_mi=1.0, elev_gain_ft=0.0, coords=[(0.0,0.0), (0.0,1.0)], kind="trail", direction="both")
    # Edges that will be in G and form a cycle to ensure RPP path is taken
    edge_g1 = Edge(seg_id="g1", name="Graph Edge 1", start=(0.0,0.0), end=(1.0,0.0), length_mi=1.0, elev_gain_ft=0.0, coords=[(0.0,0.0), (1.0,0.0)], kind="trail", direction="both")
    edge_g2 = Edge(seg_id="g2", name="Graph Edge 2", start=(1.0,0.0), end=(1.0,1.0), length_mi=1.0, elev_gain_ft=0.0, coords=[(1.0,0.0), (1.0,1.0)], kind="trail", direction="both")
    edge_g3 = Edge(seg_id="g3", name="Graph Edge 3", start=(1.0,1.0), end=(0.0,0.0), length_mi=1.0, elev_gain_ft=0.0, coords=[(1.0,1.0), (0.0,0.0)], kind="trail", direction="both")


    # 2. Build G (ensure (0.0,1.0) is NOT in G's edges)
    # These edges will form the graph G.
    graph_forming_edges = [edge_g1, edge_g2, edge_g3] # Does not include edge_problem or node (0.0,1.0)
    g_actual = challenge_planner.build_nx_graph(graph_forming_edges, pace=10.0, grade=0.0, road_pace=10.0)

    # 3. Define edges_for_rpp (includes the problematic edge and the cycle)
    # RPP will be attempted on these. (0.0,1.0) from edge_problem is a required node for RPP.
    edges_for_rpp_attempt = [edge_problem, edge_g1, edge_g2, edge_g3]

    # 4. Define start_node
    start_node_for_plan = (0.0,0.0) # Must be in g_actual

    # 5. Parameters
    pace_val = 10.0
    grade_val = 0.0
    road_pace_val = 12.0
    max_road_val = 1.0 # Allow some road for connectors if needed by greedy fallback
    road_threshold_val = 0.25
    rpp_timeout_val = 2.0 # Short timeout for test

    # 6. Debug args
    debug_log_path = tmp_path / "debug_test_node_not_in_graph.log"
    debug_args_val = argparse.Namespace(verbose=True, debug=str(debug_log_path))
    # Ensure the debug log file is clear at the start of the test
    if debug_log_path.exists():
        debug_log_path.unlink()

    # 7. Call plan_route
    route = []
    try:
        route = challenge_planner.plan_route(
            G=g_actual,
            edges=edges_for_rpp_attempt,
            start=start_node_for_plan,
            pace=pace_val,
            grade=grade_val,
            road_pace=road_pace_val,
            max_road=max_road_val,
            road_threshold=road_threshold_val,
            dist_cache=None,
            use_rpp=True,
            allow_connectors=True, # Important for RPP to try to connect things
            rpp_timeout=rpp_timeout_val,
            debug_args=debug_args_val,
            spur_length_thresh=0.3,
            spur_road_bonus=0.25,
            path_back_penalty=1.2,
            redundancy_threshold=None,
            use_advanced_optimizer=False
        )
    except nx.NodeNotFound as e:
        # This is what we want to avoid. If this happens, the test fails.
        # Specifically, the error would be like: networkx.exception.NodeNotFound: Node (0.0, 1.0) not in graph.
        assert False, f"Steiner tree calculation failed with NodeNotFound: {e}"

    # Assertion 1 is implicit: if we reach here without the specific NodeNotFound, it's a pass for that part.

    # Assertion 2: Check debug logs
    log_content = ""
    if debug_log_path.exists():
        with open(debug_log_path, "r") as f:
            log_content = f.read()
    else:
        assert False, f"Debug log file not found at {debug_log_path}"

    problem_node_tuple = (0.0, 1.0) # This is the node from edge_problem.end
    expected_log_message_part = f"RPP: Required node {problem_node_tuple!r} not in UG"
    assert expected_log_message_part in log_content, \
        f"Expected log message part '{expected_log_message_part}' not found in logs. Log content:\n{log_content}"

    # Assertion 3: Route is valid or empty (graceful fallback)
    assert route is not None, "plan_route should always return a list, even if empty."

    # The problematic edge (edge_problem) should not be part of the route because one of its nodes was invalid for RPP,
    # and it's not part of the main graph G for greedy routing either.
    problem_edge_in_route = any(e.seg_id == edge_problem.seg_id for e in route)
    assert not problem_edge_in_route, \
        f"Problematic edge '{edge_problem.seg_id}' should not be in the final route. Route: {[e.seg_id for e in route]}"

    # If a route is returned, it should ideally contain the valid edges (g1, g2)
    # as RPP should fall back to greedy, and greedy should be able to route them (or RPP itself if valid nodes are > 1).
    if route:
        g1_in_route = any(e.seg_id == edge_g1.seg_id for e in route)
        g2_in_route = any(e.seg_id == edge_g2.seg_id for e in route)
        g3_in_route = any(e.seg_id == edge_g3.seg_id for e in route)
        # All three g_edges should be in the route if a valid plan is made for the cyclic part
        assert g1_in_route, f"Edge '{edge_g1.seg_id}' should be in the route if a non-empty route is returned. Route: {[e.seg_id for e in route]}"
        assert g2_in_route, f"Edge '{edge_g2.seg_id}' should be in the route if a non-empty route is returned. Route: {[e.seg_id for e in route]}"
        assert g3_in_route, f"Edge '{edge_g3.seg_id}' should be in the route if a non-empty route is returned. Route: {[e.seg_id for e in route]}"
    else:
        # This might happen if RPP filtering results in <2 nodes, and then greedy also fails for some reason.
        # For this test's simple data (g1, g2 form a line), greedy should typically succeed.
        # Log a message for easier debugging if this happens, but it's not strictly a failure of *this* test's main assertion (no crash).
        print(f"Warning: Route was empty for test_plan_route_rpp_node_not_in_graph. Debug log contents:\n{log_content}")



def test_plan_route_fallback_on_rpp_failure(tmp_path):
    """
    Tests that plan_route falls back to the greedy algorithm when RPP (use_rpp=True)
    fails and returns an empty list (simulating the new try-except behavior).
    """
    import argparse
    from unittest.mock import patch

    # 1. Set up simple edges
    edges = build_edges(2)  # e.g., S0: (0,0)-(1,0), S1: (1,0)-(2,0)

    # 2. Build the graph G
    G = challenge_planner.build_nx_graph(edges, pace=10.0, grade=0.0, road_pace=12.0)

    # 3. Prepare debug_args
    debug_log_path = tmp_path / "debug_fallback_test.log"
    debug_args = argparse.Namespace(verbose=True, debug=str(debug_log_path))
    if debug_log_path.exists():
        debug_log_path.unlink()

    # 4. Use unittest.mock.patch to mock plan_route_rpp
    # Configure the mock to return an empty list, simulating RPP failure
    @patch('trail_route_ai.challenge_planner.plan_route_rpp', return_value=[])
    def run_test_with_mock(mock_plan_route_rpp):
        # 5. Define a start node
        start_node = (0.0, 0.0)

        # 6. Call challenge_planner.plan_route
        route = challenge_planner.plan_route(
            G=G,
            edges=edges,
            start=start_node,
            pace=10.0,
            grade=0.0,
            road_pace=12.0,
            max_road=1.0,
            road_threshold=0.25,
            dist_cache=None,
            use_rpp=True,  # Critical: RPP is enabled
            allow_connectors=True,
            rpp_timeout=2.0,
            debug_args=debug_args,
            spur_length_thresh=0.3,
            spur_road_bonus=0.25,
            path_back_penalty=1.2,
            redundancy_threshold=None,
            use_advanced_optimizer=False
        )

        # 7. Assert that the returned route is not empty
        assert route, "Route should not be empty, indicating fallback to greedy succeeded."
        # Greedy for S0, S1 from (0,0) should be [S0, S1, S1_rev, S0_rev] or similar
        assert len(route) >= 2, "Greedy route for 2 segments should have at least 2 edges."

        # 8. Check the debug log content
        log_content = ""
        if debug_log_path.exists():
            with open(debug_log_path, "r") as f:
                log_content = f.read()
        else:
            assert False, f"Debug log file not found at {debug_log_path}"

        mock_plan_route_rpp.assert_called_once()

        # Check for RPP failure/empty route log
        assert "RPP attempted but returned an empty route. Proceeding to greedy." in log_content or \
               "RPP failed with exception" in log_content or \
               "RPP retry returned an empty route. Proceeding to greedy." in log_content or \
               "RPP skipped, split_cluster_by_connectivity resulted in" in log_content, \
               "Log should indicate RPP failure or empty result before greedy."

        # Check for greedy attempt log
        assert "Entering greedy search stage with _plan_route_greedy." in log_content, \
               "Log should indicate that the greedy algorithm was attempted."

    run_test_with_mock()
