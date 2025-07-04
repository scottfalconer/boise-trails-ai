import json
import pytest
import networkx as nx
import numpy as np
import rasterio
from rasterio.transform import from_origin
from trail_route_ai import planner_utils


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


def test_bounding_box(tmp_path):
    e1 = planner_utils.Edge(
        "A", "A", (0.0, 0.0), (1.0, 1.0), 1.0, 0.0, [(0.0, 0.0), (1.0, 1.0)]
    )
    e2 = planner_utils.Edge(
        "B", "B", (2.0, 1.0), (2.0, 2.0), 1.0, 0.0, [(2.0, 1.0), (2.0, 2.0)]
    )
    bbox = planner_utils.bounding_box_from_edges([e1, e2], buffer_km=0)
    assert bbox == [0.0, 0.0, 2.0, 2.0]
    assert planner_utils.bounding_box_from_edges([], buffer_km=0) is None


def test_load_trailheads(tmp_path):
    csv_path = tmp_path / "ths.csv"
    csv_path.write_text("lat,lon,name\n1,2,TH1\n")
    res_csv = planner_utils.load_trailheads(str(csv_path))
    assert res_csv == {(2.0, 1.0): "TH1"}

    json_path = tmp_path / "ths.json"
    json.dump([{"lat": 1, "lon": 2, "name": "TH1"}], json_path.open("w"))
    res_json = planner_utils.load_trailheads(str(json_path))
    assert res_json == {(2.0, 1.0): "TH1"}


def test_load_completed_and_tracking(tmp_path):
    csv_path = tmp_path / "perf.csv"
    csv_path.write_text("seg_id,year\nA,2024\nB,2023\n")
    completed = planner_utils.load_completed(str(csv_path), 2024)
    assert completed == {"A"}

    segs_path = tmp_path / "segments.json"
    data = {"segments": [{"id": "A", "coordinates": [[0, 0], [1, 0]]}]}
    json.dump(data, segs_path.open("w"))
    track_path = tmp_path / "track.json"
    tracking = planner_utils.load_segment_tracking(str(track_path), str(segs_path))
    assert tracking == {"A": False}
    written = json.load(track_path.open())
    assert "A" in written


def test_load_segment_tracking_dashboard(tmp_path):
    segs_path = tmp_path / "segments.json"
    data = {"segments": [
        {"id": "1", "coordinates": [[0, 0], [1, 0]]},
        {"id": "2", "coordinates": [[1, 0], [2, 0]]},
    ]}
    json.dump(data, segs_path.open("w"))

    dash_path = tmp_path / "dash.json"
    json.dump({"CompletedSegmentIds": [1]}, dash_path.open("w"))

    tracking = planner_utils.load_segment_tracking(str(dash_path), str(segs_path))
    assert tracking == {"1": True, "2": False}


def test_add_elevation_from_dem(tmp_path):
    dem_path = tmp_path / "dem.tif"
    create_dem(dem_path)
    edge = planner_utils.Edge(
        "A", "A", (0.0, 1.0), (3.0, 1.0), 3.0, 0.0, [(0.0, 1.0), (3.0, 1.0)]
    )
    planner_utils.add_elevation_from_dem([edge], str(dem_path))
    assert edge.elev_gain_ft > 0


def test_add_elevation_from_flat_dem(tmp_path):
    dem_path = tmp_path / "dem.tif"
    data = np.zeros((2, 2), dtype=np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1, 1),
    ) as dst:
        dst.write(data, 1)
    edge = planner_utils.Edge(
        "A",
        "A",
        (0.0, 1.0),
        (1.0, 1.0),
        1.0,
        0.0,
        [(0.0, 1.0), (1.0, 1.0)],
    )
    planner_utils.add_elevation_from_dem([edge], str(dem_path))
    assert edge.elev_gain_ft == 0.0


def test_estimate_drive_time_minutes():
    G = nx.Graph()
    G.add_edge((0.0, 0.0), (1.0, 0.0), length_mi=1.0)
    t, dist = planner_utils.estimate_drive_time_minutes(
        (0.0, 0.0), (1.0, 0.0), G, 30.0, return_distance=True
    )
    assert pytest.approx(2.0) == t
    assert pytest.approx(1.0) == dist


def test_load_segments_multilinestring(tmp_path):
    data = {
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[0, 0], [1, 0]],
                        [[1, 0], [1, 1]],
                    ],
                },
                "properties": {"id": "A", "name": "A"},
            }
        ]
    }
    path = tmp_path / "segs.geojson"
    path.write_text(json.dumps(data))
    edges = planner_utils.load_segments(str(path))
    assert len(edges) == 2
    assert edges[0].start == (0, 0)
    assert edges[0].end == (1, 0)
    assert edges[1].start == (1, 0)
    assert edges[1].end == (1, 1)


def test_connect_trails_to_roads():
    trail = planner_utils.Edge(
        "T",
        "T",
        (0.0, 0.0),
        (0.001, 0.0),
        0.1,
        0.0,
        [(0.0, 0.0), (0.001, 0.0)],
        "trail",
        "both",
    )
    road = planner_utils.Edge(
        "R",
        "R",
        (0.0004, 0.0),
        (0.0004, 0.001),
        0.1,
        0.0,
        [(0.0004, 0.0), (0.0004, 0.001)],
        "road",
        "both",
    )

    connectors = planner_utils.connect_trails_to_roads([trail], [road], threshold_meters=50.0)
    assert connectors
    c = connectors[0]
    assert c.kind == "road"
    assert c.start in {trail.start, trail.end}
    assert c.end in {road.start, road.end}
