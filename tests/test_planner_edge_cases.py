import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from unittest.mock import patch

from trail_route_ai import planner_utils, challenge_planner


def create_dem(path):
    data = np.ones((1, 1), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data, 1)


def build_edges(n=2):
    edges = []
    for i in range(n):
        start = (float(i), 0.0)
        end = (float(i) + 0.01, 0.0)
        edges.append(
            planner_utils.Edge(
                str(i),
                str(i),
                start,
                end,
                0.01,
                0.0,
                [start, end],
                "trail",
                "both",
            )
        )
    return edges


def setup_planner_args(tmp_path, segments):
    seg_path = tmp_path / "segs.json"
    with open(seg_path, "w") as f:
        json.dump({"segments": segments}, f)
    dem_path = tmp_path / "dem.tif"
    create_dem(dem_path)
    perf_path = tmp_path / "perf.csv"
    perf_path.write_text("seg_id,year\n")
    gpx_dir = tmp_path / "gpx"
    args = [
        "--start-date",
        "2024-07-01",
        "--end-date",
        "2024-07-01",
        "--time",
        "1h",
        "--pace",
        "10",
        "--grade",
        "0",
        "--year",
        "2024",
        "--segments",
        str(seg_path),
        "--dem",
        str(dem_path),
        "--perf",
        str(perf_path),
        "--output",
        str(tmp_path / "out.csv"),
        "--gpx-dir",
        str(gpx_dir),
    ]
    return args


def test_empty_segments_file_error(tmp_path):
    args = setup_planner_args(tmp_path, [])
    with patch("trail_route_ai.plan_review.review_plan"):
        with pytest.raises(ValueError):
            challenge_planner.main(args)


def test_malformed_coordinate_array(tmp_path):
    seg_path = tmp_path / "bad.json"
    bad = {"segments": [{"id": "A", "coordinates": [[0, 0], ["x", 1]]}]}
    seg_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        planner_utils.load_segments(str(seg_path))


def test_extreme_time_budgets_cluster(tmp_path):
    edges = build_edges(3)
    small = challenge_planner.cluster_segments(
        edges, pace=10.0, grade=0.0, budget=0.01, max_clusters=10, road_pace=15.0
    )
    big = challenge_planner.cluster_segments(
        edges, pace=10.0, grade=0.0, budget=100000.0, max_clusters=10, road_pace=15.0
    )
    assert len(small) >= 1
    assert len(big) >= 1
