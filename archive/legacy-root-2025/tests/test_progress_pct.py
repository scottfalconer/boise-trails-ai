import csv
from unittest.mock import patch

from trail_route_ai import challenge_planner
from tests.test_challenge_planner import (
    build_edges,
    setup_planner_test_environment,
    create_dem,
)
from trail_route_ai import planner_utils


def test_planner_reaches_full_progress(tmp_path):
    segments = build_edges(2, prefix="P")
    total_miles = sum(e.length_mi for e in segments)
    args_list, out_csv = setup_planner_test_environment(
        tmp_path,
        segments_data=segments,
        remaining_ids_str=",".join(e.seg_id for e in segments),
        extra_args=["--end-date", "2024-07-02", "--challenge-target-distance-mi", str(total_miles)]
    )

    with patch('trail_route_ai.plan_review.review_plan'):
        challenge_planner.main(args_list)

    rows = list(csv.DictReader(open(out_csv)))
    totals = next(r for r in rows if r["date"] == "Totals")
    assert float(totals["progress_distance_pct"]) == 100.0

    all_descriptions = " ".join(r["plan_description"] for r in rows if r["date"] != "Totals")
    for seg in segments:
        assert seg.seg_id in all_descriptions


def test_progress_elevation_pct_full(tmp_path):
    segments = build_edges(3, prefix="E")
    dem_path = tmp_path / "dem.tif"
    create_dem(dem_path)
    planner_utils.add_elevation_from_dem(segments, str(dem_path))
    total_gain = round(sum(e.elev_gain_ft for e in segments))
    total_miles = sum(e.length_mi for e in segments)

    args_list, out_csv = setup_planner_test_environment(
        tmp_path,
        segments_data=segments,
        remaining_ids_str=",".join(e.seg_id for e in segments),
        extra_args=[
            "--end-date",
            "2024-07-03",
            "--challenge-target-distance-mi",
            str(total_miles),
            "--challenge-target-elevation-ft",
            str(total_gain),
        ],
    )

    with patch("trail_route_ai.plan_review.review_plan"):
        challenge_planner.main(args_list)

    rows = list(csv.DictReader(open(out_csv)))
    totals = next(r for r in rows if r["date"] == "Totals")
    assert float(totals["progress_elevation_pct"]) == 100.0

