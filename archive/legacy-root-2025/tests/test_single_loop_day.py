import csv
from unittest.mock import patch

from trail_route_ai import challenge_planner, planner_utils
from tests.test_challenge_planner import (
    write_segments,
    setup_planner_test_environment,
)


def test_prefer_single_loop_days_merges_clusters(tmp_path):
    seg1 = planner_utils.Edge(
        "A1",
        "A1",
        (0.0, 0.0),
        (1.0, 0.0),
        1.0,
        0.0,
        [(0.0, 0.0), (1.0, 0.0)],
        "trail",
        "both",
    )
    seg2 = planner_utils.Edge(
        "B1",
        "B1",
        (1.1, 0.0),
        (2.1, 0.0),
        1.0,
        0.0,
        [(1.1, 0.0), (2.1, 0.0)],
        "trail",
        "both",
    )
    connector = planner_utils.Edge(
        "CX",
        "CX",
        (1.0, 0.0),
        (1.1, 0.0),
        0.1,
        0.0,
        [(1.0, 0.0), (1.1, 0.0)],
        "trail",
        "both",
    )
    connectors_path = tmp_path / "connectors.json"
    write_segments(connectors_path, [connector])

    args_list, out_csv = setup_planner_test_environment(
        tmp_path,
        segments_data=[seg1, seg2],
        extra_args=["--connector-trails", str(connectors_path), "--prefer-single-loop-days"],
    )

    with patch("trail_route_ai.plan_review.review_plan"):
        challenge_planner.main(args_list)

    rows = list(csv.DictReader(open(out_csv)))
    first_day = rows[0]
    assert first_day["num_drives"] == "0"
    assert first_day["num_activities"] == "1"
