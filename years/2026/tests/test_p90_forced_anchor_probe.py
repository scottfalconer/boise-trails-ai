from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_forced_anchor_probe as probe  # noqa: E402


def test_rank_anchors_for_segment_uses_closest_start_end_or_center():
    segment = {
        "start": (-116.2, 43.6),
        "end": (-116.1, 43.7),
        "center": (-116.15, 43.65),
    }
    anchors = [
        {"name": "near-start", "lon": -116.201, "lat": 43.6},
        {"name": "near-end", "lon": -116.1, "lat": 43.701},
        {"name": "far", "lon": -116.5, "lat": 43.1},
    ]

    ranked = probe.rank_anchors_for_segment(segment, anchors, limit=2)

    assert [item["anchor"]["name"] for item in ranked] == ["near-start", "near-end"]
    assert ranked[0]["distance_basis"] == "start"
    assert ranked[1]["distance_basis"] == "end"


def test_summarize_rows_reports_best_bounded_and_missing_segments():
    rows = [
        {
            "seg_id": 1,
            "door_to_door_p90_minutes": 200,
            "route_status": "graph_validated",
            "track_validation_passed": True,
            "field_ready": True,
        },
        {
            "seg_id": 2,
            "door_to_door_p90_minutes": 240,
            "route_status": "graph_validated",
            "track_validation_passed": True,
            "field_ready": False,
        },
        {
            "seg_id": 3,
            "door_to_door_p90_minutes": 280,
            "route_status": "graph_validated",
            "track_validation_passed": True,
            "field_ready": True,
        },
    ]

    summary = probe.summarize_rows(rows, target_segment_ids=[1, 2, 3], p90_bound_minutes=260)

    assert summary["target_segment_count"] == 3
    assert summary["under_p90_bound_count"] == 2
    assert summary["strict_field_ready_segment_count"] == 1
    assert summary["conditional_segment_count"] == 1
    assert summary["strict_missing_segment_ids"] == [2, 3]
    assert summary["conditional_missing_segment_ids"] == [3]
