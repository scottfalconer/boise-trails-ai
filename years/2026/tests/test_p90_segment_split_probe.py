from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_segment_split_probe as probe  # noqa: E402


def test_segment_trail_preserves_one_official_segment():
    segment = {
        "seg_id": 1542,
        "seg_name": "Dry Creek Trail 1",
        "trail_name": "Dry Creek Trail",
        "official_miles": 0.57,
        "direction": "both",
        "coordinates": [(-116.0, 43.0), (-116.1, 43.1)],
        "start": (-116.0, 43.0),
        "end": (-116.1, 43.1),
        "center": (-116.05, 43.05),
    }

    trail = probe.segment_trail(segment)

    assert trail["trail_name"] == "Dry Creek Trail"
    assert trail["remaining_segment_ids"] == [1542]
    assert trail["segments"] == [segment]
    assert trail["official_miles"] == 0.57


def test_summarize_probe_rows_identifies_newly_bounded_segments():
    rows = [
        {"seg_id": 1, "door_to_door_p90_minutes": 90, "route_status": "graph_validated", "track_validation_passed": True},
        {"seg_id": 2, "door_to_door_p90_minutes": 130, "route_status": "graph_validated", "track_validation_passed": True},
        {"seg_id": 3, "door_to_door_p90_minutes": 220, "route_status": "draft", "track_validation_passed": False},
    ]

    summary = probe.summarize_probe_rows(rows, max_bound_minutes=150, weekend_bound_minutes=100)

    assert summary["probe_count"] == 3
    assert summary["under_max_bound_count"] == 2
    assert summary["under_max_bound_track_valid_graph_validated_count"] == 2
    assert summary["under_weekend_bound_count"] == 1
    assert summary["graph_validated_count"] == 2
    assert summary["track_validation_passed_count"] == 2
    assert summary["still_over_max_bound_count"] == 1
