from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import manual_access_anchor_probe as probe  # noqa: E402


def test_anchor_trailhead_uses_manual_parking_fields():
    anchor = {
        "name": "Probe",
        "lat": 43.1,
        "lon": -116.2,
        "has_parking": True,
        "parking_confidence": "manual_required",
        "source": "manual",
    }

    trailhead = probe.anchor_trailhead(anchor)

    assert trailhead["name"] == "Probe"
    assert trailhead["lat"] == 43.1
    assert trailhead["lon"] == -116.2
    assert trailhead["parking_confidence"] == "manual_required"
    assert trailhead["field_ready"] is False


def test_summary_counts_bounded_track_valid_rows_and_field_ready_rows():
    rows = [
        {"door_to_door_p90_minutes": 80, "track_validation_passed": True, "route_status": "graph_validated"},
        {"door_to_door_p90_minutes": 120, "track_validation_passed": False, "route_status": "graph_validated"},
        {"door_to_door_p90_minutes": 180, "track_validation_passed": True, "route_status": "draft"},
    ]

    summary = probe.summarize_rows(rows, p90_bound_minutes=150, field_ready=False)

    assert summary["probe_count"] == 3
    assert summary["under_p90_bound_count"] == 2
    assert summary["under_p90_bound_track_valid_graph_validated_count"] == 1
    assert summary["field_ready"] is False
