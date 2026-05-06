from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_repaired_candidate_universe_audit as audit  # noqa: E402


def official_fixture():
    return {
        "features": [
            {"properties": {"segId": 1, "segName": "A 1", "LengthFt": 5280, "direction": "both"}},
            {"properties": {"segId": 2, "segName": "B 1", "LengthFt": 5280, "direction": "both"}},
            {"properties": {"segId": 1656, "segName": "Shingle Creek Trail 1", "LengthFt": 5280, "direction": "ascent"}},
        ]
    }


def candidate(candidate_id, segment_ids, p75, p90):
    return {
        "candidate_id": candidate_id,
        "segment_ids": segment_ids,
        "official_new_miles": len(segment_ids),
        "estimated_total_on_foot_miles": len(segment_ids) + 1,
        "time_estimates_minutes": {"door_to_door_p75": p75, "door_to_door_p90": p90},
        "trailhead": {"name": "Trailhead"},
        "trail_names": [candidate_id],
        "route_status": "graph_validated",
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
    }


def probe_row(seg_id, p75, p90, *, field_ready=True):
    return {
        "candidate_id": f"probe-{seg_id}",
        "seg_id": seg_id,
        "seg_name": f"Segment {seg_id}",
        "trail_name": "Probe Trail",
        "official_miles": 1.0,
        "on_foot_miles": 2.0,
        "door_to_door_p75_minutes": p75,
        "door_to_door_p90_minutes": p90,
        "route_status": "graph_validated",
        "validation_passed": True,
        "track_validation_passed": True,
        "field_ready": field_ready,
        "trailhead": "Probe TH",
    }


def test_repaired_universe_reports_only_shingle_missing_until_exception():
    report = audit.build_report(
        official_geojson=official_fixture(),
        personal_route_menu={"route_menu": {"all_candidates": [candidate("existing", [1], 40, 50)]}},
        hybrid_route_pass={"candidate_index": {}},
        field_menu={"route_cues": {}},
        state={"availability_model": {"weekday_max_minutes": 100, "weekend_max_minutes": 60}},
        split_probe={"probe_rows": [probe_row(2, 55, 90)]},
        forced_probe={"probe_rows": [probe_row(1656, 101, 112)]},
    )

    assert report["summary"]["strict_bounded_missing_segment_ids"] == [1656]
    assert report["summary"]["completion_possible_under_current_p90_bound"] is False
    assert report["summary"]["completion_possible_if_shingle_exception_accepted"] is True


def test_forced_probe_rows_must_be_field_ready():
    rows = [
        probe_row(1, 10, 20, field_ready=False),
        probe_row(2, 10, 20, field_ready=True),
    ]

    candidates = audit.strict_probe_candidates({"probe_rows": []}, {"probe_rows": rows})

    assert [candidate["segment_ids"] for candidate in candidates] == [[2]]
