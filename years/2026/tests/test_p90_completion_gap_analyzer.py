from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_completion_gap_analyzer as analyzer  # noqa: E402


def official_fixture():
    return {
        "features": [
            {"properties": {"segId": 1, "segName": "Trail A 1", "LengthFt": 5280, "direction": "both"}},
            {"properties": {"segId": 2, "segName": "Trail B 1", "LengthFt": 2640, "direction": "ascent"}},
        ]
    }


def state_fixture():
    return {"availability_model": {"weekday_max_minutes": 100, "weekend_max_minutes": 60}}


def candidate(candidate_id, segment_ids, p75, p90, route_status="graph_validated"):
    return {
        "candidate_id": candidate_id,
        "segment_ids": segment_ids,
        "official_new_miles": len(segment_ids),
        "estimated_total_on_foot_miles": len(segment_ids) + 1,
        "time_estimates_minutes": {"door_to_door_p75": p75, "door_to_door_p90": p90},
        "trailhead": {"name": "Trailhead"},
        "trail_names": [candidate_id],
        "route_status": route_status,
        "validation": {
            "segment_coverage_passed": True,
            "ascent_direction_passed": True,
            "return_path_graph_validated": True,
        },
    }


def test_gap_report_marks_segments_missing_when_no_candidate_fits_max_bound():
    report = analyzer.build_report(
        official_geojson=official_fixture(),
        personal_route_menu={"route_menu": {"all_candidates": [candidate("a", [1], 40, 50), candidate("b", [2], 120, 130)]}},
        hybrid_route_pass={"candidate_index": {}},
        field_menu={"route_cues": {}},
        state=state_fixture(),
    )

    assert report["summary"]["target_segment_count"] == 2
    assert report["summary"]["max_bound_covered_segment_count"] == 1
    assert report["summary"]["max_bound_missing_segment_count"] == 1
    assert report["missing_under_max_bound_segments"][0]["seg_id"] == 2
    assert report["missing_under_max_bound_segments"][0]["best_existing_candidate"]["candidate_id"] == "b"


def test_gap_report_can_cover_all_segments_with_bounded_existing_candidates():
    report = analyzer.build_report(
        official_geojson=official_fixture(),
        personal_route_menu={
            "route_menu": {
                "all_candidates": [
                    candidate("a", [1], 40, 50),
                    candidate("b", [2], 55, 90),
                    candidate("combo", [1, 2], 90, 99),
                ]
            }
        },
        hybrid_route_pass={"candidate_index": {}},
        field_menu={"route_cues": {}},
        state=state_fixture(),
    )

    assert report["summary"]["completion_possible_with_existing_bounded_candidates"] is True
    assert report["maximum_bounded_coverage_solution"]["success"] is True
    assert report["maximum_bounded_coverage_solution"]["missing_segment_count"] == 0
