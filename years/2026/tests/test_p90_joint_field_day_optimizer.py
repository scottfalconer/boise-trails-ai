from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_joint_field_day_optimizer as optimizer  # noqa: E402


def field_day(day_id, day_type, segments, p75=60):
    return {
        "field_day_id": day_id,
        "day_type": day_type,
        "loop_ids": [day_id],
        "segment_ids": segments,
        "loop_count": 1,
        "p75_minutes": p75,
        "p90_minutes": p75 + 10,
        "p90_bound_minutes": 100,
        "stress": 0.7,
        "drive_minutes": 10,
        "between_drive_minutes": 0,
        "on_foot_miles": len(segments),
        "grade_adjusted_miles": len(segments),
        "parking_risk": 1,
    }


def loop_record(loop_id, segment_id, lon):
    return {
        "loop_id": loop_id,
        "parking": {"lon": lon, "lat": 43.6},
        "segment_ids": [segment_id],
        "internal_p75_minutes": 10,
        "p90_delta_minutes": 2,
        "door_to_door_p90_minutes": 20,
        "on_foot_miles": 1.0,
        "grade_adjusted_miles": 1.0,
        "parking_risk": 1,
    }


def field_day_state():
    return {
        "drive_model": {
            "origin_lon": -116.2,
            "origin_lat": 43.6,
            "straight_line_factor": 1.0,
            "minutes_per_mile": 1.0,
            "minimum_one_way_minutes": 1,
        },
        "availability_model": {"acceptable_inter_trailhead_drive_minutes": 20},
    }


def test_direct_field_day_cover_solves_segment_coverage_with_day_counts():
    field_days = [
        field_day("a", "weekday", [1, 2], p75=50),
        field_day("b", "weekend", [3], p75=40),
        field_day("c", "weekday", [1], p75=20),
    ]

    solution = optimizer.solve_direct_field_day_cover(
        field_days,
        [1, 2, 3],
        {"weekday": 1, "weekend": 1},
    )

    assert solution["success"] is True
    assert solution["field_day_count"] == 2
    assert solution["covered_segment_count"] == 3


def test_direct_field_day_cover_reports_missing_segments():
    solution = optimizer.solve_direct_field_day_cover(
        [field_day("a", "weekday", [1])],
        [1, 2],
        {"weekday": 1, "weekend": 0},
    )

    assert solution["success"] is False
    assert solution["reason"] == "not_all_segments_coverable_by_field_day_candidates"
    assert solution["missing_segment_ids"] == [2]


def test_direct_field_day_max_coverage_reports_best_partial_schedule():
    field_days = [
        field_day("a", "weekday", [1, 2], p75=50),
        field_day("b", "weekday", [3], p75=40),
    ]

    solution = optimizer.solve_direct_field_day_max_coverage(
        field_days,
        [1, 2, 3],
        {"weekday": 1, "weekend": 0},
        official_miles_by_segment={1: 1.0, 2: 1.0, 3: 1.0},
    )

    assert solution["success"] is True
    assert solution["covered_segment_count"] == 2
    assert solution["missing_segment_ids"] == [3]
    assert solution["field_day_count"] == 1


def test_direct_field_day_max_coverage_can_force_required_segment():
    field_days = [
        field_day("a", "weekday", [1, 2], p75=50),
        field_day("b", "weekday", [3], p75=40),
    ]

    solution = optimizer.solve_direct_field_day_max_coverage(
        field_days,
        [1, 2, 3],
        {"weekday": 1, "weekend": 0},
        official_miles_by_segment={1: 1.0, 2: 1.0, 3: 1.0},
        required_segment_ids=[3],
    )

    assert solution["success"] is True
    assert solution["covered_segment_count"] == 1
    assert solution["missing_segment_ids"] == [1, 2]
    assert solution["field_day_count"] == 1


def test_direct_field_day_max_coverage_rejects_uncoverable_required_segment():
    solution = optimizer.solve_direct_field_day_max_coverage(
        [field_day("a", "weekday", [1])],
        [1, 2],
        {"weekday": 1, "weekend": 0},
        official_miles_by_segment={1: 1.0, 2: 1.0},
        required_segment_ids=[2],
    )

    assert solution["success"] is False
    assert solution["reason"] == "required_segments_not_coverable_by_field_day_candidates"
    assert solution["missing_required_segment_ids"] == [2]


def test_generate_direct_field_day_candidates_honors_max_combo_size_four():
    loops = [
        loop_record("a", 1, -116.2),
        loop_record("b", 2, -116.201),
        loop_record("c", 3, -116.202),
        loop_record("d", 4, -116.203),
    ]

    field_days = optimizer.generate_direct_field_day_candidates(
        loops,
        field_day_state(),
        weekday_bound=100,
        weekend_bound=100,
        max_combo_size=4,
        neighbor_limit=4,
    )

    assert any(
        day["loop_count"] == 4 and set(day["segment_ids"]) == {1, 2, 3, 4}
        for day in field_days
    )


def test_generate_direct_field_day_candidates_allows_connected_drive_chain():
    loops = [
        loop_record("a", 1, -116.2),
        loop_record("b", 2, -115.82),
        loop_record("c", 3, -115.44),
    ]

    field_days = optimizer.generate_direct_field_day_candidates(
        loops,
        field_day_state(),
        weekday_bound=200,
        weekend_bound=200,
        max_combo_size=3,
        neighbor_limit=2,
    )

    assert any(
        day["loop_count"] == 3 and set(day["segment_ids"]) == {1, 2, 3}
        for day in field_days
    )
