from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import field_day_completion_planner as planner  # noqa: E402


def test_date_counts_splits_weekdays_and_weekends():
    assert planner.date_counts("2026-06-18", "2026-06-21") == {
        "weekday": 2,
        "weekend": 2,
        "total": 4,
    }


def test_field_day_time_chains_between_trailheads_without_driving_home_between():
    loops = [
        {
            "candidate_id": "a",
            "parking": {"lon": -116.2, "lat": 43.6},
            "internal_p75_minutes": 30,
            "p90_delta_minutes": 5,
            "on_foot_miles": 2.0,
            "grade_adjusted_miles": 2.2,
            "ascent_ft": 200,
            "parking_risk": 0,
        },
        {
            "candidate_id": "b",
            "parking": {"lon": -116.19, "lat": 43.605},
            "internal_p75_minutes": 25,
            "p90_delta_minutes": 3,
            "on_foot_miles": 1.5,
            "grade_adjusted_miles": 1.7,
            "ascent_ft": 100,
            "parking_risk": 1,
        },
    ]
    drive_model = {"straight_line_factor": 1.25, "minutes_per_mile": 2.2, "minimum_one_way_minutes": 5}

    day = planner.field_day_time(loops, (0, 1), drive_model, (-116.21, 43.61))
    separate_days = planner.field_day_time(loops, (0,), drive_model, (-116.21, 43.61))[
        "drive_minutes"
    ] + planner.field_day_time(loops, (1,), drive_model, (-116.21, 43.61))["drive_minutes"]

    assert day["order"] == ["a", "b"]
    assert day["between_drive_minutes"] > 0
    assert day["drive_minutes"] < separate_days
    assert day["p90_minutes"] == day["p75_minutes"] + 8


def test_generate_field_day_candidates_reports_loop_that_exceeds_all_bounds():
    loops = [
        {
            "candidate_id": "short",
            "label": "Short loop",
            "trailhead": "Short TH",
            "parking": {"lon": -116.2, "lat": 43.6},
            "segment_ids": [1],
            "official_miles": 1.0,
            "on_foot_miles": 2.0,
            "grade_adjusted_miles": 2.0,
            "ascent_ft": 100,
            "door_to_door_p90_minutes": 50,
            "internal_p75_minutes": 20,
            "p90_delta_minutes": 5,
            "parking_risk": 0,
        },
        {
            "candidate_id": "too-long",
            "label": "Too long",
            "trailhead": "Long TH",
            "parking": {"lon": -116.3, "lat": 43.7},
            "segment_ids": [2],
            "official_miles": 5.0,
            "on_foot_miles": 12.0,
            "grade_adjusted_miles": 14.0,
            "ascent_ft": 2000,
            "door_to_door_p90_minutes": 275,
            "internal_p75_minutes": 230,
            "p90_delta_minutes": 30,
            "parking_risk": 1,
        },
    ]
    state = {
        "drive_model": {
            "origin_lon": -116.21,
            "origin_lat": 43.61,
            "straight_line_factor": 1.25,
            "minutes_per_mile": 2.2,
            "minimum_one_way_minutes": 5,
        },
        "availability_model": {"weekday_max_minutes": 180, "weekend_max_minutes": 240},
    }

    candidates, blockers = planner.generate_field_day_candidates(loops, state, max_combo_size=1)

    assert [blocker["candidate_id"] for blocker in blockers] == ["too-long"]
    assert any(candidate["candidate_ids"] == ["short"] for candidate in candidates)
    assert not any(candidate["candidate_ids"] == ["too-long"] for candidate in candidates)


def test_solve_field_day_partition_selects_one_covering_day_per_loop():
    loops = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    candidates = [
        {
            "field_day_id": "weekday-a",
            "day_type": "weekday",
            "candidate_ids": ["a"],
            "p75_minutes": 30,
            "p90_minutes": 40,
            "stress": 0.4,
            "grade_adjusted_miles": 2.0,
            "on_foot_miles": 2.0,
            "parking_risk": 0,
        },
        {
            "field_day_id": "weekday-b",
            "day_type": "weekday",
            "candidate_ids": ["b"],
            "p75_minutes": 30,
            "p90_minutes": 40,
            "stress": 0.4,
            "grade_adjusted_miles": 2.0,
            "on_foot_miles": 2.0,
            "parking_risk": 0,
        },
        {
            "field_day_id": "weekend-a-b",
            "day_type": "weekend",
            "candidate_ids": ["a", "b"],
            "p75_minutes": 50,
            "p90_minutes": 70,
            "stress": 0.5,
            "grade_adjusted_miles": 4.0,
            "on_foot_miles": 4.0,
            "parking_risk": 1,
        },
    ]

    solution = planner.solve_field_day_partition(loops, candidates, {"weekday": 2, "weekend": 1})

    assert solution["success"] is True
    assert solution["field_day_count"] == 1
    assert solution["field_days"][0]["candidate_ids"] == ["a", "b"]
