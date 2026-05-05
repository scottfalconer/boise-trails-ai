import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calendar_scheduler.py"


def load_scheduler():
    spec = importlib.util.spec_from_file_location("calendar_scheduler", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outing(
    candidate_id,
    trail_names,
    segment_ids,
    minutes,
    official_miles,
    status="simulated_ready",
    total_on_foot_miles=None,
):
    total_on_foot_miles = total_on_foot_miles if total_on_foot_miles is not None else official_miles + 1
    return {
        "candidate_id": candidate_id,
        "trail_names": trail_names,
        "segment_ids": segment_ids,
        "execution_status": status,
        "simulated_total_minutes": minutes,
        "simulated_efficiency_score": round(official_miles / minutes, 4),
        "legs": [
            {"duration_minutes": 5},
            {"trailhead": "Test Trailhead"},
            {},
            {
                "official_new_miles": official_miles,
                "estimated_total_on_foot_miles": total_on_foot_miles,
                "ascent_ft": 100,
                "descent_ft": 80,
                "grade_adjusted_miles": total_on_foot_miles + 0.1,
                "effort_score": minutes - 10,
            },
            {},
            {"duration_minutes": 5},
        ],
    }


def test_schedule_uses_ready_non_overlapping_outings_within_daily_budget():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("fast", ["Fast"], [1, 2], 70, 4.0),
            outing("overlap", ["Overlap"], [2, 3], 80, 5.0),
            outing("second-day", ["Second"], [4], 70, 3.0),
            outing("too-long", ["Too Long"], [5], 200, 9.0),
            outing("blocked", ["Blocked"], [6], 50, 2.0, status="blocked_by_route_validation"),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-19",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    scheduled_ids = [
        outing_id
        for day in schedule["days"]
        if day["status"] == "scheduled"
        for outing_id in day["outing_ids"]
    ]
    scheduled_segments = [
        segment_id
        for day in schedule["days"]
        if day["status"] == "scheduled"
        for segment_id in day["segment_ids"]
    ]
    assert scheduled_ids == ["overlap", "second-day"]
    assert len(scheduled_segments) == len(set(scheduled_segments))
    assert schedule["summary"]["scheduled_outings"] == 2
    assert schedule["summary"]["scheduled_official_miles"] == 8.0
    assert schedule["summary"]["scheduled_ascent_ft"] == 200
    assert schedule["summary"]["scheduled_descent_ft"] == 160
    assert "fast" in schedule["unscheduled_ready_outing_ids"]


def test_schedule_skips_static_preflight_blocked_routes():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("lower-hulls", ["Lower Hull's Gulch Trail"], [1], 70, 3.0),
            outing("safe", ["Safe Trail"], [2], 80, 2.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-19",
        end_date="2026-06-19",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    day = schedule["days"][0]
    assert day["outing_id"] == "safe"
    assert "lower-hulls" in schedule["preflight_blocked_outing_ids"]


def test_schedule_inserts_recovery_day_after_long_outing():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("long", ["Long"], [1], 250, 8.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-20",
        end_date="2026-06-21",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 300,
            "start_time": "07:00",
            "long_outing_minutes": 240,
            "rest_days_after_long": 1,
        },
    )

    assert schedule["days"][0]["outing_id"] == "long"
    assert schedule["days"][1]["status"] == "recovery"


def test_schedule_preserves_segments_for_higher_value_weekend_route():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("small-weekday", ["Small"], [1], 60, 1.0),
            outing("big-weekend", ["Big"], [1, 2, 3], 180, 6.0),
            outing("other-weekday", ["Other"], [4], 60, 1.5),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-20",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    scheduled_ids = [
        outing_id
        for day in schedule["days"]
        if day["status"] == "scheduled"
        for outing_id in day["outing_ids"]
    ]
    assert "big-weekend" in scheduled_ids
    assert "small-weekday" not in scheduled_ids


def test_schedule_can_pack_multiple_outings_into_one_day_budget():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("morning", ["Morning"], [1], 50, 1.0),
            outing("lunch", ["Lunch"], [2], 45, 1.0),
            outing("too-much", ["Too Much"], [3], 80, 2.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-18",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    day = schedule["days"][0]
    assert set(day["outing_ids"]) == {"morning", "lunch"}
    assert day["simulated_total_minutes"] == 95
    assert day["segment_ids"] == [1, 2]
    assert schedule["summary"]["scheduled_segments"] == 2


def test_schedule_allows_overlap_when_outing_adds_new_segments():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("base", ["Base"], [1, 2], 55, 2.0),
            outing("overlap-adds-one", ["Overlap Adds One"], [2, 3], 55, 2.0),
            outing("small", ["Small"], [4], 55, 1.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-19",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    scheduled_ids = [
        outing_id
        for day in schedule["days"]
        if day["status"] == "scheduled"
        for outing_id in day["outing_ids"]
    ]
    assert "base" in scheduled_ids
    assert "overlap-adds-one" in scheduled_ids
    assert schedule["summary"]["scheduled_segments"] == 4
    overlap_day = next(
        day for day in schedule["days"] if "overlap-adds-one" in day.get("outing_ids", [])
    )
    overlap_outing = next(
        item for item in overlap_day["outings"] if item["outing_id"] == "overlap-adds-one"
    )
    assert overlap_outing["new_segment_ids"] == [3]
    assert overlap_outing["repeat_segment_ids"] == [2]


def test_schedule_respects_max_consecutive_scheduled_days():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("day-1", ["Day 1"], [1], 50, 1.0),
            outing("day-2", ["Day 2"], [2], 50, 1.0),
            outing("day-3", ["Day 3"], [3], 50, 1.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-20",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 120,
            "start_time": "07:00",
            "rest_days_after_long": 0,
            "max_consecutive_scheduled_days": 2,
        },
    )

    statuses = [day["status"] for day in schedule["days"]]
    assert statuses.count("scheduled") == 2
    assert statuses != ["scheduled", "scheduled", "scheduled"]


def test_minimize_overhead_prefers_lower_total_on_foot_miles():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("fast-high-overhead", ["High"], [1, 2], 80, 3.0, total_on_foot_miles=10.0),
            outing("slow-low-overhead", ["Low"], [1, 2], 100, 3.0, total_on_foot_miles=4.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-18",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
            "objective_profile": "minimize_overhead",
        },
    )

    day = schedule["days"][0]
    assert day["outing_ids"] == ["slow-low-overhead"]
    assert schedule["summary"]["scheduled_total_on_foot_miles"] == 4.0


def test_schedule_respects_latest_scheduled_date():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("allowed", ["Allowed"], [1], 50, 1.0),
            outing("would-fit-late", ["Late"], [2], 50, 1.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-19",
        constraints={
            "weekday_max_minutes": 60,
            "weekend_max_minutes": 60,
            "start_time": "07:00",
            "rest_days_after_long": 0,
            "latest_scheduled_date": "2026-06-18",
        },
    )

    assert schedule["days"][0]["status"] == "scheduled"
    assert schedule["days"][1]["status"] == "open"
    assert schedule["summary"]["scheduled_segments"] == 1


def test_schedule_records_normal_cap_exceptions_without_blocking_hard_budget():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("exception", ["Exception"], [1], 150, 2.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-18",
        constraints={
            "weekday_max_minutes": 180,
            "weekday_normal_max_minutes": 120,
            "weekend_max_minutes": 240,
            "start_time": "07:00",
            "rest_days_after_long": 0,
        },
    )

    day = schedule["days"][0]
    assert day["status"] == "scheduled"
    assert day["normal_available_minutes"] == 120
    assert day["requires_normal_cap_exception"] is True
    assert day["normal_cap_exception_minutes"] == 30
    assert schedule["summary"]["normal_cap_exception_days"] == 1


def test_required_completion_segments_is_recorded_in_optimizer_metadata():
    scheduler = load_scheduler()
    execution = {
        "outings": [
            outing("only", ["Only"], [1, 2], 50, 2.0),
        ]
    }

    schedule = scheduler.build_calendar_schedule(
        execution,
        start_date="2026-06-18",
        end_date="2026-06-18",
        constraints={
            "weekday_max_minutes": 120,
            "weekend_max_minutes": 120,
            "start_time": "07:00",
            "rest_days_after_long": 0,
            "required_completion_segments": 2,
        },
    )

    assert schedule["optimizer"]["required_completion_segments"] == 2
    assert schedule["summary"]["scheduled_segments"] == 2
