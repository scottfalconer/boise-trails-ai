from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_profile_acceptance_audit as audit  # noqa: E402


def test_bound_violations_use_day_type_specific_limits():
    days = [
        {"draft_day_number": 1, "day_type": "weekday", "p90_minutes": 261},
        {"draft_day_number": 2, "day_type": "weekday", "p90_minutes": 260},
        {"draft_day_number": 3, "day_type": "weekend", "p90_minutes": 181},
    ]

    violations = audit.bound_violations(
        days,
        weekday_bound=260,
        weekend_bound=180,
    )

    assert [row["draft_day_number"] for row in violations] == [1, 3]
    assert violations[0]["minutes_over_bound"] == 1
    assert violations[1]["minutes_over_bound"] == 1


def test_profile_match_requires_all_bounds_and_drive_limit():
    assert audit.profile_matches(
        current={"weekday_max_minutes": 260, "weekend_max_minutes": 180, "acceptable_inter_trailhead_drive_minutes": 20},
        candidate={"weekday_bound_minutes": 260, "weekend_bound_minutes": 180, "inter_trailhead_drive_minutes": 20},
    )
    assert not audit.profile_matches(
        current={"weekday_max_minutes": 260, "weekend_max_minutes": 180, "acceptable_inter_trailhead_drive_minutes": 20},
        candidate={"weekday_bound_minutes": 292, "weekend_bound_minutes": 360, "inter_trailhead_drive_minutes": 45},
    )


def test_build_report_keeps_relaxed_plan_unaccepted_when_profile_differs():
    state = {
        "availability_model": {
            "weekday_max_minutes": 260,
            "weekend_max_minutes": 180,
            "acceptable_inter_trailhead_drive_minutes": 20,
            "profile": "historical_p90_plus_logistics",
        }
    }
    draft = {
        "config": {
            "weekday_bound_minutes": 292,
            "weekend_bound_minutes": 360,
            "inter_trailhead_drive_minutes": 45,
        },
        "coverage": {"covered_segment_count": 251, "official_segment_count": 251, "missing_segment_count": 0},
        "time_and_logistics": {"field_day_count": 1, "total_p75_minutes": 200, "days_over_p90_bound": []},
        "field_days": [
            {"draft_day_number": 1, "day_type": "weekday", "p90_minutes": 280, "between_drive_minutes": 25},
        ],
    }
    calendar = {"audit": {"passed": True, "lower_hulls_even_day_violation_count": 0}}
    gpx = {"summary": {"day_level_gpx_ready": True, "day_gpx_count": 31, "day_level_gpx_failed_day_count": 0}}

    report = audit.build_report(state=state, draft=draft, calendar=calendar, gpx_readiness=gpx)

    assert report["summary"]["profile_matches_current_personal_bounds"] is False
    assert report["summary"]["accepted_as_active_personal_plan"] is False
    assert report["summary"]["current_bound_p90_violation_count"] == 1
    assert report["summary"]["inter_trailhead_drive_violation_count"] == 1
