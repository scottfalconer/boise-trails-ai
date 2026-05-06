from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_completion_decision_gate as gate  # noqa: E402


def completion_audit():
    return {
        "metrics": {
            "wide_joint_optimizer_strict_max_coverage_segments": 219,
            "wide_joint_optimizer_strict_max_coverage_missing_segments": 32,
            "strict_profile_max_coverage_plan_missing_official_miles": 41.98,
            "wide_joint_optimizer_strict_missing_segment_ids": [1656],
            "wide_joint_optimizer_shingle_292_relaxed_min_field_days": 43,
            "wide_joint_optimizer_shingle_292_min_weekdays_with_9_weekends": 37,
            "near_miss_relaxed_drive_45_neighbor_40_feasible": True,
        }
    }


def shingle_probe():
    return {
        "summary": {
            "strict_success": False,
            "anchor_count": 74,
            "under_bound_graph_track_count": 0,
            "best_field_ready_anchor": "Dry Creek / Sweet Connie roadside parking",
            "best_field_ready_p90_minutes": 292,
            "best_field_ready_p75_minutes": 260,
            "best_field_ready_on_foot_miles": 11.88,
            "minutes_over_bound_best_field_ready": 32,
        }
    }


def profile_acceptance():
    return {
        "summary": {
            "accepted_as_active_personal_plan": False,
            "current_bound_p90_violation_count": 22,
            "inter_trailhead_drive_violation_count": 1,
            "max_minutes_over_current_p90_bound": 179,
            "max_minutes_over_current_inter_trailhead_drive_bound": 7,
        }
    }


def relaxed_draft():
    return {
        "config": {
            "weekday_bound_minutes": 292,
            "weekend_bound_minutes": 360,
            "inter_trailhead_drive_minutes": 45,
        },
        "summary": {
            "field_day_count": 31,
            "total_p75_minutes": 7684,
        },
    }


def test_build_report_marks_goal_not_complete_and_lists_decisions():
    report = gate.build_report(
        completion_audit=completion_audit(),
        shingle_probe=shingle_probe(),
        profile_acceptance=profile_acceptance(),
        relaxed_draft=relaxed_draft(),
    )

    assert report["verdict"] == "decision_required_not_complete"
    assert report["strict_status"]["active_strict_completion_passed"] is False
    assert report["shingle_status"]["strict_success"] is False
    assert report["relaxed_completion_draft_status"]["full_clear_draft_exists"] is True
    assert [row["option"] for row in report["decision_options"]] == [
        "keep_active_strict_bounds",
        "accept_shingle_only_292_exception",
        "accept_relaxed_292_360_drive45_profile",
        "field_calibrate_or_redesign_shingle",
    ]


def test_shingle_only_exception_is_not_enough_by_itself():
    report = gate.build_report(
        completion_audit=completion_audit(),
        shingle_probe=shingle_probe(),
        profile_acceptance=profile_acceptance(),
        relaxed_draft=relaxed_draft(),
    )

    option = next(row for row in report["decision_options"] if row["option"] == "accept_shingle_only_292_exception")

    assert option["completion_plan_available"] is False
    assert "43 field days" in option["result"]
