from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_strict_profile_swap_audit as swap_audit  # noqa: E402


def test_build_swap_row_reports_coverage_loss_tradeoff():
    target_ids = [1, 2, 3]
    baseline = {
        "success": True,
        "covered_segment_count": 2,
        "missing_segment_ids": [3],
        "total_p75_minutes": 50,
    }
    forced = {
        "success": True,
        "covered_segment_count": 1,
        "missing_segment_ids": [1, 2],
        "total_p75_minutes": 40,
    }

    row = swap_audit.build_swap_row(
        3,
        official_row={"seg_id": 3, "seg_name": "Trail 3", "official_miles": 1.0},
        target_ids=target_ids,
        baseline_solution=baseline,
        forced_solution=forced,
        option_count=2,
    )

    assert row["classification"] == "coverage_loss_swap"
    assert row["covered_segment_delta"] == -1
    assert row["newly_recovered_missing_segments"] == [3]
    assert row["lost_previously_covered_segments"] == [1, 2]
    assert row["p75_delta_minutes"] == -10


def test_build_swap_row_reports_one_for_one_tradeoff():
    target_ids = [1, 2, 3]
    baseline = {
        "success": True,
        "covered_segment_count": 2,
        "missing_segment_ids": [3],
        "total_p75_minutes": 50,
    }
    forced = {
        "success": True,
        "covered_segment_count": 2,
        "missing_segment_ids": [2],
        "total_p75_minutes": 60,
    }

    row = swap_audit.build_swap_row(
        3,
        official_row={"seg_id": 3, "seg_name": "Trail 3", "official_miles": 1.0},
        target_ids=target_ids,
        baseline_solution=baseline,
        forced_solution=forced,
        option_count=4,
    )

    assert row["classification"] == "one_for_one_swap"
    assert row["covered_segment_delta"] == 0
    assert row["newly_recovered_missing_segments"] == [3]
    assert row["lost_previously_covered_segments"] == [2]
    assert row["p75_delta_minutes"] == 10


def test_build_swap_row_reports_no_strict_candidate():
    row = swap_audit.build_swap_row(
        3,
        official_row={"seg_id": 3, "seg_name": "Trail 3", "official_miles": 1.0},
        target_ids=[1, 2, 3],
        baseline_solution={
            "success": True,
            "covered_segment_count": 2,
            "missing_segment_ids": [3],
            "total_p75_minutes": 50,
        },
        forced_solution={
            "success": False,
            "reason": "required_segments_not_coverable_by_field_day_candidates",
            "missing_required_segment_ids": [3],
        },
        option_count=0,
    )

    assert row["classification"] == "no_strict_field_day_candidate"
    assert row["force_success"] is False
    assert row["force_failure_reason"] == "required_segments_not_coverable_by_field_day_candidates"
