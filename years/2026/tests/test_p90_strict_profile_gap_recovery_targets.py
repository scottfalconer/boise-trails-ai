from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_strict_profile_gap_recovery_targets as targets  # noqa: E402


def test_field_days_covering_segment_sorts_by_p75_then_p90():
    days = [
        {"field_day_id": "slow", "segment_ids": [1], "p75_minutes": 90, "p90_minutes": 100},
        {"field_day_id": "fast", "segment_ids": [1, 2], "p75_minutes": 60, "p90_minutes": 90},
        {"field_day_id": "other", "segment_ids": [2], "p75_minutes": 10, "p90_minutes": 20},
    ]

    result = targets.field_days_covering_segment(days, 1)

    assert [day["field_day_id"] for day in result] == ["fast", "slow"]


def test_classify_missing_segment_without_candidate_is_structural():
    row = targets.classify_missing_segment(
        1656,
        official_row={"seg_id": 1656, "seg_name": "Shingle Creek Trail 1", "trail_name": "Shingle Creek Trail", "official_miles": 4.76},
        field_days=[],
        selected_segment_ids=set(),
    )

    assert row["classification"] == "no_strict_field_day_candidate"
    assert row["best_option"] is None


def test_classify_missing_segment_with_candidate_is_schedule_tradeoff():
    row = targets.classify_missing_segment(
        1540,
        official_row={"seg_id": 1540, "seg_name": "Deer Point Trail 1", "trail_name": "Deer Point Trail", "official_miles": 1.14},
        field_days=[
            {"field_day_id": "deer", "day_type": "weekend", "segment_ids": [1540, 1], "p75_minutes": 120, "p90_minutes": 150, "p90_bound_minutes": 180, "on_foot_miles": 4.0, "loop_count": 1}
        ],
        selected_segment_ids={1, 2, 3},
    )

    assert row["classification"] == "strict_candidate_exists_but_not_selected"
    assert row["best_option"]["field_day_id"] == "deer"
    assert row["best_option"]["new_missing_segments_recovered"] == 1


def test_summarize_counts_classifications():
    summary = targets.summarize_rows(
        [
            {"classification": "a", "official_miles": 1.0},
            {"classification": "a", "official_miles": 2.0},
            {"classification": "b", "official_miles": 3.0},
        ]
    )

    assert summary["missing_segment_count"] == 3
    assert summary["classification_counts"] == {"a": 2, "b": 1}
    assert summary["missing_official_miles"] == 6.0
