from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_relaxed_drive_solution_quality as quality  # noqa: E402


def test_summarize_days_counts_car_hops_and_long_days():
    days = [
        {"loop_count": 1, "between_drive_minutes": 0, "p90_minutes": 200},
        {"loop_count": 2, "between_drive_minutes": 21, "p90_minutes": 341},
        {"loop_count": 3, "between_drive_minutes": 7, "p90_minutes": 300},
    ]

    summary = quality.summarize_days(days)

    assert summary["field_day_count"] == 3
    assert summary["multi_start_days"] == 2
    assert summary["total_between_drive_minutes"] == 28
    assert summary["max_between_drive_minutes"] == 21
    assert summary["days_with_between_drive_over_20"] == 1
    assert summary["days_with_p90_over_340"] == 1


def test_short_day_keeps_quality_fields():
    day = {
        "field_day_id": "weekday-a",
        "day_type": "weekday",
        "p75_minutes": 100,
        "p90_minutes": 120,
        "loop_count": 2,
        "segment_ids": [1, 2, 3],
        "between_drive_minutes": 5,
        "drive_minutes": 30,
        "loop_ids": ["loop-a", "loop-b"],
    }

    assert quality.short_day(day) == {
        "field_day_id": "weekday-a",
        "day_type": "weekday",
        "p75_minutes": 100,
        "p90_minutes": 120,
        "loop_count": 2,
        "segment_count": 3,
        "between_drive_minutes": 5,
        "drive_minutes": 30,
        "loop_ids": ["loop-a", "loop-b"],
    }
