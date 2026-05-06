from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_near_miss_consolidation_probe as probe  # noqa: E402


def test_close_weekend_only_days_sorts_by_excess():
    selected = [
        {"field_day_id": "w2", "day_type": "weekend", "p75_minutes": 100, "p90_minutes": 310, "loop_count": 1, "segment_ids": [2], "loop_ids": ["b"]},
        {"field_day_id": "weekday", "day_type": "weekday", "p75_minutes": 90, "p90_minutes": 120, "loop_count": 1, "segment_ids": [3], "loop_ids": ["c"]},
        {"field_day_id": "w1", "day_type": "weekend", "p75_minutes": 95, "p90_minutes": 294, "loop_count": 1, "segment_ids": [1], "loop_ids": ["a"]},
    ]

    rows = probe.close_weekend_only_days(selected, weekday_bound=292, limit=10)

    assert [row["field_day_id"] for row in rows] == ["w1", "w2"]
    assert rows[0]["minutes_over_weekday_bound"] == 2


def test_short_day_keeps_readable_fields():
    day = {
        "field_day_id": "weekday-a",
        "day_type": "weekday",
        "p75_minutes": 100,
        "p90_minutes": 120,
        "loop_count": 1,
        "segment_ids": [1, 2],
        "loop_ids": ["loop-a"],
        "extra": "ignored",
    }

    assert probe.short_day(day) == {
        "field_day_id": "weekday-a",
        "day_type": "weekday",
        "p75_minutes": 100,
        "p90_minutes": 120,
        "loop_count": 1,
        "segment_count": 2,
        "loop_ids": ["loop-a"],
    }
