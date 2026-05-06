from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_sensitivity_gap_targets as targets  # noqa: E402


def test_missing_rows_for_scenario_uses_max_coverage_missing_ids():
    scenario = {
        "scenario": "fixture",
        "max_coverage_solution": {"missing_segment_ids": [1540, 1661]},
    }
    segment_index = {
        1540: {
            "seg_name": "Deer Point Trail 1",
            "trail_name": "Deer Point Trail",
            "official_miles": 1.137,
            "direction": "both",
        },
        1661: {
            "seg_name": "Spring Creek 1",
            "trail_name": "Spring Creek",
            "official_miles": 0.078,
            "direction": "both",
        },
    }

    rows = targets.missing_rows_for_scenario(scenario, segment_index)

    assert [row["seg_id"] for row in rows] == [1540, 1661]
    assert rows[0]["trail_name"] == "Deer Point Trail"
    assert rows[1]["official_miles"] == 0.078


def test_group_missing_by_trail_sums_miles_and_orders_segment_ids():
    rows = [
        {"trail_name": "Spring Creek", "official_miles": 0.5, "seg_id": 1662},
        {"trail_name": "Deer Point Trail", "official_miles": 1.137, "seg_id": 1540},
        {"trail_name": "Spring Creek", "official_miles": 0.078, "seg_id": 1661},
    ]

    grouped = targets.group_missing_by_trail(rows)

    assert grouped == [
        {
            "trail_name": "Deer Point Trail",
            "segment_count": 1,
            "official_miles": 1.14,
            "segment_ids": [1540],
        },
        {
            "trail_name": "Spring Creek",
            "segment_count": 2,
            "official_miles": 0.58,
            "segment_ids": [1661, 1662],
        },
    ]
