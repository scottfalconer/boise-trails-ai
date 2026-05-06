from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_repaired_field_day_pack_audit as audit  # noqa: E402


def test_exact_set_cover_selects_lowest_p75_cover():
    candidates = [
        {
            "source": "fixture",
            "candidate_id": "a",
            "trailhead": "TH",
            "segment_ids": [1],
            "door_to_door_p75_minutes": 20,
            "door_to_door_p90_minutes": 30,
            "on_foot_miles": 2.0,
        },
        {
            "source": "fixture",
            "candidate_id": "b",
            "trailhead": "TH",
            "segment_ids": [1, 2],
            "door_to_door_p75_minutes": 25,
            "door_to_door_p90_minutes": 35,
            "on_foot_miles": 3.0,
        },
        {
            "source": "fixture",
            "candidate_id": "c",
            "trailhead": "TH",
            "segment_ids": [2],
            "door_to_door_p75_minutes": 20,
            "door_to_door_p90_minutes": 30,
            "on_foot_miles": 2.0,
        },
    ]

    result, selected = audit.exact_set_cover_candidates(candidates, [1, 2])

    assert result["success"] is True
    assert [candidate["candidate_id"] for candidate in selected] == ["b"]


def test_field_day_partition_packs_two_loops_into_one_day():
    loops = [
        {
            "loop_id": "a",
            "trailhead": "TH A",
            "parking": {"lon": -116.2, "lat": 43.6},
            "door_to_door_p90_minutes": 40,
            "internal_p75_minutes": 30,
            "p90_delta_minutes": 5,
            "on_foot_miles": 2.0,
            "grade_adjusted_miles": 2.0,
            "parking_risk": 1,
        },
        {
            "loop_id": "b",
            "trailhead": "TH B",
            "parking": {"lon": -116.201, "lat": 43.601},
            "door_to_door_p90_minutes": 40,
            "internal_p75_minutes": 30,
            "p90_delta_minutes": 5,
            "on_foot_miles": 2.0,
            "grade_adjusted_miles": 2.0,
            "parking_risk": 1,
        },
    ]
    state = {
        "drive_model": {
            "origin_lon": -116.21,
            "origin_lat": 43.62,
            "straight_line_factor": 1.0,
            "minutes_per_mile": 1.0,
            "minimum_one_way_minutes": 1,
        },
        "availability_model": {"acceptable_inter_trailhead_drive_minutes": 20},
    }
    candidates, blockers = audit.generate_field_day_candidates(
        loops,
        state,
        weekday_bound=100,
        weekend_bound=100,
        max_combo_size=2,
    )

    assert blockers == []
    solution = audit.solve_field_day_partition(loops, candidates, {"weekday": 1, "weekend": 0})

    assert solution["success"] is True
    assert solution["field_day_count"] == 1
    assert solution["field_days"][0]["loop_count"] == 2
